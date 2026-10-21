"""Default all-residue GQT: pKa on every residue, titratable gate, burial and residue contact edges.

Model (``models/ogqt-allres15-siamese-50``, training run
``control-v2-50-relations-4-1-allres-cut15-cysmask-gate01-ressiamese-redge01-s17``): backbone and sequence only.
Two relational residue layers over the 20 A CA graph (4 relation maps, then 1), backbone orientation edges and
dihedrals; then every residue becomes a token (residue embedding + amino-acid embedding) and passes a query block and
a token-to-token block restricted to CA-CA <= 15 A. Each token has three outputs: side chain, N-terminus, C-terminus.

Heads on the final tokens:
  pKa        8 tanh(head) is the shift from the group reference pKa (PKPDB_PK_MOD); groups D E H C Y K R (side chain),
             N-terminus, C-terminus. Supervised only on labelled titratable sites (Arg never; bonded Cys excluded).
  gate       aux_gate logit per output: is this group titratable (D/E/H/C/Y/K/R minus disulfide Cys; chain termini).
  burial     aux_burial logit per residue: 1 - clip(RSA) in that structure (bound or free state).
  edges      aux_redge: symmetric pair logit for residues on different chains being in contact (<= 10 A heavy atoms).

Graphs are production oGQT residue graphs (``pkanet.features`` or ``pkanet.differentiable``) with the backbone-v1
table appended to the node features: N, CA, C, O xyz, chain index, sequence index within the chain. The free state is
the same graph with every edge between chains masked (``state_graph``). Indexed attention uses
``pkanet.attention_backend`` (Triton on CUDA, plain JAX elsewhere).
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxpropka.parameters import ALPHABET, GROUP_AA, GROUPS
from .model import PKPDB_PK_MOD, linear, norm

DEFAULT_EXPORT = "ogqt-allres15-siamese-50"
TABLE_WIDTH = 14
FIELDS = ("nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch", "query_residue", "query_group")
_SIDE_GROUP = np.full(20, -1, np.int32)
_SIDE_GROUP[np.asarray(GROUP_AA)] = np.arange(len(GROUP_AA))


def is_allres_export(path):
    info = json.loads((Path(path) / "model.json").read_text())
    return "allres" in info.get("protocol", {}).get("production_variant", "")


def load(path):
    """(params, info) for an all-residue export; checks checksum, parameter count and heads."""
    path = Path(path); info = json.loads((path / "model.json").read_text())
    if hashlib.sha256((path / "params.npz").read_bytes()).hexdigest() != info["params_sha256"]:
        raise ValueError("Checkpoint SHA256 does not match model.json")
    variant = info.get("protocol", {}).get("production_variant", "")
    if "allres" not in variant:
        raise ValueError(f"Not an all-residue export: {path}")
    tree = {}; count = 0
    with np.load(path / "params.npz", allow_pickle=False) as data:
        for name in data.files:
            array = data[name]
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid checkpoint parameter {name}")
            count += array.size; node = tree; parts = name.split("/")
            for part in parts[:-1]: node = node.setdefault(part, {})
            node[parts[-1]] = jnp.asarray(array)
    if count != info["parameter_count"]:
        raise ValueError("Checkpoint parameter count mismatch")

    def lists(node):
        if not isinstance(node, dict): return node
        if node and all(k.isdigit() for k in node): return [lists(node[str(i)]) for i in range(len(node))]
        return {k: lists(v) for k, v in node.items()}
    params = lists(tree)
    for head in ("head", "aa_types", "terminus_maps", "aux_burial", "aux_gate", "aux_redge"):
        if head not in params: raise ValueError(f"Export lacks {head}")
    return params, info


def embedding_function(info):
    """embeddings(params, graph) -> (tokens (3N, W), query rows), as training's model_inputs built it."""
    from pkatrain.production_inputs import input_functions
    from pkatrain.production_position import annotate, functions
    arch = info["architecture"]; variant = info["protocol"]["production_variant"]
    if arch.get("orientation_edges") != 13 or arch.get("dihedral_features") != 6 or "node_dim" in arch:
        raise ValueError("Expected the backbone-only orientation + dihedral all-residue architecture")
    embeddings, _ = input_functions(functions(variant), orientation=True, table=True, dihedrals=True)
    return lambda params, graph: embeddings(params, annotate(graph, variant))


def outputs(params, embeddings, graph):
    """Per-structure outputs (unbatched graph): query shift / pKa, and per residue (N, 3) shift, pKa, gate logits
    (side chain, N-terminus, C-terminus), (N,) burial logit and (N, W) normalised edge embedding."""
    tokens, query = embeddings(params, graph); n = graph["nodes"].shape[0]
    shift_rows = 8 * jnp.tanh(linear(params["head"], tokens)[:, 0])
    reference = jnp.asarray(PKPDB_PK_MOD)
    side = jnp.asarray(_SIDE_GROUP)[jnp.argmax(graph["nodes"][:, :20], axis=1)]
    groups = jnp.stack((side, jnp.full(n, 7), jnp.full(n, 8)), axis=1)
    shift = shift_rows.reshape(3, n).T
    pka = jnp.where(groups >= 0, shift + reference[jnp.maximum(groups, 0)], jnp.nan)
    return {"query_shift": shift_rows[query], "query_pka": shift_rows[query] + reference[graph["query_group"]],
            "shift": shift, "pka": pka, "gate_logit": linear(params["aux_gate"], tokens)[:, 0].reshape(3, n).T,
            "burial_logit": linear(params["aux_burial"], tokens[:n])[:, 0],
            "edge_embedding": norm(params["aux_redge"]["norm"], tokens[:n])}


def pair_logits(params, z, a, b):
    """Contact logits for residue pairs (a[k], b[k]) from edge embeddings z (training's aux_redge scorer)."""
    e = params["aux_redge"]; za, zb = z[a], z[b]
    return linear(e["out"], jax.nn.gelu(linear(e["hidden"], jnp.concatenate((za * zb, jnp.abs(za - zb)), axis=-1))))[..., 0]


def backbone_table(backbone, chain):
    """backbone-v1 node table: N, CA, C, O xyz (12), chain index, sequence index within the chain."""
    chain = np.asarray(chain); seq = np.zeros(len(chain), np.int32); counts = {}
    for i, c in enumerate(chain): seq[i] = counts.get(int(c), 0); counts[int(c)] = seq[i] + 1
    return np.concatenate((np.asarray(backbone, np.float32).reshape(len(chain), 12), chain[:, None], seq[:, None]), axis=1).astype(np.float32)


def state_graph(graph, state):
    """Bound: unchanged. Free: residue-graph edges between chains masked (edge column 19 is the same-chain flag)."""
    if state == "bound": return graph
    if state != "free": raise ValueError("state must be bound or free")
    return {**graph, "edge_mask": graph["edge_mask"] & (graph["edge"][..., 19] > 0.5)}


class AllResidueStructureFeatures:
    """Wrap pkanet.differentiable.StructureFeatures for the all-residue model: appends the backbone table and applies
    the free-state chain isolation. Continuous in the backbone; the template fixes chains and topology."""

    def __init__(self, base, state="bound"):
        if state not in ("bound", "free"): raise ValueError("state must be bound or free")
        if base.sidechains: raise ValueError("The default all-residue model uses backbone-only features")
        self.base = base; self.state = state

    def __call__(self, sequence, backbone, sidechains):
        g = self.base(sequence, backbone, sidechains); chain = jnp.asarray(self.base.chain)
        table = jnp.concatenate((backbone.reshape((-1, 12)), chain[:, None].astype(backbone.dtype),
                                 jnp.asarray(self.base.sequence_index)[:, None].astype(backbone.dtype)), axis=1)
        g = {**{k: g[k] for k in FIELDS}, "nodes": jnp.concatenate((g["nodes"], table), axis=1)}
        return state_graph(g, self.state)


# ---------------------------------------------------------------- batched file inference (ogqt predict)
def capacities(dimensions):
    """Power-of-two (nodes, neighbour slots, queries) capacities, reused across requests."""
    n, k, q = (int(x) for x in dimensions[:3])
    return tuple(1 << (max(v, m) - 1).bit_length() for v, m in ((n, 32), (k, 8), (q, 16)))


def pad(graph, cap):
    n, k, q = cap
    shapes = dict(nodes=(n, 24 + TABLE_WIDTH), node_mask=(n,), neighbors=(n, k), edge=(n, k, 20), edge_mask=(n, k),
                  switch=(n, k), query_residue=(q,), query_group=(q,))
    out = {}
    for name, shape in shapes.items():
        value = graph[name]
        if value.ndim != len(shape) or any(a > b for a, b in zip(value.shape, shape)):
            raise ValueError(f"Bucket overflow for {name}: {value.shape} > {shape}; no truncation")
        padded = np.zeros(shape, value.dtype); padded[tuple(slice(0, s) for s in value.shape)] = value; out[name] = padded
    return out


def graph_from_features(graph):
    """Cached pkanet.features graph (schema v2: backbone and chain_index stored) -> all-residue model graph."""
    if "backbone" not in graph:
        raise ValueError("Feature cache lacks backbone coordinates (schema v1); prepare the structures again")
    out = {k: graph[k] for k in FIELDS}
    out["nodes"] = np.concatenate((graph["nodes"], backbone_table(graph["backbone"], graph["chain_index"])), axis=1)
    return out


class AllResiduePredictor:
    """Load once; batch structures by padded shape; jit per (capacity, state).

    predict(records, states) yields (input index, {state: outputs}) with arrays trimmed to the structure, plus, when
    both states are requested and the structure has several chains, the free-state interface score per residue: the
    maximum edge logit over residues on other chains.
    """

    def __init__(self, model, *, device="cpu", backend="jax", batch_size=4, compilation_cache=None):
        from .attention_backend import set_backend
        from .compilation import configure_compilation_cache
        if batch_size < 1: raise ValueError("batch_size must be positive")
        if backend not in ("jax", "triton"): raise ValueError("backend must be jax or triton")
        self.compilation_cache = None if compilation_cache is False else configure_compilation_cache(compilation_cache)
        self.device = jax.devices(device)[0]
        if backend == "triton" and self.device.platform not in ("gpu", "cuda"): raise ValueError("Triton requires a GPU")
        set_backend(backend)
        params, self.model_info = load(model)
        self.params = jax.device_put(params, self.device); self.batch_size = batch_size; self.backend = backend
        embeddings = embedding_function(self.model_info)
        self._batch = {state: jax.jit(jax.vmap(lambda p, g, s=state: outputs(p, embeddings, state_graph(g, s)), in_axes=(None, 0)))
                       for state in ("bound", "free")}
        self._pairs = jax.jit(lambda p, z, a, b: pair_logits(p, z, a, b))
        self.executables = {}; self.last_stats = {}

    def predict(self, records, states=("bound",)):
        from .features import load_features
        groups = defaultdict(list)
        for i, (_, metadata) in enumerate(records): groups[capacities(metadata["dimensions"])].append(i)
        specs = [(cap, idx[s:s + self.batch_size]) for cap, idx in sorted(groups.items()) for s in range(0, len(idx), self.batch_size)]
        stats = dict(batches=0, structures=0, compiled_shapes=0, compile_seconds=0., execution_seconds=0., backend=self.backend,
                     device=str(self.device), states=list(states), compilation_cache=str(self.compilation_cache) if self.compilation_cache else None)
        self.last_stats = stats

        def load_batch(spec):
            cap, indices = spec; items = [pad(graph_from_features(load_features(records[i][0])[0]), cap) for i in indices]
            items += [items[-1]] * (self.batch_size - len(items))
            return cap, indices, jax.tree.map(lambda *x: np.stack(x), *items)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(load_batch, specs[0]) if specs else None
            for position in range(len(specs)):
                cap, indices, host = pending.result()
                pending = pool.submit(load_batch, specs[position + 1]) if position + 1 < len(specs) else None
                batch = jax.device_put(host, self.device); result = {}
                for state in states:
                    key = (cap, state)
                    if key not in self.executables:
                        began = time.perf_counter(); self.executables[key] = self._batch[state].lower(self.params, batch).compile()
                        stats["compile_seconds"] += time.perf_counter() - began; stats["compiled_shapes"] += 1
                    began = time.perf_counter(); result[state] = jax.device_get(self.executables[key](self.params, batch))
                    stats["execution_seconds"] += time.perf_counter() - began
                stats["batches"] += 1; stats["structures"] += len(indices)
                for lane, index in enumerate(indices):
                    n, _, q = records[index][1]["dimensions"][:3]
                    out = {"query_residue": np.asarray(host["query_residue"][lane, :q]), "query_group": np.asarray(host["query_group"][lane, :q])}
                    for state, value in result.items():
                        o = {k: np.asarray(v[lane]) for k, v in value.items()}
                        o = {**{k: o[k][:q] for k in ("query_shift", "query_pka")}, **{k: o[k][:n] for k in ("shift", "pka", "gate_logit", "burial_logit", "edge_embedding")}}
                        if not np.isfinite(o["query_pka"]).all(): raise FloatingPointError(f"Nonfinite predictions for input {index}")
                        out[state] = o
                    if "free" in states:
                        chain = np.asarray(host["nodes"][lane, :n, 24 + 12]).astype(int); score = np.full(n, np.nan, np.float32)
                        if len(np.unique(chain)) > 1:
                            a, b = np.nonzero(chain[:, None] < chain[None, :]); z = jax.device_put(out["free"]["edge_embedding"], self.device)
                            logits = np.asarray(jax.device_get(self._pairs(self.params, z, a.astype(np.int32), b.astype(np.int32))))
                            score = np.full(n, -np.inf, np.float32); np.maximum.at(score, a, logits); np.maximum.at(score, b, logits)
                        out["interface_score"] = score
                    yield index, out


def residue_rows(metadata, out, states):
    """JSON-ready per-residue records from AllResiduePredictor outputs."""
    rows = []; nterm, cterm = metadata["terminal_flags"]
    for i, key in enumerate(metadata["residues"]):
        aa = ALPHABET[metadata["residue_types"][i]]; group = _SIDE_GROUP[metadata["residue_types"][i]]
        row = dict(**key, residue=aa)
        for state in states:
            o = out[state]; gate = 1 / (1 + np.exp(-o["gate_logit"][i])); suffix = "" if len(states) == 1 else f"_{state}"
            outputs_ = {"side_chain": (0, group >= 0, GROUPS[group] if group >= 0 else None), "n_terminus": (1, bool(nterm[i]), "NTR"),
                        "c_terminus": (2, bool(cterm[i]), "CTR")}
            row[f"burial{suffix}"] = float(1 / (1 + np.exp(-o["burial_logit"][i])))
            for name, (col, present, label) in outputs_.items():
                if not present: continue
                row.setdefault(name, {"group": label})
                row[name][f"pka{suffix}"] = float(o["pka"][i, col]); row[name][f"gate{suffix}"] = float(gate[col])
        if "interface_score" in out and np.isfinite(out["interface_score"][i]): row["interface_logit"] = float(out["interface_score"][i])
        rows.append(row)
    return rows
