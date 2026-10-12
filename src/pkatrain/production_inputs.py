"""Input variants for backbone-only production oGQT (2026-10-12; plan after the scaling runs).

1. Side-chain views (--sidechain-view, with --sidechains): which residues' side-chain features (production_sidechains)
   the model sees. Applied on the device to node columns 24: of each graph.
   - "titratable": only residues of the titratable types D/E/H/C/Y/K/R (jaxpropka GROUP_AA).
   - "non-titratable": every other residue.
   - "dropout": training hides all side-chain features of a structure with probability --sidechain-drop (one draw per
     PINDER complex, shared by its AB and free branches, from the engine's per-structure key); validation and
     predictions never see side chains, so selection is backbone-only.
2. Residue-edge orientation (--orientation): the residue graph's 20 edge features (pkanet.graph.geometry: Ca distance
   RBF, neighbour direction in the residue's own frame, same chain) gain the 13 orientation features that only the
   site-site edges had (site_graph_data.build_site_graph): the direction back to the residue in the neighbour's
   frame (3), the relative rotation F_i^T F_j (9) and the signed same-chain sequence separation clip(ds / 32, -1, 1)
   (1). They are computed on the device from a per-node backbone table (backbone-v1: N, CA, C, O xyz, chain index,
   sequence index), appended to the node features by the loader and stripped before the embedding. The residue blocks'
   and the site query block's edge projections gain 13 rows (drawn N(0, 1) / sqrt(33)); the rest of the forward pass
   is pkanet.site_model's.

3. Titratable side-chain geometry auxiliary (--geometry-weight W, with --sidechains and --sidechain-view dropout):
   the hidden side chains become targets for two heads on the final site embedding, for each D/E/H/C/Y/K/R query
   site (termini excluded): aux_geom_pos, the centroid of its functional atoms (D OD1/OD2, E OE1/OE2, H ND1/NE2,
   C SG, Y OH, K NZ, R NE/NH1/NH2) in its own backbone frame (A / 10; mean squared error), and aux_partner, a
   salt-bridge score clip(exp(-(d - 4) / 2), 0, 1) with d the distance to the nearest functional centroid of opposite
   charge (D/E/C/Y -1, H/K/R +1) among its residue-graph neighbours in that branch (cross-entropy). Needs the
   backbone-v1 table (global centroids), loaded with or without --orientation.

  python -m pkatrain.production_inputs build-backbone {pinder,pkpdb,benchmark-val} [--workers 72] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest
from .production_graphs import ProductionStore, output, read

BACKBONE_TABLE = "backbone-v1"
BACKBONE_WIDTH = 14  # N, CA, C, O xyz (12), chain index, sequence index within the chain
ORIENTATION_EDGE = 13
ORIENTATION_SALT = 6151
DROP_SALT = 5381
VIEWS = ("all", "titratable", "non-titratable", "dropout")
GEOMETRY_SALT = 8191
FUNCTIONAL = {"D": ("OD1", "OD2"), "E": ("OE1", "OE2"), "H": ("ND1", "NE2"), "C": ("SG",), "Y": ("OH",), "K": ("NZ",), "R": ("NE", "NH1", "NH2")}
ACIDS = "DECY"
PARTNER_CONTACT_A, PARTNER_DECAY_A = 4.0, 2.0


# ---- backbone table ----------------------------------------------------------------------------------------------

def _residues(dataset, root, cid):
    """(backbone (n, 4, 3), chain index (n,), identity (n,)) in store node order."""
    if dataset == "pkpdb":
        return _pkpdb_chains(root, cid)
    from jaxpropka.topology import load_topology
    if dataset == "pinder":
        from .gqt_paired_pinder import _read_cif_gz
        from .production_graphs import PINDER, source
        topology = load_topology(_read_cif_gz(source(root, f"{PINDER}/entries/{cid}") / "AB.cif.gz"), gap_policy="cap", freeze_disulfides=True)
    else:
        from pkabench.prep import read_cif
        from .production_graphs import BENCHMARK_SOURCE, _local
        r = read(root / BENCHMARK_SOURCE / "records" / f"{cid}.json"); cif = _local(root, r["structures"]["AB"])
        if digest(cif) != r["structure_sha256"]["AB"]: raise AssertionError((cid, "structure hash"))
        topology = load_topology(read_cif(cif), gap_policy="cap", freeze_disulfides=True)
    return np.asarray(topology.backbone), np.asarray(topology.chain_index), np.asarray(topology.native_index)


def _pkpdb_chains(root, cid):
    """As production_sidechains._pkpdb_atoms (the store's pKPDB residue filter), keeping each residue's chain."""
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from pkabench.conformers import resolve
    from pkabench.pkpdb_pilot_refs import cif
    from jaxpropka.parameters import THREE_TO_INDEX
    from pkabench.prep import CANONICAL
    from .production_graphs import PKPDB, PKPDB_STRUCTURES, source
    entry = source(root, f"{PKPDB}/entries/{cid}"); receipt = read(entry / "receipt.json"); defects = read(entry / "defects.json")
    path = source(root, f"{PKPDB_STRUCTURES}/{cid[1:3]}/{cid}.cif.gz")
    if digest(path) != receipt["source_sha256"]: raise AssertionError((cid, "source structure hash"))
    selected = [row["chain"] for row in defects["sequences"]]
    file, _ = resolve(cif(path), selected)
    label = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=False)
    author = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True)
    working = label.copy(); working.res_id = author.res_id.copy(); working.ins_code = author.ins_code.copy()
    keep = np.isin(working.res_name, list(CANONICAL)) & np.isin(working.chain_id, selected) & ~np.isin(np.char.upper(working.element), ["H", "D"])
    starts = struc.get_residue_starts(working, add_exclusive_stop=True); backbone = []; chain = []; identity = []
    for s, e in zip(starts[:-1], starts[1:]):
        if not keep[s]: continue
        a = working[s:e]; names = {str(n): i for i, n in enumerate(a.atom_name)}
        backbone.append(np.stack([a.coord[names[name]] if name in names else a.coord[names["C"]] for name in ("N", "CA", "C", "O")]))
        chain.append(selected.index(str(a.chain_id[0]))); identity.append(THREE_TO_INDEX[str(a.res_name[0])])
    return np.asarray(backbone), np.asarray(chain), np.asarray(identity)


def _sequence_index(chain):
    out = np.zeros(len(chain), np.int32); counters = {}
    for i, name in enumerate(chain): out[i] = counters.get(int(name), 0); counters[int(name)] = out[i] + 1
    return out


def _backbone_one(args):
    root, dataset, cid, store_path = args
    backbone, chain, identity = _residues(dataset, Path(root), cid)
    store = ProductionStore(Path(store_path)); raw = store.raw(cid); store.close()
    nodes, edge, neighbors, mask = (np.asarray(raw[k]) for k in ("nodes", "edge", "neighbors", "edge_mask"))
    if len(backbone) != len(nodes): raise AssertionError((cid, "node count differs from the store", len(backbone), len(nodes)))
    if not np.array_equal(identity, nodes[:, :20].argmax(1)): raise AssertionError((cid, "residue identity differs from the store"))
    features = np.concatenate((backbone.reshape(len(backbone), 12), chain[:, None], _sequence_index(chain)[:, None]), axis=1).astype(np.float32)
    # the stored edge features recomputed from the table: own-frame direction (16:19) and same-chain flag (19)
    direction, same = _stored_edges(features, neighbors)
    # and the 13 orientation features against the stored site-site edges (same definitions, site pairs only)
    sr = np.asarray(raw["site_residue"]); smask = np.asarray(raw["site_edge_mask"]); sedge = np.asarray(raw["site_edge"])
    oriented = orientation_edges(features, sr[np.asarray(raw["site_neighbors"])], rows=sr, xp=np)
    stored = np.concatenate((sedge[..., 19:31], sedge[..., 33:34]), axis=-1)
    return cid, features, {"nodes": len(nodes), "max_abs_direction_diff": float(np.max(np.abs(direction - edge[..., 16:19])[mask], initial=0.0)),
                           "same_chain_disagree": int(np.sum((same != (edge[..., 19] > 0.5))[mask])),
                           "max_abs_orientation_diff": float(np.max(np.abs(oriented - stored)[smask], initial=0.0))}


def _stored_edges(features, neighbors):
    """Own-frame neighbour direction and same-chain flag recomputed as pkanet.graph.geometry (table check)."""
    n, ca, c = features[:, 0:3], features[:, 3:6], features[:, 6:9]
    x = c - ca; xn = np.linalg.norm(x, axis=-1, keepdims=True); x = x / np.maximum(xn, 1e-6)
    y = n - ca; y = y - (y * x).sum(-1, keepdims=True) * x; yn = np.linalg.norm(y, axis=-1, keepdims=True); y = y / np.maximum(yn, 1e-6)
    valid = (xn[:, 0] > 1e-5) & (yn[:, 0] > 1e-5); frames = np.stack((x, y, np.cross(x, y)), axis=-1)
    delta = ca[neighbors] - ca[:, None]; distance = np.sqrt((delta ** 2).sum(-1) + 1e-8)
    direction = np.einsum("nkj,njl->nkl", delta, frames) / distance[..., None] * valid[:, None, None]
    same = features[neighbors, 12] == features[:, None, 12]
    return direction, same


def build_backbone(root, dataset, workers=72, limit=None):
    root = Path(root); manifest = read(output(root, dataset) / "manifest-v1.json"); store = manifest["store"]
    out = output(root, dataset) / (BACKBONE_TABLE + ("-smoke" if limit else "")); out.mkdir(parents=True, exist_ok=True); began = time.time()
    ids = [r["id"] for r in manifest["records"]][:limit]
    with ProcessPoolExecutor(workers) as ex: results = list(ex.map(_backbone_one, [(root, dataset, cid, store) for cid in ids], chunksize=4))
    offsets = np.cumsum([0] + [len(f) for _, f, _ in results]).astype(np.int64)
    np.save(out / "features.npy", np.concatenate([f for _, f, _ in results])); np.save(out / "offsets.npy", offsets)
    atomic_json(out / "ids.json", ids); checks = [c for _, _, c in results]
    summary = {"version": BACKBONE_TABLE, "dataset": dataset, "definition": "per node: N, CA, C, O xyz (A; missing O as C), chain index, sequence index within the chain; float32",
               "structures": len(ids), "nodes": int(offsets[-1]), "manifest_sha256": digest(output(root, dataset) / "manifest-v1.json"),
               "max_abs_direction_diff_vs_store_edges": max(c["max_abs_direction_diff"] for c in checks),
               "same_chain_disagreements_vs_store_edges": sum(c["same_chain_disagree"] for c in checks),
               "max_abs_orientation_diff_vs_store_site_edges": max(c["max_abs_orientation_diff"] for c in checks),
               "files": {f: digest(out / f) for f in ("features.npy", "offsets.npy", "ids.json")}, "seconds": round(time.time() - began, 1)}
    atomic_json(out / "verification.json", summary); return summary


class BackboneTable:
    """Per-node backbone table by structure id (memory-mapped float32)."""

    def __init__(self, root, dataset):
        path = output(root, dataset) / BACKBONE_TABLE; self.features = np.load(path / "features.npy", mmap_mode="r")
        self.offsets = np.load(path / "offsets.npy"); self.index = {cid: i for i, cid in enumerate(read(path / "ids.json"))}
        self.verification_sha256 = digest(path / "verification.json")

    def __call__(self, cid):
        i = self.index[cid]; return np.asarray(self.features[self.offsets[i]:self.offsets[i + 1]], np.float32)


class ConcatTable:
    """Several per-node tables appended in order (the loader's `sidechains` slot)."""

    def __init__(self, *tables):
        self.tables = tables; self.verification_sha256 = "+".join(t.verification_sha256 for t in tables)

    def __call__(self, cid): return np.concatenate([t(cid) for t in self.tables], axis=1)


# ---- model side --------------------------------------------------------------------------------------------------

def orientation_edges(table, neighbors, rows=None, xp=None):
    """(R, K, 13) for residues `rows` (default all) and their `neighbors` (R, K): direction to i in j's frame,
    F_i^T F_j, signed same-chain sequence separation / 32 (clipped); zero where a frame is degenerate. The definitions
    of site_graph_data.build_site_graph (site_edge columns 19:31 and 33). xp: jax.numpy (default) or numpy."""
    if xp is None: import jax.numpy as xp
    n, ca, c = table[:, 0:3], table[:, 3:6], table[:, 6:9]; chain = table[:, 12]; sequence = table[:, 13]
    x = c - ca; xn = xp.linalg.norm(x, axis=-1, keepdims=True); x = x / xp.maximum(xn, 1e-6)
    y = n - ca; y = y - (y * x).sum(-1, keepdims=True) * x; yn = xp.linalg.norm(y, axis=-1, keepdims=True); y = y / xp.maximum(yn, 1e-6)
    valid = (xn[:, 0] > 1e-5) & (yn[:, 0] > 1e-5); frames = xp.stack((x, y, xp.cross(x, y)), axis=-1)
    if rows is None: rows = xp.arange(table.shape[0])
    delta = ca[neighbors] - ca[rows][:, None]; distance = xp.sqrt((delta ** 2).sum(-1) + 1e-8)
    fi = frames[rows]; fj = frames[neighbors]; vi = valid[rows]; vj = valid[neighbors]
    direction_j = xp.einsum("nkc,nkcl->nkl", -delta, fj) / distance[..., None] * vj[..., None]
    rotation = xp.matmul(xp.swapaxes(fi[:, None], -1, -2), fj).reshape(tuple(neighbors.shape) + (9,)) * (vi[:, None] & vj)[..., None]
    same = chain[neighbors] == chain[rows][:, None]
    separation = xp.where(same, xp.clip((sequence[neighbors] - sequence[rows][:, None]) / 32.0, -1.0, 1.0), 0.0)
    return xp.concatenate((direction_j, rotation, separation[..., None]), axis=-1)


def _functional():
    """(20, 32) functional-atom slot weights per residue type, (20,) charge class."""
    from jaxpropka.parameters import ALPHABET
    from pkanet.graph import SIDECHAIN_ATOMS
    weights = np.zeros((20, len(SIDECHAIN_ATOMS)), np.float32); charge = np.zeros(20, np.float32)
    for aa, names in FUNCTIONAL.items():
        for name in names: weights[ALPHABET.index(aa), SIDECHAIN_ATOMS.index(name)] = 1.0
        charge[ALPHABET.index(aa)] = -1.0 if aa in ACIDS else 1.0
    return weights, charge


def geometry_targets(graph, xp=None):
    """Per query site of a loader graph with nodes [24 backbone | 128 side chain | backbone-v1]: (position target (Q, 3),
    position mask (Q,), partner score (Q,), partner mask (Q,)). Masks exclude termini sites, degenerate frames and
    residues without functional atoms."""
    if xp is None: import jax.numpy as xp
    weights, charge = (xp.asarray(x) for x in _functional())
    nodes = graph["nodes"]; table = nodes[:, -BACKBONE_WIDTH:]; side = nodes[:, 24:152].reshape(nodes.shape[0], -1, 4)
    aa = xp.argmax(nodes[:, :20], axis=1); w = weights[aa] * side[..., 3]; count = w.sum(-1)
    local = (w[..., None] * side[..., :3]).sum(1) * 10.0 / xp.maximum(count, 1.0)[:, None]
    n, ca, c = table[:, 0:3], table[:, 3:6], table[:, 6:9]
    x = c - ca; xn = xp.linalg.norm(x, axis=-1, keepdims=True); x = x / xp.maximum(xn, 1e-6)
    y = n - ca; y = y - (y * x).sum(-1, keepdims=True) * x; yn = xp.linalg.norm(y, axis=-1, keepdims=True); y = y / xp.maximum(yn, 1e-6)
    valid = (xn[:, 0] > 1e-5) & (yn[:, 0] > 1e-5); frames = xp.stack((x, y, xp.cross(x, y)), axis=-1)
    has = (count > 0) & valid; centroid = ca + xp.einsum("nij,nj->ni", frames, local)
    query = graph["query_site"]; q = graph["site_residue"][query]; titratable = graph["site_type"][query] < 7
    mask = has[q] & titratable
    neighbors = graph["neighbors"][q]; edge = graph["edge_mask"][q]
    distance = xp.sqrt(((centroid[neighbors] - centroid[q][:, None]) ** 2).sum(-1) + 1e-8)
    partner = edge & has[neighbors] & (charge[aa[neighbors]] * charge[aa[q]][:, None] < 0) & (neighbors != q[:, None])
    nearest = xp.min(xp.where(partner, distance, 1e4), axis=1)
    score = xp.clip(xp.exp(-(nearest - PARTNER_CONTACT_A) / PARTNER_DECAY_A), 0.0, 1.0)
    return local[q] / 10.0, mask, score, mask


def add_geometry_heads(params, seed):
    import jax
    import jax.numpy as jnp
    width = params["head"]["w"].shape[0]; kp, ks = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), GEOMETRY_SALT))
    head = lambda key, out: {"w": jax.random.normal(key, (width, out), jnp.float32) / jnp.sqrt(float(width)), "b": jnp.zeros(out, jnp.float32)}
    return {**params, "aux_geom_pos": head(kp, 3), "aux_partner": head(ks, 1)}


def add_orientation(params, seed):
    """Residue blocks' and the site query block's edge projections: 20 -> 33 input rows (first 20 unchanged)."""
    import jax
    import jax.numpy as jnp
    key = jax.random.fold_in(jax.random.PRNGKey(seed), ORIENTATION_SALT); keys = jax.random.split(key, len(params["blocks"]) + 1)

    def extend(block, k):
        heads = block["edge"]["w"].shape[1]
        extra = jax.random.normal(k, (ORIENTATION_EDGE, heads), jnp.float32) / jnp.sqrt(20.0 + ORIENTATION_EDGE)
        return {**block, "edge": {**block["edge"], "w": jnp.concatenate((block["edge"]["w"], extra))}}
    return {**params, "blocks": [extend(b, k) for b, k in zip(params["blocks"], keys[:-1])], "query": extend(params["query"], keys[-1])}


def input_functions(functions, sidechain_view="all", drop=0.0, orientation=False, table=None):
    """Wrap (site_embeddings, predict_shift) so each graph is transformed first: a loaded backbone table (the last
    BACKBONE_WIDTH node columns; table defaults to orientation) is stripped, and with orientation turned into 13 extra
    edge features; the side-chain view masks node columns 24:. The defaults return `functions` unchanged."""
    import jax
    import jax.numpy as jnp
    from jaxpropka.parameters import GROUP_AA
    from pkanet.model import linear
    if sidechain_view not in VIEWS: raise ValueError(sidechain_view)
    table = orientation if table is None else table
    if orientation and not table: raise ValueError("orientation needs the backbone table")
    if sidechain_view == "all" and not table: return functions
    embeddings = functions[0]; titratable = jnp.zeros(20, bool).at[jnp.asarray(GROUP_AA)].set(True)

    def transform(graph, key):
        nodes = graph["nodes"]
        if orientation:
            graph = {**graph, "edge": jnp.concatenate((graph["edge"], orientation_edges(nodes[:, -BACKBONE_WIDTH:], graph["neighbors"])), axis=-1)}
        if table: nodes = nodes[:, :-BACKBONE_WIDTH]
        if sidechain_view != "all":
            if sidechain_view == "dropout":
                keep = jnp.zeros(()) if key is None else jax.random.bernoulli(jax.random.fold_in(key, DROP_SALT), 1.0 - drop).astype(jnp.float32)
                keep = jnp.broadcast_to(keep, nodes.shape[:1])
            else:
                own = titratable[jnp.argmax(nodes[:, :20], axis=1)]
                keep = (own if sidechain_view == "titratable" else ~own).astype(jnp.float32)
            nodes = jnp.concatenate((nodes[:, :24], nodes[:, 24:] * keep[:, None]), axis=-1)
        return {**graph, "nodes": nodes}

    def wrapped(params, graph, *, key=None, dropout_rate=0.0):
        return embeddings(params, transform(graph, key), key=key, dropout_rate=dropout_rate)

    def predict_shift(params, graph, *, key=None, dropout_rate=0.0):
        tokens, query_site = wrapped(params, graph, key=key, dropout_rate=dropout_rate)
        return (8 * jnp.tanh(linear(params["head"], tokens)[:, 0]))[query_site]
    return wrapped, predict_shift


def main(argv=None):
    p = argparse.ArgumentParser(prog="pkatrain.production_inputs"); sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build-backbone"); b.add_argument("dataset", choices=("pinder", "pkpdb", "benchmark-val"))
    b.add_argument("--workers", type=int, default=72); b.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    print(json.dumps(build_backbone(Path(os.environ["PKABENCH_RUNTIME"]), a.dataset, a.workers, a.limit), indent=1))


if __name__ == "__main__":
    main()
