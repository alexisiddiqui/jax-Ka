"""Alternative free-branch geometry ("alt": minimised, AlphaFold or later ANM views) and the alt-vs-crystal ranking term
(min-aug-v1, 2026-10-31). Merged from apo-negatome-v1's production_apo (free-branch part only; the Negatome term is not
carried over) and renamed generically: --alt-source PATH --alt-prob p --val-alt --alt-rank-weight w --alt-rank-margin m.
The side-store schema is unchanged (apo-negatome-v1 FORMAT.md: store-v2 layout, fields nodes ... site_switch, backbone,
rsa_free_apo, plddt, surface_ok, residue_source, apo_ok); a store may also name the two alt-specific fields rsa_free_alt /
alt_ok, which are read as aliases.

Default off (--alt-prob 0, no --val-alt, --alt-rank-weight 0): the loaders and the engine build exactly the hotspot-mask-v2
graphs and objective.

A. Alt free branch. Each PINDER complex is two graph dicts, bound (branch 0) and free (branch 1), stacked (2, ...) and aligned
   by node index. Per complex visit, with probability p (own RNG stream [ALT_SALT, crc32(id), visit]) and when the side store
   holds a record with any alt_ok chain, the free dict's coordinate-derived arrays (ALT_FIELDS and the backbone-v1 node
   table; site-graph fields when the record carries them and they fit) are replaced by the record's (same n and node order;
   free edge mask restricted to same-chain edges) and rsa_residue[1] by rsa_free_apo (Siamese burial target = crystal bound
   minus alt free). Everything indexed by node (contacts, sampled pairs, within candidates, hotspot-mask indices, pKa targets,
   disulfide flags) stays from the native record. graph["use_alt"] (2,) float32 records the choice. A record that does not
   fit the bucket capacity (k) or whose n differs falls back to crystal and is counted. The hotspot mask (production_residue
   mask_keep_idx) indexes nodes, so the masked pass zeroes residue types on whichever free geometry the visit drew.
B. Rank term (--alt-rank-weight w, needs --alt-prob > 0): the loader also keeps the CRYSTAL free branch's coordinate fields
   (XTAL_FIELDS, keys "xtal_<field>", one copy per complex without the branch axis; on a crystal visit they equal the free
   branch). The engine runs one extra free-branch pass on them (same per-complex dropout key as the other passes) and adds
   w x margin_loss(l_alt, l_xtal, contact pairs of alt visits, margin m): mean softplus(m - (l_alt - l_xtal)) over the
   complex's sampled pairs with soft target > 0.5, per complex then over alt-visit complexes with >= 1 such pair. l_alt =
   unmasked free-branch edge logits on the alt geometry, l_xtal = the same pairs on the crystal geometry. Both sides get
   gradient (no stop_gradient): the term may lower the crystal score as well as raise the alt score.

anm-aug-v1 (2026-10-31) additions, default off (one --alt-source entry, consistency weights 0 -> min-aug-v1 behaviour):
C. Multi-source alt. --alt-source takes a comma-separated list of store[:keysuffix] entries (e.g. store-min-v1,
   store-anm-v1:a1,store-anm-v1:a2). On an alt visit (the same draw as A: first uniform of the visit stream < p) the
   entry is chosen with the stream's NEXT uniform (only consumed with >= 2 entries, so a single entry is bit-identical
   to min-aug-v1), uniformly or by --alt-weights; its record is looked up as <cid> (no suffix) or <cid>::<suffix>. A
   missing / not-ok / unfit record falls through to the next entry (cyclic), then to crystal. graph["alt_source"] (2,)
   int8 = index of the entry used (-1 crystal; added with >= 2 entries only). Per-entry counts (AltConfig.source_counts: chosen, used,
   used_as_fallback, and the fallback reasons) go to history as train_loss.pinder_alt_source_counts.
D. Symmetric consistency (--alt-cons-edge-weight we, --alt-cons-pka-weight wp; needs keep_crystal = the crystal free
   pass of B): on alt visits, edge term = per complex edge_weight-weighted mean over its sampled pairs (edge_a / edge_b,
   the weights of the free edge BCE) of (l_alt - l_xtal)^2, then mean over alt-visit complexes with weight > 0; pKa term
   = per complex mean over the free-state pKa sites (the state-loss site mask) of (pKa_alt - pKa_xtal)^2, then mean over
   alt-visit complexes with >= 1 site. l_alt / pKa_alt = the UNMASKED free pass on the drawn geometry, l_xtal / pKa_xtal
   = the crystal free pass (same per-complex dropout key). Both sides get gradient; --alt-cons-stopgrad detaches the
   crystal side (fallback).

anm-edge-v1 (2026-11-01) addition, default off (--alt-edge-only absent -> anm-aug-v1 behaviour, bit-identical):
E. Edge-only alt geometry (--alt-edge-only). The loader keeps the crystal free branch (xtal_ fields, as B) and does NOT
   substitute rsa_residue[1] (the Siamese burial target stays crystal RSA). In the engine the pKa state / paired losses,
   the Siamese burial terms (branch-1 BCE and the change term) and the gate read the CRYSTAL free pass (bound + crystal
   free = paired_all on crystal_graphs); the alt free pass (the drawn geometry) feeds only the edge head: free edge BCE and
   its distillation toward the bound branch, the within listwise term, and the hotspot-masked pass, which is built on the
   alt geometry. 4 encoder passes (bound, crystal free, alt free, masked alt free). --alt-cons-pka-weight is refused;
   --alt-cons-edge-weight (and --alt-rank-weight) compare the alt free logits with the crystal free pass's logits.
   Validation unchanged (crystal).
"""
from __future__ import annotations

import json
import threading
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from pkabench.runtime import digest

ALT_SALT = 16411  # = apo-negatome-v1 APO_SALT: the same per-visit draws as the apo arm
BACKBONE_WIDTH = 14
NODE_WIDTH = 24
ALT_FIELDS = ("nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch")
ALT_SITE_FIELDS = ("site_residue", "site_type", "site_mask", "site_neighbors", "site_edge", "site_edge_mask", "site_switch")
XTAL_FIELDS = ALT_FIELDS + ALT_SITE_FIELDS
XTAL = "xtal_"
SAME_CHAIN = 19  # raw residue-edge column: same chain flag
OK_FIELDS = ("apo_ok", "alt_ok")
RSA_FIELDS = ("rsa_free_apo", "rsa_free_alt")


def _first(record, names):
    for n in names:
        if n in record: return record[n]
    return None


# ---------------------------------------------------------------- side store adapters
def field_shape(name, dims):
    """Shape of a side-store field from a record's dims {n, k, q, s, sk} (production_graphs.shape for the store-v2 fields)."""
    from .production_graphs import shape
    n = dims["n"]; extra = {"backbone": (n, BACKBONE_WIDTH), "plddt": (n,), "rsa": (n,), "surface_ok": (n,), "residue_source": (n,),
                            "ca_rmsd": (2,), **{f: (n,) for f in RSA_FIELDS}, **{f: (2,) for f in OK_FIELDS}}
    return extra[name] if name in extra else shape(name, *(dims[d] for d in ("n", "k", "q", "s", "sk")))


class MemoryStore:
    """In-memory side store (tests): {cid: arrays}."""

    def __init__(self, complexes=None):
        self.complexes = dict(complexes or {}); self.verification_sha256 = "memory"; self.path = "memory"

    def complex(self, cid): return self.complexes.get(cid)

    def __len__(self): return len(self.complexes)

    def ids(self): return list(self.complexes)

    def close(self): pass


class SideStore:
    """Packed side store (apo-negatome-v1 FORMAT.md): store-v2 layout (records.bin, one zstd frame per record = every field's
    C-order bytes in metadata["fields"] order; index.npz ids / dims [n, k, q, s, sk] / offsets; metadata.json fields, dtypes
    and symbolic shapes; verification.json). Complex records are keyed by complex id; other keys (e.g. negatome::ACC) are
    ignored here. metadata "record_fields" {complex: [...]} if a store splits field lists."""

    def __init__(self, path):
        import os
        self.path = Path(path); self.metadata = json.loads((self.path / "metadata.json").read_text())
        verification = self.path / "verification.json"
        if verification.exists() and not json.loads(verification.read_text()).get("passed", True): raise AssertionError(self.path)
        self.verification_sha256 = digest(verification) if verification.exists() else None
        with np.load(self.path / "index.npz") as index:
            self._ids = index["ids"].astype(str); self.dims = index["dims"]; self.offsets = index["offsets"]
        self.by_id = {cid: i for i, cid in enumerate(self._ids)}; self.names = list(self.metadata.get("dims", ("n", "k", "q", "s", "sk")))
        self.kinds = self.metadata.get("record_fields"); self.dtypes = self.metadata["dtypes"]
        self.fd = os.open(self.path / "records.bin", os.O_RDONLY)

    def __len__(self): return sum("::" not in c for c in self._ids)

    def ids(self): return [c for c in self._ids if "::" not in c]

    def _fields(self):
        return self.metadata["fields"] if self.kinds is None else self.kinds["complex"]

    def raw(self, key):
        import os
        import zstandard
        i = self.by_id.get(key)
        if i is None: return None
        dims = dict(zip(self.names, (int(x) for x in self.dims[i]))); dims.setdefault("q", 1); dims.setdefault("s", 1); dims.setdefault("sk", 1)
        blob = os.pread(self.fd, int(self.offsets[i + 1]) - int(self.offsets[i]), int(self.offsets[i]))
        raw = zstandard.ZstdDecompressor().decompress(blob); out = {}; position = 0
        for f in self._fields():
            dtype = np.dtype(self.dtypes[f]); size = int(np.prod(shp := self._shape(f, dims)))
            out[f] = np.frombuffer(raw, dtype, count=size, offset=position).reshape(shp); position += size * dtype.itemsize
        if position != len(raw): raise ValueError((key, "record length does not match its dims"))
        return out

    def _shape(self, name, dims):
        symbolic = self.metadata.get("shapes", {}).get(name)
        return tuple(dims[d] if isinstance(d, str) else int(d) for d in symbolic) if symbolic is not None else field_shape(name, dims)

    def complex(self, cid): return None if "::" in cid else self.raw(cid)

    def close(self):
        import os
        if self.fd is not None: os.close(self.fd); self.fd = None


def open_store(path):
    return path if hasattr(path, "complex") else SideStore(path)


# ---------------------------------------------------------------- loader
def _pad(value, shape):
    out = np.zeros(shape, value.dtype); out[tuple(slice(0, size) for size in value.shape)] = value; return out


def parse_entry(text):
    """'path[:suffix]' -> (path, suffix or None); a colon followed by a path separator is part of the path."""
    text = text.strip()
    if ":" in text:
        path, suffix = text.rsplit(":", 1)
        if "/" not in suffix: return path, (suffix or None)
    return text, None


def open_sources(spec):
    """[(store, suffix, label)] from a store object, a path, a comma-separated 'store[:suffix],...' string, or a list of
    such strings / (store-or-path, suffix) pairs. A path named twice is opened once."""
    if hasattr(spec, "complex"): return [(spec, None, str(spec.path))]
    items = [e for e in str(spec).split(",") if e.strip()] if isinstance(spec, (str, Path)) else list(spec)
    opened = {}; out = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 3: out.append(item); continue  # already opened (store, suffix, label)
        if isinstance(item, (str, Path)): path, suffix = parse_entry(str(item)); label = str(item).strip()
        else: path, suffix = item; label = f"{getattr(path, 'path', path)}{':' + suffix if suffix else ''}"
        if hasattr(path, "complex"): store = path
        else: store = opened.get(str(path)) or opened.setdefault(str(path), SideStore(path))
        out.append((store, suffix, label))
    if not out: raise ValueError("empty --alt-source")
    if len({label for _, _, label in out}) != len(out): raise ValueError(("duplicate --alt-source entries", [label for _, _, label in out]))
    return out


def lookup(store, cid, suffix):
    """Record <cid> (suffix None, as min-aug-v1) or <cid>::<suffix>; None when absent."""
    if suffix is None: return store.complex(cid)
    key = f"{cid}::{suffix}"
    return store.raw(key) if hasattr(store, "raw") else store.complex(key)


def keyed_records(store, suffix):
    ids = list(store._ids) if hasattr(store, "_ids") else list(getattr(store, "complexes", {}))
    return sum("::" not in c for c in ids) if suffix is None else sum(c.endswith("::" + suffix) for c in ids)


class AltConfig:
    """Loader state for the alt free branch: store(s), probability, keep_crystal (rank / consistency terms), own visit
    counter, counts. store: one store / path (min-aug-v1) or several entries (open_sources); weights: per-entry choice
    weights (default uniform)."""

    def __init__(self, store, prob, keep_crystal=False, weights=None, alt_rsa=True):
        self.alt_rsa = bool(alt_rsa)  # anm-edge-v1 E: False -> rsa_residue[1] stays crystal on alt visits
        self.entries = open_sources(store); self.store = self.entries[0][0]; self.prob = float(prob); self.keep_crystal = bool(keep_crystal)
        w = np.ones(len(self.entries)) if weights is None else np.asarray([float(x) for x in weights], float)
        if w.shape != (len(self.entries),) or (w < 0).any() or w.sum() <= 0: raise ValueError(("--alt-weights must be one non-negative weight per --alt-source entry", list(w)))
        self.weights = w / w.sum()
        self.visits = {}; self.counts = Counter(); self.source_counts = {label: Counter() for _, _, label in self.entries}; self.lock = threading.Lock()

    def count(self, *keys):
        with self.lock:
            for k in keys: self.counts[k] += 1

    def count_source(self, label, key):
        with self.lock: self.source_counts[label][key] += 1

    def source_snapshot(self):
        with self.lock: return {label: dict(c) for label, c in self.source_counts.items()}

    def draw(self, cid):
        return self.draw_entry(cid)[0]

    def draw_entry(self, cid):
        """(drawn, entry index): first uniform of the visit stream < p (as min-aug-v1); with >= 2 entries the stream's
        next draw picks the entry."""
        with self.lock: self.visits[cid] = self.visits.get(cid, 0) + 1; visit = self.visits[cid]
        rng = np.random.default_rng([ALT_SALT, zlib.crc32(cid.encode()), visit]); drawn = rng.random() < self.prob
        if not drawn or len(self.entries) == 1: return drawn, 0
        return drawn, int(rng.choice(len(self.entries), p=self.weights))


def alt_ok(record):
    if record is None: return False
    ok = _first(record, OK_FIELDS)
    return bool(np.asarray(np.zeros(2, bool) if ok is None else ok).astype(bool).any())


def free_branch(graph, record, n_real, capacity, rsa=True):
    """(paired graph with branch 1 rebuilt from the alt record, None) or (unchanged graph, fallback reason). graph: one
    complex's loader graph (arrays (2, ...) padded to capacity = (n, k, q, s, sk)); node columns = 24 node features
    [+ the 14 backbone-v1 columns when the loader appends the table]."""
    n, k, q, s, sk = capacity; rec_n = int(record["nodes"].shape[0])
    if rec_n != n_real: return graph, "n_mismatch"
    if record["neighbors"].shape[1] > k: return graph, "capacity_k"
    width = graph["nodes"].shape[-1]
    if width not in (NODE_WIDTH, NODE_WIDTH + BACKBONE_WIDTH): raise ValueError(("alt free branch supports the backbone-v1 side table only", width))
    if width > NODE_WIDTH and "backbone" not in record: return graph, "no_backbone"
    bound = {name: value[0] for name, value in graph.items()}; free = {name: np.array(value[1]) for name, value in graph.items()}
    nodes = _pad(record["nodes"].astype(np.float32), (n, NODE_WIDTH))
    if width > NODE_WIDTH: nodes = np.concatenate((nodes, _pad(record["backbone"].astype(np.float32), (n, BACKBONE_WIDTH))), axis=-1)
    free["nodes"] = nodes.astype(graph["nodes"].dtype)
    free["node_mask"] = _pad(record["node_mask"], (n,)).astype(graph["node_mask"].dtype)
    free["neighbors"] = _pad(record["neighbors"], (n, k)).astype(graph["neighbors"].dtype)
    free["edge"] = _pad(record["edge"], (n, k, record["edge"].shape[-1])).astype(graph["edge"].dtype)
    same = record["edge"][..., SAME_CHAIN] > 0.5
    free["edge_mask"] = _pad(record["edge_mask"].astype(bool) & same, (n, k)).astype(graph["edge_mask"].dtype)
    free["switch"] = _pad(record["switch"], (n, k)).astype(graph["switch"].dtype)
    if all(f in record for f in ALT_SITE_FIELDS) and record["site_residue"].shape[0] <= s and record["site_neighbors"].shape[1] <= sk:
        for f in ALT_SITE_FIELDS: free[f] = _pad(record[f], graph[f].shape[1:]).astype(graph[f].dtype)
    rsa_alt = _first(record, RSA_FIELDS)
    if rsa and "rsa_residue" in free and rsa_alt is not None:  # anm-edge-v1 E: rsa=False keeps the crystal RSA
        rsa = np.full(free["rsa_residue"].shape, -1.0, np.float32); rsa[:n_real] = np.nan_to_num(rsa_alt.astype(np.float32), nan=-1.0)
        free["rsa_residue"] = rsa
    return {name: np.stack((bound[name], free[name])) for name in graph}, None


def attach_alt(graph, config, cid, n_real, capacity):
    """Loader hook: choose the free branch for this visit; adds use_alt (2,) and, with config.keep_crystal, the crystal
    free branch's XTAL_FIELDS as xtal_<field> (no branch axis)."""
    use = False; used = -1
    crystal = {XTAL + f: np.array(graph[f][1]) for f in XTAL_FIELDS} if config is not None and config.keep_crystal else None
    if config is not None:
        drawn, first = config.draw_entry(cid); reason = None; m = len(config.entries); extra = ()
        if drawn:
            for j in range(m):  # chosen entry first, then the others in cyclic order; crystal when none fits
                idx = (first + j) % m; store, suffix, label = config.entries[idx]; record = lookup(store, cid, suffix)
                if record is None: why = "no_record"
                elif not alt_ok(record): why = "not_ok"
                else: graph, why = free_branch(graph, record, n_real, capacity, getattr(config, "alt_rsa", True))
                if j == 0: reason = why; config.count_source(label, "chosen")
                if why is None:
                    use = True; used = idx; config.count_source(label, "used" if j == 0 else "used_as_fallback")
                    if j: extra = ("fallback_alt",)
                    break
                config.count_source(label, why)
        config.count("visits", *(("drawn",) if drawn else ()), "alt" if use else (reason or "crystal_by_draw"), *extra)
        if m > 1: graph["alt_source"] = np.full(2, used, np.int8)  # multi-source only: one entry keeps min-aug-v1's graph keys
    graph["use_alt"] = np.full(2, float(use), np.float32)
    if crystal is not None: graph.update(crystal)
    return graph


def spread_crystal(graph, spread_padding):
    """Apply the loader's padding spread to the xtal_ fields (same rule as the free branch; on crystal visits the result
    equals the spread free branch)."""
    if XTAL + "nodes" not in graph: return graph
    sub = spread_padding({f: graph[XTAL + f] for f in XTAL_FIELDS})
    for f in XTAL_FIELDS: graph[XTAL + f] = sub[f]
    return graph


# ---------------------------------------------------------------- engine side (jax)
def split_crystal(graphs):
    """(graphs without xtal_ keys, crystal free-branch batch dict (B, ...) or None). The crystal dict is the free branch
    (graphs[:, 1]) with every XTAL_FIELDS array taken from xtal_<field>."""
    if XTAL + "nodes" not in graphs: return graphs, None
    rest = {k: v for k, v in graphs.items() if not k.startswith(XTAL)}
    crystal = {k: v[:, 1] for k, v in rest.items()}; crystal.update({f: graphs[XTAL + f] for f in XTAL_FIELDS})
    return rest, crystal


def crystal_graphs(graphs, crystal):
    """anm-edge-v1 E: paired graphs (B, 2, ...) with branch 1's XTAL_FIELDS taken from the crystal dict (split_crystal);
    every other key (node-indexed labels, rsa_residue, use_alt) unchanged. On crystal visits the arrays equal the input."""
    import jax.numpy as jnp
    return {k: (jnp.stack((v[:, 0], crystal[k]), axis=1) if k in XTAL_FIELDS else v) for k, v in graphs.items()}


def crystalise(graphs):
    """anm-edge-v1 smoke / tests: a loader batch with xtal_ fields -> the same batch as a crystal-only visit (branch 1's
    XTAL_FIELDS from xtal_, use_alt 0, alt_source -1); the xtal_ keys are kept so the engine still finds the crystal pass."""
    import jax.numpy as jnp
    out = dict(graphs)
    for f in XTAL_FIELDS: out[f] = jnp.stack((jnp.asarray(graphs[f])[:, 0], jnp.asarray(graphs[XTAL + f])), axis=1)
    out["use_alt"] = jnp.zeros_like(jnp.asarray(graphs["use_alt"]))
    if "alt_source" in out: out["alt_source"] = jnp.full_like(jnp.asarray(graphs["alt_source"]), -1)
    return out


def rank_loss(l_alt, l_xtal, contact, use_alt, margin=0.0):
    """mean softplus(margin - (l_alt - l_xtal)) over contact pairs (B, P) of alt-visit complexes, per complex then over
    complexes with >= 1 pair (production_residue.margin_loss). No stop_gradient on either side."""
    from .production_residue import margin_loss
    return margin_loss(l_alt, l_xtal, contact * use_alt[:, None], margin)


def cons_edge_loss(l_alt, l_xtal, edge_weight, valid, use_alt):
    """anm-aug-v1 D: per complex edge_weight-weighted mean of (l_alt - l_xtal)^2 over the sampled pairs (B, P), then mean
    over valid alt-visit complexes with weight > 0 (the free edge BCE's weighting). No stop_gradient."""
    import jax.numpy as jnp
    ew = edge_weight * (valid * use_alt)[:, None]; en = jnp.sum(ew, axis=1); counted = en > 0
    per = jnp.sum(jnp.square(l_alt - l_xtal) * ew, axis=1) / jnp.maximum(en, 1)
    return jnp.sum(jnp.where(counted, per, 0.0)) / jnp.maximum(jnp.sum(counted), 1)


def cons_pka_loss(p_alt, p_xtal, mask, valid, use_alt):
    """anm-aug-v1 D: per complex mean of (pKa_alt - pKa_xtal)^2 over the free-state pKa sites (mask (B, Q), the state
    loss's site mask), then mean over valid alt-visit complexes with >= 1 site. No stop_gradient."""
    import jax.numpy as jnp
    sites = mask.astype(jnp.float32); n = jnp.sum(sites, axis=1); counted = (valid > 0) & (use_alt > 0.5) & (n > 0)
    per = jnp.sum(jnp.square(p_alt - p_xtal) * sites, axis=1) / jnp.maximum(n, 1)
    return jnp.sum(jnp.where(counted, per, 0.0)) / jnp.maximum(jnp.sum(counted), 1)


def strip(graphs):
    """graphs without the xtal_ keys (scorers / validation)."""
    return {k: v for k, v in graphs.items() if not k.startswith(XTAL)}


# ---------------------------------------------------------------- provenance
def alt_provenance(prob, store, val_alt, rank_weight=0.0, rank_margin=0.0, entries=None, weights=None, cons_edge=0.0, cons_pka=0.0, cons_stopgrad=False,
                   edge_only=False):
    out = _alt_provenance(prob, store, val_alt, rank_weight, rank_margin)
    if store is None: return out
    if edge_only:  # anm-edge-v1 E
        out["alt_edge_only"] = True
        out["alt_edge_only_definition"] = {
            "crystal_free_pass_feeds": ["pKa state loss (free branch)", "pKa paired loss", "Siamese burial BCE (branch 1) and change term", "gate (branch 1)"],
            "alt_free_pass_feeds": ["free edge BCE and its distillation toward bound", "within listwise term", "hotspot-masked pass (built on the alt geometry)",
                                    *(["edge consistency (alt vs crystal free logits)"] if cons_edge else []), *(["rank term"] if rank_weight else [])],
            "rsa_residue_free": "crystal RSA (rsa_free_apo / rsa_free_alt not substituted)",
            "passes": "bound + crystal free + alt free (drawn geometry) + hotspot-masked alt free = 4 encoder passes per complex per step",
            "validation": "unchanged (crystal)", "pka_consistency": "refused (--alt-cons-pka-weight)",
            "history": "train_loss.pinder_alt_edge_only_burial / pinder_alt_edge_only_gate (unweighted, crystal pass)"}
    if entries is not None and len(entries) > 1 and "alt_free_branch" in out:
        w = np.ones(len(entries)) / len(entries) if weights is None else np.asarray(weights, float) / np.sum(weights)
        out["alt_free_branch"]["source"] = ",".join(label for _, _, label in entries)
        out["alt_free_branch"]["sources"] = [{"index": i, "entry": label, "path": str(s.path), "key": f"<complex id>::{suffix}" if suffix else "<complex id>",
                                              "weight": float(w[i]), "store_verification_sha256": s.verification_sha256, "keyed_records": keyed_records(s, suffix)}
                                             for i, (s, suffix, label) in enumerate(entries)]
        out["alt_free_branch"]["selection"] = ("on an alt visit the entry is drawn from the visit stream's second uniform (weights above); a missing / "
                                               "not-ok / unfit record falls through to the next entry (cyclic), then crystal; graph alt_source = entry "
                                               "index used (-1 crystal); history train_loss.pinder_alt_source_counts (per epoch)")
    if cons_edge or cons_pka:
        out["alt_consistency"] = {**({"alt_edge_only": True, "crystal_side": "the crystal free pass that feeds pKa / burial / gate (no extra pass)"} if edge_only else {}),"edge_weight": float(cons_edge), "pka_weight": float(cons_pka), "stop_gradient_crystal": bool(cons_stopgrad),
            "definition": "on alt visits: + we x mean_complex[sum_pairs edge_weight (l_alt - l_xtal)^2 / sum edge_weight] over the complex's sampled pairs "
                          "(edge_a / edge_b, free edge BCE weights) + wp x mean_complex[mean over free-state pKa sites (state-loss mask) of (pKa_alt - "
                          "pKa_xtal)^2]; alt = unmasked free pass on the drawn geometry, xtal = extra free pass on the crystal geometry (loader xtal_ "
                          "fields), same per-complex dropout key",
            "gradient": "crystal side detached (--alt-cons-stopgrad)" if cons_stopgrad else "both sides (symmetric, no stop_gradient)",
            "passes": "bound + free (drawn geometry) + hotspot-masked free (drawn geometry) + crystal free = 4 encoder passes per complex per step",
            "history": "train_loss.pinder_alt_cons_edge, train_loss.pinder_alt_cons_pka (unweighted)"}
    return out


def _alt_provenance(prob, store, val_alt, rank_weight=0.0, rank_margin=0.0):
    out = {}
    if store is None: return out
    if prob: out["alt_free_branch"] = {"probability": float(prob), "source": str(store.path), "store_verification_sha256": store.verification_sha256,
        "salt": ALT_SALT, "store_records": len(store),
        "fields": list(ALT_FIELDS) + ["backbone (node table)", "rsa_free_apo (or rsa_free_alt) -> rsa_residue[1]"] + [f"{f} (when present and within capacity)" for f in ALT_SITE_FIELDS],
        "definition": "per complex visit with probability p and a side-store record with any apo_ok (alt_ok) chain: free branch (branch 1) graph from the "
                      "record (same nodes; edge mask restricted to same-chain edges), rsa_residue[1] = rsa_free_apo; contacts, pairs, within, hotspot-mask "
                      "indices, pKa targets from the native record (the masked pass masks the drawn geometry); records over the bucket k capacity or with "
                      "another n fall back to crystal (counted, history loader_counts.alt) (production_alt)"}
    if val_alt: out["val_alt"] = {"source": str(store.path), "definition": "extra PINDER validation pass with the alt free branch (prob 1) on complexes with "
                                  "a record; reported as validation.pinder_alt (alt vs crystal on the same complexes); selection unchanged"}
    if rank_weight: out["alt_rank"] = {"weight": float(rank_weight), "margin": float(rank_margin),
        "definition": "on alt visits: + w x mean softplus(m - (l_alt - l_xtal)) over the complex's sampled pairs with soft target > 0.5 (edge_a / edge_b), "
                      "per complex then over alt-visit complexes with >= 1 such pair; l_alt = unmasked free-branch edge logits on the alt geometry, l_xtal "
                      "= the same pairs from an extra free-branch pass on the crystal geometry (loader xtal_ fields), same per-complex dropout key",
        "gradient": "both sides (no stop_gradient): the term can lower the crystal score as well as raise the alt score",
        "passes": "bound + free (drawn geometry) + hotspot-masked free (drawn geometry) + crystal free = 4 encoder passes per complex per step "
                  "(every complex: shapes are fixed per batch; non-alt complexes' crystal pass equals the free pass and is masked out of the term)",
        "history": "train_loss.pinder_alt_rank_margin_loss, train_loss.pinder_alt_fraction"}
    return out
