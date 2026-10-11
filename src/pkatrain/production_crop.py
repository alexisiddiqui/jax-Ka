"""Inner-radius site crops for production GQT training (pkatrain.production_train --crop-radius/--crop-inner), 2026-10-11.

Experiment 35 cropped 25% of draws to one query's C-alpha neighbourhood and supervised only that centre query, so
cropped draws lost most of their labels. Here a cropped draw keeps every residue whose C-alpha lies within
`radius` of a centre site and supervises every originally supervised site within `inner` of the centre: by the
triangle inequality each supervised site keeps its complete neighbourhood out to radius - inner (15 A for 20 / 5,
where experiment 38's 15 A cutoff was level with 20 A). Sites beyond `inner` stay in the graph as context only.

- Geometry: the production graphs are exact 20 A C-alpha radius graphs (pkanet.graph.geometry) with no stored
  coordinates; centre-to-residue distances are decoded from the centre row's 16 RBF edge channels
  (gqt_crop_radius.decode_rbf_distance, the same 0-20 A basis), so radius <= 20.
- Masking: residues outside the crop leave node_mask; every residue edge and site-graph edge with an end outside it
  leaves edge_mask / site_edge_mask (attention masks before the softmax, so no path crosses the boundary); sites on
  removed residues leave site_mask. Array shapes and indices are unchanged.
- PINDER: one crop per complex, chosen from the AB branch's edges and applied identically to the AB and free
  branches (the free branch keeps its own partner-removed masks), so the paired loss compares like with like.
  pKPDB structures are cropped on their own draws.
- Draws: a cropped draw happens with `probability` per structure and epoch; decision and centre (uniform over the
  supervised sites) are a deterministic function of (seed, epoch, dataset, id), so resumed runs repeat them.
  Validation never crops.
- Losses are unchanged (per-structure means over supervised sites), so a cropped structure carries the same weight
  as a whole one.
"""
from __future__ import annotations

import hashlib
import threading

import numpy as np

from .gqt_crop_radius import decode_rbf_distance
from .production_loading import spread_padding

GRAPH_RADIUS = 20.0
TOLERANCE = 2e-4  # decoded-distance slack at the radius boundaries (experiment 35's)
STAT_KEYS = ("draws", "cropped", "nodes", "kept_nodes", "supervised", "kept_supervised", "edges", "kept_edges")


def check_config(radius, inner, probability):
    if not 0 < radius <= GRAPH_RADIUS: raise ValueError(f"crop radius must be in (0, {GRAPH_RADIUS}] (graph geometry)")
    if not 0 <= inner <= radius: raise ValueError("crop inner radius must be in [0, radius]")
    if not 0 < probability <= 1: raise ValueError("crop probability must be in (0, 1]")


def decision(seed, epoch, dataset, cid, probability):
    """(rng, cropped) for one structure draw; deterministic in (seed, epoch, dataset, id)."""
    token = f"production-crop-v1:{seed}:{epoch}:{dataset}:{cid}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
    return rng, bool(rng.random() < probability)


def crop(graph, supervised, centre_query, radius, inner, branched=False):
    """Crop one structure's graph in place around supervised query `centre_query`. `graph` holds the residue and
    site fields of production_loading (with a leading branch axis of 2 if `branched`, branch 0 = AB, whose edges
    define the crop); returns (new supervised mask, stats)."""
    view = {name: value[0] for name, value in graph.items()} if branched else graph
    supervised = np.asarray(supervised, bool)
    if not supervised[centre_query]: raise ValueError("crop centre is not a supervised site")
    centre = int(view["query_residue"][centre_query]); node_mask = np.array(view["node_mask"], bool)  # a copy: masked below
    row = np.asarray(view["edge_mask"][centre], bool); neighbors = np.asarray(view["neighbors"][centre][row], np.int64)
    distance = np.full(len(node_mask), np.inf); distance[neighbors] = decode_rbf_distance(view["edge"][centre][row][:, :16])
    distance[centre] = 0.0
    keep = (distance <= radius + TOLERANCE) & node_mask
    if not keep[centre]: raise AssertionError("crop removed its centre")
    outside = ~keep[:, None] | ~keep[np.asarray(view["neighbors"], np.int64)]
    site_keep = keep[np.asarray(view["site_residue"], np.int64)] & np.asarray(view["site_mask"], bool)
    site_outside = ~site_keep[:, None] | ~site_keep[np.asarray(view["site_neighbors"], np.int64)]
    edges = int(np.asarray(graph["edge_mask"], bool).sum())
    for branch in range(2 if branched else 1):
        g = {name: value[branch] for name, value in graph.items()} if branched else graph
        g["edge_mask"][outside] = False; g["switch"][outside] = 0.0; g["node_mask"][~keep] = 0
        g["site_edge_mask"][site_outside] = False; g["site_switch"][site_outside] = 0.0; g["site_mask"][~site_keep] = 0
    result = supervised & (distance[np.asarray(view["query_residue"], np.int64)] <= inner + TOLERANCE)
    result[centre_query] = True
    return result, {"nodes": int(node_mask.sum()), "kept_nodes": int(keep.sum()), "supervised": int(supervised.sum()),
                    "kept_supervised": int(result.sum()), "edges": edges, "kept_edges": int(np.asarray(graph["edge_mask"], bool).sum())}


class CroppedSource:
    """Wraps a production_loading source built with spread=False (PinderSource, AuxPinderSource or PkpdbSource: graphs
    in slot 0, the supervised mask in slot 2): crops each real structure of a batch on its draw, then applies the
    padding spread. set_epoch(epoch) before each epoch's Prefetcher; stats() returns and resets the epoch's counts."""

    def __init__(self, source, dataset, radius, inner, probability, seed, spread=True):
        if source.spread: raise ValueError("wrap a source built with spread=False (the crop must precede the spread)")
        check_config(radius, inner, probability)
        self.source = source; self.dataset = dataset; self.radius = float(radius); self.inner = float(inner)
        self.probability = float(probability); self.seed = int(seed); self.spread = spread; self.epoch = None
        self._lock = threading.Lock(); self._stats = dict.fromkeys(STAT_KEYS, 0)

    def __getattr__(self, name): return getattr(self.source, name)

    def set_epoch(self, epoch): self.epoch = int(epoch)

    def load(self, ids):
        if self.epoch is None: raise AssertionError("CroppedSource epoch was not set")
        batch = list(self.source.load(ids)); graphs, mask = batch[0], batch[2]; branched = self.dataset == "pinder"
        totals = dict.fromkeys(STAT_KEYS, 0)
        for slot, cid in enumerate(ids):
            supervised = np.asarray(mask[slot], bool); choices = np.flatnonzero(supervised); totals["draws"] += 1
            rng, cropped = decision(self.seed, self.epoch, self.dataset, cid, self.probability)
            if not cropped or not len(choices): continue
            view = {name: value[slot] for name, value in graphs.items()}
            result, stats = crop(view, supervised, int(choices[rng.integers(len(choices))]), self.radius, self.inner, branched)
            mask[slot] = result.astype(mask.dtype); totals["cropped"] += 1
            for key, value in stats.items(): totals[key] += value
        if self.spread: batch[0] = spread_padding(graphs)
        with self._lock:
            for key, value in totals.items(): self._stats[key] += value
        return tuple(batch)

    def stats(self):
        with self._lock: s = dict(self._stats); self._stats = dict.fromkeys(STAT_KEYS, 0)
        c = max(s["cropped"], 1)
        return {**s, "crop_fraction": s["cropped"] / max(s["draws"], 1), "kept_node_fraction": s["kept_nodes"] / max(s["nodes"], 1),
                "kept_edge_fraction": s["kept_edges"] / max(s["edges"], 1), "kept_supervised_fraction": s["kept_supervised"] / max(s["supervised"], 1),
                "supervised_per_crop": s["kept_supervised"] / c, "nodes_per_crop": s["kept_nodes"] / c}

    def close(self): self.source.close()

    def provenance(self):
        return {**self.source.provenance(), "spread_padding": self.spread,
                "crop": {"radius_A": self.radius, "inner_A": self.inner, "probability": self.probability, "seed": self.seed}}


def describe(radius, inner, probability):
    return (f"production_crop: on {probability:g} of training draws per structure and epoch (deterministic in seed, epoch, "
            f"dataset, id), keep residues with C-alpha within {radius:g} A of a uniformly drawn supervised site (distances "
            f"decoded from the 20 A graph's RBF edges), mask every residue/site edge crossing the boundary, supervise the "
            f"supervised sites within {inner:g} A of the centre (>= {radius - inner:g} A intact context each); PINDER AB and "
            f"free branches share one crop; pKPDB cropped on its own draws; validation uncropped")


def audit(root, fraction=0.1, radius=20.0, inner=5.0, structures=200, seed=17):
    """Real-store check (Slurm): decoded distances are symmetric (d_ij vs d_ji) and within the graph radius, every
    crop leaves no unmasked edge or site edge across its boundary, and crop statistics at (radius, inner)."""
    from pathlib import Path
    from .production_graphs import output, read
    from .production_loading import MANIFEST, PinderSource, PkpdbSource, select
    from .production_train import _with_policy
    root = Path(root); report = {"radius": radius, "inner": inner, "fraction": fraction}
    for dataset in ("pinder", "pkpdb"):
        manifest = _with_policy(read(output(root, dataset) / MANIFEST))
        source = PinderSource(manifest, fraction=fraction, spread=False) if dataset == "pinder" else PkpdbSource(manifest, spread=False)
        records = select(manifest, "train", fraction)[:structures]; rng = np.random.default_rng(seed)
        asym = []; far = []; wrapped = CroppedSource(source, dataset, radius, inner, 1.0, seed, spread=False); wrapped.set_epoch(1)
        for start in range(0, len(records), 8):
            rows = records[start:start + 8]; bucket = {}
            for r in rows: bucket.setdefault(source.policy.bucket(r["n"]), []).append(r["id"])
            for ids in bucket.values():
                plain = source.load(ids)[0]
                batch = wrapped.load(ids); graphs = batch[0]
                for slot in range(len(ids)):
                    g = {name: (value[slot][0] if dataset == "pinder" else value[slot]) for name, value in plain.items()}
                    n = int(np.asarray(g["node_mask"], bool).sum()); nb = np.asarray(g["neighbors"])[:n]; em = np.asarray(g["edge_mask"], bool)[:n]
                    d = decode_rbf_distance(np.asarray(g["edge"])[:n, :, :16])
                    table = {}
                    for i in rng.choice(n, min(n, 40), replace=False):
                        for j, dij in zip(nb[i][em[i]], d[i][em[i]]): table[(int(i), int(j))] = dij
                    for (i, j), dij in table.items():
                        if em[j].any():
                            back = d[j][em[j] & (nb[j] == i)]
                            if len(back): asym.append(abs(dij - back[0]))
                    far.append(float(d[em].max()))
                    c = {name: (value[slot][0] if dataset == "pinder" else value[slot]) for name, value in graphs.items()}
                    keep = np.asarray(c["node_mask"], bool)
                    live = np.asarray(c["edge_mask"], bool)
                    if (live & ~(keep[:, None] & keep[np.asarray(c["neighbors"])])).any(): raise AssertionError("edge crosses the crop")
                    sk = np.asarray(c["site_mask"], bool); sl = np.asarray(c["site_edge_mask"], bool)
                    if (sl & ~(sk[:, None] & sk[np.asarray(c["site_neighbors"])])).any(): raise AssertionError("site edge crosses the crop")
        report[dataset] = {"structures": len(records), "max_abs_asymmetry_A": float(np.max(asym)), "p99_abs_asymmetry_A": float(np.quantile(asym, 0.99)),
                           "max_decoded_distance_A": float(np.max(far)), **wrapped.stats()}
        source.close()
    return report


def main(argv=None):
    import json
    import os
    import sys
    from pathlib import Path
    from pkabench.runtime import atomic_json
    argv = sys.argv[1:] if argv is None else argv
    root = Path(os.environ["PKABENCH_RUNTIME"])
    if argv[0] != "audit": raise ValueError(argv[0])
    radius, inner = (float(argv[1]), float(argv[2])) if len(argv) > 2 else (20.0, 5.0)
    report = audit(root, radius=radius, inner=inner)
    from .production_graphs import VERSION
    atomic_json(root / "training" / VERSION / f"crop-audit-r{radius:g}-i{inner:g}.json", report); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
