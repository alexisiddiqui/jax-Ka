"""Ordinal auxiliary targets for production oGQT training (2026-10-11; audit in experiment note 07).

- burial: 1 - clip(RSA_free, 0, 1) as four intervals, RSA cut points 0.1 / 0.25 / 0.5 (buried, partly buried,
  intermediate, exposed); three cumulative binary targets [RSA < cut], from the stores' rsa_free (PINDER, free
  branch) and rsa (pKPDB, train_mask sites).
- interface: additive partner-contact score c = sum_j min(1, exp(-(d_j - 4) / 3)) over partner-chain residues j with
  d_j <= 10 A, d_j the residue-level minimum heavy-atom distance from the site residue in AB.cif.gz (the w_interface
  curve per partner residue; min_j d_j reproduces sites.json partner_distance_A); six cumulative binary targets
  [c > t] for t in 0 / 0.5 / 1 / 2 / 4 / 8 (t = 0 is the stored 10 A interface flag). PINDER bound branch only.

Two loss forms (production_train.AuxEngine, --aux-loss):
- "ordinal": cumulative BCE over the cut points / thresholds above (first version, 2026-10-11).
- "ce" (2026-10-11, user revision): one sigmoid output per head, unweighted cross-entropy against a soft target in
  [0, 1]: burial y = 1 - clip(RSA, 0, 1); interface y = log(1 + min(c, 8)) / log(9). No class balancing (the user
  will address interface false negatives through the Siamese loss instead).

The PINDER scores are not in store-v2: `build` writes a side table aligned to each record's store query sites,
<runtime>/training/<version>/pinder/contacts-v1/{scores.npy, offsets.npy, ids.json, verification.json}; it reads the
AB structures, so on Isambard it runs under scripts/sqfs_run.sh.

  python -m pkatrain.production_aux build [--workers 64] [--limit N (smoke table)]
  python -m pkatrain.production_aux build-rsa-bound [--workers 64] [--limit N]   (bound-state RSA, Siamese burial)
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
from .production_graphs import PINDER, ProductionStore, output, read, source

BURIAL_RSA_CUTS = (0.1, 0.25, 0.5)
INTERFACE_THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
INTERFACE_CAP = 8.0
CONTACT_A, DECAY_A, CUTOFF_A = 4.0, 3.0, 10.0
TABLE = "contacts-v1"
RSA_BOUND_TABLE = "rsa-bound-v1"


def contact_score(distances):
    d = np.asarray(distances, float); d = d[d <= CUTOFF_A]
    return float(np.sum(np.minimum(1.0, np.exp(-(d - CONTACT_A) / DECAY_A))))


def burial_targets(rsa):
    """(..., 3) cumulative targets [RSA < cut]; NaN RSA gives NaN (masked by the caller)."""
    rsa = np.asarray(rsa, np.float32)[..., None]
    return np.where(np.isfinite(rsa), (rsa < np.asarray(BURIAL_RSA_CUTS, np.float32)).astype(np.float32), np.nan)


def interface_targets(score):
    return (np.asarray(score, np.float32)[..., None] > np.asarray(INTERFACE_THRESHOLDS, np.float32)).astype(np.float32)


def burial_soft(rsa):
    """1 - clip(RSA, 0, 1); NaN RSA stays NaN (masked by the caller)."""
    return 1.0 - np.clip(np.asarray(rsa, float), 0.0, 1.0)


def interface_soft(score):
    return np.log1p(np.minimum(np.asarray(score, float), INTERFACE_CAP)) / np.log1p(INTERFACE_CAP)


def _scores_one(args):
    """Contact score and nearest partner distance per store query site of one PINDER record."""
    root, cid, store_path = args
    from scipy.spatial import cKDTree
    from jaxpropka.topology import load_topology
    from .gqt_paired_pinder import _read_cif_gz
    store = ProductionStore(Path(store_path)); raw = store.raw(cid); store.close()
    atoms = _read_cif_gz(source(root, f"{PINDER}/entries/{cid}") / "AB.cif.gz")
    keys = load_topology(atoms, gap_policy="cap", freeze_disulfides=True).keys
    heavy = atoms[(atoms.element != "H") & (atoms.element != "D") & ~atoms.hetero]
    chain = heavy.chain_id.astype(str); resid = heavy.res_id; icode = np.char.strip(heavy.ins_code.astype(str))
    trees = {}; scores = []; nearest = []
    for r in raw["query_residue"]:
        key = keys[int(r)]; own = (chain == key.chain) & (resid == key.number) & (icode == key.insertion); partner = chain != key.chain
        if not own.any(): raise AssertionError((cid, "query residue has no heavy atoms", key))
        if key.chain not in trees: trees[key.chain] = (cKDTree(heavy.coord[partner]), np.where(partner)[0])
        tree, index = trees[key.chain]; best = {}
        for coord, near in zip(heavy.coord[own], tree.query_ball_point(heavy.coord[own], CUTOFF_A + 1e-3)):
            for h in near:
                a = index[h]; k = (chain[a], int(resid[a]), icode[a]); d = float(np.linalg.norm(coord - heavy.coord[a]))
                if d < best.get(k, np.inf): best[k] = d
        scores.append(contact_score(list(best.values()))); nearest.append(min(best.values(), default=np.inf))
    scores = np.asarray(scores, np.float32); nearest = np.asarray(nearest, np.float32)
    stored = np.asarray(raw["partner_distance_A"], np.float32); both = (nearest <= CUTOFF_A) & (stored < CUTOFF_A - 0.05)
    return cid, scores, {"sites": len(scores), "max_abs_distance_diff": float(np.max(np.abs(nearest[both] - stored[both]), initial=0.0)),
                         "flag_disagree": int(np.sum((scores > 0) != np.asarray(raw["interface"], bool)))}


def build(root, workers=64, limit=None):
    """limit: smoke build of the first `limit` records into contacts-v1-smoke (never read by training)."""
    root = Path(root); manifest = read(output(root, "pinder") / "manifest-v1.json"); store = manifest["store"]
    out = output(root, "pinder") / (TABLE + ("-smoke" if limit else "")); out.mkdir(parents=True, exist_ok=True); began = time.time()
    ids = [r["id"] for r in manifest["records"]][:limit]
    with ProcessPoolExecutor(workers) as ex: results = list(ex.map(_scores_one, [(root, cid, store) for cid in ids], chunksize=8))
    offsets = np.cumsum([0] + [len(s) for _, s, _ in results]).astype(np.int64)
    np.save(out / "scores.npy", np.concatenate([s for _, s, _ in results])); np.save(out / "offsets.npy", offsets)
    atomic_json(out / "ids.json", ids)
    checks = [c for _, _, c in results]
    summary = {"version": TABLE, "definition": __doc__.split("\n\n")[0], "contact_A": CONTACT_A, "decay_A": DECAY_A, "cutoff_A": CUTOFF_A,
               "structures": len(ids), "sites": int(offsets[-1]), "manifest_sha256": digest(output(root, "pinder") / "manifest-v1.json"),
               "max_abs_distance_diff_vs_partner_distance_A": max(c["max_abs_distance_diff"] for c in checks),
               "interface_flag_disagreements": sum(c["flag_disagree"] for c in checks),
               "files": {f: digest(out / f) for f in ("scores.npy", "offsets.npy", "ids.json")}, "seconds": round(time.time() - began, 1)}
    atomic_json(out / "verification.json", summary); return summary


def _rsa_bound_one(args):
    """Bound-state RSA (sites.json rsa_bound) per store query site of one PINDER record, in the build's site order
    (production_graphs._pinder_one: paired rows whose residue maps into the AB topology); checked against the store's
    rsa_free and targets."""
    root, cid, split, store_path = args
    from jaxpropka.topology import load_topology
    from .gqt_paired_pinder import _paired_rows, _read_cif_gz
    store = ProductionStore(Path(store_path)); raw = store.raw(cid); store.close()
    src = source(root, f"{PINDER}/entries/{cid}")
    keys = load_topology(_read_cif_gz(src / "AB.cif.gz"), gap_policy="cap", freeze_disulfides=True).keys
    lookup = {(k.chain, k.number, k.insertion) for k in keys}
    rows = [row for row in _paired_rows(src, split) if row["key"][:3] in lookup]
    if len(rows) != len(raw["rsa_free"]): raise AssertionError((cid, "site count differs from the store"))
    free = np.asarray([np.nan if r["rsa_free"] is None else r["rsa_free"] for r in rows], np.float32)
    if not np.allclose(free, raw["rsa_free"], equal_nan=True): raise AssertionError((cid, "rsa_free differs from the store"))
    if not np.allclose(np.asarray([(r["target_ab"], r["target_free"]) for r in rows], np.float32).T, raw["targets"]):
        raise AssertionError((cid, "targets differ from the store"))
    bound = np.asarray([np.nan if r.get("rsa_bound") is None else r["rsa_bound"] for r in rows], np.float32)
    return cid, bound, {"sites": len(rows), "missing": int(np.sum(~np.isfinite(bound))),
                        "bound_above_free": int(np.sum(np.clip(bound, 0, 1) > np.clip(free, 0, 1) + 1e-4))}


def build_rsa_bound(root, workers=64, limit=None):
    """<runtime>/training/<version>/pinder/rsa-bound-v1: bound-state RSA aligned to store query sites (ContactTable
    layout, field scores.npy). limit: smoke build into rsa-bound-v1-smoke."""
    root = Path(root); manifest = read(output(root, "pinder") / "manifest-v1.json"); store = manifest["store"]
    out = output(root, "pinder") / (RSA_BOUND_TABLE + ("-smoke" if limit else "")); out.mkdir(parents=True, exist_ok=True); began = time.time()
    records = manifest["records"][:limit]; ids = [r["id"] for r in records]
    with ProcessPoolExecutor(workers) as ex: results = list(ex.map(_rsa_bound_one, [(root, r["id"], r["split"], store) for r in records], chunksize=8))
    offsets = np.cumsum([0] + [len(s) for _, s, _ in results]).astype(np.int64)
    np.save(out / "scores.npy", np.concatenate([s for _, s, _ in results])); np.save(out / "offsets.npy", offsets)
    atomic_json(out / "ids.json", ids); checks = [c for _, _, c in results]
    summary = {"version": RSA_BOUND_TABLE, "definition": "sites.json rsa_bound per store query site (PINDER AB state)",
               "structures": len(ids), "sites": int(offsets[-1]), "missing": sum(c["missing"] for c in checks),
               "clipped_bound_above_free": sum(c["bound_above_free"] for c in checks),
               "manifest_sha256": digest(output(root, "pinder") / "manifest-v1.json"),
               "files": {f: digest(out / f) for f in ("scores.npy", "offsets.npy", "ids.json")}, "seconds": round(time.time() - began, 1)}
    atomic_json(out / "verification.json", summary); return summary


class ContactTable:
    """Per-store-site values by structure id (memory-mapped): contact scores (default) or, with table=RSA_BOUND_TABLE,
    bound-state RSA."""

    def __init__(self, root, table=TABLE):
        path = output(root, "pinder") / table; self.scores = np.load(path / "scores.npy", mmap_mode="r")
        self.offsets = np.load(path / "offsets.npy"); self.index = {cid: i for i, cid in enumerate(read(path / "ids.json"))}
        self.verification_sha256 = digest(path / "verification.json")

    def __call__(self, cid):
        i = self.index[cid]; return np.asarray(self.scores[self.offsets[i]:self.offsets[i + 1]])


def main(argv=None):
    p = argparse.ArgumentParser(prog="pkatrain.production_aux"); sub = p.add_subparsers(dest="action", required=True)
    for name in ("build", "build-rsa-bound"):
        b = sub.add_parser(name); b.add_argument("--workers", type=int, default=64); b.add_argument("--limit", type=int)
    a = p.parse_args(argv); fn = build if a.action == "build" else build_rsa_bound
    print(json.dumps(fn(Path(os.environ["PKABENCH_RUNTIME"]), a.workers, a.limit), indent=1))


if __name__ == "__main__":
    main()
