"""Side-chain node features for the production stores (2026-10-11).

The production graphs are backbone-only (24 node features). The side-chain variant appends the established 32 named
side-chain heavy-atom slots per residue (pkanet.graph.sidechain_features: local-frame xyz / 10 A and presence; the
experiment 12/37 side-chain GQT input), giving 152 node features. They are kept as a side table aligned to each
store's node order, built from the same structures and residue selection as the store:
- pinder: AB.cif.gz through load_topology (gap cap, frozen disulfides), as production_graphs._pinder_one; the free
  branch shares the AB coordinates, as its graph does.
- pkpdb: the pKPDB source CIF with production_graphs._pkpdb_one's conformer resolution and residue filter.
- benchmark-val: the AB structure through load_topology, as production_graphs._benchmark_one.
Each structure is checked against its store record: node count and residue identity (node one-hot columns 0-19).

<runtime>/training/<version>/<dataset>/sidechains-v1/{features.npy (float16, nodes x 128), offsets.npy, ids.json,
verification.json}. Reads the dataset structures, so on Isambard it runs under scripts/sqfs_run.sh.

  python -m pkatrain.production_sidechains build {pinder,pkpdb,benchmark-val} [--workers 72] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest
from .production_graphs import BENCHMARK_SOURCE, PINDER, PKPDB, PKPDB_STRUCTURES, ProductionStore, _local, output, read, source

TABLE = "sidechains-v1"
WIDTH = 128


def _topology_atoms(topology):
    return topology.backbone, [{str(a.atom_name): a.coord for a in topology.residue(i)} for i in range(topology.n_residues)], \
        np.asarray(topology.native_index)


def _pinder_atoms(root, cid):
    from jaxpropka.topology import load_topology
    from .gqt_paired_pinder import _read_cif_gz
    return _topology_atoms(load_topology(_read_cif_gz(source(root, f"{PINDER}/entries/{cid}") / "AB.cif.gz"), gap_policy="cap", freeze_disulfides=True))


def _benchmark_atoms(root, cid):
    from jaxpropka.topology import load_topology
    from pkabench.prep import read_cif
    src = root / BENCHMARK_SOURCE; r = read(src / "records" / f"{cid}.json"); cif = _local(root, r["structures"]["AB"])
    if digest(cif) != r["structure_sha256"]["AB"]: raise AssertionError((cid, "structure hash"))
    return _topology_atoms(load_topology(read_cif(cif), gap_policy="cap", freeze_disulfides=True))


def _pkpdb_atoms(root, cid):
    """production_graphs._pkpdb_one's residue selection (same conformer resolution, canonical heavy atoms of the
    selected chains), returning per-residue atom coordinates in node order."""
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from pkabench.conformers import resolve
    from pkabench.pkpdb_pilot_refs import cif
    from jaxpropka.parameters import THREE_TO_INDEX
    from pkabench.prep import CANONICAL
    entry = source(root, f"{PKPDB}/entries/{cid}"); receipt = read(entry / "receipt.json"); defects = read(entry / "defects.json")
    path = source(root, f"{PKPDB_STRUCTURES}/{cid[1:3]}/{cid}.cif.gz")
    if digest(path) != receipt["source_sha256"]: raise AssertionError((cid, "source structure hash"))
    selected = [row["chain"] for row in defects["sequences"]]
    file, _ = resolve(cif(path), selected)
    label = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=False)
    author = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True)
    working = label.copy(); working.res_id = author.res_id.copy(); working.ins_code = author.ins_code.copy()
    keep = np.isin(working.res_name, list(CANONICAL)) & np.isin(working.chain_id, selected) & ~np.isin(np.char.upper(working.element), ["H", "D"])
    starts = struc.get_residue_starts(working, add_exclusive_stop=True); backbone = []; atoms = []; identity = []
    for s, e in zip(starts[:-1], starts[1:]):
        if not keep[s]: continue
        a = working[s:e]; names = {str(n): i for i, n in enumerate(a.atom_name)}
        backbone.append(np.stack([a.coord[names[name]] if name in names else a.coord[names["C"]] for name in ("N", "CA", "C", "O")]))
        atoms.append({str(n): a.coord[i] for n, i in names.items()}); identity.append(THREE_TO_INDEX[str(a.res_name[0])])
    return np.asarray(backbone), atoms, np.asarray(identity)


ATOMS = {"pinder": _pinder_atoms, "pkpdb": _pkpdb_atoms, "benchmark-val": _benchmark_atoms}


def _one(args):
    root, dataset, cid, store_path = args
    from pkanet.graph import sidechain_features
    backbone, atoms, identity = ATOMS[dataset](Path(root), cid)
    store = ProductionStore(Path(store_path)); nodes = store.raw(cid)["nodes"]; store.close()
    if len(atoms) != len(nodes): raise AssertionError((cid, "node count differs from the store", len(atoms), len(nodes)))
    if not np.array_equal(identity, nodes[:, :20].argmax(1)): raise AssertionError((cid, "residue identity differs from the store"))
    features = sidechain_features(backbone, atoms)
    if features.shape != (len(nodes), WIDTH): raise AssertionError((cid, "feature shape", features.shape))
    present = features.reshape(len(nodes), -1, 4)[..., 3].sum(1)
    return cid, features.astype(np.float16), {"nodes": len(nodes), "residues_without_sidechain_atoms": int(np.sum(present == 0)),
                                             "max_abs_feature": float(np.max(np.abs(features), initial=0.0))}


def build(root, dataset, workers=72, limit=None):
    root = Path(root); manifest = read(output(root, dataset) / "manifest-v1.json"); store = manifest["store"]
    out = output(root, dataset) / (TABLE + ("-smoke" if limit else "")); out.mkdir(parents=True, exist_ok=True); began = time.time()
    ids = [r["id"] for r in manifest["records"]][:limit]
    with ProcessPoolExecutor(workers) as ex: results = list(ex.map(_one, [(root, dataset, cid, store) for cid in ids], chunksize=4))
    offsets = np.cumsum([0] + [len(f) for _, f, _ in results]).astype(np.int64)
    np.save(out / "features.npy", np.concatenate([f for _, f, _ in results])); np.save(out / "offsets.npy", offsets)
    atomic_json(out / "ids.json", ids); checks = [c for _, _, c in results]
    summary = {"version": TABLE, "dataset": dataset, "definition": "pkanet.graph.sidechain_features (32 named side-chain heavy-atom slots: local-frame xyz / 10 A, presence), float16",
               "structures": len(ids), "nodes": int(offsets[-1]), "manifest_sha256": digest(output(root, dataset) / "manifest-v1.json"),
               "residues_without_sidechain_atoms": sum(c["residues_without_sidechain_atoms"] for c in checks),
               "max_abs_feature": max(c["max_abs_feature"] for c in checks),
               "files": {f: digest(out / f) for f in ("features.npy", "offsets.npy", "ids.json")}, "seconds": round(time.time() - began, 1)}
    atomic_json(out / "verification.json", summary); return summary


class SidechainTable:
    """Per-node side-chain features by structure id (memory-mapped float16)."""

    def __init__(self, root, dataset):
        path = output(root, dataset) / TABLE; self.features = np.load(path / "features.npy", mmap_mode="r")
        self.offsets = np.load(path / "offsets.npy"); self.index = {cid: i for i, cid in enumerate(read(path / "ids.json"))}
        self.verification_sha256 = digest(path / "verification.json")

    def __call__(self, cid):
        i = self.index[cid]; return np.asarray(self.features[self.offsets[i]:self.offsets[i + 1]], np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pkatrain.production_sidechains"); sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build"); b.add_argument("dataset", choices=tuple(ATOMS)); b.add_argument("--workers", type=int, default=72)
    b.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    print(json.dumps(build(Path(os.environ["PKABENCH_RUNTIME"]), a.dataset, a.workers, a.limit), indent=1))


if __name__ == "__main__":
    main()
