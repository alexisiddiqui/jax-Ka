"""FASPR-packed side-chain node features (2026-10-19, user: "build the packed side-chain table, run the checks, then
train original, packed and original+packed (mixture) on allres 15/20 at 50% seed 17, scored with side chains hidden
and packed shown").

sidechains-packed-v1 has the sidechains-v1 layout (pkanet.graph.sidechain_features: 32 named side-chain heavy-atom slots,
local-frame xyz / 10 A and presence; 128 float16 columns per node), but the side chains come from FASPR
($FASPR, _install/FASPR) given only the store's backbone (backbone-v1: N, CA, C, O, a missing O stored as C) and residue
types (node one-hot columns 0-19). Deposited side-chain atoms are never read.
- features.npy: every node of the structure packed together (PINDER: the bound complex; pKPDB / benchmark-val: the
  structure).
- free.npy (pinder only): each chain (backbone-v1 chain index) packed alone, the input of the partner-free branch, whose
  graph has no edges between chain indices (checked per complex).
FASPR input: one PDB per packing, chain letter ALPHABET[chain index % 62], residues numbered from 1 within each letter;
output atoms are matched back by (letter, number). A structure FASPR fails on keeps zero rows (no side chains) and is
listed in verification.json.

<runtime>/training/<version>/<dataset>/sidechains-packed-v1/{features.npy, free.npy (pinder), offsets.npy, ids.json,
verification.json}. Reads only the production stores and backbone-v1 (no squashfs images).

Side-chain sources for training (production_train --sidechain-source; the model always sees 128 side-chain columns):
- original: sidechains-v1 (deposited coordinates; the PINDER free branch shares the bound coordinates).
- packed:   sidechains-packed-v1 (the free branch gets free.npy).
- mix:      both tables are loaded (256 columns, original then packed); production_inputs picks one per structure with
            probability 1/2 (AB and free share the draw) before the side-chain view is applied; without a key (validation,
            prediction) the packed columns are used.

  python -m pkatrain.production_packed build {pinder,pkpdb,benchmark-val} [--workers 72] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import string
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest
from .production_graphs import ProductionStore, output, read
from .production_inputs import BackboneTable
from .production_sidechains import WIDTH, SidechainTable

TABLE = "sidechains-packed-v1"
SOURCES = ("original", "packed", "mix")
ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
         "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
W = {}


def faspr(backbone, aa, chain, binary=None):
    """Side-chain atoms {name: xyz} per residue packed by FASPR from backbone (n, 4, 3: N, CA, C, O), residue type
    indices (jaxpropka AA_TO_INDEX) and chain indices."""
    from jaxpropka.parameters import AA_TO_INDEX
    from pkanet.graph import SIDECHAIN_ATOMS
    res3 = {i: THREE[a] for a, i in AA_TO_INDEX.items() if a in THREE}; side = set(SIDECHAIN_ATOMS)
    letters = [ALPHABET[int(c) % len(ALPHABET)] for c in chain]; count = {}; keys = []
    for letter in letters: count[letter] = count.get(letter, 0) + 1; keys.append((letter, count[letter]))
    if max(count.values()) > 9999: raise ValueError("more than 9999 residues under one chain letter")
    lines = []; serial = 1
    for i, (letter, number) in enumerate(keys):
        for name, element, j in ((" N  ", "N", 0), (" CA ", "C", 1), (" C  ", "C", 2), (" O  ", "O", 3)):
            x, y, z = backbone[i, j]
            lines.append(f"ATOM  {serial % 100000:5d} {name} {res3[int(aa[i])]} {letter}{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element}"); serial += 1
    index = {key: i for i, key in enumerate(keys)}; atoms = [{} for _ in keys]
    with tempfile.TemporaryDirectory(prefix="faspr-", dir=os.environ.get("TMPDIR")) as d:
        (Path(d) / "in.pdb").write_text("\n".join(lines) + "\nEND\n")
        subprocess.run([binary or os.environ["FASPR"], "-i", str(Path(d) / "in.pdb"), "-o", str(Path(d) / "out.pdb")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=1800)
        for line in (Path(d) / "out.pdb").read_text().splitlines():
            if not line.startswith("ATOM"): continue
            name = line[12:16].strip()
            if name in side: atoms[index[(line[21], int(line[22:26]))]][name] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return atoms


def packed_features(backbone, aa, chain, groups=None):
    """sidechain_features of the FASPR packing; groups: node index arrays packed separately (default: all together)."""
    from pkanet.graph import sidechain_features
    atoms = [{} for _ in range(len(aa))]
    for rows in (groups if groups is not None else [np.arange(len(aa))]):
        for i, a in zip(rows, faspr(backbone[rows], aa[rows], chain[rows])): atoms[i] = a
    return sidechain_features(backbone, atoms)


def _init(root, dataset, store_path):
    W["store"] = ProductionStore(Path(store_path)); W["backbone"] = BackboneTable(Path(root), dataset); W["dataset"] = dataset


def _one(cid):
    raw = W["store"].raw(cid); table = W["backbone"](cid); n = len(raw["nodes"])
    if len(table) != n: raise AssertionError((cid, "backbone table does not match the store"))
    backbone = table[:, :12].reshape(n, 4, 3); chain = table[:, 12].astype(int); aa = raw["nodes"][:, :20].argmax(1)
    began = time.time(); free = None; error = None
    try:
        bound = packed_features(backbone, aa, chain)
        if W["dataset"] == "pinder":
            neighbors, mask = raw["neighbors"].astype(int), raw["branch_edge_mask"][1].astype(bool)
            if np.any(mask & (chain[neighbors] != chain[:, None])): raise AssertionError((cid, "free branch has edges between chain indices"))
            free = packed_features(backbone, aa, chain, [np.flatnonzero(chain == c) for c in np.unique(chain)])
    except (subprocess.SubprocessError, ValueError, KeyError) as e:
        error = repr(e); bound = np.zeros((n, WIDTH), np.float32); free = bound if W["dataset"] == "pinder" else None
    present = bound.reshape(n, -1, 4)[..., 3].sum(1)
    return cid, bound.astype(np.float16), None if free is None else free.astype(np.float16), \
        {"nodes": n, "seconds": time.time() - began, "error": error, "residues_without_sidechain_atoms": int(np.sum(present == 0))}


def build(root, dataset, workers=72, limit=None):
    root = Path(root); manifest = read(output(root, dataset) / "manifest-v1.json")
    out = output(root, dataset) / (TABLE + ("-smoke" if limit else "")); out.mkdir(parents=True, exist_ok=True); began = time.time()
    ids = [r["id"] for r in manifest["records"]][:limit]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(root), dataset, manifest["store"])) as ex: results = list(ex.map(_one, ids, chunksize=2))
    offsets = np.cumsum([0] + [len(f) for _, f, _, _ in results]).astype(np.int64)
    np.save(out / "features.npy", np.concatenate([f for _, f, _, _ in results])); np.save(out / "offsets.npy", offsets)
    files = ["features.npy", "offsets.npy", "ids.json"]
    if dataset == "pinder": np.save(out / "free.npy", np.concatenate([f for _, _, f, _ in results])); files.append("free.npy")
    atomic_json(out / "ids.json", ids); checks = {cid: c for cid, _, _, c in results}; failed = {cid: c["error"] for cid, c in checks.items() if c["error"]}
    binary = Path(os.environ["FASPR"])
    summary = {"version": TABLE, "dataset": dataset, "definition": __doc__.split("\n\n")[1].replace("\n", " "),
               "structures": len(ids), "nodes": int(offsets[-1]), "manifest_sha256": digest(output(root, dataset) / "manifest-v1.json"),
               "backbone_verification_sha256": BackboneTable(root, dataset).verification_sha256,
               "faspr": {"binary": str(binary), "binary_sha256": digest(binary), "commit": os.environ.get("FASPR_COMMIT")},
               "failed": len(failed), "failures": failed, "residues_without_sidechain_atoms": sum(c["residues_without_sidechain_atoms"] for c in checks.values()),
               "faspr_seconds": round(sum(c["seconds"] for c in checks.values()), 1),
               "files": {f: digest(out / f) for f in files}, "seconds": round(time.time() - began, 1)}
    atomic_json(out / "verification.json", summary); return {k: v for k, v in summary.items() if k != "failures"}


class PackedSidechainTable:
    """Per-node packed side-chain features by structure id; pinder: (2, nodes, 128) = (bound packing, each chain alone),
    otherwise (nodes, 128)."""

    def __init__(self, root, dataset, table=TABLE):
        path = output(root, dataset) / table; self.features = np.load(path / "features.npy", mmap_mode="r")
        self.free = np.load(path / "free.npy", mmap_mode="r") if (path / "free.npy").exists() else None
        self.offsets = np.load(path / "offsets.npy"); self.index = {cid: i for i, cid in enumerate(read(path / "ids.json"))}
        self.verification_sha256 = digest(path / "verification.json")

    def __call__(self, cid):
        i = self.index[cid]; a, b = self.offsets[i], self.offsets[i + 1]; bound = np.asarray(self.features[a:b], np.float32)
        return bound if self.free is None else np.stack((bound, np.asarray(self.free[a:b], np.float32)))


def sidechain_tables(root, dataset, source):
    """The loader's side-chain tables for --sidechain-source (in column order)."""
    if source not in SOURCES: raise ValueError(source)
    return {"original": lambda: [SidechainTable(root, dataset)], "packed": lambda: [PackedSidechainTable(root, dataset)],
            "mix": lambda: [SidechainTable(root, dataset), PackedSidechainTable(root, dataset)]}[source]()


def main(argv=None):
    p = argparse.ArgumentParser(prog="pkatrain.production_packed"); sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build"); b.add_argument("dataset", choices=("pinder", "pkpdb", "benchmark-val")); b.add_argument("--workers", type=int, default=72)
    b.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    print(json.dumps(build(Path(os.environ["PKABENCH_RUNTIME"]), a.dataset, a.workers, a.limit), indent=1), flush=True)


if __name__ == "__main__":
    main()
