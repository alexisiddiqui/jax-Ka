"""Rebuild the frozen 5k GQT cohort at a matched 25 A maximum radius."""
from __future__ import annotations

import concurrent.futures
import copy
import json
import os
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from .records import read


RADIUS = 25.0
MODES = ("backbone", "sidechain")
_WORKER_ROOT = None
_WORKER_PARENTS = None


def original_source(root, mode):
    if mode == "backbone":
        return root / "pretraining/gqt-pkai-parameter-sweep-v1/gqt-long/50k"
    if mode == "sidechain":
        return root / "pretraining/augmentation-sidechains-v1/source"
    raise ValueError(mode)


def destination(root, mode):
    return root / "pretraining/gqt-neighbor-cutoff-v1-triton/sources-25A" / mode


def _coordinates(root, parent, row):
    """Recover the exact C-alpha/node order used by the frozen graph."""
    if "record_sha256" in row:
        from jaxpropka.topology import load_topology
        from pkabench.prep import read_cif
        record_root = Path(parent.get("source", parent.get("validation_source")))
        if not (record_root / "records").is_dir():
            record_root = Path(read(record_root / "manifest.json")["source"])
        record_path = record_root / "records" / f"{row['complex_id']}.json"
        if digest(record_path) != row["record_sha256"]:
            raise AssertionError((row["complex_id"], "record hash"))
        record = read(record_path); cif = Path(record["structures"]["AB"])
        if digest(cif) != record["structure_sha256"]["AB"]:
            raise AssertionError((row["complex_id"], "structure hash"))
        top = load_topology(read_cif(cif), gap_policy="cap", freeze_disulfides=True)
        if top.n_residues != row["n"]:
            raise AssertionError((row["complex_id"], top.n_residues, row["n"]))
        return np.asarray(top.backbone), np.asarray(top.chain_index)

    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from pkabench.conformers import resolve
    from pkabench.pkpdb_pilot_refs import cif as read_gzip_cif, sequences
    cid = row["complex_id"]
    source = root / "pretraining/pkpdb-v1/structures" / cid[1:3] / f"{cid}.cif.gz"
    if digest(source) != row["source_sha256"]:
        raise AssertionError((cid, "source hash"))
    file = read_gzip_cif(source); chains, _ = sequences(file)
    file, _ = resolve(file, [item["chain"] for item in chains])
    label = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=False)
    author = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True)
    working = label.copy(); working.res_id = author.res_id.copy(); working.ins_code = author.ins_code.copy()
    starts = struc.get_residue_starts(working, add_exclusive_stop=True); by_key = {}
    for start, stop in zip(starts[:-1], starts[1:]):
        atoms = working[start:stop]
        key = (str(author.chain_id[start]), int(author.res_id[start]),
               str(author.ins_code[start]).strip(), str(author.res_name[start]))
        by_key[key] = {str(atom.atom_name): atom.coord for atom in atoms}
    key_file = root / "pretraining/augmentation-v1/contexts/node-keys" / f"{cid}.json"
    keys = read(key_file)["keys"]
    if len(keys) != row["n"]:
        raise AssertionError((cid, len(keys), row["n"]))
    residue_atoms = [by_key[tuple(key)] for key in keys]
    backbone = np.stack([
        np.stack([atoms[name] if name in atoms else atoms["C"] for name in ("N", "CA", "C", "O")])
        for atoms in residue_atoms
    ])
    chain_ids = {name: number for number, name in enumerate(dict.fromkeys(key[0] for key in keys))}
    chain = np.asarray([chain_ids[key[0]] for key in keys], np.int32)
    return backbone, chain


def _initialize_worker(runtime, parents):
    global _WORKER_ROOT, _WORKER_PARENTS
    _WORKER_ROOT = Path(runtime); _WORKER_PARENTS = parents


def _one(row):
    root = _WORKER_ROOT; parents = _WORKER_PARENTS; cid = row["complex_id"]
    outputs = {mode: destination(root, mode) / "data" / cid / "graph.npz" for mode in MODES}
    receipts = {mode: destination(root, mode) / "data" / cid / "receipt.json" for mode in MODES}
    if all(receipts[mode].exists() for mode in MODES):
        complete = {mode: read(receipts[mode]) for mode in MODES}
        if all(digest(outputs[mode]) == complete[mode]["sha256"] for mode in MODES):
            return complete
    from pkanet.graph import geometry
    backbone, chain = _coordinates(root, parents["backbone"], row)
    graph25, frames_valid = geometry(backbone, chain, radius=RADIUS, rbf_radius=RADIUS)
    result = {}
    for mode in MODES:
        src = original_source(root, mode); source_row = parents[mode]["by_id"][cid]
        source_path = src / "data" / cid / "graph.npz"
        if digest(source_path) != source_row["sha256"]:
            raise AssertionError((cid, mode, "parent graph hash"))
        with np.load(source_path, allow_pickle=False) as handle:
            data = {name: handle[name] for name in handle.files}
        if len(data["nodes"]) != len(backbone):
            raise AssertionError((cid, mode, len(data["nodes"]), len(backbone)))
        # Frame validity is already embedded in the backbone node features;
        # proving it unchanged catches coordinate/order drift.
        if mode == "backbone" and not np.array_equal(data["nodes"][:, 23].astype(bool), frames_valid):
            raise AssertionError((cid, "frame validity changed"))
        data.update(graph25)
        folder = outputs[mode].parent; folder.mkdir(parents=True, exist_ok=True)
        pending = folder / f"graph.npz.pending-{os.getpid()}"
        with pending.open("wb") as stream: np.savez_compressed(stream, **data)
        os.replace(pending, outputs[mode])
        receipt = copy.deepcopy(source_row)
        receipt.update(parent_graph_sha256=source_row["sha256"], k=int(graph25["neighbors"].shape[1]),
                       edge_count=int(graph25["edge_mask"].sum()), radius_A=RADIUS,
                       rbf_radius_A=RADIUS, sha256=digest(outputs[mode]))
        atomic_json(receipts[mode], receipt); result[mode] = receipt
    return result


def prepare(root):
    parents = {}
    for mode in MODES:
        src = original_source(root, mode); manifest = read(src / "manifest.json")
        parents[mode] = {**manifest, "by_id": {row["complex_id"]: row for row in manifest["records"]}}
    backbone = parents["backbone"]
    if backbone["train"] != parents["sidechain"]["train"] or backbone["val"] != parents["sidechain"]["val"]:
        raise AssertionError("Backbone and side-chain cohorts differ")
    workers = min(int(os.environ["SLURM_CPUS_PER_TASK"]), 48)
    rows = {mode: [] for mode in MODES}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_initialize_worker, initargs=(str(root), parents)
    ) as pool:
        for number, completed in enumerate(pool.map(_one, backbone["records"], chunksize=1), 1):
            for mode in MODES: rows[mode].append(completed[mode])
            if number % 50 == 0: print(json.dumps({"regraphed": number, "total": len(backbone["records"])}), flush=True)
    for mode in MODES:
        src = original_source(root, mode); out = destination(root, mode); parent = parents[mode]
        manifest = copy.deepcopy({key: value for key, value in parent.items() if key != "by_id"})
        manifest["records"] = rows[mode]
        capacities = {}
        for limit in (384, 768, 100000):
            members = [r for r in rows[mode] if (384 if r["n"] <= 384 else 768 if r["n"] <= 768 else 100000) == limit]
            if members:
                capacities[str(limit)] = [int(np.ceil(max(r[key] for r in members) / 32) * 32) for key in ("n", "k", "q")]
        manifest["capacities"] = capacities
        manifest["parent"] = {"path": str(src), "manifest_sha256": digest(src / "manifest.json")}
        manifest["config"].update(radius_A=RADIUS, rbf_radius_A=RADIUS)
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "preparation.json", {"passed": True, "records": len(rows[mode]),
                    "mode": mode, "radius_A": RADIUS, "rbf_radius_A": RADIUS,
                    "parent_manifest_sha256": digest(src / "manifest.json"),
                    "edge_count": sum(r["edge_count"] for r in rows[mode])})


def test(root):
    """One-record provenance/geometry gate; its output is reusable by prepare."""
    parents = {}
    for mode in MODES:
        manifest = read(original_source(root, mode) / "manifest.json")
        parents[mode] = {**manifest, "by_id": {row["complex_id"]: row for row in manifest["records"]}}
    row = parents["backbone"]["records"][0]
    _initialize_worker(str(root), parents); result = _one(row)
    for mode in MODES:
        receipt = result[mode]
        if receipt["radius_A"] != RADIUS or receipt["rbf_radius_A"] != RADIUS:
            raise AssertionError(receipt)
        if receipt["n"] != row["n"] or receipt["q"] != row["q"]:
            raise AssertionError(receipt)
    out = root / "pretraining/gqt-neighbor-cutoff-v1-triton"; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "regraph-tests.json", {"passed": True, "complex_id": row["complex_id"],
                "modes": list(MODES), "radius_A": RADIUS,
                "edge_counts": {mode: result[mode]["edge_count"] for mode in MODES}})


def mmap(root, mode):
    from .graph_mmap import build_bundle
    out = destination(root, mode); manifest = read(out / "manifest.json")
    result = build_bundle(out, manifest)
    print(json.dumps({"mode": mode, "mmap": str(result)}), flush=True)


if __name__ == "__main__":
    import sys
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]))
    root = Path(os.environ["PKABENCH_RUNTIME"]); action = sys.argv[1]
    if action == "test": test(root)
    elif action == "prepare": prepare(root)
    elif action == "mmap": mmap(root, sys.argv[2])
    else: raise ValueError(action)
