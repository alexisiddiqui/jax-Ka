"""Rigid-body distance/angle mutation audit for the oGQT interface head."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pkabench.runtime import atomic_json, digest, require_compute


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


SEED = 17
N_COMPLEXES = 50
CONDITIONS = {
    "native": (0.0, 0.0),
    "translate_2A": (2.0, 0.0),
    "translate_5A": (5.0, 0.0),
    "rotate_10deg": (0.0, 10.0),
    "rotate_30deg": (0.0, 30.0),
    "combined_2A_15deg": (2.0, 15.0),
}
GRAPH_FIELDS = ("nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch", "query_residue",
                "query_group", "site_residue", "site_type", "site_mask", "site_neighbors", "site_edge",
                "site_edge_mask", "site_switch", "query_site")


def _source(root): return Path(root) / "pretraining/pinder-pkai-v1"
def _out(root): return Path(root) / "training/ogqt-auxiliary-pilot-v1/interface-structural-mutations-v1"


def _read_cif(path):
    from biotite.structure.io import pdbx
    with gzip.open(path, "rt") as stream: file = pdbx.CIFFile.read(stream)
    return pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True, include_bonds=True)


def _unit(rng):
    value = rng.normal(size=3); return value / np.linalg.norm(value)


def _rotation(axis, degrees):
    angle = np.deg2rad(degrees); x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross


def _transform(atoms, partner_b, translation, angle, rng):
    from scipy.spatial import cKDTree
    moved = atoms.copy(); mask = np.isin(moved.chain_id, partner_b)
    heavy = np.char.upper(np.asarray(moved.element, str)) != "H"
    a = moved.coord[(~mask) & heavy]; original_b = moved.coord[mask]
    best = None
    for attempt in range(1 if translation == 0 and angle == 0 else 128):
        b = original_b.copy()
        if translation:
            b = b + translation * _unit(rng)
        if angle:
            signed = angle if rng.random() < .5 else -angle; center = b.mean(0)
            b = (b - center) @ _rotation(_unit(rng), signed).T + center
        moved.coord[mask] = b
        minimum = float(cKDTree(a).query(moved.coord[mask & heavy])[0].min())
        if best is None or minimum > best[1]:
            best = (moved.copy(), minimum, attempt + 1)
        if minimum >= 1.5:
            return moved, minimum, attempt + 1
    return best


def _prepare_task(task):
    from jaxpropka.parameters import GROUPS
    from jaxpropka.topology import load_topology
    from pkanet.graph import geometry
    from pkabench.annotate import annotate
    from pkatrain.gqt_paired_pinder import _bucket_n, _load_one
    from pkatrain.site_graph_data import build_site_graph
    root, base, record, condition, translation, angle = task
    root = Path(root); base = Path(base); cid = record["id"]; source = _source(root) / "entries" / cid
    sites = read(source / "sites.json"); chain_partner = {}
    for site in sites:
        chain = str(site["chain"]); partner = str(site["partner"])
        if chain in chain_partner and chain_partner[chain] != partner: raise ValueError((cid, chain, "partner conflict"))
        chain_partner[chain] = partner
    atoms = _read_cif(source / "AB.cif.gz"); chains = set(map(str, atoms.chain_id))
    if set(chain_partner) != chains: return {"id": cid, "condition": condition, "accepted": False,
        "reason": "partner mapping does not cover all chains"}
    partner_b = sorted(chain for chain, partner in chain_partner.items() if partner == "B")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{SEED}|{cid}|{condition}".encode()).digest()[:8], "little"))
    moved, minimum, attempts = _transform(atoms, partner_b, translation, angle, rng)
    if minimum < 1.5: return {"id": cid, "condition": condition, "accepted": False,
        "reason": "no nonclashing draw in 128 attempts", "min_interpartner_atom_A": minimum,
        "draw_attempts": attempts}
    topology = load_topology(moved, gap_policy="cap", freeze_disulfides=True)
    partners = {name: sorted(chain for chain, value in chain_partner.items() if value == name) for name in ("A", "B")}
    annotations, _, sasa_summary = annotate(topology, partners)
    # This is the exact contract used by pinder-pkai-v1 sites.json.  The
    # auxiliary pilot specification had incorrectly described it as dSASA >
    # 10 A^2; the stored labels are min partner heavy-atom distance <= 10 A.
    label_by_residue = {(x["chain"], x["resnum"], x["icode"]): x["min_partner_distance"] <= 10 for x in annotations}
    graph, frames_valid = geometry(topology.backbone, topology.chain_index, radius=20.0)
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[topology.native_index],
        np.stack((topology.nterm, topology.cterm, np.zeros(topology.n_residues, bool), frames_valid), axis=-1)), axis=-1)
    graph.update(nodes=nodes.astype(np.float32), node_mask=np.ones(topology.n_residues, bool))
    lookup = {(key.chain, key.number, key.insertion): i for i, key in enumerate(topology.keys)}
    queries = [(lookup[tuple(key[:3])], GROUPS.index(key[3])) for key in record["keys"]]
    query = np.asarray(queries, np.int32); graph.update(query_residue=query[:, 0], query_group=query[:, 1])
    site = build_site_graph(topology.backbone, topology.chain_index, nodes, query[:, 0], query[:, 1])
    q = len(query); raw = {**graph, **site,
        "branch_edge_mask": np.stack((graph["edge_mask"], graph["edge_mask"])),
        "branch_site_edge_mask": np.stack((site["site_edge_mask"], site["site_edge_mask"])),
        "targets": np.zeros((2, q), np.float32), "w_burial": np.ones(q, np.float32),
        "w_interface": np.ones(q, np.float32), "interface": np.asarray([
            label_by_residue[tuple(key[:3])] for key in record["keys"]], bool),
        "partner_distance_A": np.zeros(q, np.float32), "rsa_free": np.zeros(q, np.float32)}
    local = {**record, "n": topology.n_residues, "k": graph["neighbors"].shape[1], "q": q,
             "s": len(site["site_type"]), "sk": site["site_neighbors"].shape[1]}
    capacity = read(base / "manifest.json")["capacities"][_bucket_n(local["n"])]
    if any(local[name] > cap for name, cap in zip(("n", "k", "q", "s", "sk"), capacity)):
        return {"id": cid, "condition": condition, "accepted": False, "reason": "capacity exceeded"}
    packed = _load_one(raw, local, capacity, read(base / "manifest.json")["normalization"])[0]
    path = base / "interface-structural-mutations-v1/graphs" / cid / f"{condition}.npz"
    path.parent.mkdir(parents=True, exist_ok=True); pending = path.with_suffix(f".pending-{os.getpid()}")
    with pending.open("wb") as stream: np.savez(stream, **{name: packed[name][0] for name in GRAPH_FIELDS})
    os.replace(pending, path)
    return {"id": cid, "condition": condition, "accepted": True, "path": str(path), "sha256": digest(path),
        "bucket": _bucket_n(local["n"]), "q": q, "keys": record["keys"],
        "labels": raw["interface"].astype(int).tolist(), "min_interpartner_atom_A": minimum,
        "draw_attempts": attempts,
        "interface_residues": sasa_summary["interface_residues"], "half_sum_buried_area": sasa_summary["half_sum_buried_area"]}


def prepare(root):
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), allow_comp1400=True)
    root = Path(root); base = root / "training/ogqt-auxiliary-pilot-v1"; manifest = read(base / "manifest.json")
    validation = [x for x in manifest["records"] if x["split"] == "val"]
    rank = lambda row: hashlib.sha256(f"structural-mutations-v1|{row['id']}".encode()).hexdigest()
    selected = sorted(validation, key=rank)[:N_COMPLEXES]
    tasks = [(str(root), str(base), row, condition, *values) for row in selected for condition, values in CONDITIONS.items()]
    import multiprocessing as mp
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 32)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        receipts = list(pool.map(_prepare_task, tasks, chunksize=1))
    output = _out(root); output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", {"version": "interface-structural-mutations-v1", "seed": SEED,
        "complexes": [x["id"] for x in selected], "conditions": CONDITIONS, "records": receipts,
        "accepted": sum(x["accepted"] for x in receipts), "rejected": sum(not x["accepted"] for x in receipts),
        "rejection_reasons": {reason: sum(not x["accepted"] and x["reason"] == reason for x in receipts)
            for reason in sorted({x.get("reason") for x in receipts if not x["accepted"]})},
        "label": "recomputed residue minimum partner-heavy-atom distance <= 10 A after each rigid transform",
        "delta_sasa_role": "reported structural diagnostic; not the binary interface label in pinder-pkai-v1",
        "clash_threshold_A": 1.5, "test_data_included": False})


def run(root):
    import jax
    from pkanet.ogqt import initialize_auxiliary, predict_multi
    from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine
    from pkatrain.trainer import load_checkpoint
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    root = Path(root); base = root / "training/ogqt-auxiliary-pilot-v1"; manifest = read(base / "manifest.json")
    mutation = read(_out(root) / "manifest.json"); records = [x for x in mutation["records"] if x["accepted"]]
    verification = read(base / "standard/seed-17/verification.json")
    params = initialize_auxiliary(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = AuxiliaryEngine(params, 0, {"burial": 1, "interface": 1}); state = engine.optimizer.init(params)
    params, _, _ = load_checkpoint(base / "standard/seed-17/checkpoints" /
        f"epoch-{verification['selected_epoch']:03d}", (params, state))
    predict = jax.jit(jax.vmap(predict_multi, in_axes=(None, 0)))
    rows = []
    for number, record in enumerate(records, 1):
        if digest(record["path"]) != record["sha256"]: raise AssertionError("graph hash")
        with np.load(record["path"], allow_pickle=False) as handle:
            graph = {name: jax.numpy.asarray(handle[name][None]) for name in GRAPH_FIELDS}
        scores = np.asarray(predict(params, graph)["interface"])[0]
        for qi, key in enumerate(record["keys"]):
            rows.append({"complex_id": record["id"], "condition": record["condition"], "chain": key[0],
                "resnum": key[1], "icode": key[2], "group": key[3], "interface": bool(record["labels"][qi]),
                "score": float(scores[qi]), "min_interpartner_atom_A": record["min_interpartner_atom_A"],
                "half_sum_buried_area": record["half_sum_buried_area"]})
        if number % 25 == 0: print(json.dumps({"predicted": number, "total": len(records)}), flush=True)
    path = _out(root) / "predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(_out(root) / "run-verification.json", {"passed": True, "records": len(records),
        "predictions": len(rows), "predictions_sha256": digest(path), "checkpoint_epoch": verification["selected_epoch"],
        "test_data_included": False})


def _deduplicate(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[(row["complex_id"], row["condition"], row["chain"], row["resnum"], row["icode"])].append(row)
    return [{"complex_id": key[0], "condition": key[1], "chain": key[2], "resnum": key[3], "icode": key[4],
        "interface": values[0]["interface"],
        "score": float(np.mean([x["score"] for x in values]))} for key, values in grouped.items()]


def _confusion(rows, threshold):
    label = np.asarray([x["interface"] for x in rows], bool); call = np.asarray([x["score"] for x in rows]) >= threshold
    tp = int(np.sum(label & call)); fp = int(np.sum(~label & call)); fn = int(np.sum(label & ~call)); tn = int(np.sum(~label & ~call))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "fpr": fp/(fp+tn) if fp+tn else None,
            "fnr": fn/(fn+tp) if fn+tp else None}


def report(root):
    from pkabench.ogqt_auxiliary_roc import _roc
    require_compute(threads=2, allow_comp1400=True)
    root = Path(root); output = _out(root)
    with (output / "predictions.csv").open() as stream:
        rows = [{**x, "interface": x["interface"].lower() == "true", "score": float(x["score"])} for x in csv.DictReader(stream)]
    rows = _deduplicate(rows)
    conditions_by_complex = defaultdict(set)
    for row in rows:
        conditions_by_complex[row["complex_id"]].add(row["condition"])
    threshold = read(root / "training/ogqt-auxiliary-pilot-v1/interface-counterfactual/summary.json")["arms"]["standard"]["youden"]["native"]["threshold"]
    # Recomputed native labels must exactly reproduce the labels used in the
    # original validation evaluation.  This prevents annotation drift from
    # being mistaken for sensitivity to a structural mutation.
    reference_path = root / "training/ogqt-auxiliary-pilot-v1/interface-counterfactual/predictions.csv"
    with reference_path.open() as stream:
        reference_raw = [x for x in csv.DictReader(stream) if x["arm"] == "standard"]
    reference = {(x["complex_id"], x["chain"], x["resnum"], x["icode"]):
        x["interface"].lower() == "true" for x in reference_raw}
    native = [x for x in rows if x["condition"] == "native"]
    comparable = [x for x in native if (x["complex_id"], x["chain"], x["resnum"], x["icode"]) in reference]
    mismatches = sum(x["interface"] != reference[(x["complex_id"], x["chain"], x["resnum"], x["icode"])]
        for x in comparable)
    if mismatches:
        raise AssertionError(f"native interface-label mismatch: {mismatches}/{len(comparable)}")
    summary = {"threshold": threshold, "conditions": {},
        "comparison_design": "each mutation is compared on complexes with both native and mutated records",
        "native_label_check": {
        "compared_residues": len(comparable), "mismatches": mismatches, "passed": True}}
    native_by_key = {(x["complex_id"], x["chain"], x["resnum"], x["icode"]): x for x in native}
    for condition in CONDITIONS:
        available = {x["complex_id"] for x in rows if x["condition"] == condition}
        matched = available & {x["complex_id"] for x in native}
        subset = [x for x in rows if x["condition"] == condition and x["complex_id"] in matched]
        paired = [(native_by_key[(x["complex_id"], x["chain"], x["resnum"], x["icode"])], x) for x in subset]
        roc = _roc([x["interface"] for x in subset], [x["score"] for x in subset])
        per_complex = []
        per_confusion = []
        for complex_id in sorted({x["complex_id"] for x in subset}):
            block = [x for x in subset if x["complex_id"] == complex_id]
            block_roc = _roc([x["interface"] for x in block], [x["score"] for x in block])
            if block_roc is not None:
                per_complex.append(block_roc[2])
            per_confusion.append(_confusion(block, threshold))
        finite_fpr = [x["fpr"] for x in per_confusion if x["fpr"] is not None]
        finite_fnr = [x["fnr"] for x in per_confusion if x["fnr"] is not None]
        transition_deltas = defaultdict(list)
        for before, after in paired:
            transition = ("positive_to_negative" if before["interface"] and not after["interface"] else
                "negative_to_positive" if not before["interface"] and after["interface"] else
                "stable_positive" if before["interface"] else "stable_negative")
            transition_deltas[transition].append(after["score"] - before["score"])
        transition_summary = {name: {"sites": len(values), "mean_score_change": float(np.mean(values)),
            "median_score_change": float(np.median(values))} for name, values in transition_deltas.items()}
        changed = transition_deltas["positive_to_negative"] + transition_deltas["negative_to_positive"]
        responsive = (sum(value < 0 for value in transition_deltas["positive_to_negative"]) +
            sum(value > 0 for value in transition_deltas["negative_to_positive"]))
        summary["conditions"][condition] = {"residues": len(subset), "complexes": len({x["complex_id"] for x in subset}),
            "positives": sum(x["interface"] for x in subset), "roc_auc": roc[2] if roc else None,
            "macro_roc_auc": float(np.mean(per_complex)) if per_complex else None,
            "eligible_auc_complexes": len(per_complex), "confusion": _confusion(subset, threshold),
            "equal_complex_fpr": float(np.mean(finite_fpr)) if finite_fpr else None,
            "equal_complex_fnr": float(np.mean(finite_fnr)) if finite_fnr else None,
            "mean_score_change": float(np.mean([b["score"] - a["score"] for a, b in paired])) if paired else None,
            "native_positive_to_negative": sum(a["interface"] and not b["interface"] for a, b in paired),
            "native_negative_to_positive": sum(not a["interface"] and b["interface"] for a, b in paired),
            "score_change_by_label_transition": transition_summary,
            "responsive_direction_fraction_on_changed_labels": responsive / len(changed) if changed else None}
    atomic_json(output / "summary.json", summary)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    labels = list(CONDITIONS); auc = [summary["conditions"][x]["roc_auc"] for x in labels]
    axes[0].bar(np.arange(len(labels)), auc, color="#54a24b"); axes[0].axhline(.5, color="grey", ls=":")
    axes[0].set(xticks=np.arange(len(labels)), xticklabels=labels, ylim=(0, 1), ylabel="Pooled ROC-AUC",
                title="Interface discrimination after rigid mutations"); axes[0].tick_params(axis="x", rotation=35)
    fpr = [summary["conditions"][x]["confusion"]["fpr"] for x in labels]
    fnr = [summary["conditions"][x]["confusion"]["fnr"] for x in labels]; x = np.arange(len(labels)); width=.36
    axes[1].bar(x-width/2, fpr, width, label="False-positive rate", color="#e45756")
    axes[1].bar(x+width/2, fnr, width, label="False-negative rate", color="#4c78a8")
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Rate", title=f"Errors at native Youden threshold {threshold:.3f}")
    axes[1].tick_params(axis="x", rotation=35); axes[1].legend(frameon=False)
    fig.savefig(output / "structural_mutation_interface.png", dpi=220); fig.savefig(output / "structural_mutation_interface.svg"); plt.close(fig)


def main():
    import sys
    root = Path(os.environ["PKABENCH_RUNTIME"]); action = sys.argv[1]
    if action == "prepare": prepare(root)
    elif action == "run": run(root)
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
