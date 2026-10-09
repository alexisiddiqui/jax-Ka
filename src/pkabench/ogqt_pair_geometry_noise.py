"""Sensitivity of oGQT to random pair-distance and relative-angle errors."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, require_compute

SEED = 17
CONDITIONS = {
    "native": (0.0, 0.0),
    "distance_0.5A": (0.5, 0.0),
    "distance_1.0A": (1.0, 0.0),
    "angle_5deg": (0.0, 5.0),
    "angle_10deg": (0.0, 10.0),
    "combined_0.5A_5deg": (0.5, 5.0),
    "combined_1.0A_10deg": (1.0, 10.0),
}


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _root(runtime): return Path(runtime) / "training/ogqt-auxiliary-pilot-v1"
def _out(runtime): return _root(runtime) / "pair-geometry-noise-v1"


def _seed(identifier, condition):
    value = hashlib.sha256(f"{SEED}|{identifier}|{condition}".encode()).digest()[:8]
    return int.from_bytes(value, "little")


def _rotation(axis, radians):
    axis = axis / max(np.linalg.norm(axis), 1e-12); x, y, z = axis
    cross = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * np.cos(radians) + (1-np.cos(radians))*np.outer(axis, axis) + np.sin(radians)*cross


def _mutate_distances(edge, switch, mask, neighbors, sigma, rng, same_residue=None):
    """Add symmetric Gaussian error to pair distances and rebuild RBF/switch."""
    centers = np.linspace(0, 20, 16); source = np.arange(len(neighbors))[:, None]
    valid = mask & (neighbors != source)
    if same_residue is not None: valid &= ~same_residue
    rbf = edge[..., :16]
    distance = (rbf * centers).sum(-1) / np.maximum(rbf.sum(-1), 1e-12)
    draws = {}
    for i, slot in zip(*np.where(valid)):
        j = int(neighbors[i, slot]); key = (min(i, j), max(i, j))
        if key not in draws: draws[key] = float(rng.normal(0, sigma))
        distance[i, slot] = np.clip(distance[i, slot] + draws[key], 0, 20)
    edge[..., :16] = np.exp(-((distance[..., None] - centers) / 1.5) ** 2)
    switch[...] = np.where(distance < 18, 1, .5*(1+np.cos(np.pi*np.clip((distance-18)/2, 0, 1))))


def _mutate_angles(site_edge, mask, neighbors, degrees, rng):
    """Perturb each unordered site's relative frame by a reciprocal SO(3) error."""
    matrices = site_edge[..., 22:31].reshape((*site_edge.shape[:2], 3, 3))
    source = np.arange(len(neighbors))[:, None]; seen = set()
    for i, slot in zip(*np.where(mask & (neighbors != source) & (site_edge[..., 32] < .5))):
        j = int(neighbors[i, slot]); key = (min(i, j), max(i, j))
        if key in seen: continue
        seen.add(key); axis = rng.normal(size=3)
        q = _rotation(axis, rng.normal(0, np.deg2rad(degrees)))
        reverse = np.where((neighbors[j] == i) & mask[j])[0]
        matrices[i, slot] = matrices[i, slot] @ q
        if len(reverse): matrices[j, reverse[0]] = q.T @ matrices[j, reverse[0]]


def mutate(graphs, ids, condition, distance_sigma, angle_sigma):
    result = {name: np.array(value, copy=True) for name, value in graphs.items()}
    if not distance_sigma and not angle_sigma: return result
    for bi, identifier in enumerate(ids):
        rng = np.random.default_rng(_seed(identifier, condition))
        if distance_sigma:
            _mutate_distances(result["edge"][bi, 0], result["switch"][bi, 0],
                result["edge_mask"][bi, 0], result["neighbors"][bi, 0], distance_sigma, rng)
            _mutate_distances(result["site_edge"][bi, 0], result["site_switch"][bi, 0],
                result["site_edge_mask"][bi, 0], result["site_neighbors"][bi, 0], distance_sigma, rng,
                result["site_edge"][bi, 0, ..., 32] > .5)
            result["edge"][bi, 1] = result["edge"][bi, 0]; result["switch"][bi, 1] = result["switch"][bi, 0]
            result["site_edge"][bi, 1] = result["site_edge"][bi, 0]; result["site_switch"][bi, 1] = result["site_switch"][bi, 0]
        if angle_sigma:
            _mutate_angles(result["site_edge"][bi, 0], result["site_edge_mask"][bi, 0],
                result["site_neighbors"][bi, 0], angle_sigma, rng)
            result["site_edge"][bi, 1, ..., 22:31] = result["site_edge"][bi, 0, ..., 22:31]
    return result


def run(runtime):
    import jax
    from pkanet.model import PKPDB_PK_MOD
    from pkanet.ogqt import initialize_auxiliary
    from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine, _plans, _prefetched
    from pkatrain.gqt_paired_pinder import Loader, _bucket_n
    from pkatrain.trainer import load_checkpoint
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    runtime = Path(runtime); base = _root(runtime); manifest = read(base / "manifest.json")
    records = [r for r in manifest["records"] if r["split"] == "val"]; by_id = {r["id"]: r for r in records}
    params = initialize_auxiliary(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = AuxiliaryEngine(params, 0, {"burial": 1, "interface": 1}); state = engine.optimizer.init(params)
    verification = read(base / "standard/seed-17/verification.json")
    params, _, _ = load_checkpoint(base / "standard/seed-17/checkpoints" / f"epoch-{verification['selected_epoch']:03d}", (params, state))
    grouped = defaultdict(list)
    for record in records: grouped[_bucket_n(record["n"])].append(record["id"])
    plans = []
    for bucket in sorted(grouped, key=int):
        size = manifest["batch_sizes"][bucket]; plans.extend(grouped[bucket][i:i+size] for i in range(0, len(grouped[bucket]), size))
    output = _out(runtime); output.mkdir(parents=True, exist_ok=True); rows = []
    loader = Loader(base, manifest)
    for ids, batch in _prefetched(loader, plans):
        graphs, targets, mask, _, _, metadata = batch
        for condition, (distance_sigma, angle_sigma) in CONDITIONS.items():
            changed = mutate(graphs, ids, condition, distance_sigma, angle_sigma)
            prediction = jax.tree.map(np.asarray, engine.predictions(params, changed))
            for bi, cid in enumerate(ids):
                record = by_id[cid]
                for qi, key in enumerate(record["keys"]):
                    if not mask[bi, qi]: continue
                    group = changed["query_group"][bi, 0, qi]; ref = float(np.asarray(PKPDB_PK_MOD)[group])
                    expected = targets[bi, :, qi] - ref; shift = prediction["shift"][bi, :, qi]
                    rows.append({"condition": condition, "complex_id": cid,
                        **dict(zip(("chain", "resnum", "icode", "group"), key)),
                        "state_error": float(np.mean(np.abs(shift-expected))),
                        "paired_error": float((shift[0]-shift[1])-(expected[0]-expected[1])),
                        "interface": bool(metadata["interface"][bi, qi]),
                        "interface_score": float(prediction["interface"][bi, 0, qi])})
    loader.close()
    path = output / "predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(output / "run.json", {"passed": True, "conditions": CONDITIONS, "complexes": len(records),
        "predictions": len(rows), "checkpoint_epoch": verification["selected_epoch"], "test_data_included": False,
        "distance_mutation": "symmetric Gaussian error on residue and site pair distances; RBF and cutoff switch recomputed; adjacency fixed",
        "angle_mutation": "Gaussian SO(3) error on reciprocal site-pair relative-orientation matrices; direction features unchanged"})


def report(runtime):
    from pkabench.ogqt_auxiliary_roc import _roc
    require_compute(threads=2, allow_comp1400=True); output = _out(runtime)
    with (output / "predictions.csv").open() as stream:
        rows = [{**x, "state_error": float(x["state_error"]), "paired_error": float(x["paired_error"]),
            "interface": x["interface"].lower() == "true", "interface_score": float(x["interface_score"])} for x in csv.DictReader(stream)]
    threshold = read(_root(runtime) / "interface-counterfactual/summary.json")["arms"]["standard"]["youden"]["native"]["threshold"]
    summary = {"conditions": {}, "classification_threshold": threshold}
    for condition in CONDITIONS:
        values = [x for x in rows if x["condition"] == condition]; interface = [x for x in values if x["interface"]]
        residue = defaultdict(list)
        for x in values: residue[(x["complex_id"],x["chain"],x["resnum"],x["icode"])].append(x)
        reduced = [{"complex_id": k[0], "chain": k[1], "resnum": k[2], "icode": k[3],
            "interface": v[0]["interface"], "score": float(np.mean([x["interface_score"] for x in v]))} for k,v in residue.items()]
        roc = _roc([x["interface"] for x in reduced], [x["score"] for x in reduced])
        labels = np.asarray([x["interface"] for x in reduced], bool)
        calls = np.asarray([x["score"] >= threshold for x in reduced], bool)
        tp = int(np.sum(labels & calls)); fp = int(np.sum(~labels & calls))
        fn = int(np.sum(labels & ~calls)); tn = int(np.sum(~labels & ~calls))
        by_complex = defaultdict(list)
        for x in reduced: by_complex[x["complex_id"]].append(x)
        ap = []
        for block in by_complex.values():
            positives = sum(x["interface"] for x in block)
            if not positives: continue
            ordered = sorted(block, key=lambda x: (-x["score"], x["chain"], int(x["resnum"]), x["icode"]))
            hits = 0; total = 0.0
            for rank, x in enumerate(ordered, 1):
                if x["interface"]: hits += 1; total += hits/rank
            ap.append(total/positives)
        summary["conditions"][condition] = {"sites": len(values), "state_mae": float(np.mean([x["state_error"] for x in values])),
            "paired_mae": float(np.mean(np.abs([x["paired_error"] for x in values]))),
            "interface_paired_mae": float(np.mean(np.abs([x["paired_error"] for x in interface]))),
            "interface_roc_auc_pooled": roc[2] if roc else None,
            "interface_average_precision_macro": float(np.mean(ap)) if ap else None,
            "eligible_ap_complexes": len(ap), "false_positives": fp, "false_negatives": fn,
            "false_positive_rate": fp/(fp+tn), "false_negative_rate": fn/(fn+tp)}
    native = summary["conditions"]["native"]
    for condition, values in summary["conditions"].items():
        values["delta_state_mae"] = values["state_mae"] - native["state_mae"]
        values["delta_interface_paired_mae"] = values["interface_paired_mae"] - native["interface_paired_mae"]
        values["delta_interface_roc_auc"] = values["interface_roc_auc_pooled"] - native["interface_roc_auc_pooled"]
    atomic_json(output / "summary.json", summary)


def main():
    import sys
    runtime = Path(os.environ["PKABENCH_RUNTIME"])
    if sys.argv[1] == "run": run(runtime)
    elif sys.argv[1] == "report": report(runtime)
    else: raise ValueError(sys.argv[1])


if __name__ == "__main__": main()
