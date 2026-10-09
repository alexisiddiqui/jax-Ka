"""Native and separated-structure interface error audit for the oGQT pilot."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from pkanet.ogqt import initialize_auxiliary
from pkabench.runtime import atomic_json, require_compute
from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine, read
from pkatrain.gqt_paired_pinder import Loader, _bucket_n, _prefetched
from pkatrain.trainer import load_checkpoint


ARMS = ("baseline", "low", "standard")


def _plans(records, manifest):
    grouped = defaultdict(list)
    for row in records: grouped[_bucket_n(row["n"])].append(row["id"])
    result = []
    for bucket in sorted(grouped, key=int):
        size = manifest["batch_sizes"][bucket]
        ids = grouped[bucket]
        result.extend(ids[i:i + size] for i in range(0, len(ids), size))
    return result


def _predict(base, manifest, records, arm):
    run = base / arm / "seed-17"; verification = read(run / "verification.json")
    params = initialize_auxiliary(jax.random.PRNGKey(17), **manifest["architecture"])
    engine = AuxiliaryEngine(params, 0.0, {"burial": 1.0, "interface": 1.0})
    state = engine.optimizer.init(params)
    params, _, metadata = load_checkpoint(
        run / "checkpoints" / f"epoch-{verification['selected_epoch']:03d}", (params, state))
    if metadata["epoch"] != verification["selected_epoch"]: raise AssertionError("checkpoint epoch")
    by_id = {row["id"]: row for row in records}; output = []
    loader = Loader(base, manifest)
    for ids, batch in _prefetched(loader, _plans(records, manifest)):
        predicted = np.asarray(engine.predictions(params, batch[0])["interface"])
        mask = batch[2]; labels = batch[5]["interface"]
        for bi, cid in enumerate(ids):
            for qi, key in enumerate(by_id[cid]["keys"]):
                if not mask[bi, qi]: continue
                output.append({"arm": arm, "complex_id": cid, "chain": key[0], "resnum": key[1],
                    "icode": key[2], "group": key[3], "interface": bool(labels[bi, qi]),
                    "bound_score": float(predicted[bi, 0, qi]), "separated_score": float(predicted[bi, 1, qi])})
    loader.close(); return output


def _deduplicate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["complex_id"], row["chain"], row["resnum"], row["icode"])].append(row)
    result = []
    for key, sites in grouped.items():
        if len({x["interface"] for x in sites}) != 1: raise AssertionError("inconsistent residue label")
        result.append({"arm": key[0], "complex_id": key[1], "chain": key[2], "resnum": key[3],
            "icode": key[4], "interface": sites[0]["interface"],
            "bound_score": float(np.mean([x["bound_score"] for x in sites])),
            "separated_score": float(np.mean([x["separated_score"] for x in sites]))})
    return result


def _confusion(labels, scores, threshold, all_negative=False):
    labels = np.zeros(len(labels), bool) if all_negative else np.asarray(labels, bool)
    called = np.asarray(scores) >= threshold
    tp = int(np.sum(called & labels)); fp = int(np.sum(called & ~labels))
    fn = int(np.sum(~called & labels)); tn = int(np.sum(~called & ~labels))
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tp / (tp + fn) if tp + fn else None, "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None}


def _youden(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    candidates = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    best = None
    for threshold in candidates:
        row = _confusion(labels, scores, threshold)
        value = row["tpr"] - row["fpr"]
        candidate = (value, -abs(threshold - .5), threshold)
        if best is None or candidate > best[0]: best = (candidate, threshold)
    return float(best[1])


def _equal_complex_rate(rows, score, threshold, positive):
    values = []
    for cid in sorted({x["complex_id"] for x in rows}):
        subset = [x for x in rows if x["complex_id"] == cid and x["interface"] == positive]
        if subset:
            calls = np.asarray([x[score] for x in subset]) >= threshold
            values.append(float(np.mean(~calls if positive else calls)))
    return {"mean": float(np.mean(values)), "eligible_complexes": len(values)} if values else None


def main():
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    root = Path(os.environ["PKABENCH_RUNTIME"]); base = root / "training/ogqt-auxiliary-pilot-v1"
    manifest = read(base / "manifest.json"); records = [x for x in manifest["records"] if x["split"] == "val"]
    rows = _deduplicate([row for arm in ARMS for row in _predict(base, manifest, records, arm)])
    output = base / "interface-counterfactual"; output.mkdir(exist_ok=True)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"positive_definition": "native residue delta-SASA > 10 A^2",
        "augmentation": "free branch: all cross-chain residue and site edges removed; equivalent to separation beyond the 20 A model cutoff",
        "threshold_warning": "Youden thresholds are selected and evaluated on validation and are descriptive, not deployable",
        "arms": {}}
    for arm in ARMS:
        subset = [x for x in rows if x["arm"] == arm]
        labels = [x["interface"] for x in subset]; bound = [x["bound_score"] for x in subset]
        separated = [x["separated_score"] for x in subset]; threshold = _youden(labels, bound)
        arm_summary = {}
        for name, cutoff in (("fixed_0.5", .5), ("youden", threshold)):
            arm_summary[name] = {"native": _confusion(labels, bound, cutoff),
                "separated_all_negative": _confusion(labels, separated, cutoff, all_negative=True),
                "native_equal_complex_fnr": _equal_complex_rate(subset, "bound_score", cutoff, True),
                "native_equal_complex_fpr": _equal_complex_rate(subset, "bound_score", cutoff, False),
                "separated_equal_complex_fpr": {"mean": float(np.mean([
                    np.mean([x["separated_score"] >= cutoff for x in subset if x["complex_id"] == cid])
                    for cid in sorted({x["complex_id"] for x in subset})])), "eligible_complexes": 400}}
        arm_summary["score_response"] = {"mean_bound_positive": float(np.mean([x["bound_score"] for x in subset if x["interface"]])),
            "mean_separated_native_positive": float(np.mean([x["separated_score"] for x in subset if x["interface"]])),
            "mean_bound_negative": float(np.mean([x["bound_score"] for x in subset if not x["interface"]])),
            "mean_separated_native_negative": float(np.mean([x["separated_score"] for x in subset if not x["interface"]]))}
        summary["arms"][arm] = arm_summary
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__": main()
