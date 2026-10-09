"""Head-specific permutation attribution for oGQT burial/interface outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from pkanet.ogqt import initialize_auxiliary
from pkabench.runtime import atomic_json, require_compute
from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine, _prefetched
from pkatrain.gqt_paired_pinder import Loader, _bucket_n
from pkatrain.trainer import load_checkpoint

SEED = 17
CONDITIONS = {
    "native": None,
    "residue_distance": ("edge", 0, 16),
    "residue_direction": ("edge", 16, 19),
    "residue_same_chain": ("edge", 19, 20),
    "site_distance": ("site_edge", 0, 16),
    "site_direction": ("site_edge", 16, 22),
    "site_orientation": ("site_edge", 22, 31),
    "site_same_chain": ("site_edge", 31, 32),
    "site_same_residue": ("site_edge", 32, 33),
    "site_sequence_separation": ("site_edge", 33, 34),
    "residue_identity": ("nodes", 0, 20),
    "site_type": ("site_type", 0, 1),
    "remove_cross_chain_context": ("mask", 0, 0),
}


def read(path):
    with Path(path).open() as stream: return json.load(stream)


def _seed(identifier, condition):
    return int.from_bytes(hashlib.sha256(f"{SEED}|{identifier}|{condition}".encode()).digest()[:8], "little")


def permute(graphs, ids, condition):
    result = {name: np.array(value, copy=True) for name, value in graphs.items()}
    spec = CONDITIONS[condition]
    if spec is None: return result
    field, start, stop = spec
    for bi, identifier in enumerate(ids):
        rng = np.random.default_rng(_seed(identifier, condition))
        if field in ("edge", "site_edge"):
            mask_name = "edge_mask" if field == "edge" else "site_edge_mask"
            # Use the bound mask so both branches receive exactly the same
            # permuted feature values; their masks remain branch-specific.
            ii, jj = np.where(result[mask_name][bi, 0])
            values = result[field][bi, 0, ii, jj, start:stop].copy()
            values = values[rng.permutation(len(values))]
            for branch in (0, 1): result[field][bi, branch, ii, jj, start:stop] = values
        elif field == "nodes":
            active = np.flatnonzero(result["node_mask"][bi, 0]); values = result["nodes"][bi, 0, active, start:stop].copy()
            values = values[rng.permutation(len(values))]
            for branch in (0, 1): result["nodes"][bi, branch, active, start:stop] = values
        elif field == "site_type":
            active = np.flatnonzero(result["site_mask"][bi, 0]); values = result["site_type"][bi, 0, active].copy()
            values = values[rng.permutation(len(values))]
            for branch in (0, 1): result["site_type"][bi, branch, active] = values
        elif field == "mask":
            result["edge_mask"][bi, 0] &= result["edge"][bi, 0, ..., 19] > .5
            result["site_edge_mask"][bi, 0] &= result["site_edge"][bi, 0, ..., 31] > .5
        else: raise ValueError(field)
    return result


def _corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(x, y)[0, 1])


def run(runtime):
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    runtime = Path(runtime); base = runtime / "training/ogqt-auxiliary-pilot-v1"; manifest = read(base / "manifest.json")
    records = [r for r in manifest["records"] if r["split"] == "val"]; grouped = defaultdict(list)
    for record in records: grouped[_bucket_n(record["n"])].append(record["id"])
    plans = []
    for bucket in sorted(grouped, key=int):
        size = manifest["batch_sizes"][bucket]; ids = grouped[bucket]
        plans.extend(ids[i:i+size] for i in range(0, len(ids), size))
    params = initialize_auxiliary(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = AuxiliaryEngine(params, 0, {"burial": 1, "interface": 1}); state = engine.optimizer.init(params)
    verification = read(base / "standard/seed-17/verification.json")
    params, _, _ = load_checkpoint(base / "standard/seed-17/checkpoints" /
        f"epoch-{verification['selected_epoch']:03d}", (params, state))
    rows = []; loader = Loader(base, manifest)
    for number, (ids, batch) in enumerate(_prefetched(loader, plans), 1):
        graphs, targets, mask, normalized_burial, normalized_interface, _ = batch
        burial_target = (normalized_burial * manifest["normalization"]["burial"] - .4) / .6
        interface_target = (normalized_interface * manifest["normalization"]["interface"] - .05) / .95
        baseline = None
        for condition in CONDITIONS:
            changed = permute(graphs, ids, condition)
            prediction = jax.tree.map(np.asarray, engine.predictions(params, changed))
            if condition == "native": baseline = prediction
            for bi, cid in enumerate(ids):
                active = mask[bi]
                for head, branch, target in (("burial", 1, burial_target), ("interface", 0, interface_target)):
                    pred = prediction[head][bi, branch, active]; native = baseline[head][bi, branch, active]
                    truth = target[bi, active]
                    rows.extend({"condition": condition, "complex_id": cid, "head": head,
                        "target": float(t), "prediction": float(p), "native_prediction": float(n)}
                        for t, p, n in zip(truth, pred, native))
        if number % 20 == 0: print(json.dumps({"batches": number, "total": len(plans)}), flush=True)
    loader.close(); output = base / "auxiliary-feature-attribution-v1"; output.mkdir(parents=True, exist_ok=True)
    path = output / "predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"version": "auxiliary-feature-attribution-v1", "complexes": len(records),
        "method": "deterministic within-complex permutation; same feature permutation in both branches",
        "checkpoint_epoch": verification["selected_epoch"], "test_data_included": False, "heads": {}}
    for head in ("burial", "interface"):
        summary["heads"][head] = {}
        for condition in CONDITIONS:
            values = [x for x in rows if x["head"] == head and x["condition"] == condition]
            target = np.asarray([x["target"] for x in values]); predicted = np.asarray([x["prediction"] for x in values])
            native = np.asarray([x["native_prediction"] for x in values])
            by_complex = defaultdict(list)
            for x in values: by_complex[x["complex_id"]].append(abs(x["prediction"]-x["target"]))
            summary["heads"][head][condition] = {"sites": len(values),
                "mae": float(np.mean(np.abs(predicted-target))),
                "equal_complex_mae": float(np.mean([np.mean(x) for x in by_complex.values()])),
                "mean_abs_prediction_change": float(np.mean(np.abs(predicted-native))),
                "p99_abs_prediction_change": float(np.quantile(np.abs(predicted-native), .99)),
                "prediction_correlation_with_native": _corr(predicted, native),
                "target_correlation": _corr(predicted, target)}
    native = {head: summary["heads"][head]["native"]["mae"] for head in summary["heads"]}
    for head, conditions in summary["heads"].items():
        for values in conditions.values(): values["delta_mae"] = values["mae"] - native[head]
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__": run(Path(os.environ["PKABENCH_RUNTIME"]))
