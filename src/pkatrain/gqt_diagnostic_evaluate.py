"""Comparable clean train/validation metrics for every oGQT checkpoint."""
from __future__ import annotations

import json
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from pkanet.model import PKPDB_PK_MOD
from pkanet.ogqt import initialize as initialize_ogqt
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_diagnostic_campaign import DiagnosticEngine, nested_subset, read
from .gqt_paired_pinder import Loader as PairedLoader, _bucket_n, _prefetched, experiment_root as pair_root
from .gqt_site_weighting import metrics_from_rows
from .site_graph_data import SiteBatchLoader
from .trainer import load_checkpoint


def _mean(values): return float(np.mean(values)) if values else None


def _paired_rows(base, manifest, records, engine, params):
    loader = PairedLoader(base, manifest); by_id = {row["id"]: row for row in records}; grouped = defaultdict(list)
    for row in records: grouped[_bucket_n(row["n"])].append(row["id"])
    plans = []
    for bucket in sorted(grouped, key=int):
        size = manifest["batch_sizes"][bucket]; ids = grouped[bucket]
        plans.extend(ids[index:index+size] for index in range(0, len(ids), size))
    rows = []
    for ids, batch in _prefetched(loader, plans):
        graphs, targets, mask, _, _, metadata = batch; predictions = np.asarray(engine.predictions(params, graphs))
        for bi, cid in enumerate(ids):
            record = by_id[cid]; active = mask[bi]; predicted = predictions[bi][:, active]
            groups = graphs["query_group"][bi, 0, active]; reference = np.asarray(PKPDB_PK_MOD)[groups]
            for index, key in enumerate(record["keys"]):
                teacher_ab, teacher_free = targets[bi, :, index]
                predicted_ab, predicted_free = predicted[:, index] + reference[index]
                rows.append({"complex_id": cid, "group": key[3], "teacher_ab": float(teacher_ab),
                    "teacher_free": float(teacher_free), "predicted_ab": float(predicted_ab),
                    "predicted_free": float(predicted_free),
                    "teacher_state_shift_ab": float(teacher_ab-reference[index]),
                    "teacher_state_shift_free": float(teacher_free-reference[index]),
                    "predicted_state_shift_ab": float(predicted[0, index]),
                    "predicted_state_shift_free": float(predicted[1, index]),
                    "teacher_pair": float(teacher_ab-teacher_free),
                    "predicted_pair": float(predicted_ab-predicted_free),
                    "interface": bool(metadata["interface"][bi, index]),
                    "distance": float(metadata["partner_distance_A"][bi, index])})
    loader.close(); return rows


def _paired_metrics(rows):
    state_errors = [row[f"predicted_state_shift_{branch}"]-row[f"teacher_state_shift_{branch}"]
                    for row in rows for branch in ("ab", "free")]
    pair_errors = [row["predicted_pair"]-row["teacher_pair"] for row in rows]
    interface = [error for row, error in zip(rows, pair_errors) if row["interface"]]
    by_complex = defaultdict(list)
    for row, error in zip(rows, pair_errors): by_complex[row["complex_id"]].append(abs(error))
    output = {"complexes": len(by_complex), "sites": len(rows),
        "state_mae": _mean(list(map(abs, state_errors))), "state_mse": _mean([x*x for x in state_errors]),
        "paired_mae": _mean(list(map(abs, pair_errors))), "paired_mse": _mean([x*x for x in pair_errors]),
        "paired_complex_macro_mae": _mean([_mean(values) for values in by_complex.values()]),
        "interface_paired_mae": _mean(list(map(abs, interface))),
        "zero_shift_mae": _mean([abs(row["teacher_pair"]) for row in rows]),
        "zero_shift_interface_mae": _mean([abs(row["teacher_pair"]) for row in rows if row["interface"]]),
        "residue": {}, "paired_shift_bins": {}, "distance_bins": {}}
    for group in sorted({row["group"] for row in rows}):
        subset = [abs(row["predicted_pair"]-row["teacher_pair"]) for row in rows if row["group"] == group]
        output["residue"][group] = {"sites": len(subset), "paired_mae": _mean(subset)}
    for name, low, high in (("<0.1", 0, .1), ("0.1-0.5", .1, .5), ("0.5-1", .5, 1), (">=1", 1, np.inf)):
        subset = [abs(row["predicted_pair"]-row["teacher_pair"]) for row in rows if low <= abs(row["teacher_pair"]) < high]
        output["paired_shift_bins"][name] = {"sites": len(subset), "mae": _mean(subset)}
    for name, low, high in (("<=4", 0, 4), ("4-6", 4, 6), ("6-10", 6, 10), (">10", 10, np.inf)):
        subset = [abs(row["predicted_pair"]-row["teacher_pair"]) for row in rows if low < row["distance"] <= high]
        output["distance_bins"][name] = {"sites": len(subset), "mae": _mean(subset)}
    return output


def _pk_rows(base, manifest, ids, engine, params):
    loader = SiteBatchLoader(base, manifest, manifest["config"]["batch_size"], "site-orientation")
    records = {row["complex_id"]: row for row in manifest["records"]}; rows = []
    for cid in ids:
        record = records[cid]; graph, target, eligible = loader.one(cid)
        predicted = np.asarray(engine.forward(params, graph))[eligible]; observed = target[eligible]
        groups = graph["query_group"][eligible]; reference = np.asarray(PKPDB_PK_MOD)[groups]
        for key, y, value, ref in zip(record["keys"], observed, predicted, reference):
            rows.append({"complex_id": cid, "component_id": record["component_id"], "group": key[4],
                "teacher_pka": float(y), "predicted_pka": float(value), "teacher_shift": float(y-ref),
                "predicted_shift": float(value-ref)})
    loader.close(); return rows


def _pk_metrics(rows):
    overall, bins, equal = metrics_from_rows(rows); errors = [row["predicted_shift"]-row["teacher_shift"] for row in rows]
    return {"complexes": len({row["complex_id"] for row in rows}), "sites": len(rows),
        "micro_mae": _mean(list(map(abs, errors))), "micro_mse": _mean([x*x for x in errors]),
        "group_macro": overall, "bins": bins, "equal_bin_mae": equal,
        "fixed_pkmod_micro_mae": _mean([abs(row["teacher_shift"]) for row in rows])}


def _pk_train_ids(manifest, count=400):
    records = {row["complex_id"]: row for row in manifest["records"]}; groups = defaultdict(list)
    for cid in manifest["train"]: groups[records[cid]["component_id"]].append(cid)
    representatives = [min(ids, key=lambda cid: hashlib.sha256(("pk-fixed|"+cid).encode()).hexdigest())
                       for ids in groups.values()]
    return sorted(representatives, key=lambda cid: hashlib.sha256(("pk-fixed|"+cid).encode()).hexdigest())[:count]


def evaluate_run(root, run, *, smoke=False):
    root = Path(root); run = Path(run); run_info = read(run / "run.json")
    architecture = run_info.get("architecture", {"width": 44, "ff": 88})
    params = initialize_ogqt(jax.random.PRNGKey(17), **architecture); engine = DiagnosticEngine(params)
    state = engine.optimizer.init(params)
    pair_base = pair_root(root); pair_manifest = read(pair_base / "manifest.json")
    pair_train = nested_subset([row for row in pair_manifest["records"] if row["split"] == "train"],
                               4 if smoke else 400)
    pair_val = [row for row in pair_manifest["records"] if row["split"] == "val"]
    pk_base = root / "pretraining/gqt-site-weighting-v1/baseline"; pk_manifest = read(pk_base / "manifest.json")
    pk_train = _pk_train_ids(pk_manifest, 4 if smoke else 400); pk_val = pk_manifest["val"]
    if smoke:
        pair_val = pair_val[:4]; pk_val = pk_val[:4]
    output = run / ("clean-diagnostics-smoke" if smoke else "clean-diagnostics")
    output.mkdir(exist_ok=True); curves = []
    checkpoints = sorted((run / "checkpoints").glob("epoch-*"))
    if smoke: checkpoints = checkpoints[:1]
    for checkpoint in checkpoints:
        params, _, metadata = load_checkpoint(checkpoint, (params, state)); epoch = int(metadata["epoch"])
        record = {"epoch": epoch,
            "pinder_train": _paired_metrics(_paired_rows(pair_base, pair_manifest, pair_train, engine, params)),
            "pinder_validation": _paired_metrics(_paired_rows(pair_base, pair_manifest, pair_val, engine, params)),
            "pkpdb_train": _pk_metrics(_pk_rows(pk_base, pk_manifest, pk_train, engine, params)),
            "pkpdb_validation": _pk_metrics(_pk_rows(pk_base, pk_manifest, pk_val, engine, params))}
        curves.append(record); atomic_json(output / f"epoch-{epoch:03d}.json", record)
        atomic_json(output / "progress.json", {"completed": len(curves), "total": len(checkpoints), "epoch": epoch})
        print(json.dumps({"run": str(run), "epoch": epoch,
            "pinder_train_mse": record["pinder_train"]["state_mse"],
            "pinder_val_mae": record["pinder_validation"]["state_mae"],
            "pkpdb_val_mae": record["pkpdb_validation"]["group_macro"]["mae"]}), flush=True)
    atomic_json(output / "learning-curves.json", curves)
    atomic_json(output / "verification.json", {"passed": True, "checkpoints": len(checkpoints),
        "learning_curves_sha256": digest(output / "learning-curves.json"),
        "fixed_train_complexes": 4 if smoke else 400, "smoke": smoke,
        "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=True, allow_comp1400=True)
    jax.config.update("jax_enable_x64", False)
    if action == "run": evaluate_run(root, sys.argv[2])
    elif action == "smoke": evaluate_run(root, sys.argv[2], smoke=True)
    else: raise ValueError(action)


if __name__ == "__main__": main()
