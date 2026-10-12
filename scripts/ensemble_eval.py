"""Seed-ensemble validation metrics for production runs (2026-10-12).

Averages the per-site validation predictions each run saved at its selected epoch (predictions-{pinder,pkpdb,
benchmark}-epoch-NNN.csv) and recomputes the metrics with the training code's own functions (gqt_paired_pinder._metrics,
gqt_site_weighting.metrics_from_rows, production_train.pkpdb_metrics / squared_errors). With one run it reproduces that
run's selection.json.

  python scripts/ensemble_eval.py RUN [RUN ...]   (PKABENCH_RUNTIME set)
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from pkatrain.gqt_paired_pinder import _metrics
from pkatrain.gqt_site_weighting import metrics_from_rows
from pkatrain.production_graphs import read
from pkatrain.production_train import pkpdb_metrics, run_dir, squared_errors


def rows(root, run, dataset):
    out = run_dir(root, run); epoch = read(out / "selection.json")["selected_epoch"]
    with open(out / f"predictions-{dataset}-epoch-{epoch:03d}.csv") as handle: return list(csv.DictReader(handle))


def mean_of(tables, key, value):
    """First table's rows with `value` columns replaced by the mean over tables (rows matched on `key`)."""
    index = [{tuple(r[k] for k in key): r for r in t} for t in tables]; base = tables[0]
    if any(set(i) != set(index[0]) for i in index): raise AssertionError("runs validated different sites")
    return [{**r, **{v: float(np.mean([float(i[tuple(r[k] for k in key)][v]) for i in index])) for v in value}} for r in base]


def main():
    root = Path(os.environ["PKABENCH_RUNTIME"]); runs = sys.argv[1:]
    pinder = mean_of([rows(root, r, "pinder") for r in runs], ("complex_id", "site"), ("predicted_ab", "predicted_free"))
    for r in pinder:
        for k in ("teacher_ab", "teacher_free", "distance", "rsa_free"): r[k] = float(r[k])
        r["interface"] = r["interface"] == "True"
        r["state_error"] = (abs(r["predicted_ab"] - r["teacher_ab"]) + abs(r["predicted_free"] - r["teacher_free"])) / 2
        r["paired_error"] = (r["predicted_ab"] - r["predicted_free"]) - (r["teacher_ab"] - r["teacher_free"])
    bench = mean_of([rows(root, r, "benchmark") for r in runs], ("complex_id", "site"), ("predicted_shift",))
    for r in bench:
        for k in ("teacher_pka", "teacher_shift"): r[k] = float(r[k])
        r["group"] = int(r["group"]); r["site"] = int(r["site"]); r["predicted_pka"] = r["teacher_pka"] - r["teacher_shift"] + r["predicted_shift"]
    pk = mean_of([rows(root, r, "pkpdb") for r in runs], ("structure_id", "site"), ("predicted_shift",))
    for r in pk: r["teacher_shift"] = float(r["teacher_shift"])
    p = _metrics(pinder); overall, _, _ = metrics_from_rows(bench); ps, bs = squared_errors(pinder, bench); k = pkpdb_metrics(pk)
    print(json.dumps({"runs": runs, "selection": round(p["state_mae"] + p["interface_paired_mae"], 4), "state_mae": round(p["state_mae"], 4),
                      "state_mse": round(ps["state_mse"], 4), "interface_paired_mae": round(p["interface_paired_mae"], 4),
                      "interface_paired_mse": round(ps["interface_paired_mse"], 4), "pkpdb_mae": round(k["site_mae"], 4), "pkpdb_mse": round(k["site_mse"], 4),
                      "benchmark_mae": round(overall["mae"], 4), "benchmark_rmse": round(overall["rmse"], 4), "benchmark_spearman": round(overall["spearman"], 4)}))


if __name__ == "__main__":
    main()
