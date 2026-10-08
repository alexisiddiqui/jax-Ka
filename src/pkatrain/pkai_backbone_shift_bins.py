"""Post-hoc shift-bin report for the completed pKAI backbone ablation."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.frozen_score import aggregate, measures
from pkabench.runtime import atomic_json, digest, require_compute


ARMS = ("full-scratch", "full-pretrained", "backbone-scratch", "backbone-pretrained")
SEEDS = (17, 29, 43)
BIN_NAMES = ("<0.5", "0.5–1", "1–2", "≥2")


def shift_bin(value):
    return int(np.searchsorted(np.asarray((0.5, 1.0, 2.0)), abs(float(value)), side="right"))


def read_predictions(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["teacher_shift"] = float(row["teacher_shift"])
        row["predicted_shift"] = float(row["predicted_shift"])
        row["shift_bin"] = shift_bin(row["teacher_shift"])
    return rows


def group_macro(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[row["complex_id"]].append(row)
    scores = []
    for cid, subset in grouped.items():
        target = np.asarray([row["teacher_shift"] for row in subset])
        predicted = np.asarray([row["predicted_shift"] for row in subset])
        scores.append({"complex_id": cid, "component_id": subset[0]["component_id"],
                       "n": len(subset), **measures(target, predicted)})
    return aggregate(scores, replicates=2000)[0]


def run(root):
    out = root / "pretraining/pkai-backbone-ablation-v1"
    per_run = []; counts = None
    for arm in ARMS:
        for seed in SEEDS:
            folder = out / arm / f"seed-{seed}"; predictions = folder / "validation_predictions.csv"
            verification = json.loads((folder / "verification.json").read_text())
            if not verification["passed"] or digest(predictions) != verification["predictions_sha256"]:
                raise AssertionError(folder)
            rows = read_predictions(predictions)
            current_counts = [sum(row["shift_bin"] == index for row in rows) for index in range(4)]
            if counts is None: counts = current_counts
            if current_counts != counts: raise AssertionError((arm, seed, current_counts, counts))
            overall = group_macro(rows)
            bins = []
            for index, name in enumerate(BIN_NAMES):
                subset = [row for row in rows if row["shift_bin"] == index]
                metric = group_macro(subset)
                target = np.asarray([row["teacher_shift"] for row in subset])
                predicted = np.asarray([row["predicted_shift"] for row in subset])
                bins.append({"bin": name, "sites": len(subset), "mae": metric["mae"],
                             "mae_ci95": metric["mae_ci95"], "rmse": metric["rmse"],
                             "mean_abs_teacher_shift": float(np.mean(np.abs(target))),
                             "mean_abs_predicted_shift": float(np.mean(np.abs(predicted)))})
            per_run.append({"arm": arm, "seed": seed, "overall_mae": overall["mae"], "bins": bins})
    summary = []
    for arm in ARMS:
        subset = [row for row in per_run if row["arm"] == arm]
        bins = []
        for index, name in enumerate(BIN_NAMES):
            values = np.asarray([row["bins"][index]["mae"] for row in subset])
            bins.append({"bin": name, "sites": counts[index], "mean_mae": float(values.mean()),
                         "sd_mae": float(values.std(ddof=1)),
                         "mean_abs_teacher_shift": float(np.mean([row["bins"][index]["mean_abs_teacher_shift"] for row in subset])),
                         "mean_abs_predicted_shift": float(np.mean([row["bins"][index]["mean_abs_predicted_shift"] for row in subset]))})
        overall = np.asarray([row["overall_mae"] for row in subset])
        summary.append({"arm": arm, "overall_mean_mae": float(overall.mean()),
                        "overall_sd_mae": float(overall.std(ddof=1)), "bins": bins})
    atomic_json(out / "shift-bin-results.json", {"bin_edges": [0.5, 1.0, 2.0],
                "bin_definition": "absolute teacher shift from residue-specific PK_MOD",
                "counts": counts, "per_run": per_run, "summary": summary,
                "prediction_hashes_verified": True, "test_data_included": False})
    headers = [f"{name} (n={count:,})" for name, count in zip(BIN_NAMES, counts)]
    lines = ["# pKAI validation MAE by reference-shift magnitude", "",
             "Bins use the absolute teacher shift from the residue-specific `PK_MOD` reference. Values are group-macro MAE averaged over three seeds; parentheses give the seed SD.", "",
             f"| Input | Initialization | Overall | {headers[0]} | {headers[1]} | {headers[2]} | **{headers[3]}** |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        mode, initialization = row["arm"].split("-")
        values = [f"{item['mean_mae']:.3f} ({item['sd_mae']:.3f})" for item in row["bins"]]
        lines.append(f"| {mode} | {initialization} | {row['overall_mean_mae']:.3f} | {values[0]} | {values[1]} | {values[2]} | **{values[3]}** |")
    lines += ["", "## Largest shifts (|ΔpKa| ≥ 2)", "",
              "| Input | Initialization | Mean absolute teacher shift | Mean absolute predicted shift | Magnitude retained |",
              "|---|---|---:|---:|---:|"]
    for row in summary:
        mode, initialization = row["arm"].split("-"); largest = row["bins"][3]
        retained = largest["mean_abs_predicted_shift"] / largest["mean_abs_teacher_shift"]
        lines.append(f"| {mode} | {initialization} | {largest['mean_abs_teacher_shift']:.3f} | {largest['mean_abs_predicted_shift']:.3f} | {retained:.1%} |")
    lines += ["", "No test data were read.", ""]
    (out / "shift-bin-report.md").write_text("\n".join(lines))
    atomic_json(out / "shift-bin-verification.json", {"passed": True, "runs": len(per_run),
                "prediction_hashes_verified": True, "report_sha256": digest(out / "shift-bin-report.md"),
                "results_sha256": digest(out / "shift-bin-results.json"), "test_data_included": False})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]))
    run(Path(os.environ["PKABENCH_RUNTIME"]))
