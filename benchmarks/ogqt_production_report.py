#!/usr/bin/env python3
"""Summarise the production oGQT frozen-cohort evaluation with cluster CIs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, require_compute


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap_mae(data: list[dict[str, str]], error: str, replicates: int = 2000) -> dict:
    grouped = defaultdict(list)
    for row in data:
        grouped[row["cluster_id"]].append(abs(float(row[error])))
    clusters = sorted(grouped)
    rng = np.random.default_rng(20261012)
    means = np.empty(replicates)
    for draw in range(replicates):
        sample = rng.choice(clusters, len(clusters), replace=True)
        values = [value for cluster in sample for value in grouped[cluster]]
        means[draw] = np.mean(values)
    observed = float(np.mean([abs(float(row[error])) for row in data]))
    return {"sites": len(data), "clusters": len(clusters), "mae": observed,
            "cluster_bootstrap_95_ci": np.percentile(means, [2.5, 97.5]).tolist(),
            "replicates": replicates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    require_compute(threads=1, allow_comp1400=True)
    state = rows(args.audit / "matched-state.csv")
    paired = rows(args.audit / "matched-paired.csv")
    interface = [row for row in paired if row["interface"].lower() == "true"]

    bins = (("<0.5", 0.0, 0.5), ("0.5-1", 0.5, 1.0),
            ("1-2", 1.0, 2.0), (">=2", 2.0, np.inf))
    shift_bins = []
    for label, lo, hi in bins:
        subset = [row for row in paired if lo <= abs(float(row["pypka_shift"])) < hi]
        prediction = np.asarray([float(row["prediction_shift"]) for row in subset])
        target = np.asarray([float(row["pypka_shift"]) for row in subset])
        shift_bins.append({"bin": label, "sites": len(subset),
                           "mae": float(np.mean(np.abs(prediction-target))),
                           "prediction_std": float(prediction.std()),
                           "target_std": float(target.std())})

    target = np.asarray([float(row["pypka_shift"]) for row in paired])
    prediction = np.asarray([float(row["prediction_shift"]) for row in paired])
    slope = float(np.polyfit(target, prediction, 1)[0])
    payload = {
        "state": bootstrap_mae(state, "error"),
        "paired": bootstrap_mae(paired, "shift_error"),
        "interface_paired": bootstrap_mae(interface, "shift_error"),
        "shift_bins": shift_bins,
        "predicted_vs_reference_slope": slope,
        "prediction_shift_std": float(prediction.std()),
        "reference_shift_std": float(target.std()),
        "test_data_included": False,
    }
    atomic_json(args.audit / "report-summary.json", payload)

    def interval(item):
        lo, hi = item["cluster_bootstrap_95_ci"]
        return f"{item['mae']:.4f} [{lo:.4f}, {hi:.4f}]"

    lines = [
        "# Production oGQT: frozen 400-complex evaluation",
        "",
        "All intervals are 2,000-replicate cluster bootstrap intervals. The final test set was not used.",
        "",
        "| Metric | Sites | Clusters | MAE [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for label, name in (("State", "state"), ("Paired shift", "paired"),
                        ("Interface paired shift", "interface_paired")):
        item = payload[name]
        lines.append(f"| {label} | {item['sites']:,} | {item['clusters']:,} | {interval(item)} |")
    lines += ["", "| Reference | Sites | MAE | Predicted SD | Reference SD |",
              "|---|---:|---:|---:|---:|"]
    for item in shift_bins:
        lines.append(f"| {item['bin']} | {item['sites']:,} | {item['mae']:.4f} | "
                     f"{item['prediction_std']:.4f} | {item['target_std']:.4f} |")
    lines += ["", f"Predicted-versus-reference shift slope: **{slope:.4f}**.", ""]
    (args.audit / "report.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
