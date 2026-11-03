#!/usr/bin/env python3
"""Paired cluster-bootstrap comparison of smooth_propka backbone variants."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def state_key(row):
    return (row["complex_id"], row["state"], row["chain"], row["resnum"],
            row.get("icode", ""), row["group"])


def pair_key(row):
    return (row["complex_id"], row["chain"], row["resnum"], row.get("icode", ""), row["group"])


def paired_delta(model_rows, base_rows, model_field, base_field, key_fn, interface_only=False):
    def eligible(row):
        return not interface_only or row["interface"].lower() == "true"
    model = {key_fn(row): row for row in model_rows if eligible(row)}
    base = {key_fn(row): row for row in base_rows if eligible(row)}
    common = sorted(model.keys() & base.keys())
    if not common:
        raise ValueError("no common rows")
    by_cluster = defaultdict(lambda: [0.0, 0.0, 0])
    for key in common:
        candidate, baseline = model[key], base[key]
        cluster = candidate["cluster_id"]
        if cluster != baseline["cluster_id"]:
            raise AssertionError((key, cluster, baseline["cluster_id"]))
        by_cluster[cluster][0] += abs(float(candidate[model_field]))
        by_cluster[cluster][1] += abs(float(baseline[base_field]))
        by_cluster[cluster][2] += 1
    values = np.asarray(list(by_cluster.values()), np.float64)
    observed_candidate = values[:, 0].sum() / values[:, 2].sum()
    observed_baseline = values[:, 1].sum() / values[:, 2].sum()
    rng = np.random.default_rng(20261011)
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    sampled = values[indices]
    delta = (sampled[:, :, 0].sum(1) - sampled[:, :, 1].sum(1)) / sampled[:, :, 2].sum(1)
    low, high = np.percentile(delta, [2.5, 97.5])
    return {"sites": len(common), "clusters": len(values),
        "candidate_mae": float(observed_candidate), "baseline_mae": float(observed_baseline),
        "mae_delta": float(observed_candidate-observed_baseline),
        "relative_change_percent": float(100*(observed_candidate-observed_baseline)/observed_baseline),
        "cluster_bootstrap_95_ci": [float(low), float(high)], "replicates": 2000}


def main():
    require_compute(threads=1, allow_comp1400=True)
    root = Path("/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/audits")
    paths = {
        "candidate_state": root / "pkai-backbone-pinder-v1/matched-state.csv",
        "candidate_pair": root / "pkai-backbone-pinder-v1/matched-paired.csv",
        "baseline_state": root / "pinder-pkai-backbone-eval-v1/matched-state.csv",
        "baseline_pair": root / "pinder-pkai-backbone-eval-v1/matched-paired.csv",
        "fine_state": root / "pkai-feature-pinder-v1/matched-state.csv",
        "fine_pair": root / "pkai-feature-pinder-v1/matched-paired.csv",
    }
    data = {name: rows(path) for name, path in paths.items()}
    comparisons = {}
    baselines = {
        "Regular pKAI": (data["baseline_state"], data["baseline_pair"],
                          "regular_pkai_error", "regular_pkai_shift_error"),
        "Backbone-pretrained pKAI": (data["baseline_state"], data["baseline_pair"],
                                      "backbone_ensemble_error", "backbone_ensemble_shift_error"),
        "Fine-tuned pKAI-joint-aa20": (
            [row for row in data["fine_state"] if row["model"] == "pKAI-joint-aa20"],
            [row for row in data["fine_pair"] if row["model"] == "pKAI-joint-aa20"],
            "error", "shift_error"),
    }
    for model in ("pKAI-backbone-15A", "pKAI-backbone-20A"):
        state = [row for row in data["candidate_state"] if row["model"] == model]
        pair = [row for row in data["candidate_pair"] if row["model"] == model]
        comparisons[model] = {}
        for baseline, (base_state, base_pair, state_field, pair_field) in baselines.items():
            comparisons[model][baseline] = {
                "state": paired_delta(state, base_state, "error", state_field, state_key),
                "paired": paired_delta(pair, base_pair, "shift_error", pair_field, pair_key),
                "interface_paired": paired_delta(pair, base_pair, "shift_error",
                                                   pair_field, pair_key, interface_only=True),
            }
    rows_15_state = [row for row in data["candidate_state"] if row["model"] == "pKAI-backbone-15A"]
    rows_20_state = [row for row in data["candidate_state"] if row["model"] == "pKAI-backbone-20A"]
    rows_15_pair = [row for row in data["candidate_pair"] if row["model"] == "pKAI-backbone-15A"]
    rows_20_pair = [row for row in data["candidate_pair"] if row["model"] == "pKAI-backbone-20A"]
    head_to_head = {
        "state": paired_delta(rows_20_state, rows_15_state, "error", "error", state_key),
        "paired": paired_delta(rows_20_pair, rows_15_pair, "shift_error", "shift_error", pair_key),
        "interface_paired": paired_delta(rows_20_pair, rows_15_pair, "shift_error",
                                           "shift_error", pair_key, interface_only=True),
    }
    out = root / "pkai-backbone-pinder-v1"
    payload = {"difference": "candidate MAE minus baseline MAE; negative favours candidate",
        "bootstrap_unit": "PINDER cluster", "comparisons": comparisons,
        "head_to_head_20A_minus_15A": head_to_head}
    atomic_json(out / "paired-comparison.json", payload)
    lines = ["# Paired comparison of backbone-only pKAI variants", "",
        "Negative MAE deltas favour the candidate. Confidence intervals resample PINDER clusters.", "",
        "| Candidate | Baseline | Metric | Candidate MAE | Baseline MAE | Delta | Relative change | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    labels = (("state", "State"), ("paired", "Paired shift"),
              ("interface_paired", "Interface paired"))
    for model, baseline_results in comparisons.items():
        for baseline, result in baseline_results.items():
            for key, label in labels:
                item = result[key]
                lines.append(f"| {model} | {baseline} | {label} | {item['candidate_mae']:.4f} | "
                    f"{item['baseline_mae']:.4f} | {item['mae_delta']:+.4f} | "
                    f"{item['relative_change_percent']:+.1f}% | "
                    f"[{item['cluster_bootstrap_95_ci'][0]:+.4f}, {item['cluster_bootstrap_95_ci'][1]:+.4f}] |")
    for key, label in labels:
        item = head_to_head[key]
        lines.append(f"| pKAI-backbone-20A | pKAI-backbone-15A | {label} | "
            f"{item['candidate_mae']:.4f} | {item['baseline_mae']:.4f} | {item['mae_delta']:+.4f} | "
            f"{item['relative_change_percent']:+.1f}% | "
            f"[{item['cluster_bootstrap_95_ci'][0]:+.4f}, {item['cluster_bootstrap_95_ci'][1]:+.4f}] |")
    (out / "paired-comparison.md").write_text("\n".join(lines) + "\n")
    atomic_json(out / "paired-comparison-receipt.json", {
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()},
        "outputs": {name: digest(out / name) for name in ("paired-comparison.json", "paired-comparison.md")},
        "script_sha256": digest(Path(__file__)),
    })
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
