#!/usr/bin/env python3
"""Paired cluster-bootstrap comparison of pKAI feature/fork variants."""

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
    model = {key_fn(row): row for row in model_rows if not interface_only or row["interface"].lower() == "true"}
    base = {key_fn(row): row for row in base_rows if not interface_only or row["interface"].lower() == "true"}
    common = sorted(model.keys() & base.keys())
    if not common:
        raise ValueError("no common rows")
    by_cluster = defaultdict(lambda: [0.0, 0.0, 0])
    for key in common:
        m, b = model[key], base[key]
        cluster = m["cluster_id"]
        if cluster != b["cluster_id"]:
            raise AssertionError((key, cluster, b["cluster_id"]))
        by_cluster[cluster][0] += abs(float(m[model_field]))
        by_cluster[cluster][1] += abs(float(b[base_field]))
        by_cluster[cluster][2] += 1
    values = np.asarray(list(by_cluster.values()), np.float64)
    observed_model = values[:, 0].sum() / values[:, 2].sum()
    observed_base = values[:, 1].sum() / values[:, 2].sum()
    rng = np.random.default_rng(20261010)
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    sampled = values[indices]
    model_mae = sampled[:, :, 0].sum(1) / sampled[:, :, 2].sum(1)
    base_mae = sampled[:, :, 1].sum(1) / sampled[:, :, 2].sum(1)
    delta = model_mae - base_mae
    low, high = np.percentile(delta, [2.5, 97.5])
    return {"sites": len(common), "clusters": len(values), "model_mae": float(observed_model),
        "baseline_mae": float(observed_base), "mae_delta": float(observed_model-observed_base),
        "relative_change_percent": float(100*(observed_model-observed_base)/observed_base),
        "cluster_bootstrap_95_ci": [float(low), float(high)], "replicates": 2000}


def main():
    require_compute(threads=1, allow_comp1400=True)
    root = Path("/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench")
    audits = root / "audits"
    paths = {
        "new_state": audits / "pkai-feature-pinder-v1/matched-state.csv",
        "new_pair": audits / "pkai-feature-pinder-v1/matched-paired.csv",
        "regular_state": audits / "pinder-pkai-backbone-eval-v1/matched-state.csv",
        "regular_pair": audits / "pinder-pkai-backbone-eval-v1/matched-paired.csv",
        "plus_state": audits / "pinder-pkai-plus-eval-v1/matched-state.csv",
        "plus_pair": audits / "pinder-pkai-plus-eval-v1/matched-paired.csv",
    }
    data = {name: rows(path) for name, path in paths.items()}
    comparisons = {}
    specifications = (
        ("smooth-pKAI", "Regular pKAI", "regular"),
        ("pKAI-joint", "Regular pKAI", "regular"),
        ("pKAI-joint-aa20", "Regular pKAI", "regular"),
        ("smooth-pKAI+", "pKAI+", "plus"),
    )
    for model, baseline, family in specifications:
        state = [row for row in data["new_state"] if row["model"] == model]
        pair = [row for row in data["new_pair"] if row["model"] == model]
        if family == "regular":
            base_state, base_pair = data["regular_state"], data["regular_pair"]
            state_field, pair_field = "regular_pkai_error", "regular_pkai_shift_error"
        else:
            base_state, base_pair = data["plus_state"], data["plus_pair"]
            state_field, pair_field = "error", "shift_error"
        comparisons[model] = {"baseline": baseline,
            "state": paired_delta(state, base_state, "error", state_field, state_key),
            "paired": paired_delta(pair, base_pair, "shift_error", pair_field, pair_key),
            "interface_paired": paired_delta(pair, base_pair, "shift_error", pair_field,
                                               pair_key, interface_only=True)}
    out = audits / "pkai-feature-pinder-v1"
    payload = {"difference": "candidate MAE minus baseline MAE; negative favours candidate",
        "bootstrap_unit": "PINDER cluster", "comparisons": comparisons}
    atomic_json(out / "paired-comparison.json", payload)
    lines = ["# Paired comparison with released pKAI baselines", "",
        "Negative MAE deltas favour the candidate. Confidence intervals resample PINDER clusters.", "",
        "| Candidate | Baseline | Metric | Candidate MAE | Baseline MAE | Delta | Relative change | 95% CI for delta |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    labels = (("state", "State"), ("paired", "Paired shift"),
              ("interface_paired", "Interface paired"))
    for model, result in comparisons.items():
        for key, label in labels:
            item = result[key]
            lines.append(f"| {model} | {result['baseline']} | {label} | {item['model_mae']:.4f} | "
                f"{item['baseline_mae']:.4f} | {item['mae_delta']:+.4f} | "
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
