#!/usr/bin/env python3
"""Compare midpoint-derived pH linkage for all PINDER estimators.

For scalar pKa estimators, a Henderson--Hasselbalch protonated fraction is
integrated analytically.  The reported complex-level quantity is

    log10[Kd(7.4) / Kd(6.1)] = integral_6.1^7.4 (Q_AB - Q_free) dpH.

All estimators and the PypKa reference use exactly the same AB/free site keys.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute


MODELS = (
    "Regular pKAI",
    "pKAI+",
    "oGQT joint + dropout",
    "oGQT joint",
    "oGQT pKPDB-pretrained",
    "Backbone-pretrained pKAI",
    "smooth_propka",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def state_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row["complex_id"], row["state"], row["chain"], row["resnum"],
        row.get("icode", ""), row["group"],
    )


def site_key(k: tuple[str, str, str, str, str, str]) -> tuple[str, str, str, str, str]:
    return (k[0], k[2], k[3], k[4], k[5])


def hh_integral(pka: float, lo: float, hi: float) -> float:
    """Integral of 1/(1 + 10**(pH-pKa)) with stable logaddexp."""
    ln10 = math.log(10.0)

    def primitive(ph: float) -> float:
        return ph - np.logaddexp(0.0, ln10 * (ph - pka)) / ln10

    return float(primitive(hi) - primitive(lo))


def index_values(rows: list[dict[str, str]], column: str) -> dict[tuple, float]:
    result = {}
    for row in rows:
        k = state_key(row)
        value = float(row[column])
        if k in result and not math.isclose(result[k], value, abs_tol=1e-10):
            raise ValueError(f"conflicting {column} values for {k}")
        result[k] = value
    return result


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    require_compute(threads=1, allow_comp1400=True)

    audit = args.runtime / "pkabench" / "audits"
    inputs = {
        "gqt": audit / "gqt-pinder-pypka-v1" / "matched-state.csv",
        "pkai": audit / "pinder-pkai-backbone-eval-v1" / "matched-state.csv",
        "pkai_plus": audit / "pinder-pkai-plus-eval-v1" / "matched-state.csv",
        "smooth": audit / "smooth-propka-pinder-v1" / "matched-state.csv",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    gqt_rows = read_csv(inputs["gqt"])
    pkai_rows = read_csv(inputs["pkai"])
    plus_rows = read_csv(inputs["pkai_plus"])
    smooth_rows = read_csv(inputs["smooth"])

    observed_gqt = {r["model"] for r in gqt_rows}
    expected_gqt = {
        "oGQT joint + dropout", "oGQT joint", "oGQT pKPDB-pretrained"
    }
    if observed_gqt != expected_gqt:
        raise ValueError(f"unexpected GQT models: {sorted(observed_gqt)}")

    values: dict[str, dict[tuple, float]] = {
        "Regular pKAI": index_values(pkai_rows, "regular_pkai"),
        "Backbone-pretrained pKAI": index_values(pkai_rows, "backbone_ensemble"),
        "pKAI+": index_values(plus_rows, "pkai_plus"),
        "smooth_propka": index_values(smooth_rows, "smooth_propka"),
    }
    for model in expected_gqt:
        values[model] = index_values(
            [r for r in gqt_rows if r["model"] == model], "ogqt"
        )

    reference_sources = (
        index_values(pkai_rows, "pypka"),
        index_values(plus_rows, "pypka"),
        index_values(smooth_rows, "pypka"),
        index_values(gqt_rows, "pypka"),
    )
    common_states = set.intersection(
        *(set(d) for d in (*values.values(), *reference_sources))
    )
    if not common_states:
        raise RuntimeError("no state keys shared by every estimator")

    reference = {}
    for k in common_states:
        candidates = [source[k] for source in reference_sources]
        if max(candidates) - min(candidates) > 1e-9:
            raise ValueError(f"PypKa reference mismatch for {k}: {candidates}")
        reference[k] = candidates[0]

    # Retain only sites having one bound and exactly one monomer state on the
    # common support. This is the same support for every estimator.
    site_states: dict[tuple, list[tuple]] = defaultdict(list)
    for k in common_states:
        site_states[site_key(k)].append(k)
    paired = {}
    for sk, keys in site_states.items():
        bound = [k for k in keys if k[1] == "AB"]
        free = [k for k in keys if k[1] in {"A", "B"}]
        if len(bound) == 1 and len(free) == 1:
            paired[sk] = (bound[0], free[0])
    if not paired:
        raise RuntimeError("no complete common AB/free site pairs")

    metadata = {}
    for row in pkai_rows:
        metadata.setdefault(row["complex_id"], (row["cluster_id"], row["ctype"]))

    per_complex: dict[str, dict[str, float]] = defaultdict(
        lambda: {"reference": 0.0, **{m: 0.0 for m in MODELS}, "n_sites": 0}
    )
    lo, hi = 6.1, 7.4
    for sk, (bound, free) in paired.items():
        complex_id = sk[0]
        per_complex[complex_id]["n_sites"] += 1
        per_complex[complex_id]["reference"] += (
            hh_integral(reference[bound], lo, hi)
            - hh_integral(reference[free], lo, hi)
        )
        for model in MODELS:
            per_complex[complex_id][model] += (
                hh_integral(values[model][bound], lo, hi)
                - hh_integral(values[model][free], lo, hi)
            )

    args.output.mkdir(parents=True, exist_ok=True)
    per_path = args.output / "per-complex.csv"
    fields = [
        "complex_id", "cluster_id", "ctype", "n_sites",
        "reference_log10_kd_ratio",
    ]
    for model in MODELS:
        slug = model.lower().replace("+", "plus").replace(" ", "_").replace("-", "_")
        fields += [f"{slug}_log10_kd_ratio", f"{slug}_reference_minus_prediction"]
    with per_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for complex_id in sorted(per_complex):
            item = per_complex[complex_id]
            cluster, ctype = metadata[complex_id]
            row = {
                "complex_id": complex_id,
                "cluster_id": cluster,
                "ctype": ctype,
                "n_sites": int(item["n_sites"]),
                "reference_log10_kd_ratio": item["reference"],
            }
            for model in MODELS:
                slug = model.lower().replace("+", "plus").replace(" ", "_").replace("-", "_")
                row[f"{slug}_log10_kd_ratio"] = item[model]
                row[f"{slug}_reference_minus_prediction"] = item["reference"] - item[model]
            writer.writerow(row)

    complexes = sorted(per_complex)
    clusters = defaultdict(list)
    for complex_id in complexes:
        clusters[metadata[complex_id][0]].append(complex_id)
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(20261009)
    boot_indices = rng.integers(0, len(cluster_ids), size=(args.bootstrap, len(cluster_ids)))

    summary = []
    ref = np.asarray([per_complex[c]["reference"] for c in complexes])
    for model in MODELS:
        pred = np.asarray([per_complex[c][model] for c in complexes])
        diff = ref - pred
        cluster_means = np.asarray([
            np.mean([per_complex[c]["reference"] - per_complex[c][model] for c in clusters[k]])
            for k in cluster_ids
        ])
        boot = cluster_means[boot_indices].mean(axis=1)
        ci_lo, ci_hi = percentile_interval(boot)
        summary.append({
            "estimator": model,
            "n_complexes": len(complexes),
            "n_site_pairs": len(paired),
            "reference_mean_log10_kd_ratio": float(ref.mean()),
            "predicted_mean_log10_kd_ratio": float(pred.mean()),
            "mean_reference_minus_prediction": float(diff.mean()),
            "cluster_bootstrap_95_ci_low": ci_lo,
            "cluster_bootstrap_95_ci_high": ci_hi,
            "mean_absolute_difference": float(np.abs(diff).mean()),
            "median_reference_minus_prediction": float(np.median(diff)),
        })

    payload = {
        "definition": "log10[Kd(7.4)/Kd(6.1)]",
        "difference": "PypKa reference minus estimator prediction",
        "curve_reconstruction": "Henderson-Hasselbalch from scalar pKa midpoint",
        "aggregation": "arithmetic mean of per-complex signed differences",
        "bootstrap": {
            "unit": "PINDER cluster", "replicates": args.bootstrap, "seed": 20261009
        },
        "coverage": {
            "complexes": len(complexes), "clusters": len(cluster_ids),
            "common_ab_free_site_pairs": len(paired),
        },
        "results": summary,
    }
    atomic_json(args.output / "summary.json", payload)

    table = [
        "| Estimator | Mean PypKa ratio | Mean predicted ratio | Mean difference | 95% cluster-bootstrap CI | Mean absolute difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        table.append(
            f"| {row['estimator']} | {row['reference_mean_log10_kd_ratio']:.4f} | "
            f"{row['predicted_mean_log10_kd_ratio']:.4f} | "
            f"{row['mean_reference_minus_prediction']:+.4f} | "
            f"[{row['cluster_bootstrap_95_ci_low']:+.4f}, {row['cluster_bootstrap_95_ci_high']:+.4f}] | "
            f"{row['mean_absolute_difference']:.4f} |"
        )
    report = "\n".join([
        "# Midpoint-derived pH linkage comparison", "",
        "The quantity is `log10[Kd(7.4)/Kd(6.1)]`. The requested signed difference is "
        "`PypKa reference - prediction`; positive values mean the estimator's signed ratio is "
        "more negative than the PypKa ratio on average.", "",
        *table, "",
        f"Common support: {len(complexes)} complexes in {len(cluster_ids)} clusters and "
        f"{len(paired):,} matched AB/free site pairs.", "",
        "These models provide scalar pKa midpoints, so each site curve was reconstructed with "
        "the Henderson--Hasselbalch equation and integrated analytically. This is a common-support "
        "midpoint-derived linkage proxy, not native PypKa Monte Carlo curve linkage. The reported "
        "mean gives every complex equal weight; confidence intervals resample PINDER clusters.", "",
    ])
    (args.output / "report.md").write_text(report)
    atomic_json(args.output / "manifest.json", {
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in inputs.items()},
        "outputs": {
            "per_complex_sha256": digest(per_path),
            "summary_sha256": digest(args.output / "summary.json"),
            "report_sha256": digest(args.output / "report.md"),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
