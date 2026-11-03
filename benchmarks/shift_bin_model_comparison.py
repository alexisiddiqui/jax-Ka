#!/usr/bin/env python3
"""Compare binding-shift MAE by PypKa |delta-pKa| bin on common support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, require_compute


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return row["complex_id"], row["chain"], row["resnum"], row.get("icode", ""), row["group"]


def index(rows: list[dict[str, str]], prediction: str, model: str | None = None) -> dict[tuple, tuple[float, float]]:
    result = {}
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        result[key(row)] = (float(row["pypka_shift"]), float(row[prediction]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute(threads=1, allow_comp1400=True)
    audit = args.runtime / "pkabench/audits"

    pkai = read(audit / "pinder-pkai-backbone-eval-v1/matched-paired.csv")
    feature = read(audit / "pkai-feature-pinder-v1/matched-paired.csv")
    backbone = read(audit / "pkai-backbone-pinder-v1/matched-paired.csv")
    ogqt = read(audit / "ogqt-production-frozen-v2-exact/matched-paired.csv")
    models = {
        "oGQT production 50K (backbone)": index(ogqt, "prediction_shift"),
        "pKAI backbone 15A": index(backbone, "prediction_shift", "pKAI-backbone-15A"),
        "pKAI backbone 20A": index(backbone, "prediction_shift", "pKAI-backbone-20A"),
        "pKAI backbone-pretrained ensemble": index(pkai, "backbone_ensemble_shift"),
        "Regular pKAI (all atom)": index(pkai, "regular_pkai_shift"),
        "Joint-trained pKAI aa20 (all atom)": index(feature, "prediction_shift", "pKAI-joint-aa20"),
    }
    common = set.intersection(*(set(values) for values in models.values()))
    if len(common) != 10_744:
        raise AssertionError(f"expected 10,744 common sites, found {len(common)}")
    for site in common:
        references = [values[site][0] for values in models.values()]
        if max(references)-min(references) > 1e-9:
            raise AssertionError((site, references))

    bins = (("<0.5", 0.0, 0.5), ("0.5-1", 0.5, 1.0),
            ("1-2", 1.0, 2.0), (">=2", 2.0, np.inf))
    output = []
    for label, lo, hi in bins:
        sites = [site for site in common if lo <= abs(models[next(iter(models))][site][0]) < hi]
        row = {"reference_abs_shift_bin": label, "sites": len(sites)}
        for model, values in models.items():
            row[model] = float(np.mean([abs(values[site][1]-values[site][0]) for site in sites]))
        output.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "summary.json", {"common_sites": len(common), "rows": output,
                "test_data_included": False})
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    lines = ["# Binding-shift MAE by PypKa shift magnitude", "",
             "All models use the same 10,744 matched AB/free sites.", "",
             "| Reference `|ΔpKa|` | Sites | " + " | ".join(models) + " |",
             "|---|---:|" + "---:|" * len(models)]
    for row in output:
        lines.append("| " + row["reference_abs_shift_bin"] + f" | {row['sites']:,} | " +
                     " | ".join(f"{row[model]:.3f}" for model in models) + " |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
