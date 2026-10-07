"""Matched-support comparison of explicit-shift GQT and pKAI experiments."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.frozen_score import aggregate, measures
from pkabench.runtime import atomic_json, digest, require_compute


require_compute(threads=2, allow_comp1400=True)
root = Path(os.environ["PKABENCH_RUNTIME"])
base = root / "pretraining"
paths = {
    "GQT unweighted": base / "gqt-backbone-5k-pkmod-v1/unweighted/seed-17/validation_predictions.csv",
    "GQT weighted": base / "gqt-backbone-5k-pkmod-v1/weighted/seed-17/validation_predictions.csv",
    "pKAI scratch unweighted": base / "pkai-5k-shift-weight-v1/scratch-unweighted/validation_predictions.csv",
    "pKAI scratch weighted": base / "pkai-5k-shift-weight-v1/scratch-weighted/validation_predictions.csv",
    "pKAI pretrained unweighted": base / "pkai-5k-shift-weight-v1/pretrained-unweighted/validation_predictions.csv",
    "pKAI pretrained weighted": base / "pkai-5k-shift-weight-v1/pretrained-weighted/validation_predictions.csv",
}


def key(row):
    return (row["complex_id"], row["chain"], int(row["resnum"]), row["icode"], row["group"])


models = {}
for name, path in paths.items():
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    indexed = {key(row): row for row in rows}
    assert len(indexed) == len(rows)
    models[name] = indexed

support = set.intersection(*(set(rows) for rows in models.values()))
assert support
pkai_reference = models["pKAI pretrained unweighted"]
for identity in support:
    teachers = np.asarray([float(rows[identity]["teacher_pka"]) for rows in models.values()])
    assert np.max(teachers) - np.min(teachers) < 1e-6


def score(name, selected):
    grouped = defaultdict(list)
    rows = models[name]
    for identity in sorted(selected):
        reference = pkai_reference[identity]
        baseline = float(reference["model_pka"])
        target = float(reference["teacher_pka"]) - baseline
        predicted = float(rows[identity]["predicted_pka"]) - baseline
        grouped[identity[0]].append((target, predicted, reference["component_id"]))
    complexes = []
    for complex_id, values in grouped.items():
        target, predicted, _ = zip(*values)
        complexes.append({
            "complex_id": complex_id, "component_id": values[0][2], "n": len(values),
            **measures(target, predicted),
        })
    return aggregate(complexes, replicates=2000)[0]


def site_mae(name, selected):
    errors = []
    for identity in selected:
        target = float(pkai_reference[identity]["teacher_pka"])
        predicted = float(models[name][identity]["predicted_pka"])
        errors.append(abs(predicted - target))
    return float(np.mean(errors))


def shift_bin(identity):
    row = pkai_reference[identity]
    shift = abs(float(row["teacher_pka"]) - float(row["model_pka"]))
    return int(np.searchsorted([0.5, 1.0, 2.0], shift, side="right"))


overall = {name: score(name, support) for name in models}
bin_names = ("0-0.5", "0.5-1", "1-2", "2-inf")
bins = []
for index, label in enumerate(bin_names):
    selected = {identity for identity in support if shift_bin(identity) == index}
    for name in models:
        metric = score(name, selected)
        bins.append({
            "model": name, "bin": label, "sites": len(selected),
            "site_mae": site_mae(name, selected),
            "group_macro_mae": metric["mae"], "group_macro_mae_ci95": metric["mae_ci95"],
            "complexes": metric["complexes"], "groups": metric["groups"],
        })

output = base / "gqt-pkai-pkmod-comparison-v1"
output.mkdir(exist_ok=True)
artifact = {
    "support": len(support),
    "support_by_model": {name: len(rows) for name, rows in models.items()},
    "overall": overall, "bins": bins,
    "sources": {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()},
    "target": "signed shift relative to historical pKPDB/pKAI PK_MOD",
    "aggregation": "equal sites within complex, equal complexes within component, equal components; 2,000 component bootstrap replicates",
    "test_data_included": False,
}
atomic_json(output / "comparison.json", artifact)

lines = [
    "# Matched-support GQT and pKAI explicit-shift comparison", "",
    f"All models are evaluated on the same {len(support):,} validation sites. Targets are signed shifts relative to the historical pKPDB/pKAI `PK_MOD` constants.", "",
    "| Model | Group-macro MAE | 95% CI |", "|---|---:|---|",
]
for name, metric in overall.items():
    lines.append(f"| {name} | {metric['mae']:.4f} | {metric['mae_ci95']} |")
for label in bin_names:
    subset = [row for row in bins if row["bin"] == label]
    lines += ["", f"## Validation shift {label}", "",
              "| Model | Sites | Site MAE | Group-macro MAE | 95% CI |", "|---|---:|---:|---:|---|"]
    for row in subset:
        lines.append(f"| {row['model']} | {row['sites']:,} | {row['site_mae']:.4f} | {row['group_macro_mae']:.4f} | {row['group_macro_mae_ci95']} |")
lines += ["", "All models use development validation checkpoints under their registered protocols. No test data were read."]
(output / "report.md").write_text("\n".join(lines) + "\n")
atomic_json(output / "verification.json", {
    "passed": True, "support": len(support), "report_sha256": digest(output / "report.md"),
    "comparison_sha256": digest(output / "comparison.json"), "test_data_included": False,
})
print("\n".join(lines), flush=True)
