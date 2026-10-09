"""Rescore frozen oGQT PINDER predictions against current PypKa labels.

This is an evaluation-only join.  Model inference and checkpoint selection are
not repeated: the registered PINDER prediction CSVs are matched to the frozen
400-complex current-PypKa audit by explicit site identity.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

from .runtime import atomic_json, digest, require_compute


VERSION = "gqt-pinder-pypka-v1"


def _read_json(path: Path):
    return json.loads(path.read_text())


def _read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _key(row):
    return (row["complex_id"], row["chain"], int(row["resnum"]),
            row.get("icode", ""), row["group"])


def _metrics(errors):
    errors = [float(value) for value in errors]
    if not errors:
        return None
    absolute = sorted(abs(value) for value in errors)
    n = len(errors)
    quantile = lambda p: absolute[min(n - 1, max(0, math.ceil(p * n) - 1))]
    return {"sites": n, "mae": sum(absolute) / n,
            "rmse": math.sqrt(sum(value * value for value in errors) / n),
            "bias": sum(errors) / n, "p50": quantile(0.50),
            "p90": quantile(0.90), "p95": quantile(0.95),
            "max": absolute[-1]}


def _paths(root: Path):
    training = root / "training"
    return {
        "oGQT pKPDB-pretrained": training / "ogqt-multitask-replay-v1/seed-17/epoch-000-pinder.csv",
        "oGQT joint": training / "ogqt-multitask-replay-v1/seed-17/validation-pinder.csv",
        "oGQT joint + dropout": training / "ogqt-diagnostic-v1/dropout/seed-17/validation-pinder.csv",
    }


def _shared_comparators(root: Path, state_keys, pair_keys):
    """Score every comparator on exactly the sites available to oGQT."""
    audits = root / "audits"
    specs = (
        ("smooth_propka", audits / "smooth-propka-pinder-v1", "error", "shift_error"),
        ("Regular pKAI", audits / "pinder-pkai-backbone-eval-v1", "regular_pkai_error",
         "regular_pkai_shift_error"),
        ("Backbone-pretrained pKAI", audits / "pinder-pkai-backbone-eval-v1",
         "backbone_ensemble_error", "backbone_ensemble_shift_error"),
        ("pKAI+", audits / "pinder-pkai-plus-eval-v1", "error", "shift_error"),
    )
    result = []
    for label, folder, state_field, pair_field in specs:
        states = [row for row in _read_csv(folder / "matched-state.csv")
                  if (row["state"], *_key(row)) in state_keys]
        pairs = [row for row in _read_csv(folder / "matched-paired.csv") if _key(row) in pair_keys]
        interface = [row for row in pairs if _truth(row["interface"])]
        result.append((label, _metrics(row[state_field] for row in states),
                       _metrics(row[pair_field] for row in pairs),
                       _metrics(row[pair_field] for row in interface)))
    return result


def evaluate(root: Path):
    root = Path(root)
    teacher_dir = root / "audits/pinder-pkai-pypka-v1"
    out = root / "audits/gqt-pinder-pypka-v1"
    out.mkdir(parents=True, exist_ok=True)
    state_teacher = _read_csv(teacher_dir / "matched-state.csv")
    pair_teacher = _read_csv(teacher_dir / "matched-paired.csv")
    sources = _paths(root)
    manifest = {
        "version": VERSION,
        "target": "current PypKa",
        "cohort": "frozen 400-complex PINDER validation cohort",
        "teacher_state_sha256": digest(teacher_dir / "matched-state.csv"),
        "teacher_paired_sha256": digest(teacher_dir / "matched-paired.csv"),
        "models": {name: {"path": str(path), "sha256": digest(path)}
                   for name, path in sources.items()},
        "model_selection_reused": True,
        "inference_reused": True,
        "test_data_included": False,
        "code_sha256": digest(Path(__file__)),
    }
    atomic_json(out / "manifest.json", manifest)

    state_rows = []
    pair_rows = []
    summaries = {}
    shared_state_keys = None
    shared_pair_keys = None
    teacher_state_keys = {(row["state"], *_key(row)): row for row in state_teacher}
    teacher_pair_keys = {_key(row): row for row in pair_teacher}
    for model, path in sources.items():
        predictions = {_key(row): row for row in _read_csv(path)}
        model_states = []
        model_pairs = []
        for (state, *site_key), truth in teacher_state_keys.items():
            prediction = predictions.get(tuple(site_key))
            if prediction is None:
                continue
            estimate = float(prediction["predicted_ab" if state == "AB" else "predicted_free"])
            reference = float(truth["pypka"])
            error = estimate - reference
            row = {"model": model, **truth, "ogqt": estimate, "error": error}
            model_states.append(row)
            state_rows.append(row)
        for site_key, truth in teacher_pair_keys.items():
            prediction = predictions.get(site_key)
            if prediction is None:
                continue
            estimate = float(prediction["predicted_ab"]) - float(prediction["predicted_free"])
            reference = float(truth["pypka_shift"])
            error = estimate - reference
            row = {"model": model, **truth, "ogqt_shift": estimate, "shift_error": error}
            model_pairs.append(row)
            pair_rows.append(row)
        interface = [row for row in model_pairs if _truth(row["interface"])]
        this_state_keys = {(row["state"], *_key(row)) for row in model_states}
        this_pair_keys = {_key(row) for row in model_pairs}
        shared_state_keys = this_state_keys if shared_state_keys is None else shared_state_keys & this_state_keys
        shared_pair_keys = this_pair_keys if shared_pair_keys is None else shared_pair_keys & this_pair_keys
        summaries[model] = {
            "prediction_sites": len(predictions),
            "matched_state_sites": len(model_states),
            "matched_paired_sites": len(model_pairs),
            "eligible_state_sites": len(state_teacher),
            "eligible_paired_sites": len(pair_teacher),
            "state_match_fraction": len(model_states) / len(state_teacher),
            "paired_match_fraction": len(model_pairs) / len(pair_teacher),
            "state": _metrics(row["error"] for row in model_states),
            "paired": _metrics(row["shift_error"] for row in model_pairs),
            "interface_paired": _metrics(row["shift_error"] for row in interface),
        }

    def write_csv(name, rows):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("matched-state.csv", state_rows)
    write_csv("matched-paired.csv", pair_rows)
    shared = _shared_comparators(root, shared_state_keys, shared_pair_keys)
    summary = {"version": VERSION, "models": summaries,
        "shared_site_comparators": {name: {"state": state, "paired": paired,
            "interface_paired": interface} for name, state, paired, interface in shared},
        "shared_state_sites": len(shared_state_keys), "shared_paired_sites": len(shared_pair_keys),
        "test_data_included": False}
    atomic_json(out / "summary.json", summary)

    comparison = []
    for name, row in summaries.items():
        comparison.append((name, row["state"]["mae"], row["paired"]["mae"],
                           row["interface_paired"]["mae"]))
    comparison += [(name, state["mae"], paired["mae"], interface["mae"])
                   for name, state, paired, interface in shared]
    lines = ["# oGQT against current PypKa on frozen PINDER validation", "",
        "Frozen oGQT prediction files were rescored without rerunning inference or selecting a new checkpoint. "
        "All rows below use current PypKa labels on the same 400-complex validation cohort and the exact same "
        f"{len(shared_state_keys):,} state / {len(shared_pair_keys):,} paired sites.", "",
        "| Estimator | State MAE | Paired-shift MAE | Interface paired-shift MAE |",
        "|---|---:|---:|---:|"]
    lines += [f"| {name} | {state:.3f} | {paired:.3f} | {interface:.3f} |"
              for name, state, paired, interface in comparison]
    lines += ["", "## Coverage", "",
        "| oGQT checkpoint | State sites | Paired sites | State coverage | Pair coverage |",
        "|---|---:|---:|---:|---:|"]
    for name, row in summaries.items():
        lines.append(f"| {name} | {row['matched_state_sites']:,}/{row['eligible_state_sites']:,} | "
                     f"{row['matched_paired_sites']:,}/{row['eligible_paired_sites']:,} | "
                     f"{row['state_match_fraction']:.1%} | {row['paired_match_fraction']:.1%} |")
    lines += ["", "The pKPDB-pretrained row is the frozen epoch-zero parent. The joint rows were trained with "
        "both pKPDB and PINDER/pKAI supervision, so their agreement with PypKa measures cross-teacher retention. "
        "PypKa is a computational teacher, not experimental truth.", "",
        "The exact-site table differs slightly from the estimators' standalone reports, which use all "
        "22,384 current-PypKa state sites and 11,032 paired sites available to those methods.", "",
        "Per-site values are in `matched-state.csv` and `matched-paired.csv`; exact metrics and provenance are "
        "in `summary.json` and `manifest.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "evaluate":
        raise SystemExit("usage: python -m pkabench.gqt_pinder_pypka_eval evaluate")
    require_compute(threads=min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
                    allow_comp1400=True)
    evaluate(Path(os.environ["PKABENCH_RUNTIME"]))


if __name__ == "__main__":
    main()
