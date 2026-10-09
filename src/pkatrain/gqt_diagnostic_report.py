"""Aggregate the preregistered oGQT diagnostic campaign into reviewable gates."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute


SEED = 20261009
REPLICATES = 2000


def read(path): return json.loads(Path(path).read_text())


def _complex_scores(path):
    grouped = defaultdict(list)
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream): grouped[row["complex_id"]].append(row)
    result = {}
    for cid, rows in grouped.items():
        state = np.mean([abs(float(row["state_error"])) for row in rows])
        interface = [abs(float(row["paired_error"])) for row in rows
                     if row["interface"].lower() == "true"]
        if interface: result[cid] = float(state + np.mean(interface))
    return result


def _paired_bootstrap(reference, candidate):
    common = sorted(set(reference) & set(candidate)); rng = np.random.default_rng(SEED)
    differences = np.asarray([candidate[cid] - reference[cid] for cid in common])
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = np.mean(differences[rng.integers(0, len(differences), len(differences))])
    return {"complexes": len(common), "candidate_minus_baseline": float(np.mean(differences)),
            "ci95": np.quantile(draws, (.025, .975)).tolist(), "replicates": REPLICATES,
            "resampling_unit": "PINDER validation cluster (one complex per frozen cluster)"}


def _run(run):
    verification = read(run / "verification.json"); final = read(run / "final.json")
    if not verification["passed"] or verification["test_data_included"]: raise AssertionError(run)
    scores = _complex_scores(run / "validation-pinder.csv")
    pair = final["pinder"]
    return {"run": str(run), "selected_epoch": final["selected_epoch"],
            "state_mae": pair["state_mae"], "paired_mae": pair["paired_mae"],
            "interface_paired_mae": pair["interface_paired_mae"],
            "selection": pair["state_mae"] + pair["interface_paired_mae"],
            "pkpdb_mae": final["pkpdb"]["overall"]["mae"],
            "pkpdb_limit": final["pkpdb_mae_limit"], "complex_scores": scores}


def _clean(run, epoch):
    path = run / "clean-diagnostics" / f"epoch-{epoch:03d}.json"
    return read(path) if path.exists() else None


def report(root):
    root = Path(root); base = root / "training/ogqt-multitask-replay-v1/seed-17"
    diagnostic = root / "training/ogqt-diagnostic-v1"
    paths = {"baseline": base, **{arm: diagnostic / arm / "seed-17"
        for arm in ("dropout", "context", "data25", "data50")}}
    runs = {name: _run(path) for name, path in paths.items()}
    reference = runs["baseline"]["complex_scores"]
    comparisons = {name: _paired_bootstrap(reference, row["complex_scores"])
                   for name, row in runs.items() if name != "baseline"}
    for row in runs.values(): row.pop("complex_scores")

    regularization = {}
    for name in ("dropout", "context"):
        delta = comparisons[name]
        regularization[name] = {"passed": (runs["baseline"]["selection"] - runs[name]["selection"] >= .01
            and delta["ci95"][1] < 0 and runs[name]["pkpdb_mae"] <= runs[name]["pkpdb_limit"]),
            "improvement": runs["baseline"]["selection"] - runs[name]["selection"], "bootstrap": delta}
    data_values = [runs[name]["selection"] for name in ("data25", "data50", "baseline")]
    data_gate = {"passed": bool(data_values[0] >= data_values[1] >= data_values[2]
                                  and comparisons["data25"]["ci95"][0] > 0),
                 "selection_25_50_100": data_values,
                 "endpoint_bootstrap_25_minus_100": comparisons["data25"]}

    capacity = {"passed": None, "reason": "282k capacity arm cancelled by user; replaced by batch-size audit"}

    under_path = diagnostic / "undertraining/seed-17"
    under = read(under_path / "verification.json") if (under_path / "verification.json").exists() else None
    teacher_path = root / "audits/pinder-pkai-pypka-v1/summary.json"
    teacher = read(teacher_path) if teacher_path.exists() else None
    batch_path = root / "training/ogqt-joint-batch-v1/report.json"
    batch = read(batch_path) if batch_path.exists() else None
    result = {"version": "ogqt-diagnostic-report-v1", "runs": runs, "comparisons": comparisons,
              "gates": {"regularization": regularization, "more_data": data_gate,
                        "more_capacity": capacity, "undertraining": under},
              "teacher_audit": teacher, "batch_audit": batch, "bootstrap_seed": SEED,
              "test_data_included": False}
    atomic_json(diagnostic / "diagnostic-report.json", result)

    lines = ["# oGQT multitask diagnostic results", "",
        "All rows use the same frozen validation cohorts. Lower values are better. The selection score is PINDER state MAE plus interface paired MAE.", "",
        "| Arm | Epoch | State MAE | Paired MAE | Interface paired MAE | Selection | pKPDB MAE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "dropout", "context", "data25", "data50"):
        row = runs[name]; lines.append(f"| {name} | {row['selected_epoch']} | {row['state_mae']:.4f} | "
            f"{row['paired_mae']:.4f} | {row['interface_paired_mae']:.4f} | {row['selection']:.4f} | {row['pkpdb_mae']:.4f} |")
    lines += ["", "| Comparison | Cluster-bootstrap change vs baseline | 95% CI |",
              "|---|---:|---|"]
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['candidate_minus_baseline']:+.4f} | "
                     f"[{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] |")
    if teacher:
        lines += ["", "## Teacher comparison", "",
            "| Matched states | State coverage | State MAE | Matched pairs | Pair coverage | Paired MAE | Interface paired MAE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {teacher['matched_state_sites']} | {teacher['state_match_fraction']:.1%} | {teacher['state_teacher_mae']:.4f} | "
            f"{teacher['matched_paired_sites']} | {teacher['paired_match_fraction']:.1%} | {teacher['paired_teacher_mae']:.4f} | "
            f"{teacher['interface_paired_teacher_mae']:.4f} |"]
    if batch:
        lines += ["", "## Batch-size audit", "",
            f"Joint gradient-noise scale: **{batch['gradient_noise']['joint']['noise_scale_structures_per_task']:.2f}** structures per task batch.", "",
            "| Scale | pKPDB batch | LR | Selection | pKPDB MAE | Wall min |",
            "|---:|---:|---:|---:|---:|---:|"]
        for row in batch["runs"]:
            lines.append(f"| {row['scale']:g} | {row['pkpdb_batch_size']} | {row['learning_rate']:.2g} | "
                         f"{row['selection']:.4f} | {row['pkpdb_mae']:.4f} | {row['seconds']/60:.1f} |")
    destination = diagnostic / "diagnostic-report.md"; destination.write_text("\n".join(lines) + "\n")
    atomic_json(diagnostic / "report-verification.json", {"passed": True,
        "json_sha256": digest(diagnostic / "diagnostic-report.json"),
        "markdown_sha256": digest(destination), "test_data_included": False})


def main():
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    report(Path(os.environ["PKABENCH_RUNTIME"]))


if __name__ == "__main__": main()
