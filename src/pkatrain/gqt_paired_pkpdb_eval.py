"""Frozen pKPDB evaluation of scratch and pretrained Siamese oGQT checkpoints."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import jax
import numpy as np

from pkanet.site_model import initialize_site
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_site_weighting import WeightedSiteEngine, evaluate, metrics_from_rows
from .trainer import load_checkpoint


def read(path): return json.loads(Path(path).read_text())


def _micro(path):
    with Path(path).open(newline="") as stream: rows = list(csv.DictReader(stream))
    errors = [abs(float(row["predicted_shift"]) - float(row["teacher_shift"])) for row in rows]
    return {"sites": len(rows), "site_micro_mae": float(np.mean(errors))}


def main():
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=True, allow_comp1400=True)
    jax.config.update("jax_enable_x64", False)
    runtime = Path(os.environ["PKABENCH_RUNTIME"])
    pkpdb = runtime / "pretraining/gqt-site-weighting-v1"
    paired = runtime / "training/ogqt-pinder-factorial-v1"
    output = paired / "pkpdb-evaluation"; output.mkdir(parents=True, exist_ok=True)
    manifest = read(pkpdb / "baseline/manifest.json")
    architecture = manifest["config"]["architecture"]
    sources = (
        ("pKPDB-pretrained", pkpdb / "baseline/seed-17"),
        ("scratch vanilla Siamese", paired / "vanilla/seed-17"),
        ("pretrained Siamese, standard LR", paired / "pretrained-vanilla-standard/seed-17"),
        ("pretrained Siamese, low LR", paired / "pretrained-vanilla-low/seed-17"),
    )
    rows = []
    for name, run in sources:
        verification = read(run / "verification.json")
        if not verification["passed"] or verification["test_data_included"]: raise AssertionError(name)
        epoch = int(verification["selected_epoch"])
        checkpoint = run / "checkpoints" / f"epoch-{epoch:03d}"
        params = initialize_site(jax.random.PRNGKey(17), **architecture)
        engine = WeightedSiteEngine(params, (1.0, 1.0, 1.0, 1.0)); state = engine.optimizer.init(params)
        params, _, metadata = load_checkpoint(checkpoint, (params, state))
        predictions = output / (name.lower().replace(" ", "-").replace(",", "") + ".csv")
        metrics = evaluate(pkpdb / "baseline", manifest, "baseline", engine, params,
            predictions=predictions, bootstrap=True)
        rows.append({"configuration": name, "selected_epoch": epoch,
            "checkpoint_sha256": digest(checkpoint / "state.npz"), "predictions_sha256": digest(predictions),
            **_micro(predictions), **metrics})

    with (output / "pkpdb-pretrained.csv").open(newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    for row in baseline_rows:
        row["teacher_shift"] = float(row["teacher_shift"])
        row["predicted_shift"] = 0.0
    baseline_overall, baseline_bins, _ = metrics_from_rows(baseline_rows)
    pkmod = {"configuration": "fixed PK_MOD", "sites": len(baseline_rows),
        "site_micro_mae": float(np.mean([abs(float(row["teacher_shift"])) for row in baseline_rows])),
        "overall": baseline_overall, "bins": baseline_bins}
    atomic_json(output / "summary.json", {"pkmod_baseline": pkmod, "models": rows})
    lines = ["# Frozen pKPDB evaluation after Siamese training", "",
        "All checkpoints are scored on the unchanged 142-complex PypKa validation set. Group-macro MAE is the historical primary metric; site-micro MAE is included for comparison with the PINDER state report. No pKPDB fitting or test data are used here.", "",
        "| Configuration | Group-macro MAE | Site-micro MAE | <0.5 | 0.5-1 | 1-2 | >=2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| fixed PK_MOD | {pkmod['overall']['mae']:.4f} | {pkmod['site_micro_mae']:.4f} | "
        f"{pkmod['bins'][0]['mae']:.4f} | {pkmod['bins'][1]['mae']:.4f} | {pkmod['bins'][2]['mae']:.4f} | {pkmod['bins'][3]['mae']:.4f} |"]
    for row in rows:
        bins = row["bins"]
        lines.append(f"| {row['configuration']} | {row['overall']['mae']:.4f} | {row['site_micro_mae']:.4f} | "
            f"{bins[0]['mae']:.4f} | {bins[1]['mae']:.4f} | {bins[2]['mae']:.4f} | {bins[3]['mae']:.4f} |")
    report = output / "report.md"; report.write_text("\n".join(lines) + "\n")
    atomic_json(output / "verification.json", {"passed": True, "models": len(rows),
        "report_sha256": digest(report), "test_data_included": False})


if __name__ == "__main__": main()
