"""Three-seed 5 A query-centred crop extension with resource accounting."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import jax
import numpy as np

from jaxpropka.parameters import GROUPS
from pkanet.model import initialize
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_crop_radius import (
    CROP_PROBABILITY, EPOCHS, MIN_DELTA, PATIENCE, SEEDS,
    CropBatchLoader, learning_rate, source_hashes as parent_hashes,
)
from .gqt_regularization import RegularizedShiftEngine
from .graph_batches import epoch_batches
from .graph_experiment import evaluate
from .graph_pkmod_compare import digest_array_tree
from .records import read
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


RADIUS = 5.0


def experiment_root(root):
    return root / "pretraining" / "gqt-50k-crop-5a-v1-triton"


def code_hashes():
    return {str(Path(__file__)): digest(Path(__file__)), **parent_hashes()}


def peak_device_memory():
    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_pool_bytes": stats.get("peak_pool_bytes"),
        "raw": {key: value for key, value in stats.items() if isinstance(value, (int, float))},
    }


def register(root):
    out = experiment_root(root)
    out.mkdir(parents=True, exist_ok=True)
    source = root / "pretraining/gqt-50k-crop-radius-v1-triton/full"
    parent = read(source / "manifest.json")
    protocol = {
        "radius_A": RADIUS, "crop_probability": CROP_PROBABILITY,
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "optimizer": "AdamW", "weight_decay": 1e-4,
        "schedule": "1e-3 through epoch 10; per-update cosine to 1e-5 in epochs 11-20",
        "selection": f"minimum validation group-macro MAE; min_delta {MIN_DELTA}; patience {PATIENCE}",
        "input": "strict backbone 50k GQT; 20 A C-alpha encoder graph",
        "crop": "25% of structure draws; uniformly selected query; retain nodes within 5 A C-alpha distance; supervise centre only",
        "validation": "complete unmodified validation graphs", "test_data_included": False,
        "resource_metrics": "monotonic end-to-end run time and JAX device allocator peak",
    }
    atomic_json(out / "protocol.json", protocol)
    manifest = copy.deepcopy(parent)
    manifest["parent"] = {"path": str(source), "manifest_sha256": digest(source / "manifest.json")}
    manifest["crop5_protocol_sha256"] = digest(out / "protocol.json")
    manifest["config"].update(
        seed=None, epochs=EPOCHS, constant_epochs=10, end_learning_rate=1e-5,
        schedule="hold 10 epochs then per-update cosine", optimizer="AdamW",
        weight_decay=1e-4, weight_decay_mask="ndim>=2", crop_radius_A=RADIUS,
        crop_probability=CROP_PROBABILITY, augmentation_arm="crop-5A",
    )
    data = out / "data"
    if not data.exists():
        data.symlink_to((source / "data").resolve(), target_is_directory=True)
    for name in ("preparation.json", "train_type_means.json"):
        if not (out / name).exists():
            shutil.copy2(source / name, out / name)
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "registration.json", {
        "passed": True, "protocol_sha256": digest(out / "protocol.json"),
        "manifest_sha256": digest(out / "manifest.json"), "code_hashes": code_hashes(),
    })


def train(root, seed, *, smoke=False):
    if seed not in SEEDS:
        raise ValueError(seed)
    out = experiment_root(root)
    registration = read(out / "registration.json")
    if not registration["passed"] or registration["code_hashes"] != code_hashes():
        raise AssertionError("Registration/code provenance mismatch")
    stored_manifest = out / "manifest.json"
    manifest = read(stored_manifest)
    manifest["config"]["seed"] = seed
    cfg = manifest["config"]
    run_started = time.monotonic()
    params = initialize(jax.random.PRNGKey(seed), **cfg["architecture"])
    engine = RegularizedShiftEngine(params, weight_decay=cfg["weight_decay"])
    state = engine.optimizer.init(params)
    rng = np.random.default_rng(seed)
    run = out / (f"smoke-seed-{seed}" if smoke else f"seed-{seed}")
    run.mkdir(parents=True, exist_ok=True)
    by_id = {row["complex_id"]: row for row in manifest["records"]}
    records = [by_id[cid] for cid in manifest["train"]]
    loader = CropBatchLoader(
        out, manifest, cfg["batch_size"], backend="mmap", seed=seed,
        crop_radius=RADIUS, crop_probability=CROP_PROBABILITY,
    )
    means = np.asarray([read(out / "train_type_means.json")[group] for group in GROUPS])
    provenance = {
        "manifest_sha256": digest(stored_manifest), "seed": seed,
        "code_hashes": code_hashes(), "parameter_count": cfg["parameter_count"],
    }
    atomic_json(run / "run.json", {**provenance, "config": cfg, "loader": loader.provenance()})
    history, best, stalled = [], None, 0
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        order = sample_epoch(records, rng)
        plans = epoch_batches(order, by_id, rng, cfg["batch_size"])
        if smoke:
            plans = plans[:1]
        loader.set_epoch(epoch)
        losses, began = [], time.monotonic()
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            lr = learning_rate(epoch, number, len(plans))
            params, state, value = engine.audited_batch_update(params, state, *batch, learning_rate=lr)
            losses.append(value)
        if smoke:
            history.append({"epoch": 1, "train_shift_mse": float(np.mean(losses))})
            break
        metrics = evaluate(out, manifest, engine, params, means)
        mae = float(metrics["graph_query"]["mae"])
        improved = best is None or mae < best["mae"] - MIN_DELTA
        if improved:
            best, stalled = {"epoch": epoch, "mae": mae}, 0
            atomic_json(run / "best.json", best)
        else:
            stalled += 1
        row = {
            "epoch": epoch, "train_shift_mse": float(np.mean(losses)), "validation": metrics,
            "seconds": time.monotonic() - began, "updates": len(plans), "learning_rate": lr,
            "best_epoch": best["epoch"], "stalled_epochs": stalled,
        }
        history.append(row)
        atomic_json(run / "history.json", history)
        atomic_json(run / "crop-stats.json", loader.crop_stats())
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"seed": seed, **{k: row[k] for k in ("epoch", "train_shift_mse", "seconds", "best_epoch")}, "val_mae": mae}), flush=True)
        if stalled >= PATIENCE:
            break
    loader.close()
    if smoke:
        atomic_json(run / "verification.json", {
            "passed": True, "finite_update": True, "crop_stats": loader.crop_stats(),
            "device_memory": peak_device_memory(), **provenance,
        })
        return
    checkpoint = run / "checkpoints" / f"epoch-{best['epoch']:03d}"
    params, _, _ = load_checkpoint(checkpoint, (params, state))
    final = evaluate(out, manifest, engine, params, means, replicates=2000,
                     predictions=run / "validation_predictions.csv")
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {
        "passed": True, "seed": seed, "epochs_completed": len(history),
        "selected_epoch": best["epoch"], "selected_validation_mae": final["graph_query"]["mae"],
        "wall_seconds": time.monotonic() - run_started, "device_memory": peak_device_memory(),
        "crop_stats": loader.crop_stats(), "test_data_included": False,
        "predictions_sha256": digest(run / "validation_predictions.csv"), **provenance,
    })


def report(root):
    out = experiment_root(root)
    rows = []
    for seed in SEEDS:
        run = out / f"seed-{seed}"
        verify = read(run / "verification.json")
        metric = read(run / "final.json")["graph_query"]
        peak = verify["device_memory"]["peak_bytes_in_use"]
        rows.append({
            "seed": seed, "mae": metric["mae"], "mae_ci95": metric["mae_ci95"],
            "selected_epoch": verify["selected_epoch"], "epochs": verify["epochs_completed"],
            "wall_seconds": verify["wall_seconds"], "peak_vram_bytes": peak,
            "crop_fraction": verify["crop_stats"]["observed_crop_fraction"],
            "retained_node_fraction": verify["crop_stats"]["mean_retained_node_fraction_on_crops"],
        })
    maes = np.asarray([row["mae"] for row in rows])
    wall = np.asarray([row["wall_seconds"] for row in rows])
    peaks = np.asarray([row["peak_vram_bytes"] for row in rows], dtype=float)
    summary = {
        "mean_mae": float(maes.mean()), "sd_mae": float(maes.std(ddof=1)),
        "mean_wall_seconds": float(wall.mean()), "range_wall_seconds": [float(wall.min()), float(wall.max())],
        "mean_peak_vram_gib": float(peaks.mean() / 2**30), "max_peak_vram_gib": float(peaks.max() / 2**30),
        "mean_retained_node_fraction": float(np.mean([row["retained_node_fraction"] for row in rows])),
    }
    atomic_json(out / "results.json", rows)
    atomic_json(out / "summary.json", summary)
    lines = [
        "# 50k GQT 5 A query-centred crop", "",
        "Three seeds; AdamW 1e-4; 1e-3 held through epoch 10 then cosine decay; 25% crop draws; full validation.", "",
        "| Seed | Selected epoch | Validation MAE | Wall time (min) | Peak VRAM (GiB) | Nodes retained on crops |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['selected_epoch']} | {row['mae']:.4f} | {row['wall_seconds']/60:.1f} | {row['peak_vram_bytes']/2**30:.2f} | {row['retained_node_fraction']:.3f} |")
    lines += ["", f"Mean MAE: **{summary['mean_mae']:.4f} +/- {summary['sd_mae']:.4f}**.",
              f"Mean wall time: **{summary['mean_wall_seconds']/60:.1f} min/run**. Maximum measured peak VRAM: **{summary['max_peak_vram_gib']:.2f} GiB**.", ""]
    (out / "report.md").write_text("\n".join(lines))
    atomic_json(out / "verification.json", {"passed": True, "runs": 3, "report_sha256": digest(out / "report.md"), "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]
    root = Path(os.environ["PKABENCH_RUNTIME"])
    gpu = action in ("smoke", "train")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu, allow_comp1400=gpu)
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "smoke": train(root, 17, smoke=True)
    elif action == "train": train(root, int(sys.argv[2]))
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
