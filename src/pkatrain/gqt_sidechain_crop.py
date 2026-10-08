"""Matched side-chain GQT crop-radius screen with resource accounting."""
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
    CropBatchLoader, learning_rate, source_hashes as crop_source_hashes,
)
from .gqt_regularization import RegularizedShiftEngine
from .graph_batches import epoch_batches
from .graph_experiment import evaluate
from .graph_mmap import build_bundle, default_bundle
from .graph_pkmod_compare import digest_array_tree
from .records import read
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


ARMS = ("full", "crop-5A", "crop-15A", "crop-20A")
RADII = {"full": None, "crop-5A": 5.0, "crop-15A": 15.0, "crop-20A": 20.0}


def experiment_root(root):
    return root / "pretraining/gqt-sidechain-crop-radius-v1-triton"


def source(root):
    return root / "pretraining/augmentation-sidechains-v1/source"


def code_hashes():
    mmap_code = Path(__file__).with_name("graph_mmap.py")
    return {
        str(Path(__file__)): digest(Path(__file__)),
        str(mmap_code): digest(mmap_code),
        **crop_source_hashes(),
    }


def peak_device_memory():
    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"), "bytes_in_use": stats.get("bytes_in_use"),
        "raw": {key: value for key, value in stats.items() if isinstance(value, (int, float))},
    }


def register(root):
    out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    src = source(root); parent = read(src / "manifest.json")
    if parent["config"].get("strict_backbone"):
        raise AssertionError("Side-chain source retained a backbone-only loader flag")
    protocol = {
        "architecture": "side-chain GQT, 50,001 parameters, node_dim 152",
        "input": "backbone plus 32 named native side-chain heavy-atom coordinate/presence slots",
        "arms": list(ARMS), "crop_radii_A": RADII, "crop_probability": CROP_PROBABILITY,
        "crop_geometry": "C-alpha distance; one uniformly sampled eligible query; supervise centre query on crop draws",
        "seeds": list(SEEDS), "max_epochs": EPOCHS,
        "optimizer": "AdamW, weight decay 1e-4 on learned matrices/embeddings",
        "dropout": 0.0, "schedule": "1e-3 through epoch 10; per-update cosine to 1e-5 in epochs 11-20",
        "selection": f"minimum validation group-macro MAE; min_delta {MIN_DELTA}; patience {PATIENCE}",
        "fixed": "same cleaned 5k cohort and historical-pKPDB explicit-shift target; batch 8; full validation",
        "attention_backend": "indexed Triton encoder; native query attention", "test_data_included": False,
        "resource_metrics": "end-to-end monotonic time and JAX allocator peak device memory",
    }
    atomic_json(out / "protocol.json", protocol)
    means = root / "pretraining/augmentation-sidechains-v1/gqt-baseline/train_type_means.json"
    for arm in ARMS:
        dest = out / arm; dest.mkdir(exist_ok=True)
        manifest = copy.deepcopy(parent)
        manifest["parent"] = {"path": str(src), "manifest_sha256": digest(src / "manifest.json")}
        manifest["sidechain_crop_protocol_sha256"] = digest(out / "protocol.json")
        manifest["config"].update(
            seed=None, epochs=EPOCHS, learning_rate=1e-3, constant_epochs=10,
            end_learning_rate=1e-5, schedule="hold 10 epochs then per-update cosine",
            batch_size=8, accumulation=1, matmul_precision="highest", dropout_rate=0.0,
            context_mask_probability=0.0, optimizer="AdamW", weight_decay=1e-4,
            weight_decay_mask="ndim>=2", crop_radius_A=RADII[arm],
            crop_probability=CROP_PROBABILITY if RADII[arm] else 0.0,
            objective="explicit signed pKPDB PK_MOD shift MSE; uniform eligible sites within structure",
            selection=f"minimum validation group-macro MAE; min_delta={MIN_DELTA}; patience={PATIENCE}",
            augmentation_arm=arm, graph_backend="verified read-only mmap",
            attention_backend="indexed Triton encoder; native query attention",
        )
        data = dest / "data"
        if not data.exists(): data.symlink_to((src / "data").resolve(), target_is_directory=True)
        if not (dest / "preparation.json").exists(): shutil.copy2(src / "preparation.json", dest / "preparation.json")
        if not (dest / "train_type_means.json").exists(): shutil.copy2(means, dest / "train_type_means.json")
        atomic_json(dest / "manifest.json", manifest)
    atomic_json(out / "registration.json", {
        "passed": True, "protocol_sha256": digest(out / "protocol.json"),
        "source_manifest_sha256": digest(src / "manifest.json"), "code_hashes": code_hashes(),
    })


def prepare_mmap(root):
    out = experiment_root(root); registration = read(out / "registration.json")
    if registration["code_hashes"] != code_hashes(): raise AssertionError("Registration/code mismatch")
    src = source(root); manifest = read(src / "manifest.json")
    destination = build_bundle(src, manifest)
    receipt = read(destination / "verification.json")
    atomic_json(out / "mmap-verification.json", {
        "passed": bool(receipt["passed"] and receipt["all_records_and_fields_bit_identical"]),
        "path": str(destination), "bytes": receipt["bytes"],
        "verification_sha256": digest(destination / "verification.json"),
        "source_manifest_sha256": digest(src / "manifest.json"), "code_hashes": code_hashes(),
    })


def train(root, arm, seed, *, smoke=False):
    if arm not in ARMS or seed not in SEEDS: raise ValueError((arm, seed))
    base = experiment_root(root); registration = read(base / "registration.json"); mmap = read(base / "mmap-verification.json")
    if registration["code_hashes"] != code_hashes() or mmap["code_hashes"] != code_hashes() or not mmap["passed"]:
        raise AssertionError("Side-chain crop provenance mismatch")
    out = base / arm; stored_manifest = out / "manifest.json"; manifest = read(stored_manifest)
    manifest["config"]["seed"] = seed; cfg = manifest["config"]
    run_started = time.monotonic()
    params = initialize(jax.random.PRNGKey(seed), **cfg["architecture"])
    engine = RegularizedShiftEngine(params, weight_decay=cfg["weight_decay"])
    state = engine.optimizer.init(params); rng = np.random.default_rng(seed)
    run = out / (f"smoke-seed-{seed}" if smoke else f"seed-{seed}"); run.mkdir(parents=True, exist_ok=True)
    by_id = {row["complex_id"]: row for row in manifest["records"]}; records = [by_id[cid] for cid in manifest["train"]]
    loader = CropBatchLoader(out, manifest, cfg["batch_size"], backend="mmap", seed=seed,
                             crop_radius=RADII[arm], crop_probability=cfg["crop_probability"])
    means = np.asarray([read(out / "train_type_means.json")[group] for group in GROUPS])
    provenance = {"manifest_sha256": digest(stored_manifest), "seed": seed, "arm": arm,
                  "code_hashes": code_hashes(), "parameter_count": cfg["parameter_count"],
                  "mmap_verification_sha256": mmap["verification_sha256"]}
    atomic_json(run / "run.json", {**provenance, "config": cfg, "loader": loader.provenance(),
                                    "initial_parameter_digest": digest_array_tree(params)})
    history, best, stalled = [], None, 0
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        plans = epoch_batches(sample_epoch(records, rng), by_id, rng, cfg["batch_size"])
        if smoke: plans = plans[:1]
        loader.set_epoch(epoch); losses = []; began = time.monotonic()
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            lr = learning_rate(epoch, number, len(plans))
            params, state, value = engine.audited_batch_update(params, state, *batch, learning_rate=lr)
            losses.append(value)
        if smoke:
            history.append({"epoch": 1, "train_shift_mse": float(np.mean(losses))}); break
        metrics = evaluate(out, manifest, engine, params, means); mae = float(metrics["graph_query"]["mae"])
        if best is None or mae < best["mae"] - MIN_DELTA:
            best, stalled = {"epoch": epoch, "mae": mae}, 0; atomic_json(run / "best.json", best)
        else: stalled += 1
        row = {"epoch": epoch, "train_shift_mse": float(np.mean(losses)), "validation": metrics,
               "seconds": time.monotonic() - began, "updates": len(plans), "learning_rate": lr,
               "best_epoch": best["epoch"], "stalled_epochs": stalled}
        history.append(row); atomic_json(run / "history.json", history); atomic_json(run / "crop-stats.json", loader.crop_stats())
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch, "val_mae": mae,
                          "train_shift_mse": row["train_shift_mse"], "best_epoch": best["epoch"], "seconds": row["seconds"]}), flush=True)
        if stalled >= PATIENCE: break
    loader.close()
    if smoke:
        atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                    "crop_stats": loader.crop_stats(), "device_memory": peak_device_memory(), **provenance}); return
    params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final = evaluate(out, manifest, engine, params, means, replicates=2000,
                     predictions=run / "validation_predictions.csv")
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {
        "passed": True, "epochs_completed": len(history), "selected_epoch": best["epoch"],
        "wall_seconds": time.monotonic() - run_started, "device_memory": peak_device_memory(),
        "crop_stats": loader.crop_stats(), "predictions_sha256": digest(run / "validation_predictions.csv"),
        "test_data_included": False, **provenance,
    })


def smoke(root):
    for arm in ARMS: train(root, arm, 17, smoke=True)
    atomic_json(experiment_root(root) / "smoke-verification.json", {
        "passed": True, "arms": list(ARMS), "finite_update_each_arm": True, "code_hashes": code_hashes()})


def report(root):
    base = experiment_root(root); rows = []
    for arm in ARMS:
        for seed in SEEDS:
            run = base / arm / f"seed-{seed}"; verify = read(run / "verification.json")
            metric = read(run / "final.json")["graph_query"]; peak = verify["device_memory"]["peak_bytes_in_use"]
            rows.append({"arm": arm, "radius_A": RADII[arm], "seed": seed, "mae": metric["mae"],
                         "mae_ci95": metric["mae_ci95"], "selected_epoch": verify["selected_epoch"],
                         "epochs": verify["epochs_completed"], "wall_seconds": verify["wall_seconds"],
                         "peak_vram_bytes": peak, "crop_fraction": verify["crop_stats"]["observed_crop_fraction"],
                         "retained_node_fraction": verify["crop_stats"]["mean_retained_node_fraction_on_crops"]})
    baseline = {row["seed"]: row["mae"] for row in rows if row["arm"] == "full"}; summary = []
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]; values = np.asarray([row["mae"] for row in subset])
        summary.append({"arm": arm, "mean_mae": float(values.mean()), "sd_mae": float(values.std(ddof=1)),
                        "mean_delta_vs_full": float(np.mean([row["mae"] - baseline[row["seed"]] for row in subset])),
                        "selected_epochs": [row["selected_epoch"] for row in subset],
                        "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in subset])),
                        "max_peak_vram_gib": float(max(row["peak_vram_bytes"] for row in subset) / 2**30),
                        "mean_retained_node_fraction": None if arm == "full" else float(np.mean([row["retained_node_fraction"] for row in subset]))})
    winner = min(summary, key=lambda row: row["mean_mae"])["arm"]
    atomic_json(base / "results.json", rows); atomic_json(base / "summary.json", {"winner": winner, "arms": summary})
    lines = ["# Side-chain GQT query-centred crop-radius screen", "",
             "Three seeds per arm; matched 50k side-chain GQT, AdamW 1e-4, 10-epoch learning-rate hold, 25% crop draws, full validation.", "",
             "| Arm | Validation MAE | Seed SD | Delta vs full | Selected epochs | Mean time/run (min) | Peak VRAM (GiB) | Nodes retained |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for row in summary:
        retained = "—" if row["mean_retained_node_fraction"] is None else f"{row['mean_retained_node_fraction']:.3f}"
        lines.append(f"| {row['arm']} | {row['mean_mae']:.4f} | {row['sd_mae']:.4f} | {row['mean_delta_vs_full']:+.4f} | {', '.join(map(str,row['selected_epochs']))} | {row['mean_wall_seconds']/60:.1f} | {row['max_peak_vram_gib']:.2f} | {retained} |")
    lines += ["", f"Lowest three-seed mean: **{winner}**.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "verification.json", {"passed": True, "runs": len(rows), "winner": winner,
                                               "report_sha256": digest(base / "report.md"), "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"]); gpu = action in ("smoke", "train")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu, allow_comp1400=gpu)
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "mmap": prepare_mmap(root)
    elif action == "smoke": smoke(root)
    elif action == "train": train(root, sys.argv[2], int(sys.argv[3]))
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
