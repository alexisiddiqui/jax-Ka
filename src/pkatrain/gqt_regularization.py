"""Three-seed regularization screen for the 50k explicit-shift GQT."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxpropka.parameters import GROUPS
from pkanet.model import (
    PKPDB_PK_MOD, initialize, predict_pkpdb_indexed, predict_shift_indexed,
)
from pkabench.runtime import atomic_json, digest, require_compute
from .graph_batches import BatchLoader, epoch_batches
from .graph_experiment import evaluate
from .graph_pkmod_compare import digest_array_tree
from .records import read
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


ARMS = ("baseline", "mask", "adamw", "mask-adamw")
SEEDS = (17, 29, 43)
EPOCHS = 20
PATIENCE = 8
MIN_DELTA = 0.001


def source_hashes():
    root = Path(__file__).parents[1]
    paths = (
        Path(__file__), root / "pkanet/model.py", root / "pkatrain/graph_batches.py",
        root / "pkatrain/context_augmentation.py", root / "pkatrain/graph_experiment.py",
        root / "pkatrain/trainer.py",
    )
    return {str(path): digest(path) for path in paths}


def cosine_lr(epoch, batch, batches, *, start=1e-3, end=1e-5):
    if EPOCHS == 1:
        fraction = 1.0
    else:
        within = (batch - 1) / max(batches - 1, 1)
        fraction = ((epoch - 1) + within) / (EPOCHS - 1)
    return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * fraction)))


def decay_mask(params):
    """Decay learned matrices/embeddings, excluding biases and norm vectors."""
    return jax.tree.map(lambda value: value.ndim >= 2, params)


class RegularizedShiftEngine:
    def __init__(self, params, *, weight_decay):
        self.baseline = jnp.asarray(PKPDB_PK_MOD, jnp.float32)
        transform = (
            optax.adamw(1.0, weight_decay=weight_decay, mask=decay_mask(params))
            if weight_decay else optax.adam(1.0)
        )
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), transform)
        self.forward = jax.jit(predict_pkpdb_indexed)
        self.batch_forward = jax.jit(jax.vmap(predict_pkpdb_indexed, in_axes=(None, 0)))

        def one_loss(p, graph, target, eligible):
            target_shift = target - self.baseline[graph["query_group"]]
            error = jnp.where(eligible, predict_shift_indexed(p, graph) - target_shift, 0.0)
            return jnp.sum(error * error) / jnp.maximum(jnp.sum(eligible), 1)

        def batch_loss(p, graphs, targets, eligible, valid):
            losses = jax.vmap(one_loss, in_axes=(None, 0, 0, 0))(p, graphs, targets, eligible)
            return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        def step(p, state, graphs, targets, eligible, valid, learning_rate):
            loss, gradient = jax.value_and_grad(batch_loss)(p, graphs, targets, eligible, valid)
            finite = jnp.isfinite(loss) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)
            ]))
            updates, new_state = self.optimizer.update(gradient, state, p)
            updates = jax.tree.map(lambda value: value * learning_rate, updates)
            return optax.apply_updates(p, updates), new_state, loss, finite

        self.batch_step = jax.jit(step)

    def audited_batch_update(self, params, state, inputs, targets, eligible, valid, learning_rate):
        params, state, loss, finite = self.batch_step(
            params, state, inputs, targets, eligible, valid, learning_rate
        )
        if not bool(finite):
            raise FloatingPointError("Nonfinite regularization-screen update")
        return params, state, float(loss)


def experiment_root(root):
    return root / "pretraining" / "gqt-50k-regularization-v2-triton"


def register(root):
    destination = experiment_root(root)
    destination.mkdir(parents=True, exist_ok=True)
    source = root / "pretraining" / "gqt-pkai-parameter-sweep-v1" / "gqt-long" / "50k"
    parent = read(source / "manifest.json")
    contexts = root / "pretraining" / "augmentation-v1" / "contexts"
    context_hash = digest(contexts / "plan.json")
    protocol = {
        "architecture": "GQT 49,709 parameters, strict backbone",
        "arms": list(ARMS), "seeds": list(SEEDS), "max_epochs": EPOCHS,
        "schedule": "per-update cosine 1e-3 to 1e-5 from the first update",
        "selection": f"minimum validation group-macro MAE; min_delta {MIN_DELTA}; patience {PATIENCE}",
        "augmentation": "5% deterministic context-residue masking in mask arms",
        "weight_decay": "AdamW 1e-4 on learned matrices/embeddings in adamw arms; biases and norm vectors excluded",
        "fixed": "5k cohort, explicit historical-pKPDB signed-shift MSE, uniform site weights, batch 8, no dropout, full unaugmented validation",
        "attention_backend": "validated float32 indexed Triton encoder; native query attention",
        "test_data_included": False,
    }
    atomic_json(destination / "protocol.json", protocol)
    for arm in ARMS:
        out = destination / arm
        out.mkdir(exist_ok=True)
        manifest = copy.deepcopy(parent)
        manifest["parent"] = {"path": str(source), "manifest_sha256": digest(source / "manifest.json")}
        manifest["regularization_protocol_sha256"] = digest(destination / "protocol.json")
        manifest["context_path"] = str(contexts)
        manifest["context_plan_sha256"] = context_hash
        manifest["config"].update(
            seed=None, epochs=EPOCHS, learning_rate=1e-3, end_learning_rate=1e-5,
            schedule="immediate per-update cosine", dropout_rate=0.0,
            context_mask_probability=0.05 if "mask" in arm else 0.0,
            optimizer="AdamW" if "adamw" in arm else "Adam",
            weight_decay=1e-4 if "adamw" in arm else 0.0,
            weight_decay_mask="ndim>=2" if "adamw" in arm else None,
            objective="explicit signed pKPDB PK_MOD shift MSE; uniform eligible sites within structure",
            selection=f"minimum validation group-macro MAE; min_delta={MIN_DELTA}; patience={PATIENCE}",
            augmentation_arm=arm, graph_backend="verified read-only mmap",
            attention_backend="indexed Triton encoder; native query attention",
        )
        data = out / "data"
        if not data.exists():
            data.symlink_to((source / "data").resolve(), target_is_directory=True)
        for name in ("preparation.json", "train_type_means.json"):
            if not (out / name).exists():
                shutil.copy2(source / name, out / name)
        atomic_json(out / "manifest.json", manifest)
    atomic_json(destination / "registration.json", {
        "passed": True, "protocol_sha256": digest(destination / "protocol.json"),
        "source_hashes": source_hashes(), "context_plan_sha256": context_hash,
    })
    print(json.dumps({"registered": str(destination), "arms": ARMS, "seeds": SEEDS}))


def configured_manifest(path, seed):
    manifest = read(path)
    manifest["config"]["seed"] = seed
    return manifest


def train(root, arm, seed, *, smoke=False):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError((arm, seed))
    base = experiment_root(root)
    registration = read(base / "registration.json")
    if not registration["passed"] or registration["source_hashes"] != source_hashes():
        raise AssertionError("Registration/code provenance mismatch")
    out = base / arm
    stored_manifest = out / "manifest.json"
    manifest = configured_manifest(stored_manifest, seed)
    cfg = manifest["config"]
    params = initialize(jax.random.PRNGKey(seed), **cfg["architecture"])
    engine = RegularizedShiftEngine(params, weight_decay=cfg["weight_decay"])
    state = engine.optimizer.init(params)
    rng = np.random.default_rng(seed)
    run = out / (f"smoke-seed-{seed}" if smoke else f"seed-{seed}")
    run.mkdir(parents=True, exist_ok=True)
    by_id = {row["complex_id"]: row for row in manifest["records"]}
    records = [by_id[cid] for cid in manifest["train"]]
    loader = BatchLoader(out, manifest, cfg["batch_size"], backend="mmap")
    means = np.asarray([read(out / "train_type_means.json")[group] for group in GROUPS])
    manifest_identity = digest(stored_manifest)
    provenance = {
        "manifest_sha256": manifest_identity, "seed": seed,
        "source_hashes": source_hashes(), "parameter_count": cfg["parameter_count"],
    }
    atomic_json(run / "run.json", {
        **provenance, "config": cfg, "loader": loader.provenance(),
        "initial_parameter_digest": digest_array_tree(params),
    })
    history = []
    best = None
    stalled = 0
    max_epochs = 1 if smoke else EPOCHS
    for epoch in range(1, max_epochs + 1):
        order = sample_epoch(records, rng)
        plans = epoch_batches(order, by_id, rng, cfg["batch_size"])
        if smoke:
            plans = plans[:1]
        losses = []
        began = time.monotonic()
        mask_hash = loader.set_epoch(epoch)
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            learning_rate = cosine_lr(epoch, number, len(plans))
            params, state, value = engine.audited_batch_update(
                params, state, *batch, learning_rate=learning_rate
            )
            losses.append(value)
            if number % 20 == 0:
                atomic_json(run / "progress.json", {
                    "arm": arm, "seed": seed, "epoch": epoch, "epochs": max_epochs,
                    "batches": number, "total_batches": len(plans),
                    "learning_rate": learning_rate, "elapsed_seconds": time.monotonic() - began,
                })
        if smoke:
            row = {"epoch": 1, "train_shift_mse": float(np.mean(losses)),
                   "learning_rate": learning_rate, "mask_sha256": mask_hash,
                   "seconds": time.monotonic() - began}
            history.append(row)
            break
        metrics = evaluate(out, manifest, engine, params, means)
        mae = float(metrics["graph_query"]["mae"])
        improved = best is None or mae < best["mae"] - MIN_DELTA
        if improved:
            stalled = 0
            best = {"epoch": epoch, "mae": mae}
            atomic_json(run / "best.json", best)
        else:
            stalled += 1
        row = {
            "epoch": epoch, "train_shift_mse": float(np.mean(losses)),
            "validation": metrics, "seconds": time.monotonic() - began,
            "updates": len(plans), "learning_rate": learning_rate,
            "mask_sha256": mask_hash, "best_epoch": best["epoch"], "stalled_epochs": stalled,
        }
        history.append(row)
        atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {
            **provenance, "epoch": epoch, "rng": rng.bit_generator.state,
            "validation_mae": mae, "mask_sha256": mask_hash,
        })
        print(json.dumps({
            "arm": arm, "seed": seed, "epoch": epoch, "train_shift_mse": row["train_shift_mse"],
            "val_mae": mae, "best_epoch": best["epoch"], "stalled": stalled,
            "seconds": row["seconds"], "learning_rate": learning_rate,
        }), flush=True)
        if stalled >= PATIENCE:
            break
    loader.close()
    atomic_json(run / "history.json", history)
    if smoke:
        atomic_json(run / "verification.json", {
            "passed": True, "smoke": True, "finite_update": True,
            "arm": arm, "seed": seed, **provenance,
        })
        return

    checkpoint = run / "checkpoints" / f"epoch-{best['epoch']:03d}"
    params, _, metadata = load_checkpoint(checkpoint, (params, state))
    if metadata["manifest_sha256"] != manifest_identity:
        raise AssertionError("Selected checkpoint manifest mismatch")
    final = evaluate(
        out, manifest, engine, params, means, replicates=2000,
        predictions=run / "validation_predictions.csv",
    )
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {
        "passed": True, "arm": arm, "seed": seed, "epochs_completed": len(history),
        "early_stopped": len(history) < EPOCHS, "selected_epoch": best["epoch"],
        "selected_validation_mae": final["graph_query"]["mae"],
        "finite_gradients": True, "final_parameter_digest": digest_array_tree(params),
        "predictions_sha256": digest(run / "validation_predictions.csv"),
        "test_data_included": False, **provenance,
    })


def smoke(root):
    for arm in ARMS:
        train(root, arm, 17, smoke=True)
    output = experiment_root(root) / "smoke-verification.json"
    atomic_json(output, {
        "passed": True, "arms": list(ARMS), "finite_update_each_arm": True,
        "source_hashes": source_hashes(),
    })


def report(root):
    base = experiment_root(root)
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            run = base / arm / f"seed-{seed}"
            verification = read(run / "verification.json")
            if not verification["passed"] or verification["test_data_included"]:
                raise AssertionError((arm, seed, verification))
            metric = read(run / "final.json")["graph_query"]
            rows.append({
                "arm": arm, "seed": seed, "selected_epoch": verification["selected_epoch"],
                "epochs_completed": verification["epochs_completed"],
                "mae": metric["mae"], "mae_ci95": metric["mae_ci95"],
                "rmse": metric["rmse"], "spearman": metric["spearman"],
                "sites": metric["sites"], "groups": metric["groups"],
            })
    summary = []
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        values = np.asarray([row["mae"] for row in subset])
        baseline = {row["seed"]: row["mae"] for row in rows if row["arm"] == "baseline"}
        deltas = np.asarray([row["mae"] - baseline[row["seed"]] for row in subset])
        summary.append({
            "arm": arm, "seeds": len(subset), "mean_mae": float(values.mean()),
            "sd_mae": float(values.std(ddof=1)), "mean_delta_vs_baseline": float(deltas.mean()),
            "selected_epochs": [row["selected_epoch"] for row in subset],
        })
    winner = min(summary, key=lambda row: row["mean_mae"])["arm"]
    atomic_json(base / "results.json", rows)
    atomic_json(base / "summary.json", {"winner": winner, "arms": summary})
    lines = [
        "# 50k GQT regularization screen", "",
        "Three seeds per arm; maximum 20 epochs; immediate cosine decay; validation-MAE checkpoint selection with patience eight. No test data were read.", "",
        "| Arm | Mean validation MAE | Seed SD | Mean delta vs baseline | Selected epochs |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['arm']} | {row['mean_mae']:.4f} | {row['sd_mae']:.4f} | "
            f"{row['mean_delta_vs_baseline']:+.4f} | {', '.join(map(str, row['selected_epochs']))} |"
        )
    lines += ["", f"Lowest three-seed mean: **{winner}**.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "verification.json", {
        "passed": True, "runs": len(rows), "winner": winner,
        "results_sha256": digest(base / "results.json"),
        "summary_sha256": digest(base / "summary.json"),
        "report_sha256": digest(base / "report.md"), "test_data_included": False,
    })
    print(json.dumps({"winner": winner, "summary": summary}, indent=2))


def main():
    import sys
    action = sys.argv[1]
    root = Path(os.environ["PKABENCH_RUNTIME"])
    gpu = action in ("train", "smoke")
    require_compute(
        threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        gpu_benchmark=gpu, allow_comp1400=gpu,
    )
    jax.config.update("jax_enable_x64", False)
    if action == "register":
        register(root)
    elif action == "smoke":
        smoke(root)
    elif action == "train":
        train(root, sys.argv[2], int(sys.argv[3]))
    elif action == "report":
        report(root)
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
