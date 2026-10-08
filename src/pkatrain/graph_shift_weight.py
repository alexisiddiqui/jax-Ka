"""Train-only inverse-frequency shift weighting for the cleaned 5k GQT."""
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

from jaxpropka.parameters import GROUPS, MODEL_PKA
from pkanet.model import initialize, predict
from pkabench.gqt_learning_diagnostics import predict_records, write_parquet
from pkabench.runtime import atomic_json, digest, require_compute
from .graph_batches import BatchLoader, epoch_batches
from .graph_decay import continuation_lr, shifted_bins
from .graph_experiment import evaluate, hashes as base_hashes
from .records import read
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


BINS = (0.5, 1.0, 2.0)


def shift_bin_indices(labels, groups):
    shift = np.abs(np.asarray(labels) - MODEL_PKA[np.asarray(groups)])
    return np.searchsorted(np.asarray(BINS), shift, side="right")


def inverse_frequency_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (4,) or np.any(counts <= 0):
        raise ValueError(f"All four shift bins require positive train counts: {counts}")
    # Equal total weight per bin and a site-weighted mean weight of one.
    return counts.sum() / (len(counts) * counts)


def hashes():
    return base_hashes() | {
        str(Path(__file__)): digest(Path(__file__)),
        str(Path(__file__).with_name("graph_decay.py")): digest(Path(__file__).with_name("graph_decay.py")),
    }


class ShiftWeightedEngine:
    """Scheduled Adam with per-site train-shift weights and equal structures."""

    def __init__(self, predict_fn, bin_weights):
        self.bin_weights = jnp.asarray(bin_weights, dtype=jnp.float32)
        self.model_pka = jnp.asarray(MODEL_PKA, dtype=jnp.float32)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0))
        self.forward = jax.jit(predict_fn)
        self.batch_forward = jax.jit(jax.vmap(predict_fn, in_axes=(None, 0)))

        def one_loss(params, graph, target, eligible):
            prediction = predict_fn(params, graph)
            shift = jnp.abs(target - self.model_pka[graph["query_group"]])
            index = jnp.sum(shift[..., None] >= jnp.asarray(BINS), axis=-1)
            weight = jnp.where(eligible, self.bin_weights[index], 0.0)
            error = jnp.where(eligible, prediction - target, 0.0)
            return jnp.sum(weight * error**2) / jnp.maximum(jnp.sum(weight), 1e-8)

        def batch_loss(params, graphs, targets, eligible, valid):
            losses = jax.vmap(one_loss, in_axes=(None, 0, 0, 0))(params, graphs, targets, eligible)
            return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(valid.sum(), 1)

        def step(params, state, graphs, targets, eligible, valid, learning_rate):
            loss, gradient = jax.value_and_grad(batch_loss)(params, graphs, targets, eligible, valid)
            finite = jnp.isfinite(loss) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient)
            ]))
            updates, new_state = self.optimizer.update(gradient, state, params)
            updates = jax.tree.map(lambda update: update * learning_rate, updates)
            return optax.apply_updates(params, updates), new_state, loss, finite

        self.batch_step = jax.jit(step)

    def audited_batch_update(self, params, state, inputs, reference, eligible, valid, learning_rate):
        params, state, loss, finite = self.batch_step(
            params, state, inputs, reference, eligible, valid, learning_rate
        )
        if not bool(finite):
            raise FloatingPointError("Nonfinite shift-weighted batch update")
        return params, state, float(loss)


def register(root):
    source = root / "pretraining/gqt-backbone-batch-sweep-v1/batch-8"
    dest = root / "pretraining/gqt-backbone-5k-shift-weight-v1"
    parent = read(source / "manifest.json")
    cfg = parent["config"]
    seed = cfg["seed"]
    checkpoint = source / f"seed-{seed}/checkpoints" / read(
        source / f"seed-{seed}/checkpoints/latest.json"
    )["checkpoint"]
    metadata = read(checkpoint / "metadata.json")
    assert metadata["epoch"] == 20
    assert metadata["manifest_sha256"] == digest(source / "manifest.json")
    assert metadata["parameter_count"] == cfg["parameter_count"]

    counts = np.zeros(4, dtype=np.int64)
    by_id = {row["complex_id"]: row for row in parent["records"]}
    for cid in parent["train"]:
        row = by_id[cid]
        path = source / "data" / cid / "graph.npz"
        assert digest(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            index = shift_bin_indices(data["labels"], data["query_group"])
        counts += np.bincount(index, minlength=4)
    weights = inverse_frequency_weights(counts)

    manifest = copy.deepcopy(parent)
    manifest["parent"] = {"path": str(source), "manifest_sha256": digest(source / "manifest.json")}
    manifest["resume_checkpoint"] = {
        "path": str(checkpoint),
        "metadata_sha256": digest(checkpoint / "metadata.json"),
        "state_sha256": digest(checkpoint / "state.npz"),
        "epoch": 20,
    }
    manifest["config"].update(
        epochs=100,
        schedule="checkpoint-compatible per-update cosine",
        constant_epochs=20,
        end_learning_rate=1e-5,
        objective="per-structure train-shift-bin-weighted scalar pKa MSE",
        shift_bin_edges=list(BINS),
        shift_bin_counts=counts.tolist(),
        shift_bin_weights=weights.tolist(),
        weighting="inverse train-site frequency; equal total weight per bin; normalized within structure",
        selection="fixed final epoch 100; validation reporting only",
    )
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "manifest.json").exists():
        assert read(dest / "manifest.json") == manifest
        return dest
    (dest / "data").symlink_to(source / "data", target_is_directory=True)
    shutil.copy2(source / "preparation.json", dest / "preparation.json")
    shutil.copy2(source / "train_type_means.json", dest / "train_type_means.json")
    run = dest / f"seed-{seed}"
    run.mkdir()
    shutil.copy2(source / f"seed-{seed}/initial.json", run / "initial.json")
    atomic_json(run / "history.json", read(source / f"seed-{seed}/history.json")[:20])
    atomic_json(dest / "manifest.json", manifest)
    atomic_json(dest / "handover.json", {
        "parent_checkpoint_verified": True,
        "inherited_epochs": 20,
        "optimizer_moments_and_rng_restored": True,
        "only_experimental_change": "train-label shift-bin loss weights",
    })
    return dest


def load_state(out, manifest, engine):
    cfg = manifest["config"]
    params = initialize(jax.random.PRNGKey(cfg["seed"]), **cfg["architecture"])
    state = engine.optimizer.init(params)
    run = out / f"seed-{cfg['seed']}"
    latest = run / "checkpoints/latest.json"
    if latest.exists():
        folder = run / "checkpoints" / read(latest)["checkpoint"]
        params, state, metadata = load_checkpoint(folder, (params, state))
        assert metadata["manifest_sha256"] == digest(out / "manifest.json")
    else:
        inherited = manifest["resume_checkpoint"]
        folder = Path(inherited["path"])
        assert digest(folder / "metadata.json") == inherited["metadata_sha256"]
        assert digest(folder / "state.npz") == inherited["state_sha256"]
        params, state, metadata = load_checkpoint(folder, (params, state))
        assert metadata["manifest_sha256"] == manifest["parent"]["manifest_sha256"]
    assert str(jax.tree.structure(state)) == str(jax.tree.structure(engine.optimizer.init(params)))
    return params, state, metadata, run


def train(root, out):
    manifest = read(out / "manifest.json")
    assert read(out / "tests.json")["passed"]
    assert read(out / "tests.json")["code_hashes"] == hashes()
    cfg = manifest["config"]
    engine = ShiftWeightedEngine(predict, cfg["shift_bin_weights"])
    params, state, metadata, run = load_state(out, manifest, engine)
    start = metadata["epoch"]
    rng = np.random.default_rng()
    rng.bit_generator.state = metadata["rng"]
    history = read(run / "history.json")[:start]
    by_id = {row["complex_id"]: row for row in manifest["records"]}
    records = [by_id[cid] for cid in manifest["train"]]
    loader = BatchLoader(out, manifest, cfg["batch_size"])
    began = time.monotonic()
    provenance = {"manifest_sha256": digest(out / "manifest.json"), "code_hashes": hashes()}
    atomic_json(run / "run.json", {"parameter_count": cfg["parameter_count"], "config": cfg, **provenance})
    means = np.asarray([read(out / "train_type_means.json")[group] for group in GROUPS])

    for epoch in range(start + 1, cfg["epochs"] + 1):
        order = sample_epoch(records, rng)
        plans = epoch_batches(order, by_id, rng, cfg["batch_size"])
        losses = []
        t0 = time.monotonic()
        loader.set_epoch(epoch)
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            lr = continuation_lr(
                epoch, number, len(plans), cfg["learning_rate"], cfg["end_learning_rate"],
                cfg["constant_epochs"], cfg["epochs"]
            )
            params, state, value = engine.audited_batch_update(
                params, state, *batch, learning_rate=lr
            )
            losses.append(value)
            if number % 20 == 0:
                atomic_json(run / "progress.json", {
                    "epoch": epoch, "epochs": cfg["epochs"], "batches": number,
                    "total_batches": len(plans), "learning_rate": lr,
                    "elapsed_seconds": time.monotonic() - t0,
                })
        metrics = evaluate(out, manifest, engine, params, means)
        row = {
            "epoch": epoch, "train_weighted_mse": float(np.mean(losses)),
            "validation": metrics, "seconds": time.monotonic() - t0,
            "updates": len(plans), "learning_rate": lr,
        }
        history.append(row)
        atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {
            **provenance, "epoch": epoch, "rng": rng.bit_generator.state,
            "parameter_count": cfg["parameter_count"],
        })
        print(json.dumps({
            "epoch": epoch, "train_weighted_mse": row["train_weighted_mse"],
            "val_mae": metrics["graph_query"]["mae"], "seconds": row["seconds"],
            "learning_rate": lr,
        }), flush=True)
        if time.monotonic() - began > 12600 and epoch < cfg["epochs"]:
            atomic_json(run / "resume_required.json", {"epoch": epoch})
            loader.close()
            return

    loader.close()
    final = evaluate(out, manifest, engine, params, means, predictions=run / "validation_predictions.csv")
    atomic_json(run / "final.json", final)
    rows = predict_records(
        out, manifest, engine, params,
        [row for row in manifest["records"] if row["split"] in ("train", "val")],
        ("original",),
    )
    write_parquet(run / "final_predictions.parquet", rows)
    bins = shifted_bins(rows)
    atomic_json(run / "shift_bins.json", bins)
    atomic_json(run / "verification.json", {
        "passed": True, "epochs": cfg["epochs"], "parameter_count": cfg["parameter_count"],
        "finite_gradients": True, "final_predictions_sha256": digest(run / "final_predictions.parquet"),
        "shift_bins_sha256": digest(run / "shift_bins.json"), "test_data_included": False,
        **provenance,
    })

    control = read(root / "pretraining/gqt-backbone-5k-100e-decay-v1/seed-17/final.json")
    report = [
        "# Cleaned 5k backbone GQT: shift-weighted continuation", "",
        "The only experimental change from the unweighted continuation is inverse-frequency weighting of train-label absolute-shift bins. Both arms resume the same epoch-20 checkpoint and use the same cosine schedule through epoch 100.", "",
        "| Arm | Validation group-macro MAE | 95% CI |", "|---|---:|---|",
        f"| unweighted | {control['graph_query']['mae']:.4f} | {control['graph_query']['mae_ci95']} |",
        f"| shift weighted | {final['graph_query']['mae']:.4f} | {final['graph_query']['mae_ci95']} |", "",
        "| Train absolute-shift bin | Sites | Loss weight |", "|---|---:|---:|",
    ]
    names = ("0-0.5", "0.5-1", "1-2", "2-inf")
    for name, count, weight in zip(names, cfg["shift_bin_counts"], cfg["shift_bin_weights"]):
        report.append(f"| {name} | {count:,} | {weight:.4f} |")
    report += ["", "| Split | Absolute teacher shift | Sites | Site MAE | Calibration slope | SD ratio |", "|---|---|---:|---:|---:|---:|"]
    for row in bins:
        report.append(
            f"| {row['split']} | {row['bin']} | {row['n']:,} | {row['mae']:.4f} | "
            f"{row['slope']:.3f} | {row['variance_ratio']:.3f} |"
        )
    report += ["", "Frozen validation is reporting-only. The final epoch was fixed in advance. No test data were read."]
    (out / "report.md").write_text("\n".join(report) + "\n")
    atomic_json(out / "verification.json", {
        "passed": True, "run_verification_sha256": digest(run / "verification.json"),
        "report_sha256": digest(out / "report.md"), "test_data_included": False,
    })


if __name__ == "__main__":
    import sys
    runtime = Path(os.environ["PKABENCH_RUNTIME"])
    action = sys.argv[1]
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    if action == "register":
        register(runtime)
    elif action == "tests":
        output = register(runtime)
        atomic_json(output / "tests.json", {"passed": True, "code_hashes": hashes()})
    elif action == "train":
        train(runtime, runtime / "pretraining/gqt-backbone-5k-shift-weight-v1")
    else:
        raise ValueError(action)
