"""Matched unweighted/weighted pKPDB-shift training for backbone GQT."""
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
from pkanet.model import PKPDB_PK_MOD, initialize, predict_pkpdb, predict_shift
from pkabench.gqt_learning_diagnostics import predict_records, write_parquet
from pkabench.runtime import atomic_json, digest, require_compute
from .graph_batches import BatchLoader, epoch_batches
from .graph_experiment import evaluate, hashes as base_hashes
from .records import read
from .trainer import sample_epoch, save_checkpoint


BINS = (0.5, 1.0, 2.0)
ARMS = ("unweighted", "weighted")


def hashes():
    return base_hashes() | {
        str(Path(__file__)): digest(Path(__file__)),
        str(Path(__file__).parents[1] / "pkanet/model.py"): digest(Path(__file__).parents[1] / "pkanet/model.py"),
    }


def shift_bin_indices(labels, groups):
    baseline = np.asarray(PKPDB_PK_MOD)[np.asarray(groups)]
    if not np.isfinite(baseline).all():
        raise ValueError("pKPDB has no registered PK_MOD for an observed group")
    return np.searchsorted(np.asarray(BINS), np.abs(np.asarray(labels) - baseline), side="right")


def inverse_frequency_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (4,) or np.any(counts <= 0):
        raise ValueError(f"All four pKPDB shift bins require positive train counts: {counts}")
    return counts.sum() / (len(counts) * counts)


class ExplicitShiftEngine:
    """Optimize signed pKPDB shifts; reconstruct absolute pKa only for evaluation."""

    def __init__(self, bin_weights, learning_rate=1e-3):
        self.bin_weights = jnp.asarray(bin_weights, jnp.float32)
        self.baseline = jnp.asarray(PKPDB_PK_MOD, jnp.float32)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
        self.forward = jax.jit(predict_pkpdb)
        self.batch_forward = jax.jit(jax.vmap(predict_pkpdb, in_axes=(None, 0)))

        def one_loss(params, graph, absolute_target, eligible):
            group = graph["query_group"]
            target_shift = absolute_target - self.baseline[group]
            predicted_shift = predict_shift(params, graph)
            index = jnp.sum(jnp.abs(target_shift)[..., None] >= jnp.asarray(BINS), axis=-1)
            weight = jnp.where(eligible, self.bin_weights[index], 0.0)
            error = jnp.where(eligible, predicted_shift - target_shift, 0.0)
            return jnp.sum(weight * error**2) / jnp.maximum(jnp.sum(weight), 1e-8)

        def batch_loss(params, graphs, targets, eligible, valid):
            losses = jax.vmap(one_loss, in_axes=(None, 0, 0, 0))(
                params, graphs, targets, eligible
            )
            return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(valid.sum(), 1)

        def step(params, state, graphs, targets, eligible, valid):
            loss, gradient = jax.value_and_grad(batch_loss)(params, graphs, targets, eligible, valid)
            finite = jnp.isfinite(loss) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient)
            ]))
            updates, new_state = self.optimizer.update(gradient, state, params)
            return optax.apply_updates(params, updates), new_state, loss, finite

        self.batch_step = jax.jit(step)

    def audited_batch_update(self, params, state, inputs, reference, eligible, valid):
        params, state, loss, finite = self.batch_step(
            params, state, inputs, reference, eligible, valid
        )
        if not bool(finite):
            raise FloatingPointError("Nonfinite explicit-shift update")
        return params, state, float(loss)


def register(root):
    source = root / "pretraining/gqt-backbone-batch-sweep-v1/batch-8"
    destination = root / "pretraining/gqt-backbone-5k-pkmod-v1"
    parent = read(source / "manifest.json")
    by_id = {row["complex_id"]: row for row in parent["records"]}
    counts = np.zeros(4, dtype=np.int64)
    group_counts = np.zeros(len(GROUPS), dtype=np.int64)
    for cid in parent["train"]:
        row = by_id[cid]
        path = source / "data" / cid / "graph.npz"
        assert digest(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            labels, groups = data["labels"], data["query_group"]
        group_counts += np.bincount(groups, minlength=len(GROUPS))
        counts += np.bincount(shift_bin_indices(labels, groups), minlength=4)
    assert group_counts[GROUPS.index("ARG")] == 0
    weights = inverse_frequency_weights(counts)
    protocol = {
        "source": str(source), "source_manifest_sha256": digest(source / "manifest.json"),
        "seed": parent["config"]["seed"], "epochs": 20, "batch_size": 8,
        "target": "signed pKa - historical pKPDB PK_MOD",
        "pk_mod": {group: float(PKPDB_PK_MOD[i]) for i, group in enumerate(GROUPS) if group != "ARG"},
        "arg_policy": "fail registration if an ARG label is present; historical pKPDB PK_MOD is undefined",
        "shift_bin_edges": list(BINS), "shift_bin_counts": counts.tolist(),
        "weighted_arm_weights": weights.tolist(),
        "comparison": "identical data, initialization, sampler, batches and optimizer; loss weights only",
        "selection": "fixed epoch 20; validation reporting only; no test data",
    }
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "protocol.json").exists():
        assert read(destination / "protocol.json") == protocol
    else:
        atomic_json(destination / "protocol.json", protocol)
    for arm in ARMS:
        out = destination / arm
        out.mkdir(exist_ok=True)
        manifest = copy.deepcopy(parent)
        manifest["parent"] = {"path": str(source), "manifest_sha256": digest(source / "manifest.json")}
        manifest["pkmod_protocol_sha256"] = digest(destination / "protocol.json")
        manifest["config"].update(
            epochs=20, batch_size=8, accumulation=1, arm=arm,
            objective="explicit signed pKPDB PK_MOD shift MSE",
            output="PK_MOD + predicted signed shift",
            shift_bin_edges=list(BINS), shift_bin_counts=counts.tolist(),
            shift_bin_weights=(np.ones(4) if arm == "unweighted" else weights).tolist(),
            weighting=("uniform eligible sites within each structure" if arm == "unweighted" else
                       "inverse train-shift-bin frequency; normalized within structure"),
            selection="fixed final epoch 20; validation reporting only",
        )
        if (out / "manifest.json").exists():
            assert read(out / "manifest.json") == manifest
            continue
        (out / "data").symlink_to(source / "data", target_is_directory=True)
        shutil.copy2(source / "preparation.json", out / "preparation.json")
        shutil.copy2(source / "train_type_means.json", out / "train_type_means.json")
        atomic_json(out / "manifest.json", manifest)
    return destination


def shifted_bins(rows):
    result = []
    baseline = np.asarray(PKPDB_PK_MOD)

    def summarize(target, predicted, group):
        target = np.asarray(target, float) - baseline[np.asarray(group, int)]
        predicted = np.asarray(predicted, float) - baseline[np.asarray(group, int)]
        slope = float(np.cov(target, predicted, ddof=0)[0, 1] / np.var(target)) if np.var(target) > 0 else None
        return {
            "n": len(target), "mae": float(np.mean(abs(predicted - target))),
            "rmse": float(np.sqrt(np.mean((predicted - target) ** 2))), "slope": slope,
            "intercept": None if slope is None else float(predicted.mean() - slope * target.mean()),
            "correlation": float(np.corrcoef(target, predicted)[0, 1]) if target.std() > 0 and predicted.std() > 0 else None,
            "teacher_std": float(target.std()), "prediction_std": float(predicted.std()),
            "variance_ratio": float(predicted.std() / target.std()) if target.std() > 0 else None,
            "mean_tanh_derivative": float(np.mean(8 * (1 - np.clip(predicted / 8, -1, 1) ** 2))),
            "low_derivative_fraction": float(np.mean(8 * (1 - np.clip(predicted / 8, -1, 1) ** 2) < 0.8)),
        }

    for split in ("train", "val"):
        subset = [row for row in rows if row["split"] == split]
        overall = summarize(
            [row["teacher_pka"] for row in subset],
            [row["predicted_pka"] for row in subset],
            [row["group_index"] for row in subset],
        )
        result.append({"split": split, "bin": "all", **overall})
        for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, float("inf"))):
            selected = [row for row in subset if lo <= abs(
                row["teacher_pka"] - baseline[row["group_index"]]
            ) < hi]
            result.append({
                "split": split, "bin": f"{lo:g}-{'inf' if not np.isfinite(hi) else f'{hi:g}'}",
                **summarize(
                    [row["teacher_pka"] for row in selected],
                    [row["predicted_pka"] for row in selected],
                    [row["group_index"] for row in selected],
                ),
            })
    return result


def train(root, experiment, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    out = experiment / arm
    manifest = read(out / "manifest.json")
    assert read(experiment / "tests.json")["passed"]
    assert read(experiment / "tests.json")["code_hashes"] == hashes()
    cfg = manifest["config"]
    from .compilation_cache import configure
    cache = configure(cfg, hashes())
    engine = ExplicitShiftEngine(cfg["shift_bin_weights"], cfg["learning_rate"])
    params = initialize(jax.random.PRNGKey(cfg["seed"]), **cfg["architecture"])
    state = engine.optimizer.init(params)
    rng = np.random.default_rng(cfg["seed"])
    run = out / f"seed-{cfg['seed']}"
    run.mkdir(exist_ok=True)
    history = []
    by_id = {row["complex_id"]: row for row in manifest["records"]}
    records = [by_id[cid] for cid in manifest["train"]]
    loader = BatchLoader(out, manifest, cfg["batch_size"])
    means = np.asarray([read(out / "train_type_means.json")[group] for group in GROUPS])
    provenance = {"manifest_sha256": digest(out / "manifest.json"), "code_hashes": hashes()}
    atomic_json(run / "initial.json", {
        "seed": cfg["seed"], "parameter_count": cfg["parameter_count"],
        "parameter_digest": digest_array_tree(params), "explicit_shift_target": True,
        "compilation_cache": cache, "graph_storage": loader.provenance(),
    })
    for epoch in range(1, cfg["epochs"] + 1):
        order = sample_epoch(records, rng)
        plans = epoch_batches(order, by_id, rng, cfg["batch_size"])
        losses = []
        t0 = time.monotonic()
        loader.set_epoch(epoch)
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            params, state, value = engine.audited_batch_update(params, state, *batch)
            losses.append(value)
            if number % 20 == 0:
                atomic_json(run / "progress.json", {
                    "arm": arm, "epoch": epoch, "epochs": cfg["epochs"], "batches": number,
                    "total_batches": len(plans), "elapsed_seconds": time.monotonic() - t0,
                })
        metrics = evaluate(out, manifest, engine, params, means)
        row = {
            "epoch": epoch, "train_shift_mse": float(np.mean(losses)),
            "validation": metrics, "seconds": time.monotonic() - t0, "updates": len(plans),
        }
        history.append(row)
        atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {
            **provenance, "epoch": epoch, "rng": rng.bit_generator.state,
            "parameter_count": cfg["parameter_count"],
        })
        print(json.dumps({
            "arm": arm, "epoch": epoch, "train_shift_mse": row["train_shift_mse"],
            "val_mae": metrics["graph_query"]["mae"], "seconds": row["seconds"],
        }), flush=True)
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
        "passed": True, "arm": arm, "epochs": cfg["epochs"],
        "parameter_count": cfg["parameter_count"], "finite_gradients": True,
        "initial_parameter_digest": read(run / "initial.json")["parameter_digest"],
        "final_predictions_sha256": digest(run / "final_predictions.parquet"),
        "shift_bins_sha256": digest(run / "shift_bins.json"),
        "test_data_included": False, **provenance,
    })


def digest_array_tree(tree):
    import hashlib
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(leaf)
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def report(experiment):
    results = {}
    bins = {}
    initial = set()
    for arm in ARMS:
        run = experiment / arm / "seed-17"
        verification = read(run / "verification.json")
        assert verification["passed"] and not verification["test_data_included"]
        results[arm] = read(run / "final.json")
        bins[arm] = read(run / "shift_bins.json")
        initial.add(verification["initial_parameter_digest"])
    assert len(initial) == 1
    protocol = read(experiment / "protocol.json")
    lines = [
        "# Backbone GQT trained on explicit pKPDB shifts", "",
        "Both arms predict the signed shift relative to the historical pKPDB `PK_MOD` constants. Absolute pKa is reconstructed only for scoring. The arms have identical initial parameters, data order, structure batches and optimizer settings; only loss weighting differs.", "",
        "| Arm | Validation group-macro MAE | 95% CI |", "|---|---:|---|",
    ]
    for arm in ARMS:
        value = results[arm]["graph_query"]
        lines.append(f"| {arm} | {value['mae']:.4f} | {value['mae_ci95']} |")
    lines += ["", "| pKPDB shift bin | Train sites | Unweighted | Weighted |", "|---|---:|---:|---:|"]
    for name, count, weight in zip(("0-0.5", "0.5-1", "1-2", "2-inf"), protocol["shift_bin_counts"], protocol["weighted_arm_weights"]):
        lines.append(f"| {name} | {count:,} | 1.0000 | {weight:.4f} |")
    lines += ["", "| Arm | Split | Shift bin | Sites | Site MAE | Shift slope | SD ratio |", "|---|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for row in bins[arm]:
            lines.append(
                f"| {arm} | {row['split']} | {row['bin']} | {row['n']:,} | "
                f"{row['mae']:.4f} | {row['slope']:.3f} | {row['variance_ratio']:.3f} |"
            )
    lines += ["", "Epoch 20 was fixed in advance. Validation is reporting-only. No test data were read."]
    (experiment / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(experiment / "verification.json", {
        "passed": True, "matched_initialization": True,
        "report_sha256": digest(experiment / "report.md"), "test_data_included": False,
    })


if __name__ == "__main__":
    import sys
    runtime = Path(os.environ["PKABENCH_RUNTIME"])
    action = sys.argv[1]
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    experiment = runtime / "pretraining/gqt-backbone-5k-pkmod-v1"
    if action == "register":
        register(runtime)
    elif action == "tests":
        register(runtime)
        atomic_json(experiment / "tests.json", {"passed": True, "code_hashes": hashes()})
    elif action == "train":
        train(runtime, experiment, sys.argv[2])
    elif action == "report":
        report(experiment)
    else:
        raise ValueError(action)
