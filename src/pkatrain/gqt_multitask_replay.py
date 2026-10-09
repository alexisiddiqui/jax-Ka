"""Joint pKPDB pretraining and paired-complex training for backbone oGQT.

Each optimizer update combines one pKPDB single-state gradient with one PINDER
state-plus-binding-shift gradient.  The run starts from the selected pKPDB
checkpoint and retains it as an eligible epoch-zero baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pkanet.model import PKPDB_PK_MOD
from pkanet.ogqt import initialize as initialize_ogqt
from pkanet.ogqt import predict_pkpdb, predict_shift
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_crop_radius import EPOCHS, learning_rate
from .gqt_paired_pinder import (
    Loader as PairedLoader,
    _plans as paired_plans,
    _prefetched as paired_prefetched,
    evaluate as evaluate_paired,
    experiment_root as paired_root,
    pretrained_root,
)
from .gqt_regularization import decay_mask
from .gqt_site_weighting import evaluate as evaluate_pkpdb
from .graph_batches import epoch_batches
from .site_graph_data import SiteBatchLoader
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


SEED = 17
PKPDB_TOLERANCE = 0.05
MIN_DELTA = 0.001


def read(path):
    return json.loads(Path(path).read_text())


def experiment_root(root):
    return Path(root) / "training/ogqt-multitask-replay-v1"


def pkpdb_root(root):
    return Path(root) / "pretraining/gqt-site-weighting-v1/baseline"


def code_hashes():
    src = Path(__file__).parents[1]
    paths = (
        Path(__file__), src / "pkanet/model.py", src / "pkanet/site_model.py",
        src / "pkanet/ogqt.py", src / "pkanet/triton_attention.py",
        src / "pkatrain/gqt_paired_pinder.py", src / "pkatrain/gqt_site_weighting.py",
        src / "pkatrain/site_graph_data.py", src / "pkatrain/graph_batches.py",
        src / "pkatrain/trainer.py",
    )
    return {str(path): digest(path) for path in paths}


def register(root):
    root = Path(root); out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    pk_manifest = pkpdb_root(root) / "manifest.json"
    pair_manifest = paired_root(root) / "manifest.json"
    source_run = pretrained_root(root)
    verification = read(source_run / "verification.json")
    if not verification["passed"] or verification["test_data_included"]:
        raise AssertionError("invalid pKPDB parent checkpoint")
    source_epoch = int(verification["selected_epoch"])
    checkpoint = source_run / "checkpoints" / f"epoch-{source_epoch:03d}"
    protocol = {
        "version": "ogqt-multitask-replay-v1",
        "model": "backbone-only oGQT, 20 A residue and site cutoffs",
        "initialization": {"checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint / "state.npz"), "selected_epoch": source_epoch},
        "training_step": "one pKPDB batch plus one PINDER batch; sum their gradients before one AdamW update",
        "objective": "pKPDB state-shift MSE + PINDER AB/free state-shift MSE + PINDER binding-shift MSE",
        "objective_coefficients": {"pkpdb_state": 1.0, "pinder_state": 1.0, "pinder_paired": 1.0},
        "sampling": "one independently sampled pKPDB batch per PINDER batch; full PINDER epoch",
        "optimizer": "AdamW, weight decay 1e-4, global gradient clip 1 after gradient summation",
        "schedule": "1e-4 through epoch 10, cosine to 1e-6 by epoch 20",
        "augmentation": "none; canonical backbone graphs and existing uncertainty masks",
        "checkpoint_selection": (
            "minimum PINDER validation state MAE + interface paired MAE among checkpoints whose "
            f"pKPDB group-macro MAE is no more than epoch zero + {PKPDB_TOLERANCE}"
        ),
        "epoch_zero_eligible": True,
        "test_data_included": False,
        "sources": {"pkpdb_manifest": str(pk_manifest), "pkpdb_manifest_sha256": digest(pk_manifest),
            "pinder_manifest": str(pair_manifest), "pinder_manifest_sha256": digest(pair_manifest)},
    }
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "registration.json", {"passed": True, "code_hashes": code_hashes(),
        "protocol_sha256": digest(out / "protocol.json"), "test_data_included": False})


class JointEngine:
    """One shared parameter tree and one optimizer for both data sources."""
    def __init__(self, params):
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0),
            optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32)

        def paired_predictions(p, graphs):
            batch, branches = graphs["nodes"].shape[:2]
            flat = jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs)
            result = jax.vmap(predict_shift, in_axes=(None, 0))(p, flat)
            return result.reshape(batch, branches, -1)

        def paired_objective(p, graphs, targets, mask, wb, wi, valid):
            del wb, wi
            predicted = paired_predictions(p, graphs)
            expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
            state_error = jnp.square(predicted - expected) * mask[:, None, :]
            pair_error = jnp.square((predicted[:, 0] - predicted[:, 1]) -
                                    (expected[:, 0] - expected[:, 1])) * mask
            state_loss = jnp.sum(state_error, axis=(1, 2)) / jnp.maximum(2 * jnp.sum(mask, axis=1), 1)
            pair_loss = jnp.sum(pair_error, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
            denominator = jnp.maximum(jnp.sum(valid), 1)
            state_loss = jnp.sum(jnp.where(valid, state_loss, 0.0)) / denominator
            pair_loss = jnp.sum(jnp.where(valid, pair_loss, 0.0)) / denominator
            return state_loss + pair_loss, (state_loss, pair_loss)

        def pkpdb_objective(p, graphs, targets, eligible, valid):
            expected = targets - reference[graphs["query_group"]]
            predicted = jax.vmap(predict_shift, in_axes=(None, 0))(p, graphs)
            squared = jnp.square(predicted - expected) * eligible
            per_structure = jnp.sum(squared, axis=1) / jnp.maximum(jnp.sum(eligible, axis=1), 1)
            return jnp.sum(jnp.where(valid, per_structure, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        def apply(p, state, pair_gradient, pk_gradient, rate):
            gradient = jax.tree.map(lambda a, b: a + b, pair_gradient, pk_gradient)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            updates = jax.tree.map(lambda value: value * rate, updates)
            return optax.apply_updates(p, updates), state, finite

        self.predictions = jax.jit(paired_predictions)
        self.forward = jax.jit(predict_pkpdb)
        self.paired_value_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
        self.pkpdb_value_grad = jax.jit(jax.value_and_grad(pkpdb_objective))
        self.apply = jax.jit(apply)

    def update(self, params, state, pair_batch, pk_batch, rate):
        graphs, targets, mask, wb, wi, _ = pair_batch
        valid = np.ones(len(targets), bool)
        (pair_total, (state_loss, pair_loss)), pair_gradient = self.paired_value_grad(
            params, graphs, targets, mask, wb, wi, valid)
        pk_loss, pk_gradient = self.pkpdb_value_grad(params, *pk_batch)
        params, state, finite = self.apply(params, state, pair_gradient, pk_gradient, rate)
        values = np.asarray((pair_total, state_loss, pair_loss, pk_loss), float)
        if not bool(finite) or not np.all(np.isfinite(values)):
            raise FloatingPointError("nonfinite joint oGQT update")
        return params, state, {"total": float(pair_total + pk_loss), "pkpdb": float(pk_loss),
            "pinder_state": float(state_loss), "pinder_paired": float(pair_loss)}


def _pkpdb_plans(records, by_id, rng, batch_size, count):
    plans = []
    while len(plans) < count:
        plans.extend(epoch_batches(sample_epoch(records, rng), by_id, rng, batch_size))
    return plans[:count]


def _digest_plans(pair, pk):
    raw = json.dumps({"pinder": pair, "pkpdb": pk}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_parent(root, params, state):
    source_run = pretrained_root(root); verification = read(source_run / "verification.json")
    if not verification["passed"] or verification["test_data_included"]:
        raise AssertionError("invalid pKPDB parent")
    epoch = int(verification["selected_epoch"])
    checkpoint = source_run / "checkpoints" / f"epoch-{epoch:03d}"
    params, _, metadata = load_checkpoint(checkpoint, (params, state))
    count = sum(value.size for value in jax.tree.leaves(params))
    if int(metadata["parameter_count"]) != count:
        raise AssertionError("parent parameter count mismatch")
    return params, {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint / "state.npz"),
        "selected_epoch": epoch, "verification_sha256": digest(source_run / "verification.json")}


def train(root, *, smoke=False):
    root = Path(root); register(root); out = experiment_root(root)
    registration = read(out / "registration.json")
    if not registration["passed"] or registration["code_hashes"] != code_hashes():
        raise AssertionError("registration/code mismatch")
    pair_base = paired_root(root); pair_manifest = read(pair_base / "manifest.json")
    pk_base = pkpdb_root(root); pk_manifest = read(pk_base / "manifest.json")
    architecture = pk_manifest["config"]["architecture"]
    if {"width": pair_manifest["architecture"]["width"], "ff": pair_manifest["architecture"]["ff"]} != architecture:
        raise AssertionError("architecture mismatch")
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **architecture)
    engine = JointEngine(params); state = engine.optimizer.init(params)
    params, parent = _load_parent(root, params, state); state = engine.optimizer.init(params)
    run = out / ("smoke" if smoke else "seed-17"); run.mkdir(parents=True, exist_ok=True)
    pair_split = {name: [row for row in pair_manifest["records"] if row["split"] == name]
                  for name in ("train", "val")}
    pk_by_id = {row["complex_id"]: row for row in pk_manifest["records"]}
    pk_train = [pk_by_id[cid] for cid in pk_manifest["train"]]
    pair_loader = PairedLoader(pair_base, pair_manifest)
    pk_loader = SiteBatchLoader(pk_base, pk_manifest, pk_manifest["config"]["batch_size"], "site-orientation")
    provenance = {"seed": SEED, "parent": parent, "protocol_sha256": digest(out / "protocol.json"),
        "code_hashes": code_hashes(), "pinder_manifest_sha256": digest(pair_base / "manifest.json"),
        "pkpdb_manifest_sha256": digest(pk_base / "manifest.json"),
        "loaders": {"pinder": str(pair_loader.store.path), "pkpdb": pk_loader.provenance()},
        "parameter_count": sum(value.size for value in jax.tree.leaves(params)), "test_data_included": False}
    atomic_json(run / "run.json", provenance)
    pair_rng = np.random.default_rng(SEED); pk_rng = np.random.default_rng(1701)

    if smoke:
        pair_plan = paired_plans(pair_split["train"], pair_rng, pair_manifest)[:1]
        pk_plan = _pkpdb_plans(pk_train, pk_by_id, pk_rng, pk_manifest["config"]["batch_size"], 1)
        _, pair_batch = next(iter(paired_prefetched(pair_loader, pair_plan)))
        _, pk_batch = next(iter(pk_loader.iterate(pk_plan)))
        params, state, losses = engine.update(params, state, pair_batch, pk_batch, 1e-4)
        atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
            "losses": losses, "batch_plan_digest": _digest_plans(pair_plan, pk_plan),
            "device_memory": jax.local_devices()[0].memory_stats(), **provenance})
        pair_loader.close(); pk_loader.close(); return

    started = time.monotonic(); history = []
    initial_pair = evaluate_paired(pair_base, pair_manifest, pair_split["val"], engine, params,
                                   run / "epoch-000-pinder.csv")
    initial_pk = evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params,
                               predictions=run / "epoch-000-pkpdb.csv")
    threshold = float(initial_pk["overall"]["mae"] + PKPDB_TOLERANCE)
    best = {"epoch": 0, "selection": initial_pair["state_mae"] + initial_pair["interface_paired_mae"],
        "pinder": initial_pair, "pkpdb": initial_pk, "pkpdb_mae_limit": threshold}
    atomic_json(run / "best.json", best)
    save_checkpoint(run / "checkpoints/epoch-000", params, state, {**provenance, "epoch": 0})

    for epoch in range(1, EPOCHS + 1):
        pair_plan = paired_plans(pair_split["train"], pair_rng, pair_manifest)
        pk_plan = _pkpdb_plans(pk_train, pk_by_id, pk_rng, pk_manifest["config"]["batch_size"], len(pair_plan))
        plan_digest = _digest_plans(pair_plan, pk_plan); began = time.monotonic(); losses = []
        pk_iterator = iter(pk_loader.iterate(pk_plan))
        for number, (_, pair_batch) in enumerate(paired_prefetched(pair_loader, pair_plan), 1):
            _, pk_batch = next(pk_iterator)
            rate = learning_rate(epoch, number, len(pair_plan), start=1e-4, end=1e-6)
            params, state, values = engine.update(params, state, pair_batch, pk_batch, rate)
            losses.append(values)
        try:
            next(pk_iterator)
            raise AssertionError("unconsumed pKPDB batch")
        except StopIteration:
            pass
        pair_validation = evaluate_paired(pair_base, pair_manifest, pair_split["val"], engine, params)
        pk_validation = evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params)
        selection = pair_validation["state_mae"] + pair_validation["interface_paired_mae"]
        feasible = pk_validation["overall"]["mae"] <= threshold
        if feasible and selection < best["selection"] - MIN_DELTA:
            best = {"epoch": epoch, "selection": selection, "pinder": pair_validation,
                "pkpdb": pk_validation, "pkpdb_mae_limit": threshold}
            atomic_json(run / "best.json", best)
        row = {"epoch": epoch,
            "train": {key: float(np.mean([item[key] for item in losses])) for key in losses[0]},
            "pinder_validation": pair_validation, "pkpdb_validation": pk_validation,
            "selection": selection, "pkpdb_feasible": feasible, "best_epoch": best["epoch"],
            "batch_plan_digest": plan_digest, "updates": len(pair_plan), "seconds": time.monotonic() - began}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state,
                        {**provenance, "epoch": epoch})
        print(json.dumps({"experiment": "ogqt-multitask-replay-v1", **row}), flush=True)

    pair_loader.close(); pk_loader.close()
    params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final_pair = evaluate_paired(pair_base, pair_manifest, pair_split["val"], engine, params,
                                 run / "validation-pinder.csv")
    final_pk = evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params,
                              predictions=run / "validation-pkpdb.csv", bootstrap=True)
    final = {"selected_epoch": best["epoch"], "pinder": final_pair, "pkpdb": final_pk,
        "pkpdb_mae_limit": threshold}
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {"passed": True, "epochs_completed": EPOCHS,
        "selected_epoch": best["epoch"], "wall_seconds": time.monotonic() - started,
        "pinder_predictions_sha256": digest(run / "validation-pinder.csv"),
        "pkpdb_predictions_sha256": digest(run / "validation-pkpdb.csv"),
        "device_memory": jax.local_devices()[0].memory_stats(), **provenance})


def report(root):
    root = Path(root); run = experiment_root(root) / "seed-17"
    verification = read(run / "verification.json"); final = read(run / "final.json")
    if not verification["passed"] or verification["test_data_included"]:
        raise AssertionError("invalid joint run")
    if digest(run / "validation-pinder.csv") != verification["pinder_predictions_sha256"]:
        raise AssertionError("PINDER prediction hash")
    if digest(run / "validation-pkpdb.csv") != verification["pkpdb_predictions_sha256"]:
        raise AssertionError("pKPDB prediction hash")
    pair = final["pinder"]; pk = final["pkpdb"]
    lines = ["# Joint pKPDB + paired-complex oGQT training", "",
        "The shared backbone-only oGQT was initialized from the selected pKPDB checkpoint. Each optimizer update summed gradients from one pKPDB structure batch and one PINDER paired batch before a single AdamW update. No random masking or test data were used.", "",
        "| Selected epoch | PINDER state MAE | PINDER paired MAE | Interface paired MAE | pKPDB group-macro MAE | pKPDB limit |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {final['selected_epoch']} | {pair['state_mae']:.4f} | {pair['paired_mae']:.4f} | "
        f"{pair['interface_paired_mae']:.4f} | {pk['overall']['mae']:.4f} | {final['pkpdb_mae_limit']:.4f} |", "",
        "Checkpoint selection minimized PINDER state MAE plus interface paired MAE while requiring pKPDB group-macro MAE to remain within 0.05 of epoch zero.", ""]
    destination = experiment_root(root) / "report.md"; destination.write_text("\n".join(lines))
    atomic_json(experiment_root(root) / "report-verification.json", {"passed": True,
        "report_sha256": digest(destination), "selected_epoch": final["selected_epoch"],
        "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    gpu_benchmark=action in ("smoke", "train"), allow_comp1400=True)
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "smoke": train(root, smoke=True)
    elif action == "train": train(root)
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
