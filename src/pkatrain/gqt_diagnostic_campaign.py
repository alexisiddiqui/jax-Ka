"""Controlled one-seed diagnostics for joint pKPDB/PINDER oGQT training."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pkanet.model import PKPDB_PK_MOD
from pkanet.ogqt import initialize as initialize_ogqt, predict_pkpdb, predict_shift
from pkabench.runtime import atomic_json, digest, require_compute
from .context_augmentation import residue_mask
from .gqt_crop_radius import EPOCHS, learning_rate
from .gqt_multitask_replay import JointEngine, pkpdb_root
from .gqt_paired_pinder import (
    Loader as PairedLoader, _plans, _prefetched, evaluate as evaluate_paired,
    experiment_root as paired_root,
)
from .gqt_regularization import decay_mask
from .gqt_site_weighting import evaluate as evaluate_pkpdb
from .graph_batches import epoch_batches
from .site_graph_data import SiteBatchLoader
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


SEED = 17
ARMS = ("dropout", "context", "data25", "data50")
DROP_RATE = 0.10
MASK_RATE = 0.05
PKPDB_TOLERANCE = 0.05
MIN_DELTA = 0.001


def read(path): return json.loads(Path(path).read_text())
def experiment_root(root): return Path(root) / "training/ogqt-diagnostic-v1"
def large_root(root): return Path(root) / "pretraining/ogqt-large-site-v1"


def code_hashes():
    src = Path(__file__).parents[1]
    paths = (Path(__file__), src / "pkanet/model.py", src / "pkanet/site_model.py",
        src / "pkanet/ogqt.py", src / "pkanet/triton_attention.py",
        src / "pkatrain/context_augmentation.py", src / "pkatrain/gqt_paired_pinder.py",
        src / "pkatrain/gqt_multitask_replay.py", src / "pkatrain/site_graph_data.py")
    return {str(path): digest(path) for path in paths}


def _rank(identifier):
    return hashlib.sha256(("ogqt-diagnostic-v1|" + identifier).encode()).hexdigest()


def nested_subset(records, count):
    """Label-independent, complex-type-stratified nested subset."""
    if count == len(records): return sorted(records, key=lambda row: _rank(row["id"]))
    grouped = defaultdict(list)
    for row in records: grouped[row["ctype"]].append(row)
    for rows in grouped.values(): rows.sort(key=lambda row: _rank(row["id"]))
    exact = {name: count * len(rows) / len(records) for name, rows in grouped.items()}
    quotas = {name: int(np.floor(value)) for name, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(grouped, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]: quotas[name] += 1
    selected = [row for name, rows in grouped.items() for row in rows[:quotas[name]]]
    if len(selected) != count: raise AssertionError((len(selected), count))
    return sorted(selected, key=lambda row: _rank(row["id"]))


def repeat_plans(plans, count):
    if not plans: raise ValueError("empty plan")
    output = []
    while len(output) < count: output.extend(plans)
    return output[:count]


def _mask_one(graph, eligible, identifier, epoch, *, branches=False):
    """Mask residue and site context while preserving every supervised centre."""
    active_nodes = np.asarray(graph["node_mask"][0] if branches else graph["node_mask"], bool)
    queries = np.asarray(graph["query_residue"][0] if branches else graph["query_residue"], int)
    eligible = np.asarray(eligible, bool)
    # Padded queries point to zero; use the explicit supervision mask.
    protected = np.unique(queries[eligible])
    masked = residue_mask(len(active_nodes), protected, seed=SEED, epoch=epoch,
                          complex_id=identifier, probability=MASK_RATE) & active_nodes
    if np.any(masked[protected]): raise AssertionError("masked supervised residue")

    branch_range = range(graph["nodes"].shape[0]) if branches else (None,)
    for branch in branch_range:
        def value(name): return graph[name][branch] if branches else graph[name]
        neighbors = value("neighbors"); affected = masked[:, None] | masked[neighbors]
        value("edge")[affected, :19] = 0.0
        value("edge_mask")[affected] = False; value("switch")[affected] = 0.0
        value("nodes")[masked, 23] = 0.0
        site_residue = value("site_residue"); removed_sites = masked[site_residue]
        site_neighbors = value("site_neighbors")
        site_affected = removed_sites[:, None] | removed_sites[site_neighbors]
        value("site_edge_mask")[site_affected] = False; value("site_switch")[site_affected] = 0.0
        value("site_mask")[removed_sites] = False
        if np.any(~value("site_mask")[value("query_site")[eligible]]):
            raise AssertionError("masked supervised site token")
    return int(masked.sum())


def mask_paired_batch(batch, ids, epoch):
    graphs, eligible = batch[0], batch[2]
    for index, identifier in enumerate(ids):
        one = {name: value[index] for name, value in graphs.items()}
        _mask_one(one, eligible[index], identifier, epoch, branches=True)
    return batch


def mask_pkpdb_batch(batch, ids, epoch):
    graphs, eligible = batch[0], batch[2]
    for index, identifier in enumerate(ids):
        one = {name: value[index] for name, value in graphs.items()}
        _mask_one(one, eligible[index], identifier, epoch, branches=False)
    return batch


def _tree_stats(left, right):
    dot = aa = bb = 0.0
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        dot += jnp.vdot(a, b); aa += jnp.vdot(a, a); bb += jnp.vdot(b, b)
    return jnp.sqrt(aa), jnp.sqrt(bb), dot / jnp.maximum(jnp.sqrt(aa * bb), 1e-30)


class DiagnosticEngine:
    def __init__(self, params, dropout_rate=0.0):
        self.dropout_rate = float(dropout_rate)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0),
            optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32)

        def paired_predictions(p, graphs):
            batch, branches = graphs["nodes"].shape[:2]
            flat = jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs)
            values = jax.vmap(predict_shift, in_axes=(None, 0))(p, flat)
            return values.reshape(batch, branches, -1)

        def paired_training_predictions(p, graphs, key):
            batch, branches = graphs["nodes"].shape[:2]
            flat = jax.tree.map(
                lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs)
            # Repeat each complex key across its AB/free branches. This preserves
            # coordinated dropout while keeping a single outer vmap, which is
            # the batching layout supported by the Triton attention primitive.
            complex_keys = jax.random.split(key, batch)
            keys = jnp.repeat(complex_keys, branches, axis=0)
            values = jax.vmap(lambda graph, one_key: predict_shift(
                p, graph, key=one_key, dropout_rate=self.dropout_rate))(flat, keys)
            return values.reshape(batch, branches, -1)

        def paired_objective(p, graphs, targets, mask, valid, key):
            predicted = paired_training_predictions(p, graphs, key)
            expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
            state = jnp.sum(jnp.square(predicted - expected) * mask[:, None, :], axis=(1, 2)) / jnp.maximum(2 * mask.sum(1), 1)
            pair = jnp.sum(jnp.square((predicted[:, 0] - predicted[:, 1]) -
                (expected[:, 0] - expected[:, 1])) * mask, axis=1) / jnp.maximum(mask.sum(1), 1)
            denominator = jnp.maximum(valid.sum(), 1)
            state = jnp.sum(jnp.where(valid, state, 0.0)) / denominator
            pair = jnp.sum(jnp.where(valid, pair, 0.0)) / denominator
            return state + pair, (state, pair)

        def pk_objective(p, graphs, targets, eligible, valid, key):
            keys = jax.random.split(key, graphs["nodes"].shape[0])
            predicted = jax.vmap(lambda graph, one_key: predict_shift(
                p, graph, key=one_key, dropout_rate=self.dropout_rate))(graphs, keys)
            expected = targets - reference[graphs["query_group"]]
            per = jnp.sum(jnp.square(predicted - expected) * eligible, 1) / jnp.maximum(eligible.sum(1), 1)
            return jnp.sum(jnp.where(valid, per, 0.0)) / jnp.maximum(valid.sum(), 1)

        def apply(p, state, pair_gradient, pk_gradient, rate):
            norm_pair, norm_pk, cosine = _tree_stats(pair_gradient, pk_gradient)
            gradient = jax.tree.map(lambda a, b: a + b, pair_gradient, pk_gradient)
            norm_sum = optax.global_norm(gradient)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            updates = jax.tree.map(lambda value: value * rate, updates)
            return optax.apply_updates(p, updates), state, finite, (norm_pair, norm_pk, norm_sum, cosine)

        self.predictions = jax.jit(paired_predictions)
        self.forward = jax.jit(predict_pkpdb)
        self.pair_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
        self.pk_grad = jax.jit(jax.value_and_grad(pk_objective))
        self.apply = jax.jit(apply)

    def update(self, params, state, pair_batch, pk_batch, rate, key):
        graphs, targets, mask, _, _, _ = pair_batch
        valid = np.ones(len(targets), bool); pair_key, pk_key = jax.random.split(key)
        (pair_total, (state_loss, pair_loss)), pair_gradient = self.pair_grad(
            params, graphs, targets, mask, valid, pair_key)
        pk_loss, pk_gradient = self.pk_grad(params, *pk_batch, pk_key)
        params, state, finite, stats = self.apply(params, state, pair_gradient, pk_gradient, rate)
        values = np.asarray((pair_total, state_loss, pair_loss, pk_loss, *stats), float)
        if not bool(finite) or not np.isfinite(values).all(): raise FloatingPointError("nonfinite diagnostic update")
        names = ("pinder_total", "pinder_state", "pinder_paired", "pkpdb", "pair_grad_norm",
                 "pk_grad_norm", "sum_grad_norm", "gradient_cosine")
        return params, state, dict(zip(names, map(float, values)))


def register(root):
    root = Path(root); out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    pair_manifest = read(paired_root(root) / "manifest.json")
    train = [row for row in pair_manifest["records"] if row["split"] == "train"]
    subsets = {"data25": nested_subset(train, 1250), "data50": nested_subset(train, 2500)}
    if not {r["id"] for r in subsets["data25"]} <= {r["id"] for r in subsets["data50"]}:
        raise AssertionError("subsets are not nested")
    protocol = {"version": "ogqt-diagnostic-v1", "seed": SEED, "arms": list(ARMS),
        "dropout_rate": DROP_RATE, "context_mask_probability": MASK_RATE,
        "data_counts": {name: len(rows) for name, rows in subsets.items()},
        "matched_updates": True, "epochs": EPOCHS,
        "objective": "equal pKPDB state + PINDER state + PINDER paired MSE",
        "test_data_included": False}
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "subsets.json", {name: [row["id"] for row in rows] for name, rows in subsets.items()})
    atomic_json(out / "registration.json", {"passed": True, "code_hashes": code_hashes(),
        "protocol_sha256": digest(out / "protocol.json"), "subsets_sha256": digest(out / "subsets.json"),
        "ctype_counts": {name: dict(Counter(row["ctype"] for row in rows)) for name, rows in subsets.items()},
        "test_data_included": False})


def _parent(root, params, state, *, large=False):
    if large:
        source = large_root(root); verification = read(source / "verification.json")
        epoch = int(verification["selected_epoch"]); run = source / "seed-17"
    else:
        run = Path(root) / "pretraining/gqt-site-weighting-v1/baseline/seed-17"
        verification = read(run / "verification.json"); epoch = int(verification["selected_epoch"])
    if not verification["passed"] or verification["test_data_included"]: raise AssertionError("parent")
    checkpoint = run / "checkpoints" / f"epoch-{epoch:03d}"
    params, _, metadata = load_checkpoint(checkpoint, (params, state))
    return params, {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint / "state.npz"),
        "epoch": epoch, "metadata": metadata}


def train(root, arm, *, smoke=False):
    if arm not in (*ARMS, "large"): raise ValueError(arm)
    root = Path(root); register(root); base = experiment_root(root); registration = read(base / "registration.json")
    if registration["code_hashes"] != code_hashes(): raise AssertionError("code changed after registration")
    pair_base = paired_root(root); pair_manifest = read(pair_base / "manifest.json")
    pk_base = pkpdb_root(root); pk_manifest = read(pk_base / "manifest.json")
    architecture = {"width": 92, "ff": 184} if arm == "large" else pk_manifest["config"]["architecture"]
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **architecture)
    engine = DiagnosticEngine(params, DROP_RATE if arm == "dropout" else 0.0)
    state = engine.optimizer.init(params); params, parent = _parent(root, params, state, large=arm == "large")
    state = engine.optimizer.init(params)
    pair_all = [row for row in pair_manifest["records"] if row["split"] == "train"]
    pair_val = [row for row in pair_manifest["records"] if row["split"] == "val"]
    if arm in ("data25", "data50"):
        selected = set(read(base / "subsets.json")[arm]); pair_train = [row for row in pair_all if row["id"] in selected]
    else: pair_train = pair_all
    pk_by_id = {row["complex_id"]: row for row in pk_manifest["records"]}
    pk_train = [pk_by_id[cid] for cid in pk_manifest["train"]]
    run = base / arm / ("smoke" if smoke else "seed-17"); run.mkdir(parents=True, exist_ok=True)
    pair_loader = PairedLoader(pair_base, pair_manifest)
    pk_loader = SiteBatchLoader(pk_base, pk_manifest, pk_manifest["config"]["batch_size"], "site-orientation")
    provenance = {"arm": arm, "seed": SEED, "parent": parent, "architecture": architecture,
        "parameter_count": sum(x.size for x in jax.tree.leaves(params)), "code_hashes": code_hashes(),
        "protocol_sha256": digest(base / "protocol.json"), "test_data_included": False}
    atomic_json(run / "run.json", provenance)
    pair_rng = np.random.default_rng(SEED); pk_rng = np.random.default_rng(1701)
    full_count = len(_plans(pair_all, np.random.default_rng(SEED), pair_manifest))
    history = []; best = None; threshold = None; started = time.monotonic(); update_index = 0
    epochs = 1 if smoke else EPOCHS
    if not smoke:
        p0 = evaluate_paired(pair_base, pair_manifest, pair_val, engine, params)
        k0 = evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params)
        threshold = float(k0["overall"]["mae"] + PKPDB_TOLERANCE)
        best = {"epoch": 0, "selection": p0["state_mae"] + p0["interface_paired_mae"],
                "pinder": p0, "pkpdb": k0, "pkpdb_mae_limit": threshold}
        atomic_json(run / "best.json", best)
        save_checkpoint(run / "checkpoints/epoch-000", params, state, {**provenance, "epoch": 0})
    for epoch in range(1, epochs + 1):
        pair_plan = repeat_plans(_plans(pair_train, pair_rng, pair_manifest), full_count)
        pk_plan = []
        while len(pk_plan) < full_count:
            pk_plan.extend(epoch_batches(sample_epoch(pk_train, pk_rng), pk_by_id, pk_rng,
                                         pk_manifest["config"]["batch_size"]))
        pk_plan = pk_plan[:full_count]; pk_iterator = iter(pk_loader.iterate(pk_plan)); logs = []
        began = time.monotonic()
        for number, (pair_ids, pair_batch) in enumerate(_prefetched(pair_loader, pair_plan), 1):
            pk_ids, pk_batch = next(pk_iterator)
            if arm == "context":
                pair_batch = mask_paired_batch(pair_batch, pair_ids, epoch)
                pk_batch = mask_pkpdb_batch(pk_batch, pk_ids, epoch)
            rate = learning_rate(epoch, number, full_count, start=1e-4, end=1e-6)
            key = jax.random.fold_in(jax.random.PRNGKey(SEED), update_index); update_index += 1
            params, state, values = engine.update(params, state, pair_batch, pk_batch, rate, key); logs.append(values)
            if number % 20 == 0:
                atomic_json(run / "progress.json", {"arm": arm, "epoch": epoch, "batch": number,
                    "batches": full_count, "elapsed_seconds": time.monotonic() - began, "latest": values})
            if smoke: break
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                "losses": logs[-1], "peak_memory": jax.local_devices()[0].memory_stats(), **provenance})
            pair_loader.close(); pk_loader.close(); return
        pval = evaluate_paired(pair_base, pair_manifest, pair_val, engine, params)
        kval = evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params)
        selection = pval["state_mae"] + pval["interface_paired_mae"]; feasible = kval["overall"]["mae"] <= threshold
        if feasible and selection < best["selection"] - MIN_DELTA:
            best = {"epoch": epoch, "selection": selection, "pinder": pval, "pkpdb": kval,
                    "pkpdb_mae_limit": threshold}; atomic_json(run / "best.json", best)
        row = {"epoch": epoch, "train": {name: float(np.mean([x[name] for x in logs])) for name in logs[0]},
            "pinder_validation": pval, "pkpdb_validation": kval, "selection": selection,
            "pkpdb_feasible": feasible, "best_epoch": best["epoch"], "updates": len(logs),
            "seconds": time.monotonic() - began}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"experiment": "ogqt-diagnostic-v1", "arm": arm, **row}), flush=True)
    pair_loader.close(); pk_loader.close()
    params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final = {"selected_epoch": best["epoch"],
        "pinder": evaluate_paired(pair_base, pair_manifest, pair_val, engine, params, run / "validation-pinder.csv"),
        "pkpdb": evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params,
                                 predictions=run / "validation-pkpdb.csv", bootstrap=True),
        "pkpdb_mae_limit": threshold}
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {"passed": True, "selected_epoch": best["epoch"],
        "epochs_completed": len(history), "wall_seconds": time.monotonic() - started,
        "peak_memory": jax.local_devices()[0].memory_stats(), **provenance})


def pretrain_large(root, *, smoke=False):
    root = Path(root); base = large_root(root); base.mkdir(parents=True, exist_ok=True)
    pk_base = pkpdb_root(root); manifest = read(pk_base / "manifest.json"); architecture = {"width": 92, "ff": 184}
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **architecture)
    from .gqt_site_weighting import WeightedSiteEngine
    engine = WeightedSiteEngine(params, (1.0, 1.0, 1.0, 1.0)); state = engine.optimizer.init(params)
    by_id = {row["complex_id"]: row for row in manifest["records"]}; records = [by_id[cid] for cid in manifest["train"]]
    loader = SiteBatchLoader(pk_base, manifest, manifest["config"]["batch_size"], "site-orientation")
    run = base / ("smoke" if smoke else "seed-17"); run.mkdir(parents=True, exist_ok=True)
    provenance = {"seed": SEED, "architecture": architecture,
        "parameter_count": sum(x.size for x in jax.tree.leaves(params)), "code_hashes": code_hashes(),
        "pkpdb_manifest_sha256": digest(pk_base / "manifest.json"), "test_data_included": False}
    atomic_json(run / "run.json", provenance); rng = np.random.default_rng(SEED); history = []; best = None
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        plans = epoch_batches(sample_epoch(records, rng), by_id, rng, manifest["config"]["batch_size"])
        if smoke: plans = plans[:1]
        losses = []; began = time.monotonic()
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            rate = learning_rate(epoch, number, len(plans), start=1e-3, end=1e-5)
            params, state, loss = engine.update(params, state, batch, rate); losses.append(loss)
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                "loss": float(np.mean(losses)), "peak_memory": jax.local_devices()[0].memory_stats(), **provenance})
            loader.close(); return
        validation = evaluate_pkpdb(pk_base, manifest, "baseline", engine, params)
        score = validation["selection_equal_bin_mae"]
        if best is None or score < best["score"] - MIN_DELTA:
            best = {"epoch": epoch, "score": score, "validation": validation}; atomic_json(run / "best.json", best)
        row = {"epoch": epoch, "train_mse": float(np.mean(losses)), "validation": validation,
               "seconds": time.monotonic() - began, "best_epoch": best["epoch"]}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"experiment": "ogqt-large-site-v1", **row}), flush=True)
    loader.close(); params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final = evaluate_pkpdb(pk_base, manifest, "baseline", engine, params,
                           predictions=run / "validation_predictions.csv", bootstrap=True)
    atomic_json(run / "final.json", final)
    atomic_json(base / "verification.json", {"passed": True, "selected_epoch": best["epoch"],
        "wall_seconds": sum(row["seconds"] for row in history), "test_data_included": False, **provenance})


def extend_baseline(root):
    """Conditionally continue the exact epoch-20 baseline state through epoch 40."""
    root = Path(root); source_run = root / "training/ogqt-multitask-replay-v1/seed-17"
    history = read(source_run / "history.json")
    if len(history) != 20: raise AssertionError("baseline has not completed 20 epochs")
    feasible = [row for row in history if row["pkpdb_feasible"]]
    best_late = min(feasible, key=lambda row: row["selection"]) if feasible else None
    improvement = history[15]["selection"] - min(row["selection"] for row in history[16:])
    train_improvement = history[15]["train"]["total"] - min(row["train"]["total"] for row in history[16:])
    out = experiment_root(root) / "undertraining" / "seed-17"; out.mkdir(parents=True, exist_ok=True)
    gate = {"passed": bool(best_late and best_late["epoch"] >= 17 and improvement >= 0.005
                           and train_improvement >= 0.005),
        "best_feasible_epoch": None if best_late is None else best_late["epoch"],
        "epoch16_to_20_validation_improvement": float(improvement),
        "epoch16_to_20_training_improvement": float(train_improvement),
        "required_improvement_each": 0.005}
    atomic_json(out / "gate.json", gate)
    if not gate["passed"]:
        atomic_json(out / "verification.json", {"passed": True, "status": "not_run_no_endpoint_improvement",
            "gate": gate, "test_data_included": False}); return

    pair_base = paired_root(root); pair_manifest = read(pair_base / "manifest.json")
    pk_base = pkpdb_root(root); pk_manifest = read(pk_base / "manifest.json")
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **pk_manifest["config"]["architecture"])
    engine = JointEngine(params); state = engine.optimizer.init(params)
    checkpoint = source_run / "checkpoints/epoch-020"; params, state, metadata = load_checkpoint(checkpoint, (params, state))
    pair_split = {name: [row for row in pair_manifest["records"] if row["split"] == name]
                  for name in ("train", "val")}
    pk_by_id = {row["complex_id"]: row for row in pk_manifest["records"]}; pk_train = [pk_by_id[cid] for cid in pk_manifest["train"]]
    pair_rng = np.random.default_rng(SEED); pk_rng = np.random.default_rng(1701)
    from .gqt_multitask_replay import _pkpdb_plans
    # Reconstruct the exact RNG state after the first 20 epochs.
    for _ in range(20):
        plan = _plans(pair_split["train"], pair_rng, pair_manifest)
        _pkpdb_plans(pk_train, pk_by_id, pk_rng, pk_manifest["config"]["batch_size"], len(plan))
    pair_loader = PairedLoader(pair_base, pair_manifest)
    pk_loader = SiteBatchLoader(pk_base, pk_manifest, pk_manifest["config"]["batch_size"], "site-orientation")
    threshold = read(source_run / "best.json")["pkpdb_mae_limit"]
    best = {"epoch": 20, "selection": history[-1]["selection"],
            "pinder": history[-1]["pinder_validation"], "pkpdb": history[-1]["pkpdb_validation"]}
    continuation = []
    for epoch in range(21, 41):
        pair_plan = _plans(pair_split["train"], pair_rng, pair_manifest)
        pk_plan = _pkpdb_plans(pk_train, pk_by_id, pk_rng, pk_manifest["config"]["batch_size"], len(pair_plan))
        pk_iterator = iter(pk_loader.iterate(pk_plan)); losses = []; began = time.monotonic()
        for number, (_, pair_batch) in enumerate(_prefetched(pair_loader, pair_plan), 1):
            _, pk_batch = next(pk_iterator)
            fraction = ((epoch-21) + (number-1)/max(len(pair_plan)-1, 1)) / 19
            # Continue smoothly from the epoch-20 endpoint; do not restart the
            # learning rate and confound the undertraining diagnostic.
            rate = float(1e-7 + .5*(1e-6-1e-7)*(1+np.cos(np.pi*fraction)))
            params, state, loss = engine.update(params, state, pair_batch, pk_batch, rate); losses.append(loss)
            if number % 20 == 0:
                atomic_json(out / "progress.json", {"epoch": epoch, "batch": number, "batches": len(pair_plan),
                    "elapsed_seconds": time.monotonic()-began, "loss": loss})
        pval = evaluate_paired(pair_base, pair_manifest, pair_split["val"], engine, params)
        kval = evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params)
        selection = pval["state_mae"] + pval["interface_paired_mae"]
        if kval["overall"]["mae"] <= threshold and selection < best["selection"] - MIN_DELTA:
            best = {"epoch": epoch, "selection": selection, "pinder": pval, "pkpdb": kval}
            atomic_json(out / "best.json", best)
        train_metrics = {name: float(np.mean([item[name] for item in losses])) for name in losses[0]}
        row = {"epoch": epoch, "train": train_metrics, "pinder_validation": pval,
            "pkpdb_validation": kval, "selection": selection, "pkpdb_feasible": kval["overall"]["mae"] <= threshold,
            "best_epoch": best["epoch"], "seconds": time.monotonic()-began}
        continuation.append(row); atomic_json(out / "history.json", continuation)
        save_checkpoint(out / "checkpoints" / f"epoch-{epoch:03d}", params, state,
                        {"epoch": epoch, "parent": str(checkpoint), "parameter_count": sum(x.size for x in jax.tree.leaves(params))})
        print(json.dumps({"experiment": "ogqt-undertraining", **row}), flush=True)
    pair_loader.close(); pk_loader.close()
    continued_training_improvement = history[-1]["train"]["total"] - min(
        row["train"]["total"] for row in continuation)
    continued_validation_improvement = history[-1]["selection"] - min(
        row["selection"] for row in continuation)
    atomic_json(out / "verification.json", {"passed": True, "status": "completed", "gate": gate,
        "selected_epoch": best["epoch"], "epochs_completed": 20,
        "continued_training_improvement": float(continued_training_improvement),
        "continued_validation_improvement": float(continued_validation_improvement),
        "undertraining_supported": bool(continued_training_improvement > 0
                                        and continued_validation_improvement > 0),
        "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=True, allow_comp1400=True)
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "smoke": train(root, sys.argv[2], smoke=True)
    elif action == "train": train(root, sys.argv[2])
    elif action == "pretrain-large-smoke": pretrain_large(root, smoke=True)
    elif action == "pretrain-large": pretrain_large(root)
    elif action == "extend-baseline": extend_baseline(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
