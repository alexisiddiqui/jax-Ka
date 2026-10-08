"""One-seed controlled experiment for explicit titratable-site communication."""
from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxpropka.parameters import GROUPS
from pkanet.model import PKPDB_PK_MOD
from pkanet.site_model import initialize_site, predict_site_pkpdb_indexed, predict_site_shift_indexed, predict_site_with_trace
from pkabench.frozen_score import aggregate, measures, write_csv
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_crop_radius import EPOCHS, MIN_DELTA, PATIENCE, learning_rate
from .gqt_regularization import decay_mask
from .graph_batches import epoch_batches
from .graph_pkmod_compare import digest_array_tree
from .records import read
from .site_graph_data import SiteBatchLoader, experiment_root, pad_site, sidecar
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


ARMS = ("site-distance", "site-orientation", "no-arg")
SEED = 17
BIN_NAMES = ("<0.5", "0.5–1", "1–2", "≥2")


def source(root): return root / "pretraining/gqt-neighbor-cutoff-v1-triton/backbone/cutoff-20A"
def baseline(root): return source(root) / "seed-17"


def code_hashes():
    here = Path(__file__); src = here.parents[1]
    paths = (here, src / "pkanet/site_model.py", src / "pkanet/model.py", src / "pkanet/triton_attention.py",
             src / "pkatrain/site_graph_data.py", src / "pkatrain/graph_batches.py",
             src / "pkatrain/gqt_neighbor_cutoff.py", src / "pkatrain/trainer.py")
    return {str(path): digest(path) for path in paths}


def parameter_count(params): return int(sum(value.size for value in jax.tree.leaves(params)))


def register(root):
    out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    parent = read(source(root) / "manifest.json"); site_manifest = read(sidecar(root) / "manifest.json")
    preparation = read(sidecar(root) / "preparation.json")
    if not preparation["passed"] or len(site_manifest["records"]) != len(parent["records"]): raise AssertionError("site preparation")
    site_mmap = read(sidecar(root) / "mmap-v1/verification.json")
    if not site_mmap["passed"]: raise AssertionError("site mmap")
    params = initialize_site(jax.random.PRNGKey(SEED), **parent["config"]["architecture"])
    count = parameter_count(params)
    protocol = {"arms": ["current-gqt", *ARMS], "seed": SEED, "cutoff_A": 20.0,
        "candidate_sites": list(GROUPS), "arg_supervision": False,
        "site_edge_features": ["distance_rbf", "direction_in_source_frame", "reverse_direction_in_target_frame",
            "relative_orientation_3x3", "same_chain", "same_residue", "signed_sequence_separation", "directed_site_type_pair"],
        "optimizer": "AdamW, weight decay 1e-4", "schedule": "1e-3 through epoch 10, cosine to 1e-5",
        "selection": f"minimum validation group-macro MAE; min_delta {MIN_DELTA}; patience {PATIENCE}",
        "batch_size": 8, "dtype": "float32", "parameter_count_site_models": count,
        "baseline": str(baseline(root)), "baseline_verification_sha256": digest(baseline(root) / "verification.json"),
        "test_data_included": False}
    atomic_json(out / "protocol.json", protocol)
    for arm in ARMS:
        destination = out / arm; destination.mkdir(exist_ok=True)
        manifest = copy.deepcopy(parent)
        manifest["parent"] = {"path": str(source(root)), "manifest_sha256": digest(source(root) / "manifest.json")}
        manifest["site_source"] = str(sidecar(root)); manifest["site_manifest_sha256"] = digest(sidecar(root) / "manifest.json")
        manifest["site_protocol_sha256"] = digest(out / "protocol.json")
        manifest["config"].update(seed=SEED, epochs=EPOCHS, parameter_count=count, model_arm=arm,
            architecture={**parent["config"]["architecture"]}, edge_cutoff_A=20.0,
            objective="explicit signed pKPDB PK_MOD shift MSE on labelled sites; ARG context unsupervised",
            site_tokens=True, site_orientation=(arm != "site-distance"), arg_context=(arm != "no-arg"))
        data = destination / "data"
        if not data.exists(): data.symlink_to((source(root) / "data").resolve(), target_is_directory=True)
        for name in ("preparation.json", "train_type_means.json"):
            if not (destination / name).exists(): shutil.copy2(source(root) / name, destination / name)
        atomic_json(destination / "manifest.json", manifest)
    atomic_json(out / "registration.json", {"passed": True, "parameter_count": count,
                "protocol_sha256": digest(out / "protocol.json"), "code_hashes": code_hashes()})


class SiteEngine:
    def __init__(self, params):
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0),
            optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        self.forward = jax.jit(predict_site_pkpdb_indexed)
        baseline_values = jnp.asarray(PKPDB_PK_MOD, jnp.float32)
        def one_loss(p, graph, target, eligible):
            expected = target - baseline_values[graph["query_group"]]
            error = jnp.where(eligible, predict_site_shift_indexed(p, graph) - expected, 0.0)
            return jnp.sum(error * error) / jnp.maximum(jnp.sum(eligible), 1)
        def batch_loss(p, graphs, targets, eligible, valid):
            losses = jax.vmap(one_loss, in_axes=(None, 0, 0, 0))(p, graphs, targets, eligible)
            return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
        def step(p, state, graphs, targets, eligible, valid, rate):
            loss, gradient = jax.value_and_grad(batch_loss)(p, graphs, targets, eligible, valid)
            finite = jnp.isfinite(loss) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            updates = jax.tree.map(lambda value: value * rate, updates)
            return optax.apply_updates(p, updates), state, loss, finite
        self.step = jax.jit(step)
    def update(self, params, state, batch, rate):
        params, state, loss, finite = self.step(params, state, *batch, rate)
        if not bool(finite): raise FloatingPointError("Nonfinite site-token update")
        return params, state, float(loss)


def evaluate(out, manifest, arm, engine, params, *, predictions=None):
    loader = SiteBatchLoader(out, manifest, manifest["config"]["batch_size"], arm); rows = []; scores = []
    for record in manifest["records"]:
        if record["split"] != "val": continue
        graph, target, eligible = loader.one(record["complex_id"])
        predicted = np.asarray(engine.forward(params, graph))[eligible]; target = target[eligible]
        groups = graph["query_group"][eligible]; reference = np.asarray(PKPDB_PK_MOD)[groups]
        scores.append({"component_id": record["component_id"], "complex_id": record["complex_id"], "n": len(target),
                       **measures(target - reference, predicted - reference)})
        if predictions is not None:
            for key, observed, value, ref in zip(record["keys"], target, predicted, reference):
                rows.append(dict(zip(("complex_id", "chain", "resnum", "icode", "group"), key),
                    component_id=record["component_id"], teacher_pka=float(observed), predicted_pka=float(value),
                    teacher_shift=float(observed - ref), predicted_shift=float(value - ref)))
    loader.close(); metric = aggregate(scores, replicates=2000)[0]
    if predictions is not None: write_csv(predictions, rows)
    return metric


def rotation_gate(root):
    registration = read(experiment_root(root) / "registration.json")
    if registration["code_hashes"] != code_hashes(): raise AssertionError("registration/code mismatch")
    out = experiment_root(root) / "site-orientation"; manifest = read(out / "manifest.json")
    record = manifest["records"][0]; loader = SiteBatchLoader(out, manifest, 1, "site-orientation")
    graph, _, _ = loader.one(record["complex_id"])
    rotated_raw = {name: value for name, value in np.load(experiment_root(root) / "rotation-probe.npz", allow_pickle=False).items()}
    from .graph_data import bucket
    rotated_site = pad_site(rotated_raw, read(sidecar(root) / "manifest.json")["capacities"][bucket(record)], graph["query_group"].shape[0])
    rotated = {key: value.copy() for key, value in graph.items()}; rotated.update(rotated_site)
    params = initialize_site(jax.random.PRNGKey(SEED), **manifest["config"]["architecture"])
    original_trace = predict_site_with_trace(params, graph); rotated_trace = predict_site_with_trace(params, rotated)
    prediction_difference = float(np.max(np.abs(np.asarray(original_trace["predicted_shift"] - rotated_trace["predicted_shift"]))))
    attention_difference = float(np.max(np.abs(np.asarray(original_trace["site_attention"]["weights"] - rotated_trace["site_attention"]["weights"]))))
    if prediction_difference > 2e-4 or attention_difference > 2e-5:
        raise AssertionError((prediction_difference, attention_difference))
    loader.close()
    atomic_json(experiment_root(root) / "rotation-gate.json", {"passed": True,
        "prediction_max_abs_difference": prediction_difference,
        "site_attention_max_abs_difference": attention_difference,
        "rigid_transform": "random proper rotation plus translation", "code_hashes": code_hashes()})


def train(root, arm, *, smoke=False):
    if arm not in ARMS: raise ValueError(arm)
    base = experiment_root(root); registration = read(base / "registration.json")
    if registration["code_hashes"] != code_hashes(): raise AssertionError("registration/code mismatch")
    out = base / arm; stored = out / "manifest.json"; manifest = read(stored); cfg = manifest["config"]
    params = initialize_site(jax.random.PRNGKey(SEED), **cfg["architecture"])
    if parameter_count(params) != cfg["parameter_count"]: raise AssertionError("parameter count")
    engine = SiteEngine(params); state = engine.optimizer.init(params); rng = np.random.default_rng(SEED)
    run = out / ("smoke-seed-17" if smoke else "seed-17"); run.mkdir(parents=True, exist_ok=True)
    by_id = {row["complex_id"]: row for row in manifest["records"]}; records = [by_id[cid] for cid in manifest["train"]]
    loader = SiteBatchLoader(out, manifest, cfg["batch_size"], arm)
    provenance = {"arm": arm, "seed": SEED, "manifest_sha256": digest(stored),
                  "code_hashes": code_hashes(), "parameter_count": parameter_count(params)}
    atomic_json(run / "run.json", {**provenance, "config": cfg, "loader": loader.provenance(),
                "initial_parameter_digest": digest_array_tree(params)})
    best = None; stalled = 0; history = []; started = time.monotonic()
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        plans = epoch_batches(sample_epoch(records, rng), by_id, rng, cfg["batch_size"])
        if smoke: plans = plans[:1]
        losses = []; began = time.monotonic()
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            rate = learning_rate(epoch, number, len(plans)); params, state, loss = engine.update(params, state, batch, rate)
            losses.append(loss)
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                "loss": float(np.mean(losses)), "device_memory": jax.local_devices()[0].memory_stats(), **provenance})
            loader.close(); return
        metric = evaluate(out, manifest, arm, engine, params); mae = float(metric["mae"])
        if best is None or mae < best["mae"] - MIN_DELTA:
            best = {"epoch": epoch, "mae": mae}; stalled = 0
            atomic_json(run / "best.json", best)
        else: stalled += 1
        row = {"epoch": epoch, "train_shift_mse": float(np.mean(losses)), "validation": metric,
               "seconds": time.monotonic() - began, "best_epoch": best["epoch"], "stalled_epochs": stalled}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"arm": arm, "epoch": epoch, "train_shift_mse": row["train_shift_mse"],
                          "val_mae": mae, "best_epoch": best["epoch"], "seconds": row["seconds"]}), flush=True)
        if stalled >= PATIENCE: break
    loader.close(); params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final = evaluate(out, manifest, arm, engine, params, predictions=run / "validation_predictions.csv")
    atomic_json(run / "final.json", {"graph_query": final})
    atomic_json(run / "verification.json", {"passed": True, "epochs_completed": len(history),
        "selected_epoch": best["epoch"], "wall_seconds": time.monotonic() - started,
        "device_memory": jax.local_devices()[0].memory_stats(),
        "predictions_sha256": digest(run / "validation_predictions.csv"), "test_data_included": False, **provenance})


def read_predictions(path):
    with Path(path).open(newline="") as stream: rows = list(csv.DictReader(stream))
    for row in rows:
        row["teacher_shift"] = float(row.get("teacher_shift") or (float(row["teacher_pka"]) - float(PKPDB_PK_MOD[GROUPS.index(row["group"])])))
        row["predicted_shift"] = float(row.get("predicted_shift") or (float(row["predicted_pka"]) - float(PKPDB_PK_MOD[GROUPS.index(row["group"])])))
    return rows


def macro(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[row["complex_id"]].append(row)
    values = []
    for cid, subset in grouped.items():
        target = np.asarray([r["teacher_shift"] for r in subset]); predicted = np.asarray([r["predicted_shift"] for r in subset])
        values.append({"complex_id": cid, "component_id": subset[0]["component_id"], "n": len(subset), **measures(target, predicted)})
    return aggregate(values, replicates=2000)[0]


def summarize_predictions(path):
    rows = read_predictions(path); bins = []
    for low, high, name in ((0, .5, BIN_NAMES[0]), (.5, 1, BIN_NAMES[1]), (1, 2, BIN_NAMES[2]), (2, np.inf, BIN_NAMES[3])):
        subset = [row for row in rows if low <= abs(row["teacher_shift"]) < high]
        bins.append({"bin": name, "sites": len(subset), "mae": macro(subset)["mae"]})
    return {"overall": macro(rows)["mae"], "bins": bins}


def report(root):
    base = experiment_root(root); rows = []
    control_verification = read(baseline(root) / "verification.json")
    if not control_verification["passed"]: raise AssertionError("baseline")
    control = summarize_predictions(baseline(root) / "validation_predictions.csv")
    rows.append({"arm": "current-gqt", "parameter_count": 49709, "selected_epoch": control_verification["selected_epoch"],
                 "wall_seconds": control_verification["wall_seconds"], "peak_vram": control_verification["device_memory"]["peak_bytes_in_use"], **control})
    for arm in ARMS:
        run = base / arm / "seed-17"; verification = read(run / "verification.json")
        if not verification["passed"] or digest(run / "validation_predictions.csv") != verification["predictions_sha256"]: raise AssertionError(run)
        rows.append({"arm": arm, "parameter_count": verification["parameter_count"],
            "selected_epoch": verification["selected_epoch"], "wall_seconds": verification["wall_seconds"],
            "peak_vram": verification["device_memory"]["peak_bytes_in_use"],
            **summarize_predictions(run / "validation_predictions.csv")})
    atomic_json(base / "results.json", rows)
    counts = rows[0]["bins"]
    lines = ["# Explicit titratable-site token pilot", "",
        "Backbone-only residue encoder, global 20 Å cutoff, frozen split, seed 17. ARG is never supervised; the no-ARG arm removes it only from site context.", "",
        f"| Arm | Parameters | Overall MAE | <0.5 (n={counts[0]['sites']:,}) | 0.5–1 (n={counts[1]['sites']:,}) | 1–2 (n={counts[2]['sites']:,}) | **≥2 (n={counts[3]['sites']:,})** | Epoch | Time (min) | Peak VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [item["mae"] for item in row["bins"]]
        lines.append(f"| {row['arm']} | {row['parameter_count']:,} | {row['overall']:.4f} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | **{values[3]:.4f}** | {row['selected_epoch']} | {row['wall_seconds']/60:.1f} | {row['peak_vram']/2**30:.2f} |")
    orientation = next(row for row in rows if row["arm"] == "site-orientation")
    distance = next(row for row in rows if row["arm"] == "site-distance")
    no_arg = next(row for row in rows if row["arm"] == "no-arg")
    lines += ["", f"Orientation minus site-distance MAE: **{orientation['overall']-distance['overall']:+.4f}**.",
              f"No-ARG minus orientation MAE: **{no_arg['overall']-orientation['overall']:+.4f}**.",
              "One seed only; promising differences require confirmation. No test data were read.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "verification.json", {"passed": True, "runs": len(rows),
        "rotation_gate_sha256": digest(base / "rotation-gate.json"), "report_sha256": digest(base / "report.md"),
        "test_data_included": False})


def verify_smokes(root):
    rows = []
    for arm in ARMS:
        value = read(experiment_root(root) / arm / "smoke-seed-17/verification.json")
        if not value["passed"] or not value["finite_update"] or value["code_hashes"] != code_hashes(): raise AssertionError(arm)
        rows.append({"arm": arm, "loss": value["loss"]})
    rotation = read(experiment_root(root) / "rotation-gate.json")
    if not rotation["passed"] or rotation["code_hashes"] != code_hashes(): raise AssertionError("rotation")
    atomic_json(experiment_root(root) / "smoke-verification.json", {"passed": True, "rows": rows,
                "rotation": rotation, "code_hashes": code_hashes()})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    gpu = action in ("gate", "smoke", "train")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu, allow_comp1400=gpu)
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "gate": rotation_gate(root)
    elif action == "smoke": train(root, sys.argv[2], smoke=True)
    elif action == "verify-smokes": verify_smokes(root)
    elif action == "train": train(root, sys.argv[2])
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
