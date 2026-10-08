"""Matched global neighbour-cutoff screen for backbone and side-chain GQT."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from jaxpropka.parameters import GROUPS, MODEL_PKA
from pkanet.model import initialize
from pkabench.frozen_score import aggregate, measures, write_csv
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_crop_radius import EPOCHS, MIN_DELTA, PATIENCE, SEEDS, decode_rbf_distance, learning_rate
from .gqt_regularization import RegularizedShiftEngine
from .graph_batches import BatchLoader, epoch_batches
from .graph_data import bucket
from .graph_mmap import build_bundle
from .graph_pkmod_compare import digest_array_tree
from .records import read
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


CUTOFFS = {"backbone": (5.0, 10.0, 15.0, 20.0), "sidechain": (5.0, 15.0, 20.0)}


def experiment_root(root):
    return root / "pretraining/gqt-neighbor-cutoff-v1-triton"


def source(root, mode):
    if mode == "backbone":
        return root / "pretraining/gqt-pkai-parameter-sweep-v1/gqt-long/50k"
    if mode == "sidechain":
        return root / "pretraining/augmentation-sidechains-v1/source"
    raise ValueError(mode)


def source_means(root, mode):
    if mode == "backbone": return source(root, mode) / "train_type_means.json"
    return root / "pretraining/augmentation-sidechains-v1/gqt-baseline/train_type_means.json"


def code_hashes():
    here = Path(__file__); parent = here.parent; model = parent.parent / "pkanet/model.py"
    paths = (here, parent / "gqt_crop_radius.py", parent / "gqt_regularization.py",
             parent / "graph_batches.py", parent / "graph_mmap.py", parent / "trainer.py", model)
    return {str(path): digest(path) for path in paths}


def apply_cutoff(graph, cutoff):
    """Mask edges globally while preserving nodes, queries and the 20 A RBF basis."""
    if not 0 < cutoff <= 20: raise ValueError(cutoff)
    if cutoff == 20:
        return graph, {"original_edges": int(np.asarray(graph["edge_mask"], bool).sum()),
                       "retained_edges": int(np.asarray(graph["edge_mask"], bool).sum())}
    active = np.asarray(graph["edge_mask"], bool)
    distance = np.zeros(active.shape, np.float64)
    distance[active] = decode_rbf_distance(np.asarray(graph["edge"][..., :16])[active])
    keep = active & (distance <= cutoff + 2e-4)
    removed = active & ~keep
    graph["edge"][removed, :19] = 0.0
    graph["edge_mask"][removed] = False
    switch = np.where(distance < cutoff - 2, 1.0,
                      0.5 * (1 + np.cos(np.pi * np.clip((distance - cutoff + 2) / 2, 0, 1))))
    graph["switch"][:] = np.where(keep, switch, 0.0).astype(graph["switch"].dtype)
    return graph, {"original_edges": int(active.sum()), "retained_edges": int(keep.sum())}


class CutoffBatchLoader(BatchLoader):
    def __init__(self, *args, cutoff, **kwargs):
        super().__init__(*args, **kwargs); self.cutoff = float(cutoff)
        self.edge_original = 0; self.edge_retained = 0

    def one(self, cid, capacities=None):
        graph, targets, eligible = super().one(cid, capacities)
        original_nodes = np.asarray(graph["node_mask"], bool).copy()
        original_eligible = np.asarray(eligible, bool).copy()
        graph, stats = apply_cutoff(graph, self.cutoff)
        if not np.array_equal(graph["node_mask"], original_nodes) or not np.array_equal(eligible, original_eligible):
            raise AssertionError("A cutoff changed nodes or supervised queries")
        self.edge_original += stats["original_edges"]; self.edge_retained += stats["retained_edges"]
        return graph, targets, eligible

    def cutoff_stats(self):
        return {"original_edges": self.edge_original, "retained_edges": self.edge_retained,
                "retained_edge_fraction": self.edge_retained / max(self.edge_original, 1)}


def test_cutoffs(root):
    src = source(root, "backbone"); manifest = read(src / "manifest.json"); row = manifest["records"][0]
    with np.load(src / "data" / row["complex_id"] / "graph.npz", allow_pickle=False) as handle:
        graph = {key: handle[key].copy() for key in handle.files if key != "labels"}
    counts = []; previous = None
    for cutoff in (5.0, 10.0, 15.0):
        candidate = {key: value.copy() for key, value in graph.items()}
        candidate, stats = apply_cutoff(candidate, cutoff); counts.append(stats["retained_edges"])
        if previous is not None and stats["retained_edges"] <= previous: raise AssertionError("Cutoff edge sets are not nested")
        previous = stats["retained_edges"]
    identity = {key: value.copy() for key, value in graph.items()}; transformed, _ = apply_cutoff(identity, 20.0)
    for key in graph: np.testing.assert_array_equal(transformed[key], graph[key])
    out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "tests.json", {"passed": True, "complex_id": row["complex_id"],
                "retained_edges_5_10_15": counts, "cutoff20_bit_identical": True,
                "nested_edge_sets": True, "nodes_and_queries_preserved_by_loader": True,
                "code_hashes": code_hashes()})


def register(root):
    out = experiment_root(root); tests = read(out / "tests.json")
    if not tests["passed"] or tests["code_hashes"] != code_hashes(): raise AssertionError("Cutoff tests/code mismatch")
    protocol = {"modes": {key: list(value) for key, value in CUTOFFS.items()}, "seeds": list(SEEDS),
        "cutoff": "global C-alpha edge cutoff applied to every train and validation graph; all nodes and queries retained",
        "distance_features": "fixed original 0-20 A 16-channel RBF basis; only adjacency and cutoff switch change",
        "switch": "one through cutoff-2 A, cosine taper to zero at cutoff",
        "optimizer": "AdamW, weight decay 1e-4", "dropout": 0.0,
        "schedule": "1e-3 through epoch 10; cosine to 1e-5 during epochs 11-20",
        "selection": f"minimum validation group-macro MAE; min_delta {MIN_DELTA}; patience {PATIENCE}",
        "crop_probability": 0.0, "test_data_included": False}
    atomic_json(out / "protocol.json", protocol)
    for mode, cutoffs in CUTOFFS.items():
        src = source(root, mode); parent = read(src / "manifest.json")
        if mode == "sidechain" and parent["config"].get("strict_backbone"): raise AssertionError("Invalid side-chain source")
        for cutoff in cutoffs:
            dest = out / mode / f"cutoff-{int(cutoff)}A"; dest.mkdir(parents=True, exist_ok=True)
            manifest = copy.deepcopy(parent)
            manifest["parent"] = {"path": str(src), "manifest_sha256": digest(src / "manifest.json")}
            manifest["cutoff_protocol_sha256"] = digest(out / "protocol.json")
            manifest["config"].update(seed=None, epochs=EPOCHS, learning_rate=1e-3, constant_epochs=10,
                end_learning_rate=1e-5, schedule="hold 10 epochs then per-update cosine", batch_size=8,
                accumulation=1, matmul_precision="highest", dropout_rate=0.0, context_mask_probability=0.0,
                optimizer="AdamW", weight_decay=1e-4, weight_decay_mask="ndim>=2",
                edge_cutoff_A=cutoff, crop_probability=0.0, augmentation_arm="none",
                objective="explicit signed pKPDB PK_MOD shift MSE; uniform eligible sites within structure",
                selection=f"minimum validation group-macro MAE; min_delta={MIN_DELTA}; patience={PATIENCE}",
                graph_backend="verified read-only mmap", attention_backend="indexed Triton encoder; native query attention")
            data = dest / "data"
            if not data.exists(): data.symlink_to((src / "data").resolve(), target_is_directory=True)
            if not (dest / "preparation.json").exists(): shutil.copy2(src / "preparation.json", dest / "preparation.json")
            if not (dest / "train_type_means.json").exists(): shutil.copy2(source_means(root, mode), dest / "train_type_means.json")
            atomic_json(dest / "manifest.json", manifest)
    atomic_json(out / "registration.json", {"passed": True, "protocol_sha256": digest(out / "protocol.json"),
                "tests_sha256": digest(out / "tests.json"), "code_hashes": code_hashes()})


def prepare_sidechain_mmap(root):
    out = experiment_root(root); registration = read(out / "registration.json")
    if registration["code_hashes"] != code_hashes(): raise AssertionError("Registration/code mismatch")
    src = source(root, "sidechain"); manifest = read(src / "manifest.json"); destination = build_bundle(src, manifest)
    receipt = read(destination / "verification.json")
    atomic_json(out / "sidechain-mmap.json", {"passed": receipt["passed"] and receipt["all_records_and_fields_bit_identical"],
                "path": str(destination), "bytes": receipt["bytes"], "verification_sha256": digest(destination / "verification.json"),
                "source_manifest_sha256": digest(src / "manifest.json"), "code_hashes": code_hashes()})


def evaluate(root, mode, cutoff, out, manifest, engine, params, means, *, predictions=None):
    loader = CutoffBatchLoader(out, manifest, manifest["config"]["batch_size"], backend="mmap", cutoff=cutoff)
    rows = []; scores = {name: [] for name in ("graph_query", "model_values", "train_type_mean")}
    for record in manifest["records"]:
        if record["split"] != "val": continue
        graph, target, eligible = loader.one(record["complex_id"])
        pred = np.asarray(engine.forward(params, graph))[eligible]; target = target[eligible]
        group = graph["query_group"][eligible]; baseline = MODEL_PKA[group]
        for name, value in (("graph_query", pred), ("model_values", baseline), ("train_type_mean", means[group])):
            scores[name].append({"component_id": record["component_id"], "complex_id": record["complex_id"],
                                 "n": len(target), **measures(target - baseline, value - baseline)})
        if predictions is not None:
            for key, observed, predicted, mean in zip(record["keys"], target, pred, means[group]):
                rows.append(dict(zip(("complex_id", "chain", "resnum", "icode", "group"), key),
                                 component_id=record["component_id"], teacher_pka=float(observed),
                                 predicted_pka=float(predicted), train_type_mean=float(mean)))
    result = {name: aggregate(values, replicates=2000)[0] for name, values in scores.items()}
    stats = loader.cutoff_stats(); loader.close()
    if predictions is not None: write_csv(predictions, rows)
    return result, stats


def peak_device_memory():
    stats = jax.local_devices()[0].memory_stats() or {}
    return {"peak_bytes_in_use": stats.get("peak_bytes_in_use"), "bytes_in_use": stats.get("bytes_in_use")}


def train(root, mode, cutoff, seed, *, smoke=False):
    cutoff = float(cutoff)
    if cutoff not in CUTOFFS[mode] or seed not in SEEDS: raise ValueError((mode, cutoff, seed))
    base = experiment_root(root); registration = read(base / "registration.json")
    if registration["code_hashes"] != code_hashes(): raise AssertionError("Registration/code mismatch")
    if mode == "sidechain":
        mmap = read(base / "sidechain-mmap.json")
        if not mmap["passed"] or mmap["code_hashes"] != code_hashes(): raise AssertionError("Side-chain mmap mismatch")
    out = base / mode / f"cutoff-{int(cutoff)}A"; stored_manifest = out / "manifest.json"; manifest = read(stored_manifest)
    manifest["config"]["seed"] = seed; cfg = manifest["config"]; run_started = time.monotonic()
    params = initialize(jax.random.PRNGKey(seed), **cfg["architecture"]); engine = RegularizedShiftEngine(params, weight_decay=1e-4)
    state = engine.optimizer.init(params); rng = np.random.default_rng(seed)
    run = out / (f"smoke-seed-{seed}" if smoke else f"seed-{seed}"); run.mkdir(parents=True, exist_ok=True)
    by_id = {row["complex_id"]: row for row in manifest["records"]}; records = [by_id[cid] for cid in manifest["train"]]
    loader = CutoffBatchLoader(out, manifest, cfg["batch_size"], backend="mmap", cutoff=cutoff)
    means = np.asarray([read(out / "train_type_means.json")[group] for group in GROUPS])
    provenance = {"manifest_sha256": digest(stored_manifest), "mode": mode, "cutoff_A": cutoff, "seed": seed,
                  "code_hashes": code_hashes(), "parameter_count": cfg["parameter_count"]}
    atomic_json(run / "run.json", {**provenance, "config": cfg, "loader": loader.provenance(),
                                    "initial_parameter_digest": digest_array_tree(params)})
    history, best, stalled = [], None, 0
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        plans = epoch_batches(sample_epoch(records, rng), by_id, rng, cfg["batch_size"])
        if smoke: plans = plans[:1]
        losses = []; began = time.monotonic()
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            lr = learning_rate(epoch, number, len(plans)); params, state, value = engine.audited_batch_update(params, state, *batch, learning_rate=lr)
            losses.append(value)
        if smoke: history.append({"epoch": 1, "train_shift_mse": float(np.mean(losses))}); break
        metrics, validation_stats = evaluate(root, mode, cutoff, out, manifest, engine, params, means)
        mae = float(metrics["graph_query"]["mae"])
        if best is None or mae < best["mae"] - MIN_DELTA:
            best, stalled = {"epoch": epoch, "mae": mae}, 0; atomic_json(run / "best.json", best)
        else: stalled += 1
        row = {"epoch": epoch, "train_shift_mse": float(np.mean(losses)), "validation": metrics,
               "validation_cutoff_stats": validation_stats, "seconds": time.monotonic() - began,
               "updates": len(plans), "learning_rate": lr, "best_epoch": best["epoch"], "stalled_epochs": stalled}
        history.append(row); atomic_json(run / "history.json", history); atomic_json(run / "training-cutoff-stats.json", loader.cutoff_stats())
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"mode": mode, "cutoff_A": cutoff, "seed": seed, "epoch": epoch,
                          "train_shift_mse": row["train_shift_mse"], "val_mae": mae,
                          "best_epoch": best["epoch"], "seconds": row["seconds"]}), flush=True)
        if stalled >= PATIENCE: break
    loader.close()
    if smoke:
        atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                    "training_cutoff_stats": loader.cutoff_stats(), "device_memory": peak_device_memory(), **provenance}); return
    params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final, validation_stats = evaluate(root, mode, cutoff, out, manifest, engine, params, means,
                                       predictions=run / "validation_predictions.csv")
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {"passed": True, "epochs_completed": len(history), "selected_epoch": best["epoch"],
                "wall_seconds": time.monotonic() - run_started, "device_memory": peak_device_memory(),
                "training_cutoff_stats": loader.cutoff_stats(), "validation_cutoff_stats": validation_stats,
                "predictions_sha256": digest(run / "validation_predictions.csv"), "test_data_included": False, **provenance})


def smoke(root):
    for mode, cutoffs in CUTOFFS.items():
        for cutoff in cutoffs: train(root, mode, cutoff, 17, smoke=True)
    atomic_json(experiment_root(root) / "smoke-verification.json", {"passed": True, "configurations": 7,
                "finite_update_each": True, "code_hashes": code_hashes()})


def report(root):
    base = experiment_root(root); rows = []
    for mode, cutoffs in CUTOFFS.items():
        for cutoff in cutoffs:
            for seed in SEEDS:
                run = base / mode / f"cutoff-{int(cutoff)}A" / f"seed-{seed}"; verify = read(run / "verification.json")
                metric = read(run / "final.json")["graph_query"]
                rows.append({"mode": mode, "cutoff_A": cutoff, "seed": seed, "mae": metric["mae"],
                    "mae_ci95": metric["mae_ci95"], "selected_epoch": verify["selected_epoch"],
                    "epochs": verify["epochs_completed"], "wall_seconds": verify["wall_seconds"],
                    "peak_vram_bytes": verify["device_memory"]["peak_bytes_in_use"],
                    "validation_retained_edge_fraction": verify["validation_cutoff_stats"]["retained_edge_fraction"]})
    summaries = []
    for mode, cutoffs in CUTOFFS.items():
        reference = {row["seed"]: row["mae"] for row in rows if row["mode"] == mode and row["cutoff_A"] == 20}
        for cutoff in cutoffs:
            subset = [row for row in rows if row["mode"] == mode and row["cutoff_A"] == cutoff]; values = np.asarray([row["mae"] for row in subset])
            summaries.append({"mode": mode, "cutoff_A": cutoff, "mean_mae": float(values.mean()),
                "sd_mae": float(values.std(ddof=1)), "mean_delta_vs_20A": float(np.mean([row["mae"] - reference[row["seed"]] for row in subset])),
                "selected_epochs": [row["selected_epoch"] for row in subset],
                "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in subset])),
                "max_peak_vram_gib": float(max(row["peak_vram_bytes"] for row in subset) / 2**30),
                "mean_retained_edge_fraction": float(np.mean([row["validation_retained_edge_fraction"] for row in subset]))})
    atomic_json(base / "results.json", rows); atomic_json(base / "summary.json", summaries)
    lines = ["# GQT global neighbour-cutoff screen", "",
             "Every training and validation graph uses the registered global edge cutoff. Nodes and supervised queries are unchanged; no crop augmentation is used.", "",
             "| Input | Cutoff | Validation MAE | Seed SD | Delta vs 20 A | Edges retained | Selected epochs | Mean time/run (min) | Peak VRAM (GiB) |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['mode']} | {int(row['cutoff_A'])} A | {row['mean_mae']:.4f} | {row['sd_mae']:.4f} | {row['mean_delta_vs_20A']:+.4f} | {row['mean_retained_edge_fraction']:.3f} | {', '.join(map(str,row['selected_epochs']))} | {row['mean_wall_seconds']/60:.1f} | {row['max_peak_vram_gib']:.2f} |")
    lines += ["", "Full float32; AdamW 1e-4; three seeds; full validation under the same cutoff; no test data.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "verification.json", {"passed": True, "runs": len(rows),
                "report_sha256": digest(base / "report.md"), "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"]); gpu = action in ("smoke", "train")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu, allow_comp1400=gpu)
    jax.config.update("jax_enable_x64", False)
    if action == "test": test_cutoffs(root)
    elif action == "register": register(root)
    elif action == "mmap": prepare_sidechain_mmap(root)
    elif action == "smoke": smoke(root)
    elif action == "train": train(root, sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
