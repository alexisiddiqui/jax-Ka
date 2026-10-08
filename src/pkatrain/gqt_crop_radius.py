"""Matched query-centred crop-radius experiment for the 50k GQT."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import jax
import numpy as np

from jaxpropka.parameters import GROUPS
from pkanet.model import initialize
from pkabench.runtime import atomic_json, digest, require_compute
from .graph_batches import BatchLoader, epoch_batches
from .graph_experiment import evaluate
from .graph_pkmod_compare import digest_array_tree
from .gqt_regularization import RegularizedShiftEngine
from .records import read
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


ARMS = ("full", "crop-10A", "crop-15A", "crop-20A")
RADII = {"full": None, "crop-10A": 10.0, "crop-15A": 15.0, "crop-20A": 20.0}
SEEDS = (17, 29, 43)
EPOCHS = 20
HOLD_EPOCHS = 10
PATIENCE = 8
MIN_DELTA = 0.001
CROP_PROBABILITY = 0.25
RBF_CENTERS = np.linspace(0.0, 20.0, 16, dtype=np.float64)
RBF_WIDTH = 1.5


def source_hashes():
    root = Path(__file__).parents[1]
    paths = (
        Path(__file__), root / "pkatrain/gqt_regularization.py",
        root / "pkanet/model.py", root / "pkanet/triton_attention.py",
        root / "pkatrain/graph_batches.py", root / "pkatrain/graph_experiment.py",
        root / "pkatrain/trainer.py",
    )
    return {str(path): digest(path) for path in paths}


def learning_rate(epoch, batch, batches, *, start=1e-3, end=1e-5):
    """Hold 1e-3 through epoch 10, then decay to 1e-5 by epoch 20."""
    if epoch <= HOLD_EPOCHS:
        return float(start)
    within = (batch - 1) / max(batches - 1, 1)
    fraction = ((epoch - HOLD_EPOCHS - 1) + within) / (EPOCHS - HOLD_EPOCHS - 1)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * fraction)))


def decode_rbf_distance(rbf):
    """Recover the generating C-alpha distance from adjacent Gaussian RBFs."""
    value = np.asarray(rbf, dtype=np.float64)
    if value.shape[-1] != len(RBF_CENTERS):
        raise ValueError(value.shape)
    flat = value.reshape((-1, value.shape[-1]))
    first = np.argmax(flat, axis=1)
    second = np.where(first < len(RBF_CENTERS) - 1, first + 1, first - 1)
    rows = np.arange(len(flat))
    a = np.maximum(flat[rows, first], np.finfo(np.float32).tiny)
    b = np.maximum(flat[rows, second], np.finfo(np.float32).tiny)
    ca = RBF_CENTERS[first]
    cb = RBF_CENTERS[second]
    log_ratio = np.log(a) - np.log(b)
    distance = (cb * cb - ca * ca - log_ratio * RBF_WIDTH**2) / (2 * (cb - ca))
    return distance.reshape(value.shape[:-1])


def crop_decision(seed, epoch, complex_id, probability=CROP_PROBABILITY):
    token = f"gqt-crop-v1:{seed}:{epoch}:{complex_id}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
    return rng, bool(rng.random() < probability)


def crop_query_graph(graph, eligible, radius, query_number):
    """Retain one query's C-alpha neighbourhood and supervise only that query."""
    if radius <= 0 or radius > 20:
        raise ValueError(radius)
    eligible = np.asarray(eligible).copy()
    choices = np.flatnonzero(eligible)
    if not len(choices):
        raise ValueError("Crop draw has no eligible query")
    q = int(choices[int(query_number) % len(choices)])
    center = int(graph["query_residue"][q])
    row_mask = np.asarray(graph["edge_mask"][center], bool)
    neighbors = np.asarray(graph["neighbors"][center, row_mask], np.int64)
    distances = decode_rbf_distance(np.asarray(graph["edge"][center, row_mask, :16]))
    original_nodes = int(np.asarray(graph["node_mask"], bool).sum())
    keep = np.zeros(len(graph["node_mask"]), bool)
    keep[neighbors[distances <= radius + 2e-4]] = True
    keep[center] = True
    keep &= np.asarray(graph["node_mask"], bool)
    if not keep[center]:
        raise AssertionError("Crop removed its supervised centre")

    # pad() already returned mutable arrays. Remove every geometric path across
    # the crop boundary while retaining original residue identity and indexing.
    affected = (~keep)[:, None] | (~keep)[graph["neighbors"]]
    graph["edge"][affected, :19] = 0.0
    graph["edge_mask"][affected] = False
    graph["switch"][affected] = 0.0
    graph["node_mask"] &= keep
    eligible[:] = False
    eligible[q] = True
    return graph, eligible, {
        "query_index": q, "query_residue": center,
        "original_nodes": original_nodes, "retained_nodes": int(keep.sum()),
        "original_queries": int(len(choices)), "retained_queries": 1,
    }


class CropBatchLoader(BatchLoader):
    def __init__(self, *args, crop_radius=None, crop_probability=CROP_PROBABILITY, seed=17, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_radius = crop_radius
        self.crop_probability = crop_probability
        self.seed = seed
        self.epoch = None
        self._crop_lock = threading.Lock()
        self._crop_stats = {
            "draws": 0, "cropped_draws": 0, "full_draws": 0,
            "retained_nodes": 0, "original_nodes": 0,
            "retained_queries": 0, "original_queries": 0,
        }

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        return super().set_epoch(epoch)

    def one(self, cid, capacities=None):
        graph, targets, eligible = super().one(cid, capacities)
        if self.crop_radius is None:
            with self._crop_lock:
                self._crop_stats["draws"] += 1
                self._crop_stats["full_draws"] += 1
            return graph, targets, eligible
        if self.epoch is None:
            raise AssertionError("Crop loader epoch was not set")
        rng, selected = crop_decision(self.seed, self.epoch, cid, self.crop_probability)
        if not selected:
            with self._crop_lock:
                self._crop_stats["draws"] += 1
                self._crop_stats["full_draws"] += 1
            return graph, targets, eligible
        query_count = int(np.asarray(eligible, bool).sum())
        graph, eligible, stats = crop_query_graph(
            graph, eligible, self.crop_radius, int(rng.integers(0, query_count))
        )
        with self._crop_lock:
            self._crop_stats["draws"] += 1
            self._crop_stats["cropped_draws"] += 1
            for key in ("retained_nodes", "original_nodes", "retained_queries", "original_queries"):
                self._crop_stats[key] += stats[key]
        return graph, targets, eligible

    def crop_stats(self):
        with self._crop_lock:
            result = dict(self._crop_stats)
        cropped = result["cropped_draws"]
        result["observed_crop_fraction"] = cropped / max(result["draws"], 1)
        result["mean_retained_node_fraction_on_crops"] = (
            result["retained_nodes"] / max(result["original_nodes"], 1)
        )
        return result


def experiment_root(root):
    return root / "pretraining" / "gqt-50k-crop-radius-v1-triton"


def test_geometry(root):
    distances = np.asarray([0.0001, 1.0, 5.0, 9.999, 10.001, 15.0, 19.999])
    encoded = np.exp(-((distances[:, None] - RBF_CENTERS[None]) / RBF_WIDTH) ** 2).astype(np.float32)
    decoded = decode_rbf_distance(encoded)
    np.testing.assert_allclose(decoded, distances, rtol=0, atol=2e-5)

    n = 5
    d = np.asarray([0.0001, 5.0, 10.0, 15.0, 19.0])
    rbf = np.exp(-((d[:, None] - RBF_CENTERS[None]) / RBF_WIDTH) ** 2).astype(np.float32)
    graph = {
        "nodes": np.zeros((n, 24), np.float32), "node_mask": np.ones(n, bool),
        "neighbors": np.tile(np.arange(n, dtype=np.int32), (n, 1)),
        "edge": np.zeros((n, n, 20), np.float32), "edge_mask": np.ones((n, n), bool),
        "switch": np.ones((n, n), np.float32),
        "query_residue": np.asarray([0, 2], np.int32), "query_group": np.asarray([0, 1], np.int32),
    }
    graph["edge"][0, :, :16] = rbf
    # Other rows only need valid encodings for the crop-boundary operation.
    graph["edge"][1:, :, :16] = rbf[None]
    cropped, eligible, stats = crop_query_graph(graph, np.asarray([True, True]), 10.0, 0)
    assert cropped["node_mask"].tolist() == [True, True, True, False, False]
    assert eligible.tolist() == [True, False] and stats["retained_nodes"] == 3
    assert not cropped["edge_mask"][~cropped["node_mask"]].any()
    assert not cropped["edge_mask"][:, ~cropped["node_mask"]].any()

    rng1, yes1 = crop_decision(17, 3, "example", 1.0)
    rng2, yes2 = crop_decision(17, 3, "example", 1.0)
    assert yes1 and yes2 and rng1.integers(0, 10**6) == rng2.integers(0, 10**6)
    out = experiment_root(root)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "geometry-tests.json", {
        "passed": True, "distance_max_abs_error": float(np.max(np.abs(decoded - distances))),
        "crop_boundary_edges_removed": True, "supervised_query_preserved": True,
        "deterministic_draws": True, "source_hashes": source_hashes(),
    })


def register(root):
    destination = experiment_root(root)
    destination.mkdir(parents=True, exist_ok=True)
    tests = read(destination / "geometry-tests.json")
    if not tests["passed"] or tests["source_hashes"] != source_hashes():
        raise AssertionError("Crop geometry tests/code mismatch")
    source = root / "pretraining" / "gqt-pkai-parameter-sweep-v1" / "gqt-long" / "50k"
    parent = read(source / "manifest.json")
    protocol = {
        "architecture": "GQT 49,709 parameters, strict backbone, 20 A C-alpha graph",
        "arms": list(ARMS), "crop_radii_A": RADII, "crop_probability": CROP_PROBABILITY,
        "crop_unit": "one uniformly sampled eligible query; retain C-alpha neighbourhood; supervise centre query only",
        "crop_draw_matching": "deterministic by seed, epoch and complex; identical decision/query across radius arms",
        "seeds": list(SEEDS), "max_epochs": EPOCHS,
        "optimizer": "AdamW, weight decay 1e-4 on learned matrices/embeddings",
        "schedule": "1e-3 through epoch 10; per-update cosine to 1e-5 during epochs 11-20",
        "selection": f"minimum validation group-macro MAE; min_delta {MIN_DELTA}; patience {PATIENCE}",
        "fixed": "5k cohort, explicit historical-pKPDB signed-shift MSE, uniform structure loss, batch 8, no dropout/masking, full unaugmented validation",
        "attention_backend": "validated float32 indexed Triton encoder; native query attention",
        "implementation": "shape-preserving edge/node masks isolate augmentation effect; no claimed throughput gain",
        "test_data_included": False,
    }
    atomic_json(destination / "protocol.json", protocol)
    for arm in ARMS:
        out = destination / arm
        out.mkdir(exist_ok=True)
        manifest = copy.deepcopy(parent)
        manifest["parent"] = {"path": str(source), "manifest_sha256": digest(source / "manifest.json")}
        manifest["crop_protocol_sha256"] = digest(destination / "protocol.json")
        manifest["config"].update(
            seed=None, epochs=EPOCHS, learning_rate=1e-3, constant_epochs=HOLD_EPOCHS,
            end_learning_rate=1e-5, schedule="hold 10 epochs then per-update cosine",
            dropout_rate=0.0, context_mask_probability=0.0,
            optimizer="AdamW", weight_decay=1e-4, weight_decay_mask="ndim>=2",
            crop_radius_A=RADII[arm], crop_probability=CROP_PROBABILITY if RADII[arm] else 0.0,
            crop_supervision="uniformly sampled centre query only on crop draws",
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
        "geometry_tests_sha256": digest(destination / "geometry-tests.json"),
        "source_hashes": source_hashes(),
    })


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
    loader = CropBatchLoader(
        out, manifest, cfg["batch_size"], backend="mmap", seed=seed,
        crop_radius=cfg["crop_radius_A"], crop_probability=cfg["crop_probability"],
    )
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
        loader.set_epoch(epoch)
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            lr = learning_rate(epoch, number, len(plans))
            params, state, value = engine.audited_batch_update(
                params, state, *batch, learning_rate=lr
            )
            losses.append(value)
            if number % 20 == 0:
                atomic_json(run / "progress.json", {
                    "arm": arm, "seed": seed, "epoch": epoch, "epochs": max_epochs,
                    "batches": number, "total_batches": len(plans),
                    "learning_rate": lr, "elapsed_seconds": time.monotonic() - began,
                })
        if smoke:
            history.append({"epoch": 1, "train_shift_mse": float(np.mean(losses)), "learning_rate": lr})
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
            "updates": len(plans), "learning_rate": lr,
            "best_epoch": best["epoch"], "stalled_epochs": stalled,
        }
        history.append(row)
        atomic_json(run / "history.json", history)
        atomic_json(run / "crop-stats.json", loader.crop_stats())
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {
            **provenance, "epoch": epoch, "rng": rng.bit_generator.state,
            "validation_mae": mae,
        })
        print(json.dumps({
            "arm": arm, "seed": seed, "epoch": epoch,
            "train_shift_mse": row["train_shift_mse"], "val_mae": mae,
            "best_epoch": best["epoch"], "stalled": stalled,
            "seconds": row["seconds"], "learning_rate": lr,
        }), flush=True)
        if stalled >= PATIENCE:
            break
    loader.close()
    atomic_json(run / "history.json", history)
    atomic_json(run / "crop-stats.json", loader.crop_stats())
    if smoke:
        atomic_json(run / "verification.json", {
            "passed": True, "smoke": True, "finite_update": True,
            "arm": arm, "seed": seed, "crop_stats": loader.crop_stats(), **provenance,
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
        "crop_stats_sha256": digest(run / "crop-stats.json"),
        "test_data_included": False, **provenance,
    })


def smoke(root):
    for arm in ARMS:
        train(root, arm, 17, smoke=True)
    atomic_json(experiment_root(root) / "smoke-verification.json", {
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
            metric = read(run / "final.json")["graph_query"]
            crop_stats = read(run / "crop-stats.json")
            if not verification["passed"] or verification["test_data_included"]:
                raise AssertionError((arm, seed))
            rows.append({
                "arm": arm, "radius_A": RADII[arm], "seed": seed,
                "selected_epoch": verification["selected_epoch"],
                "epochs_completed": verification["epochs_completed"],
                "mae": metric["mae"], "mae_ci95": metric["mae_ci95"],
                "rmse": metric["rmse"], "spearman": metric["spearman"],
                "sites": metric["sites"], "groups": metric["groups"],
                "observed_crop_fraction": crop_stats["observed_crop_fraction"],
                "retained_node_fraction": crop_stats["mean_retained_node_fraction_on_crops"],
            })
    summary = []
    baseline = {row["seed"]: row["mae"] for row in rows if row["arm"] == "full"}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        values = np.asarray([row["mae"] for row in subset])
        deltas = np.asarray([row["mae"] - baseline[row["seed"]] for row in subset])
        summary.append({
            "arm": arm, "radius_A": RADII[arm], "mean_mae": float(values.mean()),
            "sd_mae": float(values.std(ddof=1)), "mean_delta_vs_full": float(deltas.mean()),
            "selected_epochs": [row["selected_epoch"] for row in subset],
            "mean_crop_fraction": float(np.mean([row["observed_crop_fraction"] for row in subset])),
            "mean_retained_node_fraction": (
                None if arm == "full" else
                float(np.mean([row["retained_node_fraction"] for row in subset]))
            ),
        })
    winner = min(summary, key=lambda row: row["mean_mae"])["arm"]
    atomic_json(base / "results.json", rows)
    atomic_json(base / "summary.json", {"winner": winner, "arms": summary})
    lines = [
        "# 50k GQT query-centred crop-radius screen", "",
        "Three seeds per arm; AdamW 1e-4; learning rate held at 1e-3 through epoch 10 and then cosine-decayed to 1e-5. Crops affect 25% of training draws. Full validation is unchanged; no test data were read.", "",
        "| Arm | Mean validation MAE | Seed SD | Delta vs full | Crop draws | Nodes retained on crops | Selected epochs |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['arm']} | {row['mean_mae']:.4f} | {row['sd_mae']:.4f} | "
            f"{row['mean_delta_vs_full']:+.4f} | {row['mean_crop_fraction']:.3f} | "
            f"{('—' if row['mean_retained_node_fraction'] is None else format(row['mean_retained_node_fraction'], '.3f'))} | "
            f"{', '.join(map(str, row['selected_epochs']))} |"
        )
    lines += ["", f"Lowest three-seed mean: **{winner}**.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "verification.json", {
        "passed": True, "runs": len(rows), "winner": winner,
        "results_sha256": digest(base / "results.json"),
        "summary_sha256": digest(base / "summary.json"),
        "report_sha256": digest(base / "report.md"), "test_data_included": False,
    })


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
    if action == "test":
        test_geometry(root)
    elif action == "register":
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
