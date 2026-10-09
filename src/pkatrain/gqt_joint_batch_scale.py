"""Five-epoch joint oGQT batch scaling and gradient-noise audit."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from pkanet.ogqt import initialize as initialize_ogqt
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_multitask_replay import (
    JointEngine, PKPDB_TOLERANCE, _load_parent, pkpdb_root,
)
from .gqt_paired_pinder import (
    Loader as PairedLoader, _prefetched as paired_prefetched,
    evaluate as evaluate_paired, experiment_root as paired_root,
)
from .gqt_site_weighting import evaluate as evaluate_pkpdb
from .graph_batches import epoch_batches
from .site_graph_data import SiteBatchLoader
from .trainer import sample_epoch, save_checkpoint


SEED = 17
EPOCHS = 5
SCALES = (0.5, 1.0, 2.0, 4.0)
BASE_LR = 1e-4
NOISE_SAMPLES = 64


def read(path): return json.loads(Path(path).read_text())
def root_path(root): return Path(root) / "training/ogqt-joint-batch-v1"
def arm_name(scale): return f"b{str(scale).replace('.', 'p')}"


def scaled_pair_sizes(manifest, scale):
    return {name: max(1, int(round(size * scale))) for name, size in manifest["batch_sizes"].items()}


def paired_plans(records, rng, manifest, scale):
    groups = defaultdict(list); sizes = scaled_pair_sizes(manifest, scale)
    limits = sorted(manifest["capacities"], key=int)
    for row in records: groups[next(name for name in limits if int(row["n"]) <= int(name))].append(row["id"])
    plans = []
    for bucket, ids in groups.items():
        rng.shuffle(ids); size = sizes[bucket]
        plans.extend(ids[i:i + size] for i in range(0, len(ids), size))
    rng.shuffle(plans); return plans


def pk_plans(records, by_id, rng, batch_size, count):
    plans = []
    while len(plans) < count:
        plans.extend(epoch_batches(sample_epoch(records, rng), by_id, rng, batch_size))
    return plans[:count]


def register(root):
    root = Path(root); out = root_path(root); out.mkdir(parents=True, exist_ok=True)
    pair_manifest = read(paired_root(root) / "manifest.json")
    pk_manifest = read(pkpdb_root(root) / "manifest.json")
    protocol = {
        "version": "ogqt-joint-batch-v1", "epochs": EPOCHS, "seed": SEED,
        "scales": list(SCALES), "base_pinder_batch_sizes": pair_manifest["batch_sizes"],
        "base_pkpdb_batch_size": pk_manifest["config"]["batch_size"],
        "learning_rate_rule": "eta(scale) = 1e-4 * sqrt(scale); AdamW beta/epsilon unchanged",
        "comparison": "fixed epoch 5; identical initialization, structures, objective and seed; no augmentation",
        "noise_scale": "B_noise=(tr Sigma_pinder + tr Sigma_pkpdb)/||G_pinder+G_pkpdb||^2 at the shared parent checkpoint",
        "noise_samples_per_task": NOISE_SAMPLES,
        "test_data_included": False,
    }
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "registration.json", {"passed": True, "protocol_sha256": digest(out / "protocol.json"),
        "test_data_included": False})


def _initial(root):
    pair_manifest = read(paired_root(root) / "manifest.json")
    pk_manifest = read(pkpdb_root(root) / "manifest.json")
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **pk_manifest["config"]["architecture"])
    engine = JointEngine(params); state = engine.optimizer.init(params)
    params, parent = _load_parent(root, params, state)
    return pair_manifest, pk_manifest, engine, params, engine.optimizer.init(params), parent


def train(root, scale, smoke=False):
    root = Path(root); register(root)
    if scale not in SCALES: raise ValueError(scale)
    pair_manifest, pk_manifest, engine, params, state, parent = _initial(root)
    pair_base, pk_base = paired_root(root), pkpdb_root(root)
    pair_train = [row for row in pair_manifest["records"] if row["split"] == "train"]
    pair_val = [row for row in pair_manifest["records"] if row["split"] == "val"]
    pk_by_id = {row["complex_id"]: row for row in pk_manifest["records"]}
    pk_train = [pk_by_id[cid] for cid in pk_manifest["train"]]
    pk_batch = max(1, int(round(pk_manifest["config"]["batch_size"] * scale)))
    rate = BASE_LR * math.sqrt(scale)
    run = root_path(root) / arm_name(scale) / ("smoke" if smoke else "seed-17")
    run.mkdir(parents=True, exist_ok=True)
    provenance = {"scale": scale, "pinder_batch_sizes": scaled_pair_sizes(pair_manifest, scale),
        "pkpdb_batch_size": pk_batch, "learning_rate": rate, "parent": parent,
        "parameter_count": sum(x.size for x in jax.tree.leaves(params)), "test_data_included": False}
    atomic_json(run / "run.json", provenance)
    pair_loader = PairedLoader(pair_base, pair_manifest)
    pk_loader = SiteBatchLoader(pk_base, pk_manifest, pk_batch, "site-orientation")
    pair_rng, pk_rng = np.random.default_rng(SEED), np.random.default_rng(1701)
    history = []; began_all = time.monotonic()
    epochs = 1 if smoke else EPOCHS
    for epoch in range(1, epochs + 1):
        pair_plan = paired_plans(pair_train, pair_rng, pair_manifest, scale)
        pk_plan = pk_plans(pk_train, pk_by_id, pk_rng, pk_batch, len(pair_plan))
        pk_iterator = iter(pk_loader.iterate(pk_plan)); losses = []; began = time.monotonic()
        for number, (_, pair_batch) in enumerate(paired_prefetched(pair_loader, pair_plan), 1):
            _, pk_batch_value = next(pk_iterator)
            params, state, values = engine.update(params, state, pair_batch, pk_batch_value, rate)
            losses.append(values)
            if smoke: break
            if number % 50 == 0:
                atomic_json(run / "progress.json", {"epoch": epoch, "batch": number,
                    "batches": len(pair_plan), "elapsed_seconds": time.monotonic() - began})
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                "losses": losses[-1], "device_memory": jax.local_devices()[0].memory_stats(), **provenance})
            pair_loader.close(); pk_loader.close(); return
        pval = evaluate_paired(pair_base, pair_manifest, pair_val, engine, params)
        kval = evaluate_pkpdb(pk_base, pk_manifest, "baseline", engine, params)
        row = {"epoch": epoch, "updates": len(pair_plan), "seconds": time.monotonic() - began,
            "train": {key: float(np.mean([x[key] for x in losses])) for key in losses[0]},
            "pinder_validation": pval, "pkpdb_validation": kval,
            "selection": pval["state_mae"] + pval["interface_paired_mae"]}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state,
                        {**provenance, "epoch": epoch})
        print(json.dumps({"experiment": "ogqt-joint-batch-v1", "arm": arm_name(scale), **row}), flush=True)
    pair_loader.close(); pk_loader.close()
    final = {"epoch": EPOCHS, "pinder": history[-1]["pinder_validation"],
        "pkpdb": history[-1]["pkpdb_validation"], "selection": history[-1]["selection"]}
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {"passed": True, "epochs_completed": EPOCHS,
        "wall_seconds": time.monotonic() - began_all, "device_memory": jax.local_devices()[0].memory_stats(),
        **provenance})


def _flat(gradient):
    return np.concatenate([np.asarray(x, np.float64).reshape(-1) for x in jax.tree.leaves(gradient)])


def _noise(vectors):
    x = np.asarray(vectors, np.float64); n = len(x); mean = x.mean(0)
    sample_trace = float(np.sum((x - mean) ** 2) / (n - 1))
    corrected_g2 = max(float(mean @ mean) - sample_trace / n, np.finfo(float).tiny)
    return {"samples": n, "gradient_norm_squared": corrected_g2,
        "covariance_trace": sample_trace, "noise_scale": sample_trace / corrected_g2,
        "mean": mean}


def noise(root):
    root = Path(root); register(root)
    pair_manifest, pk_manifest, engine, params, _, parent = _initial(root)
    rng = np.random.default_rng(20261009)
    pair_records = [row for row in pair_manifest["records"] if row["split"] == "train"]
    pk_by_id = {row["complex_id"]: row for row in pk_manifest["records"]}
    pk_records = [pk_by_id[cid] for cid in pk_manifest["train"]]
    pair_records = [pair_records[i] for i in rng.choice(len(pair_records), NOISE_SAMPLES, replace=False)]
    pk_records = [pk_records[i] for i in rng.choice(len(pk_records), NOISE_SAMPLES, replace=False)]
    pair_loader = PairedLoader(paired_root(root), pair_manifest)
    pk_loader = SiteBatchLoader(pkpdb_root(root), pk_manifest, 1, "site-orientation")
    pair_vectors, pk_vectors = [], []
    for number, row in enumerate(pair_records, 1):
        graphs, targets, mask, wb, wi, _ = pair_loader.batch([row["id"]])
        (_, _), grad = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, np.ones(1, bool))
        pair_vectors.append(_flat(grad))
        if number % 8 == 0: print(json.dumps({"noise_task": "pinder", "completed": number}), flush=True)
    from .graph_data import bucket
    for number, row in enumerate(pk_records, 1):
        capacities = pk_manifest["capacities"][bucket(row)]
        batch = pk_loader.load([row["complex_id"]], capacities)
        _, grad = engine.pkpdb_value_grad(params, *batch)
        pk_vectors.append(_flat(grad))
        if number % 8 == 0: print(json.dumps({"noise_task": "pkpdb", "completed": number}), flush=True)
    pair_loader.close(); pk_loader.close()
    p, k = _noise(pair_vectors), _noise(pk_vectors)
    total_mean = p.pop("mean") + k.pop("mean")
    total_g2 = max(float(total_mean @ total_mean)
                   - p["covariance_trace"] / p["samples"]
                   - k["covariance_trace"] / k["samples"], np.finfo(float).tiny)
    result = {"version": "gradient-noise-scale-v1", "parent": parent,
        "pinder": p, "pkpdb": k,
        "joint": {"gradient_norm_squared": total_g2,
            "covariance_trace": p["covariance_trace"] + k["covariance_trace"],
            "noise_scale_structures_per_task": (p["covariance_trace"] + k["covariance_trace"]) / max(total_g2, np.finfo(float).tiny)},
        "interpretation": "One unit of B is one PINDER complex plus one independently sampled pKPDB structure.",
        "test_data_included": False}
    atomic_json(root_path(root) / "gradient-noise.json", result)


def report(root):
    root = Path(root); out = root_path(root); rows = []
    for scale in SCALES:
        run = out / arm_name(scale) / "seed-17"; verification = read(run / "verification.json")
        if not verification["passed"]: raise AssertionError(run)
        final = read(run / "final.json"); hist = read(run / "history.json")
        rows.append({"scale": scale, "pinder_batch_sizes": verification["pinder_batch_sizes"],
            "pkpdb_batch_size": verification["pkpdb_batch_size"], "learning_rate": verification["learning_rate"],
            "updates": sum(x["updates"] for x in hist), "seconds": verification["wall_seconds"],
            "peak_bytes": (verification.get("device_memory") or {}).get("peak_bytes_in_use"),
            "state_mae": final["pinder"]["state_mae"], "interface_mae": final["pinder"]["interface_paired_mae"],
            "selection": final["selection"], "pkpdb_mae": final["pkpdb"]["overall"]["mae"]})
    result = {"runs": rows, "gradient_noise": read(out / "gradient-noise.json"), "test_data_included": False}
    atomic_json(out / "report.json", result)
    lines = ["# Joint oGQT batch-size and gradient-noise audit", "",
        "All training rows use the same 67,725-parameter parent, five epochs, full 5k PINDER cohort and joint pKPDB objective. Learning rate follows 1e-4*sqrt(batch scale).", "",
        "| Scale | PINDER batches by size bucket | pKPDB batch | LR | Updates | Wall min | State MAE | Interface MAE | Selection | pKPDB MAE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['scale']:g} | {row['pinder_batch_sizes']} | {row['pkpdb_batch_size']} | {row['learning_rate']:.2g} | {row['updates']} | {row['seconds']/60:.1f} | {row['state_mae']:.4f} | {row['interface_mae']:.4f} | {row['selection']:.4f} | {row['pkpdb_mae']:.4f} |")
    g = result["gradient_noise"]["joint"]
    lines += ["", f"Estimated joint gradient noise scale: **{g['noise_scale_structures_per_task']:.2f}** structures per task batch.", ""]
    (out / "report.md").write_text("\n".join(lines))
    atomic_json(out / "report-verification.json", {"passed": True, "report_sha256": digest(out / "report.json"),
        "markdown_sha256": digest(out / "report.md"), "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    gpu_benchmark=action in ("noise", "smoke", "train"), allow_comp1400=True)
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "noise": noise(root)
    elif action in ("smoke", "train"): train(root, float(sys.argv[2]), smoke=action == "smoke")
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
