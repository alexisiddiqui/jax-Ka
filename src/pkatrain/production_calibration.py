"""GPU memory calibration of the production GQT batch sizes (2026-10-10).

Padded shapes are fixed per size bucket (production_loading manifest capacities), so peak memory depends only on the
bucket and the batch size. For every dataset and bucket, one training gradient step of gqt_multitask_replay.JointEngine
(the joint pKPDB + PINDER engine: PINDER AB/free state + binding shift, pKPDB state shift; current backbone oGQT,
width 44 / ff 88) is compiled and run in a fresh process (JAX peak memory is per process) at two batch sizes; a straight
line through the two peaks predicts the largest batch under `fraction` of the device memory, which is then verified with
its own run. Steps are timed after compilation (median of 3).

  python -m pkatrain.production_calibration run [fraction]     -> <runtime>/training/gqt-production-v1/calibration.json
  python -m pkatrain.production_calibration sweep              -> calibration-sweep.json (batch 4..64, every bucket)
  python -m pkatrain.production_calibration measure DATASET BUCKET BATCH   (one measurement; prints JSON)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json
from .loading import BucketPolicy
from .production_graphs import output, read
from .production_loading import MANIFEST, PinderSource, PkpdbSource, normalization

ARCHITECTURE = {"width": 44, "ff": 88}
MAX_BATCH = 96


def measure(root, dataset, bucket, batch):
    import jax
    import jax.numpy as jnp
    from pkanet.ogqt import initialize as initialize_ogqt
    from .gqt_multitask_replay import JointEngine
    root = Path(root); manifest = read(output(root, dataset) / MANIFEST)
    policy = BucketPolicy.from_json(manifest["bucket_policy"])
    members = sorted((r for r in manifest["records"] if policy.bucket(r["n"]) == bucket), key=lambda r: -r["n"])
    if not members: return {"dataset": dataset, "bucket": bucket, "batch": batch, "skipped": "empty bucket"}
    ids = [members[i % len(members)]["id"] for i in range(min(batch, len(members)))]
    manifest = {**manifest, "bucket_policy": BucketPolicy(policy.bounds, (batch,) * len(policy.bounds)).to_json()}
    source = PinderSource(manifest, norms=normalization(manifest)) if dataset == "pinder" else PkpdbSource(manifest)
    host = source.load(ids); source.close()
    params = initialize_ogqt(jax.random.PRNGKey(17), **ARCHITECTURE); engine = JointEngine(params)
    state = engine.optimizer.init(params); rate = jnp.asarray(1e-4, jnp.float32)
    if dataset == "pinder":
        graphs, targets, mask, wb, wi, _, valid = host; device = jax.device_put((graphs, targets, mask, wb, wi, valid))
        def step(p, s):
            (loss, _), gradient = engine.paired_value_grad(p, *device)
            p, s, finite = engine.apply(p, s, gradient, jax.tree.map(jnp.zeros_like, gradient), rate); return p, s, loss
    else:
        device = jax.device_put(host)
        def step(p, s):
            loss, gradient = engine.pkpdb_value_grad(p, *device)
            p, s, finite = engine.apply(p, s, jax.tree.map(jnp.zeros_like, gradient), gradient, rate); return p, s, loss
    began = time.time(); params, state, loss = step(params, state); jax.block_until_ready(loss); compile_seconds = time.time() - began
    times = []
    for _ in range(3):
        began = time.time(); params, state, loss = step(params, state); jax.block_until_ready(loss); times.append(time.time() - began)
    stats = jax.devices()[0].memory_stats() or {}
    capacity = manifest["capacities"][bucket]
    return {"dataset": dataset, "bucket": bucket, "batch": batch, "capacity": capacity, "real_structures": len(ids),
            "peak_bytes": int(stats.get("peak_bytes_in_use", -1)), "limit_bytes": int(stats.get("bytes_limit", -1)),
            "compile_seconds": round(compile_seconds, 1), "step_seconds": round(float(np.median(times)), 4),
            "loss_finite": bool(np.isfinite(float(loss))), "device": str(jax.devices()[0].device_kind)}


def _run(dataset, bucket, batch):
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    result = subprocess.run([sys.executable, "-m", "pkatrain.production_calibration", "measure", dataset, bucket, str(batch)],
                            capture_output=True, text=True, env=env)
    lines = [l for l in result.stdout.splitlines() if l.startswith("{")]
    if result.returncode != 0 or not lines:
        oom = "RESOURCE_EXHAUSTED" in result.stderr or "out of memory" in result.stderr.lower()
        return {"dataset": dataset, "bucket": bucket, "batch": batch, "failed": "oom" if oom else result.stderr[-800:]}
    value = json.loads(lines[-1]); print(json.dumps(value), flush=True); return value


def run(root, fraction=0.85):
    root = Path(root); results = {"fraction": fraction, "architecture": ARCHITECTURE, "measurements": [], "batch_sizes": {}}
    for dataset in ("pinder", "pkpdb"):
        manifest = read(output(root, dataset) / MANIFEST); results["batch_sizes"][dataset] = {}
        for bucket in manifest["capacities"]:
            small = _run(dataset, bucket, 1); results["measurements"].append(small)
            large = _run(dataset, bucket, 4); results["measurements"].append(large)
            if "peak_bytes" not in small or "peak_bytes" not in large:
                results["batch_sizes"][dataset][bucket] = {"batch": 1, "note": "measurement failed", "large": large}; continue
            slope = max((large["peak_bytes"] - small["peak_bytes"]) / 3, 1); base = small["peak_bytes"] - slope
            target = fraction * small["limit_bytes"]; chosen = int(max(1, min(MAX_BATCH, (target - base) // slope)))
            check = _run(dataset, bucket, chosen); results["measurements"].append(check)
            while ("failed" in check or check["peak_bytes"] > target) and chosen > 1:
                chosen = max(1, int(chosen * 0.8)); check = _run(dataset, bucket, chosen); results["measurements"].append(check)
            results["batch_sizes"][dataset][bucket] = {"batch": chosen, "peak_GB": round(check.get("peak_bytes", 0) / 1e9, 1),
                "limit_GB": round(small["limit_bytes"] / 1e9, 1), "per_structure_GB": round(slope / 1e9, 2),
                "step_seconds": check.get("step_seconds"), "structures_per_second": round(chosen / check["step_seconds"], 1) if check.get("step_seconds") else None,
                "batch_1_step_seconds": small["step_seconds"], "batch_4_step_seconds": large["step_seconds"]}
            print(json.dumps({dataset: {bucket: results["batch_sizes"][dataset][bucket]}}), flush=True)
            atomic_json(output(root, "pinder").parent / "calibration.json", results)
    return results


def sweep(root, batches=(4, 8, 16, 32, 64)):
    """Throughput and peak memory per bucket and batch size (the model is small enough that memory rarely binds;
    the batch size is then an optimisation and throughput choice)."""
    root = Path(root); results = {"architecture": ARCHITECTURE, "batches": list(batches), "rows": []}
    for dataset in ("pinder", "pkpdb"):
        manifest = read(output(root, dataset) / MANIFEST)
        for bucket in manifest["capacities"]:
            for batch in batches:
                row = _run(dataset, bucket, batch)
                if "step_seconds" in row: row["structures_per_second"] = round(batch / row["step_seconds"], 1)
                results["rows"].append(row)
                atomic_json(output(root, "pinder").parent / "calibration-sweep.json", results)
                if "failed" in row: break
    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(os.environ["PKABENCH_RUNTIME"])
    if argv[0] == "measure": print(json.dumps(measure(root, argv[1], argv[2], int(argv[3]))))
    elif argv[0] == "run": run(root, float(argv[1]) if len(argv) > 1 else 0.85)
    elif argv[0] == "sweep": sweep(root)
    else: raise ValueError(argv[0])


if __name__ == "__main__":
    main()
