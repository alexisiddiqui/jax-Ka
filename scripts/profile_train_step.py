"""Profile production joint GQT training steps (pkatrain.production_train) on one GPU (2026-10-10).

Replays real training steps of the production loop: the epoch-1 plan of the given fraction and batch, Prefetcher +
JointSource, JointEngine paired/pKPDB gradients, apply, and the per-step finite sync. Every bucket shape is compiled
before the profiled window, so the window has no compilation. Only the window is captured (cuProfilerStart/Stop for
`nsys profile --capture-range=cudaProfilerApi`). Also writes, per compiled executable, the optimized HLO (to map
kernel names to JAX op names and source lines) and XLA's cost analysis (FLOPs, bytes accessed).

  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o OUT/profile \\
      python scripts/profile_train_step.py OUT [--batch 16] [--fraction 0.1] [--steps 60] [--skip 40]
"""
import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pkanet.ogqt import initialize as initialize_ogqt
from pkatrain.gqt_multitask_replay import JointEngine
from pkatrain.loading import LoaderConfig, Prefetcher
from pkatrain.production_graphs import output, read
from pkatrain.production_loading import MANIFEST, PinderSource, PkpdbSource, normalization
from pkatrain.production_train import ARCHITECTURE, SEED, JointSource, _with_policy, chunked, epoch_plans, lr_scale, to_device


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("out"); parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=0.1); parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--skip", type=int, default=40)
    args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if chunked(args.batch): raise SystemExit("profile the unchunked path (batch <= 64)")
    root = Path(os.environ["PKABENCH_RUNTIME"])
    jax.config.update("jax_compilation_cache_dir", str(root / "training/gqt-production-v1/runs/compilation-cache"))
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb")}
    norms = normalization(manifests["pinder"], args.fraction); config = LoaderConfig()
    source = JointSource(PinderSource(_with_policy(manifests["pinder"], args.batch), config=config, norms=norms),
                         PkpdbSource(_with_policy(manifests["pkpdb"], args.batch), config=config))
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **ARCHITECTURE); engine = JointEngine(params)
    state = engine.optimizer.init(params); scale = lr_scale(args.batch)
    pair, pk = epoch_plans(manifests, args.fraction, 1, args.batch); specs = list(zip(pair, pk))
    by_id = {d: {r["id"]: r for r in manifests[d]["records"]} for d in manifests}
    policy = {d: source.pinder.policy if d == "pinder" else source.pkpdb.policy for d in manifests}
    bucket_of = lambda d, ids: policy[d].bucket(by_id[d][ids[0]]["n"])

    def step(params, state, batch, rate):
        (pair_batch, pk_batch) = batch; graphs, targets, mask, wb, wi, _, valid = pair_batch
        (total, _), pair_gradient = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, valid)
        pk_loss, pk_gradient = engine.pkpdb_value_grad(params, *pk_batch)
        params, state, finite = engine.apply(params, state, pair_gradient, pk_gradient, rate)
        if not bool(finite): raise FloatingPointError("nonfinite")
        return params, state, total + pk_loss

    # Compile every bucket shape (one spec per PINDER bucket and per pKPDB bucket) and record HLO + cost analysis.
    warm = {}; rate = jnp.asarray(1e-4 * scale, jnp.float32)
    for p_ids, k_ids in specs:
        warm.setdefault(("pinder", bucket_of("pinder", p_ids)), (p_ids, k_ids)); warm.setdefault(("pkpdb", bucket_of("pkpdb", k_ids)), (p_ids, k_ids))
    costs = []; began = time.time()
    for (dataset, bucket), spec in sorted(warm.items()):
        batch = to_device(source.load(spec)); graphs, targets, mask, wb, wi, _, valid = batch[0]
        if dataset == "pinder": args_ = (params, graphs, targets, mask, wb, wi, valid); fn = engine.paired_value_grad
        else: args_ = (params, *batch[1]); fn = engine.pkpdb_value_grad
        compiled = fn.lower(*args_).compile()
        (out / f"hlo-{dataset}-{bucket}.txt").write_text(compiled.as_text())
        cost = compiled.cost_analysis(); cost = cost[0] if isinstance(cost, list) else cost
        jax.block_until_ready(fn(*args_)); times = []
        for _ in range(5):
            t = time.perf_counter(); jax.block_until_ready(fn(*args_)); times.append(time.perf_counter() - t)
        costs.append({"dataset": dataset, "bucket": bucket, "flops": float(cost.get("flops", -1)),
                      "bytes_accessed": float(cost.get("bytes accessed", -1)), "step_seconds": float(np.median(times))})
        print(json.dumps(costs[-1]), flush=True)
    (out / f"hlo-apply.txt").write_text(engine.apply.lower(params, state, params, params, rate).compile().as_text())
    print(json.dumps({"warmup_seconds": round(time.time() - began, 1)}), flush=True)

    window = specs[args.skip:args.skip + args.steps]
    cuda = ctypes.CDLL("libcuda.so.1")
    for profiled in (False, True):  # an unprofiled pass first: steady-state wall time without capture overhead
        jax.block_until_ready(params); began = time.perf_counter(); waits = []
        if profiled: cuda.cuProfilerStart()
        prefetcher = Prefetcher(source, window, config, to_device); p, s = params, state
        for _, batch in prefetcher: p, s, loss = step(p, s, batch, rate)
        jax.block_until_ready(loss)
        if profiled: cuda.cuProfilerStop()
        wall = time.perf_counter() - began; tele = prefetcher.telemetry.summary()
        result = {"profiled": profiled, "steps": len(window), "wall_seconds": wall, "seconds_per_step": wall / len(window),
                  "loader_wait_fraction": tele.get("wait_fraction"),
                  "pinder_buckets": [bucket_of("pinder", a) for a, _ in window], "pkpdb_buckets": [bucket_of("pkpdb", b) for _, b in window],
                  "pinder_structures": int(sum(len(a) for a, _ in window)), "pkpdb_structures": int(sum(len(b) for _, b in window))}
        print(json.dumps({k: v for k, v in result.items() if "buckets" not in k}), flush=True)
        (out / f"window-{'profiled' if profiled else 'plain'}.json").write_text(json.dumps(result, indent=1))
    (out / "costs.json").write_text(json.dumps({"batch": args.batch, "fraction": args.fraction, "rows": costs,
        "device": jax.devices()[0].device_kind, "jax": jax.__version__}, indent=1))
    source.close()


if __name__ == "__main__":
    main()
