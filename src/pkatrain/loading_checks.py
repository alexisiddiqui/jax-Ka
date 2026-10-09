"""Equivalence and throughput checks for pkatrain.loading (old loader path vs the shared protocol), 2026-10-09.

  paired-equivalence (CPU): PairedSource batches are bit-identical to gqt_paired_pinder.Loader.batch for a fixed plan.
  paired-timing (GPU):      same plan, initial parameters and learning rate through the old loop (Loader + _prefetched +
                            PairedEngine.update) and the new one (PairedSource + Prefetcher(device_put) + JaxStepRunner +
                            DeferredScalars); losses must match; reports steps/s and loader wait fraction.
  torch (GPU):              experiment 48 data path on synthetic packed arrays: old per-step indexing/transfer loop vs
                            PackedSiteSource (resident and streaming) + Prefetcher; losses must match; reports steps/s.
Writes <runtime>/audits/loader-protocol-v1/<action>.json.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json

OUT = "audits/loader-protocol-v1"


def _factorial(root):
    from .gqt_paired_pinder import experiment_root, read
    base = experiment_root(root); manifest = read(base / "manifest.json")
    return base, manifest, [row for row in manifest["records"] if row["split"] == "train"]


def paired_equivalence(root, batches=60):
    from .gqt_paired_pinder import Loader, _plans
    from .loading import LoaderConfig
    from .loading_gqt import PairedSource
    base, manifest, train = _factorial(root)
    plan = _plans(train, np.random.default_rng(17), manifest)[:batches]
    old = Loader(base, manifest); new = PairedSource(manifest, os.environ.get("PKATRAIN_PAIRED_MMAP", base / "mmap-v1"), LoaderConfig(workers=4))
    import jax
    for ids in plan:
        a = jax.tree.leaves(old.batch(ids)); b = jax.tree.leaves(new.load(ids))
        if len(a) != len(b) or not all(x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b)):
            raise AssertionError(f"batch differs: {ids}")
    old.close(); new.close()
    return {"passed": True, "batches": len(plan), "bit_identical": True}


def paired_timing(root, batches=200, warmup=10):
    import jax
    from pkanet.ogqt import initialize as initialize_ogqt
    from .gqt_paired_pinder import Loader, PairedEngine, _plans, _prefetched
    from .loading import DeferredScalars, LoaderConfig, Prefetcher
    from .loading_gqt import JaxStepRunner, PairedSource, paired_to_device
    base, manifest, train = _factorial(root)
    plan = _plans(train, np.random.default_rng(17), manifest)[:batches]; arch = manifest["architecture"]
    initial = initialize_ogqt(jax.random.PRNGKey(17), width=arch["width"], ff=arch["ff"])
    rate = 1e-4; results = {}
    # old loop
    engine = PairedEngine(initial, "both"); params, state = initial, engine.optimizer.init(initial)
    loader = Loader(base, manifest); losses = []; began = None
    for number, (_, batch) in enumerate(_prefetched(loader, plan), 1):
        if number == warmup + 1: began = time.perf_counter()
        params, state, loss = engine.update(params, state, batch, rate); losses.append(loss)
    jax.block_until_ready(params); old_seconds = time.perf_counter() - began; loader.close()
    results["old"] = {"steps_per_second": (len(plan) - warmup) / old_seconds}
    # new loop
    engine = PairedEngine(initial, "both"); runner = JaxStepRunner(engine); params, state = initial, engine.optimizer.init(initial)
    config = LoaderConfig(); source = PairedSource(manifest, os.environ.get("PKATRAIN_PAIRED_MMAP", base / "mmap-v1"), config)
    deferred = DeferredScalars(every=50); prefetcher = Prefetcher(source, plan, config, paired_to_device); began = None
    for number, (_, batch) in enumerate(prefetcher, 1):
        if number == warmup + 1: began = time.perf_counter()
        params, state, loss = runner.paired(params, state, batch, rate); deferred.add(loss)
    jax.block_until_ready(params); new_seconds = time.perf_counter() - began; new_losses = deferred.flush(); source.close()
    results["new"] = {"steps_per_second": (len(plan) - warmup) / new_seconds, "workers": config.workers, "prefetch": config.prefetch,
                      "loader": prefetcher.telemetry.summary()}
    difference = float(np.max(np.abs(np.asarray(losses) - np.asarray(new_losses))))
    results.update(batches=len(plan), warmup=warmup, max_abs_loss_difference=difference, passed=difference <= 1e-6,
                   speedup=results["new"]["steps_per_second"] / results["old"]["steps_per_second"])
    return results


def torch_check(root, rows=60000, steps=300, warmup=10):
    from .pkai_joint_scale import BATCH_SIZE, LEARNING_RATE, SEED, _epoch_specs, _weighted_mse
    from .pkai_scratch import model_class, native
    from .loading import DeferredScalars, LoaderConfig, Prefetcher
    from .loading_torch import CombinedSource, PackedSiteSource
    torch, _ = native(); torch.backends.cuda.matmul.allow_tf32 = False
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))); rng = np.random.default_rng(5)
    def array(name, shape):
        value = np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, shape[0], 5000): value[start:start + 5000] = rng.standard_normal((min(5000, shape[0] - start),) + shape[1:], dtype=np.float32) * 0.1
        value.flush(); return np.load(tmp / f"{name}.npy", mmap_mode="r")
    pk = {"x": array("pkx", (rows, 4008)), "y": array("pky", (rows,)), "w": np.abs(array("pkw", (rows,))) + 0.4}
    pi = {k: array(f"pi{k}", (rows, 4008) if k.startswith("x") else (rows,)) for k in ("xa", "xf", "ya", "yf", "wb", "wi")}
    pi["wb"] = np.abs(pi["wb"]) + 0.4; pi["wi"] = np.abs(pi["wi"]) + 0.05
    order = np.random.default_rng(SEED); specs = _epoch_specs(order.permutation(rows), order.permutation(rows))[:steps]
    def step(model, opt, b):
        lpk = _weighted_mse(torch, model(b["pk"]["x"]), b["pk"]["y"], b["pk"]["w"]); q = b["pi"]; pa = model(q["xa"]); pf = model(q["xf"])
        loss = (lpk + (_weighted_mse(torch, pa, q["ya"], q["wb"]) + _weighted_mse(torch, pf, q["yf"], q["wb"])) / 2 + _weighted_mse(torch, pa - pf, q["ya"] - q["yf"], q["wi"])) / 3
        opt.zero_grad(set_to_none=True); loss.backward(); return loss
    def fresh():
        torch.manual_seed(SEED); model = model_class(torch)().cuda().train()
        return model, torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    results = {"rows": rows, "steps": len(specs), "batch_size": BATCH_SIZE}
    # old: per-array synchronous transfers, per-parameter finite checks, float(loss) each step
    model, opt = fresh(); old_losses = []; began = None
    for number, spec in enumerate(specs, 1):
        if number == warmup + 1: torch.cuda.synchronize(); began = time.perf_counter()
        b = {"pk": {k: torch.tensor(np.asarray(v[spec["pk"]]), device="cuda") for k, v in pk.items()},
             "pi": {k: torch.tensor(np.asarray(v[spec["pi"]]), device="cuda") for k, v in pi.items()}}
        loss = step(model, opt, b)
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError
        opt.step(); old_losses.append(float(loss.detach()))
    torch.cuda.synchronize(); results["old"] = {"steps_per_second": (len(specs) - warmup) / (time.perf_counter() - began)}
    for mode in ("resident", "stream"):
        model, opt = fresh(); params = list(model.parameters()); config = LoaderConfig(workers=2, prefetch=3)
        source = CombinedSource({"pk": PackedSiteSource(pk, mode=mode, config=config), "pi": PackedSiteSource(pi, mode=mode, config=config)})
        deferred = DeferredScalars(every=50); prefetcher = Prefetcher(source, specs, config, source.to_device, source.on_consume); began = None
        for number, (_, b) in enumerate(prefetcher, 1):
            if number == warmup + 1: torch.cuda.synchronize(); began = time.perf_counter()
            loss = step(model, opt, b)
            finite = torch.isfinite(loss) & torch.stack([torch.isfinite(p.grad).all() for p in params]).all()
            if not bool(finite): raise FloatingPointError
            opt.step(); deferred.add(loss.detach())
        torch.cuda.synchronize(); seconds = time.perf_counter() - began; source.close(); new_losses = deferred.flush()
        difference = float(np.max(np.abs(np.asarray(old_losses) - np.asarray(new_losses))))
        results[mode] = {"steps_per_second": (len(specs) - warmup) / seconds, "max_abs_loss_difference": difference,
                         "identical": difference == 0.0, "loader": prefetcher.telemetry.summary()}
    results["passed"] = all(results[m]["max_abs_loss_difference"] <= 1e-6 for m in ("resident", "stream"))
    return results


def main():
    root = Path(os.environ["PKABENCH_RUNTIME"]); action = sys.argv[1]
    result = {"paired-equivalence": paired_equivalence, "paired-timing": paired_timing, "torch": torch_check}[action](root)
    out = root / OUT; out.mkdir(parents=True, exist_ok=True); atomic_json(out / f"{action}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "loader"}, default=str), flush=True)
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
