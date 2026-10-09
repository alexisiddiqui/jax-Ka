"""Portable synthetic GQT training profile using the production mmap pipeline.

Example (from the repository root):
  JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python \
    benchmarks/profile_gqt_synthetic.py --output reports/gqt-synthetic-cpu \
    --data-root /private/tmp/gqt-synthetic-data

No real training data are read. Cold compilation, synchronized attribution,
normal prefetched training, validation and instrumented tracing are separate.
"""
import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import time

import jax
import numpy as np

from jaxpropka.parameters import GROUPS
from pkanet.model import PKPDB_PK_MOD, attend, encode, initialize, linear
from pkabench.runtime import atomic_json, digest
from pkatrain.graph_batches import BatchLoader, LoaderTelemetry, epoch_batches
from pkatrain.graph_data import bucket, load, mask_features
from pkatrain.graph_experiment import evaluate
from pkatrain.graph_mmap import GraphMMap, build_bundle
from pkatrain.graph_pkmod_compare import ExplicitShiftEngine
from pkatrain.trainer import sample_epoch
from profile_gqt_fullgraph import quantiles, ready, summarize_telemetry
from profile_gqt_mmap import compare_tree


ARCHITECTURES = {"50k": (44, 88, 49709), "200k": (92, 184, 209645)}
CAPACITIES = {"384": [384, 64, 128], "768": [768, 96, 256],
              "100000": [1024, 128, 384]}


def prepare(root, seed, per_bucket, batch_size, context_probability):
    """Create valid index/mask/dtype arrays and learnable synthetic targets."""
    specification = dict(seed=seed, per_bucket=per_bucket, batch_size=batch_size,
                         context_probability=context_probability, capacities=CAPACITIES,
                         format="synthetic-gqt-profile-v1")
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["synthetic_specification"] != specification:
            raise ValueError("Existing synthetic data has a different specification; use another data root")
        return manifest
    rng = np.random.default_rng(seed)
    baseline = np.asarray(PKPDB_PK_MOD)
    groups_available = np.asarray([0, 1, 2, 3, 4, 5, 7, 8], np.int32)
    records, contexts, sums, counts = [], {}, np.zeros(9), np.zeros(9)
    offset = 0
    for name, (nc, kc, qc) in CAPACITIES.items():
        for i in range(per_bucket + 3):
            split = "train" if i < per_bucket else "val"
            cid = f"synthetic-{name}-{i:04d}"
            n = int(rng.integers(max(32, int(nc * .78), 769 if name == "100000" else 385 if name == "768" else 1), nc + 1))
            k = int(rng.integers(kc - 24, kc + 1))
            q = int(rng.integers(qc * 2 // 3, qc + 1))
            aa = rng.integers(0, 20, n)
            nodes = np.concatenate((np.eye(20, dtype=np.float32)[aa], np.zeros((n, 3), np.float32), np.ones((n, 1), np.float32)), axis=1)
            nodes[0, 20] = 1; nodes[-1, 21] = 1
            neighbors = rng.integers(0, n, (n, k), dtype=np.int32)
            neighbors[:, 0] = np.arange(n)
            distances = rng.uniform(0, 20, (n, k)).astype(np.float32)
            directions = rng.normal(size=(n, k, 3)).astype(np.float32)
            directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
            radial = np.exp(-((distances[..., None] - np.linspace(0, 20, 16, dtype=np.float32)) / 1.5)**2)
            same_chain = (neighbors // max(n // 2, 1) == np.arange(n)[:, None] // max(n // 2, 1))
            edge = np.concatenate((radial, directions, same_chain[..., None]), axis=-1).astype(np.float32)
            edge_mask = rng.random((n, k)) < .9; edge_mask[:, 0] = True
            switch = np.where(distances < 18, 1., .5 * (1 + np.cos(np.pi * np.clip((distances - 18) / 2, 0, 1)))).astype(np.float32)
            query_residue = rng.choice(n, q, replace=False).astype(np.int32)
            query_group = rng.choice(groups_available, q)
            shift = (.7 * np.sin(aa[query_residue]) + .3 * (query_group % 3 - 1) + rng.normal(0, .1, q)).astype(np.float32)
            labels = (baseline[query_group] + shift).astype(np.float32)
            folder = root / "data" / cid; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "graph.npz", nodes=nodes, node_mask=np.ones(n, bool),
                neighbors=neighbors, edge=edge, edge_mask=edge_mask, switch=switch,
                query_residue=query_residue, query_group=query_group, labels=labels)
            records.append(dict(complex_id=cid, component_id=cid, split=split,
                                n=n, k=k, q=q, sha256=digest(folder / "graph.npz")))
            contexts[cid] = dict(start=offset, stop=offset+n, protected=query_residue.tolist())
            offset += n
            if split == "train":
                np.add.at(sums, query_group, labels); np.add.at(counts, query_group, 1)
    context_path = root / "context"
    atomic_json(context_path / "plan.json", dict(residues=offset, structures=contexts))
    manifest = dict(synthetic_specification=specification, records=records, capacities=CAPACITIES,
        train=[r["complex_id"] for r in records if r["split"] == "train"],
        val=[r["complex_id"] for r in records if r["split"] == "val"],
        context_path=str(context_path.resolve()), context_plan_sha256=digest(context_path / "plan.json"),
        config=dict(seed=seed, batch_size=batch_size, learning_rate=.001,
                    strict_backbone=True, context_mask_probability=context_probability))
    means = np.divide(sums, counts, out=np.nan_to_num(baseline.astype(float)), where=counts > 0)
    atomic_json(root / "train_type_means.json", dict(zip(GROUPS, means.tolist())))
    atomic_json(root / "manifest.json", manifest)
    return manifest


def plans_for(manifest, seed):
    byid = {r["complex_id"]: r for r in manifest["records"]}
    records = [byid[cid] for cid in manifest["train"]]
    rng = np.random.default_rng(seed)
    return epoch_batches(sample_epoch(records, rng), byid, rng, manifest["config"]["batch_size"])


def loaders(root, manifest):
    return {name: BatchLoader(root, manifest, manifest["config"]["batch_size"], backend=name)
            for name in ("npz", "mmap")}


def check_equivalence(root, manifest, engine, params, state):
    pair = loaders(root, manifest); rows = []
    try:
        for loader in pair.values(): loader.set_epoch(1)
        for name in CAPACITIES:
            ids = [r["complex_id"] for r in manifest["records"] if r["split"] == "train" and bucket(r) == name][:manifest["config"]["batch_size"]]
            a = pair["npz"].load(ids); b = pair["mmap"].load(ids)
            host = compare_tree(a, b)
            if not host["exact"]: raise AssertionError((name, "host mismatch", host))
            a, b = ready(jax.device_put(a)), ready(jax.device_put(b))
            t = time.perf_counter(); left = ready(engine.batch_step(params, state, *a))
            cold = time.perf_counter() - t
            right = ready(engine.batch_step(params, state, *b))
            if not bool(left[3]) or not bool(right[3]): raise FloatingPointError("Nonfinite warm-up gradient")
            update = compare_tree(left, right)
            if update["max_absolute"] > 1e-5 or update["relative_l2"] > 5e-6:
                raise AssertionError((name, "update mismatch", update))
            fwd = compare_tree(ready(engine.batch_forward(params, a[0])), ready(engine.batch_forward(params, b[0])))
            if fwd["max_absolute"] > 1e-5: raise AssertionError((name, "forward mismatch", fwd))
            rows.append(dict(bucket=name, host=host, update=update, forward=fwd,
                             cold_compile_and_first_step_seconds=cold, loss=float(left[2])))
            print(json.dumps(dict(warmup_bucket=name, seconds=cold)), flush=True)
    finally:
        for loader in pair.values(): loader.close()
    return rows


def compiled_statistics(engine, params, state, batch):
    compiled = engine.batch_step.lower(params, state, *batch).compile()
    result = {}
    try:
        analysis = compiled.cost_analysis()
        result["cost"] = {k: float(v) for k, v in analysis.items()} if isinstance(analysis, dict) else analysis
    except (NotImplementedError, RuntimeError, AttributeError) as exc:
        result["cost_unavailable"] = str(exc)
    try:
        memory = compiled.memory_analysis()
        names = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes", "generated_code_size_in_bytes")
        result["memory"] = {k: int(getattr(memory, k)) for k in names if getattr(memory, k, None) is not None}
    except (NotImplementedError, RuntimeError, AttributeError) as exc:
        result["memory_unavailable"] = str(exc)
    return result


def attributed(root, manifest, engine, params, state, repeats):
    rows = []; pair = loaders(root, manifest)
    encoder = jax.jit(jax.vmap(encode, in_axes=(None, 0)))
    def query_only(p, graph, h):
        i = graph["query_residue"]
        query = h[i] + p["groups"][graph["query_group"]]
        z = attend(p["query"], query, h, graph["neighbors"][i], graph["edge"][i],
                   graph["edge_mask"][i], graph["switch"][i])
        return 8 * jax.numpy.tanh(linear(p["head"], z)[:, 0])
    query_head = jax.jit(jax.vmap(query_only, in_axes=(None, 0, 0)))
    try:
        for loader in pair.values(): loader.set_epoch(1)
        for name, cap in CAPACITIES.items():
            ids = [r["complex_id"] for r in manifest["records"] if r["split"] == "train" and bucket(r) == name][:manifest["config"]["batch_size"]]
            loads = {backend: [] for backend in pair}
            transfers, steps, forwards, encoders, queries = [], [], [], [], []
            warm = ready(jax.device_put(pair["mmap"].load(ids)))
            h = ready(encoder(params, warm[0])); ready(query_head(params, warm[0], h))
            p, s = params, state
            for _ in range(repeats):
                for backend, loader in pair.items():
                    t = time.perf_counter(); host = loader.load(ids); loads[backend].append(time.perf_counter()-t)
                t = time.perf_counter(); batch = ready(jax.device_put(host)); transfers.append(time.perf_counter()-t)
                t = time.perf_counter(); ready(engine.batch_forward(p, batch[0])); forwards.append(time.perf_counter()-t)
                t = time.perf_counter(); h = ready(encoder(p, batch[0])); encoders.append(time.perf_counter()-t)
                t = time.perf_counter(); ready(query_head(p, batch[0], h)); queries.append(time.perf_counter()-t)
                t = time.perf_counter(); p, s, loss = engine.audited_batch_update(p, s, *batch)
                ready((p, s)); steps.append(time.perf_counter()-t)
            row = dict(bucket=name, capacities=cap, host_load={k: quantiles(v) for k, v in loads.items()},
                transfer=quantiles(transfers), forward=quantiles(forwards),
                isolated_encoder=quantiles(encoders), isolated_query_head=quantiles(queries),
                fused_update=quantiles(steps),
                batch_bytes=sum(x.nbytes for x in jax.tree.leaves(host)),
                actual_nodes=int(host[0]["node_mask"].sum()), padded_nodes=host[0]["node_mask"].size,
                actual_edges=int(host[0]["edge_mask"].sum()), padded_edges=host[0]["edge_mask"].size,
                actual_queries=int(host[2].sum()), padded_queries=host[2].size,
                compiled=compiled_statistics(engine, params, state, batch))
            rows.append(row)
            print(json.dumps(dict(attributed_bucket=name, warm_step_median=row["fused_update"]["median"])), flush=True)
    finally:
        for loader in pair.values(): loader.close()
    return rows


def train_epochs(root, manifest, engine, initial_params, initial_state, backend, epochs, means):
    telemetry = LoaderTelemetry()
    t = time.perf_counter()
    loader = BatchLoader(root, manifest, manifest["config"]["batch_size"], backend=backend, telemetry=telemetry)
    open_seconds = time.perf_counter()-t
    params, state, rows = initial_params, initial_state, []
    try:
        for epoch in range(1, epochs + 1):
            plans = plans_for(manifest, manifest["config"]["seed"] + epoch)
            t = time.perf_counter(); loader.set_epoch(epoch); context_seconds = time.perf_counter()-t
            t = time.perf_counter(); losses = []; calls = []
            for _, batch in loader.iterate(plans):
                started = time.perf_counter()
                params, state, loss = engine.audited_batch_update(params, state, *batch)
                calls.append(time.perf_counter()-started); losses.append(loss)
            ready((params, state)); train_seconds = time.perf_counter()-t
            t = time.perf_counter(); metrics = evaluate(root, manifest, engine, params, means, replicates=2000)
            validation_seconds = time.perf_counter()-t
            rows.append(dict(epoch=epoch, updates=len(plans), seconds=train_seconds,
                context_seconds=context_seconds, validation_seconds=validation_seconds,
                train_plus_validation_seconds=train_seconds+validation_seconds+context_seconds,
                update_call=quantiles(calls), mean_loss=float(np.mean(losses)),
                sampled_structures=sum(map(len, plans)), loss_first=losses[0], loss_last=losses[-1],
                synthetic_validation_mae=metrics["graph_query"]["mae"]))
            print(json.dumps(dict(backend=backend, **rows[-1])), flush=True)
    finally:
        loader.close()
    delta = compare_tree(initial_params, params)
    if delta["max_absolute"] <= 0: raise AssertionError("Parameters did not change")
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((params, state))):
        raise FloatingPointError("Nonfinite final parameters or optimizer state")
    optimizer_updates = int(np.asarray(state[1][0].count))
    expected_updates = sum(row["updates"] for row in rows)
    if optimizer_updates != expected_updates:
        raise AssertionError(("Adam state was not carried across steps", optimizer_updates, expected_updates))
    return dict(backend=backend, loader_open_seconds=open_seconds, epochs=rows,
                loader=summarize_telemetry(telemetry.snapshot()), parameters_changed=delta,
                optimizer_updates=optimizer_updates,
                final_parameter_checksum=float(sum(np.asarray(x, dtype=float).sum() for x in jax.tree.leaves(params))))


def instrumented_epoch(root, manifest, engine, params, state, destination, trace):
    loader = BatchLoader(root, manifest, manifest["config"]["batch_size"], backend="mmap")
    loader.set_epoch(3); plans = plans_for(manifest, manifest["config"]["seed"] + 3)
    profiler = cProfile.Profile(); tracing = False
    try:
        if trace:
            jax.profiler.start_trace(str(destination / "trace")); tracing = True
        profiler.enable()
        for i, (_, batch) in enumerate(loader.iterate(plans)):
            with jax.profiler.StepTraceAnnotation("train", step_num=i):
                params, state, _ = engine.audited_batch_update(params, state, *batch)
        ready((params, state)); profiler.disable()
    finally:
        profiler.disable()
        if tracing: jax.profiler.stop_trace()
        loader.close()
    profiler.dump_stats(str(destination / "host-profile.pstats"))
    stream = io.StringIO(); pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(45)
    (destination / "host-profile.txt").write_text(stream.getvalue())


def write_summary(destination, result):
    lines = ["Synthetic GQT training profile", "", f"Device: {result['environment']['devices']}",
             "float32, highest matmul precision; batch 8 by default; production fused loss/gradient/clipped Adam.",
             "Synthetic graph features/labels are for runtime measurement, not scientific accuracy.",
             "Stage timings synchronize arrays; throughput uses production prefetch and scalar audits.",
             "Compilation, setup, equivalence and instrumentation excluded from warm epoch timings.",
             "Warm epoch comparisons exclude epoch 1 when a round has multiple epochs.", ""]
    for name, model in result["models"].items():
        medians = model["median_epoch_seconds"]
        lines += [f"{name}: {model['parameter_count']:,} parameters",
            f"  Train: NPZ {medians['npz']:.3f}s; mmap {medians['mmap']:.3f}s; reduction {model['train_time_reduction_fraction']:.1%}",
            f"  Train + validation: NPZ {model['median_total_seconds']['npz']:.3f}s; mmap {model['median_total_seconds']['mmap']:.3f}s", "  bucket / mmap host load / transfer / forward / fused training update (median ms):"]
        for backend, stats in model["warm_epoch_seconds"].items():
            lines.append(f"  {backend} warm epoch range: {stats['min']:.3f}--{stats['max']:.3f}s ({stats['n']} epochs)")
        for row in model["attributed"]:
            lines.append(f"  {row['bucket']:>6}: {1000*row['host_load']['mmap']['median']:.2f} / {1000*row['transfer']['median']:.2f} / {1000*row['forward']['median']:.2f} / {1000*row['fused_update']['median']:.2f}")
        for backend in ("npz", "mmap"):
            rounds = [r for r in model["rounds"] if r["backend"] == backend]
            waits = sum(r["loader"]["steady_wait"]["total"] for r in rounds)
            elapsed = sum(e["seconds"] for r in rounds for e in r["epochs"])
            lines.append(f"  {backend} steady loader wait: {waits/elapsed:.2%}")
        lines.append("")
    lines += ["All NPZ/mmap batches exact, finite gradients and changing parameters checked.",
              "ABBA backend order; each round starts at the same parameters/state and carries both across epochs.",
              "Validation retains production NPZ hashing/loading and 2,000 bootstrap replicates every epoch.",
              "Filesystem cache is warm; no OS cache flush. Synthetic size/padding is not the historical cohort.",
              "Loader per-structure totals overlap across workers; they are not additive wall-time attribution.",
              "Isolated encoder and query timings break fusion and are diagnostic, not additive fractions of the fused update.",
              "Host cProfile and optional JAX trace are instrumented extra runs, excluded from measured throughput."]
    (destination / "summary.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--structures-per-bucket", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--backend-cycles", type=int, default=1, help="number of matched NPZ/mmap/mmap/NPZ cycles")
    parser.add_argument("--context-probability", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.repeats, args.backend_cycles) < 1 or args.structures_per_bucket < args.batch_size:
        parser.error("Positive counts and at least one full batch per bucket required")
    if not 0 <= args.context_probability < 1: parser.error("context probability must be in [0,1)")
    if os.environ.get("SLURM_JOB_ID"):
        from pkabench.runtime import require_compute
        require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                        gpu_benchmark=jax.default_backend() in ("gpu", "cuda"), allow_comp1400=True)
    for name in ("PKATRAIN_GRAPH_MMAP_DIR", "PKATRAIN_GRAPH_MMAP_VERIFY"):
        if os.environ.get(name): parser.error(f"Unset {name} so the synthetic profile uses its own bundle")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=False, synthetic=True, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        environment=dict(python=platform.python_version(), platform=platform.platform(), jax=jax.__version__,
                         devices=list(map(str, jax.devices())), backend=jax.default_backend(),
                         mps_async_dispatch=os.environ.get("JAX_MPS_ASYNC_DISPATCH", "0")), models={})
    import optax
    result["environment"].update(numpy=np.__version__, optax=optax.__version__,
        cpu_count=os.cpu_count(), xla_flags=os.environ.get("XLA_FLAGS", ""))
    t = time.perf_counter(); manifest = prepare(args.data_root, args.seed, args.structures_per_bucket, args.batch_size, args.context_probability)
    result["prepare_seconds"] = time.perf_counter()-t
    t = time.perf_counter(); bundle = build_bundle(args.data_root, manifest)
    result["mmap"] = dict(path=str(bundle), build_seconds=time.perf_counter()-t,
                          receipt=json.loads((bundle / "verification.json").read_text()))
    result["manifest_sha256"] = digest(args.data_root / "manifest.json")
    result["code_sha256"] = {str(p): digest(p) for p in [Path(__file__), Path("src/pkanet/model.py"),
        Path("src/pkatrain/graph_pkmod_compare.py"), Path("src/pkatrain/graph_batches.py"), Path("src/pkatrain/graph_mmap.py")]}
    means = np.asarray([json.loads((args.data_root / "train_type_means.json").read_text())[g] for g in GROUPS])
    for name in args.models:
        width, ff, expected_count = ARCHITECTURES[name]
        engine = ExplicitShiftEngine([1, 1, 1, 1], manifest["config"]["learning_rate"])
        params = initialize(jax.random.PRNGKey(args.seed), width=width, ff=ff)
        count = sum(x.size for x in jax.tree.leaves(params)); assert count == expected_count
        state = engine.optimizer.init(params)
        ready((params, state))
        print(json.dumps(dict(model=name, parameters=count)), flush=True)
        checks = check_equivalence(args.data_root, manifest, engine, params, state)
        # Warm validation signatures independently of measured training epochs.
        for capacity in CAPACITIES:
            row = next(r for r in manifest["records"] if r["split"] == "val" and bucket(r) == capacity)
            graph, _, _ = load(args.data_root, row, CAPACITIES[capacity])
            ready(engine.forward(params, mask_features(graph, manifest["config"])))
        stages = attributed(args.data_root, manifest, engine, params, state, args.repeats)
        model = dict(parameter_count=count, architecture=dict(width=width, ff=ff), equivalence=checks, attributed=stages, rounds=[])
        result["models"][name] = model
        for backend in ("npz", "mmap", "mmap", "npz") * args.backend_cycles:
            model["rounds"].append(train_epochs(args.data_root, manifest, engine, params, state, backend, args.epochs, means))
            atomic_json(args.output / "profile.json", result)
        warm = {b: [e for r in model["rounds"] if r["backend"] == b for e in r["epochs"]
                    if args.epochs == 1 or e["epoch"] > 1] for b in ("npz", "mmap")}
        model["warm_epoch_seconds"] = {b: dict(quantiles([e["seconds"] for e in values]),
                                              min=min(e["seconds"] for e in values)) for b, values in warm.items()}
        model["median_epoch_seconds"] = {b: float(np.median([e["seconds"] for e in values])) for b, values in warm.items()}
        model["median_total_seconds"] = {b: float(np.median([e["train_plus_validation_seconds"] for e in values])) for b, values in warm.items()}
        model["train_time_reduction_fraction"] = 1 - model["median_epoch_seconds"]["mmap"] / model["median_epoch_seconds"]["npz"]
        trace_path = args.output / name; trace_path.mkdir(exist_ok=True)
        instrumented_epoch(args.data_root, manifest, engine, params, state, trace_path, args.trace)
        atomic_json(args.output / "profile.json", result)
    store = GraphMMap(bundle, manifest["records"], verify_files=True); store.close()
    result["mmap_unchanged_after_augmentation"] = True
    result["max_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    result["complete"] = True
    atomic_json(args.output / "profile.json", result); write_summary(args.output, result)
    print((args.output / "summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
