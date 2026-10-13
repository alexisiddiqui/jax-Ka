"""Materialising host-read and synchronised GPU preflights; never updates weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_store import AugmentationStore
from .augmentation_graphs import INPUT_FIELDS
from .loading import LoaderConfig, Prefetcher


class PilotSource:
    def __init__(self, path, workers, batch, *, pad_transform=None):
        # Full verification is a preflight prerequisite; the hot path follows
        # ProductionStore's one-read/one-decompression policy without rehashing.
        self.store = AugmentationStore(path, verify_reads=False); self.pool = ThreadPoolExecutor(workers); self.batch = batch
        self.pad_transform = pad_transform
        alignment = json.loads((Path(path)/"graph-alignment.json").read_text())
        self.capacities = alignment["capacities"]
        if alignment.get("manifest_sha256") != digest(Path(path)/"manifest.json"):
            self.store.close(); self.pool.shutdown(); raise ValueError("capacities belong to a different graph release")
        self.native_capacities = {}
        self.by_bucket = {}
        for row in self.store.rows:
            meta = row["summary"]; n = meta["dims"][0]
            bucket = next(k for k in sorted(self.capacities, key=int) if n <= int(k))
            self.by_bucket.setdefault(bucket, {}).setdefault(meta["source_parent"], {})[meta["candidate"]] = row["id"]
            if meta["kind"] == "native":
                dims = [int(bucket), *meta["dims"][1:]]
                self.native_capacities[bucket] = [max(a, b) for a, b in zip(self.native_capacities.get(bucket, [0]*5), dims)]

    def plans(self, bucket, kind, steps):
        parents = sorted(self.by_bucket[bucket]); plans = []
        for step in range(steps):
            ids = []
            for slot in range(self.batch):
                parent = parents[(step*self.batch+slot) % len(parents)]
                choices = self.by_bucket[bucket][parent]
                candidates = sorted(c for c in choices if c > 0)
                candidate = 0 if kind == "native" else candidates[(step+slot) % len(candidates)]
                ids.append(choices[candidate])
            plans.append((bucket, ids))
        return plans

    def load(self, spec):
        bucket, ids = spec
        native = all(self.store.by_id[cid]["summary"]["kind"] == "native" for cid in ids)
        n, k, q, s, sk = (self.native_capacities if native else self.capacities)[bucket]
        shapes = dict(nodes=(n, 38), node_mask=(n,), neighbors=(n, k), edge=(n, k, 20),
            edge_mask=(n, k), switch=(n, k), query_residue=(q,), query_group=(q,),
            site_residue=(s,), site_type=(s,), site_mask=(s,), site_neighbors=(s, sk),
            site_edge=(s, sk, 34), site_edge_mask=(s, sk), site_switch=(s, sk), query_site=(q,))
        integer = {"neighbors", "query_residue", "query_group", "site_residue", "site_type", "site_neighbors", "query_site"}
        graph = {name: np.zeros((len(ids), 2, *shapes[name]),
                 np.bool_ if name.endswith("mask") else np.int32 if name in integer else np.float32)
                 for name in INPUT_FIELDS}
        mask = np.zeros((len(ids), q), bool)
        def one(slot):
            record = self.store.raw(ids[slot]); arrays = record["arrays"]
            for name, value in arrays.items():
                if name in graph:
                    graph[name][(slot, *(slice(0, size) for size in value.shape))] = value
                else:
                    # Materialise and touch non-model sidechain bytes too; no index-only timing.
                    side = np.array(value, copy=True)
                    if not np.isfinite(side).all(): raise ValueError("nonfinite sidechain payload")
            mask[slot, :record["metadata"]["dims"][2]] = arrays["query_site"][0] >= 0
            return sum(a.nbytes for a in arrays.values())
        touched = sum(self.pool.map(one, range(len(ids))))
        if self.pad_transform is not None: graph = self.pad_transform(graph)
        return graph, mask, touched

    def provenance(self): return self.store.manifest
    def close(self): self.pool.shutdown(); self.store.close()


def read_benchmark(path, output, *, workers=(8, 16, 32), prefetch=(2, 4), steps=20, repeats=3, batch=4):
    results = []
    for w in workers:
        source = PilotSource(path, w, batch)
        try:
            for depth in prefetch:
                for bucket in sorted(source.capacities, key=int):
                    for kind in ("native", "augmented"):
                        for repeat in range(repeats):
                            plans = source.plans(bucket, kind, steps); config = LoaderConfig(workers=w, prefetch=depth)
                            stream = Prefetcher(source, plans, config); started = time.perf_counter(); touched = 0
                            checksum = 0.0
                            for _, (graph, _, count) in stream:
                                touched += count
                                checksum += sum(float(np.sum(a, dtype=np.float64)) for a in graph.values())
                            elapsed = time.perf_counter()-started
                            results.append(dict(workers=w, prefetch=depth, bucket=bucket, kind=kind, repeat=repeat,
                                cache="first observed pass; cache state uncontrolled" if not results else "subsequent pass; cache state uncontrolled",
                                structures=steps*batch, seconds=elapsed, structures_per_second=steps*batch/elapsed,
                                materialised_bytes=touched, checksum=checksum, telemetry=stream.telemetry.summary(),
                                capacity=(source.native_capacities if kind == "native" else source.capacities)[bucket]))
        finally: source.close()
    report = dict(kind="materialising-host-read-preflight-v1", manifest_sha256=digest(Path(path)/"manifest.json"),
                  results=results, peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"), infrastructure_ready=False,
                  pending=["effective Isambard quota", "synchronised Isambard GPU benchmark"],
                  note="Host-only timings cannot establish the 5% GPU loader-wait criterion; no cache flush attempted.")
    atomic_json(output, report); return report


def budget(snapshot, *, additional_bytes, additional_files, max_release_files=5000):
    """Fail closed when effective limits are unknown; snapshot includes project defaults."""
    reasons = []
    if additional_bytes < 0 or additional_files < 0: reasons.append("negative requested budget")
    if not snapshot.get("effective_limits_verified"): reasons.append("effective project/default limits unverified")
    if additional_files > max_release_files: reasons.append("release file-count cap exceeded")
    for quantity, added in (("bytes", additional_bytes), ("files", additional_files)):
        used = snapshot.get(f"used_{quantity}"); limit = snapshot.get(f"limit_{quantity}")
        if type(used) is not int or type(limit) is not int or used < 0 or limit <= 0:
            reasons.append(f"unknown effective {quantity} quota")
        elif used+added > int(limit*0.90): reasons.append(f"insufficient {quantity} quota headroom")
    return dict(passed=not reasons, reasons=reasons, additional_bytes=additional_bytes,
                additional_files=additional_files, reserve_fraction=0.10, snapshot=snapshot)


def gpu_benchmark(path, output, *, workers=(8, 16, 32), prefetch=(2, 4), steps=20, repeats=3, batch=256, protocol=None):
    import jax
    import jax.numpy as jnp
    from pkanet.site_model import initialize_site_auxiliary
    from pkanet.model import linear
    from .production_inputs import add_orientation, add_dihedral_inputs, input_functions
    from .production_depth import model_functions
    from .production_train import micro_size
    from .production_loading import spread_padding
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu": raise RuntimeError("preflight requires exactly one visible GPU")
    params = initialize_site_auxiliary(jax.random.PRNGKey(17), width=64, ff=128)
    params = add_dihedral_inputs(add_orientation(params, 17), 17)
    embeddings, _ = input_functions(model_functions(), orientation=True, dihedrals=True)
    def objective(p, graph, mask, key):
        b = graph["nodes"].shape[0]
        flat = jax.tree.map(lambda a: a.reshape((b*2, *a.shape[2:])), graph)
        keys = jnp.repeat(jax.random.split(key, b), 2, axis=0)
        def single(g, draw):
            tokens, query = embeddings(p, g, key=draw, dropout_rate=0.1)
            shifted = 8*jnp.tanh(linear(p["head"], tokens)[:, 0])[query]
            burial = linear(p["auxiliary"]["burial"], tokens)[:, 0][query]
            return shifted, burial
        prediction, burial = jax.vmap(single)(flat, keys)
        prediction = prediction.reshape(b, 2, -1); burial = burial.reshape(b, 2, -1)
        # Synthetic loss only: representative paired+burial compute, no new labels or updates.
        m = mask[:, None, :]
        return (jnp.sum(prediction**2*m) + jnp.sum(jax.nn.softplus(burial)*m) +
                jnp.sum((prediction[:, 0]-prediction[:, 1])**2*mask))/jnp.maximum(mask.sum(), 1)
    step = jax.jit(jax.value_and_grad(objective)); results = []; finite_probes = []
    for w in workers:
        source = PilotSource(path, w, batch, pad_transform=spread_padding)
        try:
            for bucket in sorted(source.capacities, key=int):
                micro = micro_size(batch, int(bucket))
                for kind in ("native", "augmented"):
                    spec_bucket, spec_ids = source.plans(bucket, kind, 1)[0]
                    graph, mask, _ = source.load((spec_bucket, spec_ids[:micro]))
                    device_graph = jax.device_put(jax.tree.map(lambda a: a[:micro], graph)); device_mask = jax.device_put(mask[:micro])
                    value, grads = step(params, device_graph, device_mask, jax.random.PRNGKey(17)); jax.block_until_ready((value, grads))
                    if not np.isfinite(float(value)) or not all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grads)):
                        raise ValueError("nonfinite forward/backward bucket probe")
                    finite_probes.append(dict(bucket=bucket, kind=kind, micro=micro, passed=True))
                    del graph, mask, device_graph, device_mask, grads, value
                for depth in prefetch:
                    for repeat in range(repeats):
                        # Alternate ordering to reduce systematic cache/time drift.
                        for kind in (("native", "augmented") if repeat % 2 == 0 else ("augmented", "native")):
                            stream = Prefetcher(source, source.plans(bucket, kind, steps), LoaderConfig(workers=w, prefetch=depth))
                            started = time.perf_counter(); touched = 0; losses = []
                            for _, (host_graph, host_mask, count) in stream:
                                touched += count
                                for start in range(0, batch, micro):
                                    g = jax.device_put(jax.tree.map(lambda a: a[start:start+micro], host_graph))
                                    m = jax.device_put(host_mask[start:start+micro])
                                    key = jax.random.fold_in(jax.random.PRNGKey(17), len(losses))
                                    value, grads = step(params, g, m, key); jax.block_until_ready((value, grads))
                                    losses.append(float(value)); del g, m, grads, value
                            elapsed = time.perf_counter()-started; telemetry = stream.telemetry.summary()
                            if not np.isfinite(losses).all(): raise ValueError("nonfinite timed objective")
                            results.append(dict(workers=w, prefetch=depth, bucket=bucket, kind=kind, repeat=repeat,
                                batch=batch, micro=micro, structures_per_second=steps*batch/elapsed, seconds=elapsed,
                                materialised_bytes=touched, telemetry=telemetry,
                                capacity=(source.native_capacities if kind == "native" else source.capacities)[bucket]))
        finally: source.close()
    comparisons = []
    for w in workers:
        for d in prefetch:
            buckets = sorted({r["bucket"] for r in results}, key=int)
            for b in buckets:
                groups = {k: [r for r in results if (r["workers"], r["prefetch"], r["bucket"], r["kind"]) == (w, d, b, k)] for k in ("native", "augmented")}
                ratio = float(np.median([r["structures_per_second"] for r in groups["augmented"]])/np.median([r["structures_per_second"] for r in groups["native"]]))
                wait = max(r["telemetry"]["wait_fraction"] for r in groups["augmented"])
                comparisons.append(dict(workers=w, prefetch=d, bucket=b, throughput_ratio=ratio,
                                        worst_wait_fraction=wait, passed=ratio >= 0.90 and wait <= 0.05))
    buckets = {r["bucket"] for r in comparisons}
    successful = [(w, d) for w in workers for d in prefetch if all(any(r["passed"] and
                  (r["workers"], r["prefetch"], r["bucket"]) == (w, d, b) for r in comparisons) for b in buckets)]
    report = dict(kind="synchronised-base-A-compute-preflight-v1", passed=bool(successful), viable_settings=successful,
        finite_probes=finite_probes, comparisons=comparisons, results=results, manifest_sha256=digest(Path(path)/"manifest.json"),
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2, gpu=str(devices[0]),
        device_memory_stats=devices[0].memory_stats(), job=os.environ.get("SLURM_JOB_ID"),
        production_protocol_sha256=digest(protocol) if protocol else None,
        padding="production_loading.spread_padding rotate; masked slots only",
        cache="Pilot-sized working set warmed by verification and compile probes; no cache flush; full-dataset random I/O unmeasured",
        infrastructure_ready=False, pending=["effective quota and total release staging budget"],
        note="Matched compute-only paired/burial proxy, width64 orientation+dihedrals dropout0.1; no labels, optimiser or saved weights.")
    atomic_json(output, report); return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reads", "gpu", "budget")); parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path); parser.add_argument("--workers", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--prefetch", type=int, nargs="+", default=[2, 4]); parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3); parser.add_argument("--batch", type=int)
    parser.add_argument("--protocol", type=Path, help="live base-A protocol; supplies the GPU batch size")
    parser.add_argument("--additional-bytes", type=int); parser.add_argument("--additional-files", type=int)
    args = parser.parse_args(argv)
    if args.command == "budget":
        if args.additional_bytes is None or args.additional_files is None: parser.error("budget requires additional bytes/files")
        result = budget(json.loads(args.input.read_text()), additional_bytes=args.additional_bytes, additional_files=args.additional_files)
        atomic_json(args.output, result)
    else:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        require_compute(threads=cpus, gpu_benchmark=args.command == "gpu")
        if max(args.workers) > cpus: parser.error("workers exceed allocated CPUs")
        if args.steps < 20 or args.repeats < 3: parser.error("preflight requires at least 20 steps and 3 repeats")
        extra = {}
        if args.command == "gpu":
            if args.protocol is None: parser.error("GPU preflight requires the live base-A --protocol")
            live = json.loads(args.protocol.read_text()); arch = live["architecture"]
            if arch.get("width") != 64 or arch.get("ff") != 128 or arch.get("orientation_edges") != 13 or \
                    arch.get("dihedral_features") != 6 or arch.get("node_dim", 24) != 24 or \
                    arch.get("depth", 2) != 2 or arch.get("loops", 1) != 1 or \
                    not live.get("dropout", "").startswith("oGQT residual dropout 0.1 "):
                parser.error("live protocol differs from the supported base-A compute probe")
            if args.batch is not None and args.batch != live["batch"]: parser.error("batch differs from production protocol")
            args.batch = live["batch"]; extra["protocol"] = args.protocol
        result = (gpu_benchmark if args.command == "gpu" else read_benchmark)(args.input, args.output,
            workers=args.workers, prefetch=args.prefetch, steps=args.steps, repeats=args.repeats,
            batch=args.batch or (256 if args.command == "gpu" else 4), **extra)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
