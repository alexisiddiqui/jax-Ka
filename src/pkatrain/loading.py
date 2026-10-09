"""Shared batch-loading protocol for production training (2026-10-09).

Every production loader is a `BatchSource`: `load(spec)` assembles one host batch from a plan entry (thread-safe),
`close()` releases stores, `provenance()` describes where the data came from. Plans (which ids form which batch, in
which order) come unchanged from the existing plan functions; this module never samples.

`Prefetcher` runs `load` (and an optional `to_device` transfer) ahead of the consumer in a bounded, in-order pipeline,
so assembly and host-to-device copies overlap the current step. `DeferredScalars` keeps per-step losses on the device
and materialises them in groups, so the loop does not synchronise on every loss value.

The code-hashed experiment loaders (gqt_paired_pinder, graph_batches, site_graph_data, trainer) are not modified; the
sources in loading_gqt / loading_torch wrap their stores.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Iterator, Protocol, runtime_checkable

WAIT_FRACTION_FLAG = 0.05  # experiment 29 criterion: loader waits above 5% of wall time are worth fixing


def _default_workers():
    return max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) - 1))


@dataclass(frozen=True)
class LoaderConfig:
    """workers: assembly threads inside a source; prefetch: batches in flight ahead of the consumer."""
    workers: int = field(default_factory=lambda: int(os.environ.get("PKATRAIN_LOADER_WORKERS", _default_workers())))
    prefetch: int = field(default_factory=lambda: int(os.environ.get("PKATRAIN_PREFETCH_DEPTH", 2)))
    record_waits: bool = True

    def __post_init__(self):
        if self.workers < 1 or self.prefetch < 1: raise ValueError(f"workers and prefetch must be >= 1: {self}")


@runtime_checkable
class BatchSource(Protocol):
    def load(self, spec: Any) -> Any: ...
    def close(self) -> None: ...
    def provenance(self) -> dict: ...


class LoaderTelemetry:
    """Per-batch assembly time (worker side) and consumer wait time (how long the step waited for data)."""
    def __init__(self):
        self._lock = Lock(); self.rows = []; self.started = time.perf_counter()

    def add(self, row):
        with self._lock: self.rows.append(row)

    def summary(self):
        with self._lock: rows = list(self.rows)
        elapsed = time.perf_counter() - self.started
        steady = [r["wait_seconds"] for r in rows[1:]]  # the first batch is pipeline start-up
        waited = sum(steady)
        return {"batches": len(rows), "elapsed_seconds": elapsed,
                "startup_wait_seconds": rows[0]["wait_seconds"] if rows else 0.0,
                "steady_wait_seconds": waited, "wait_fraction": waited / elapsed if elapsed > 0 else 0.0,
                "mean_assembly_seconds": sum(r["assembly_seconds"] for r in rows) / len(rows) if rows else 0.0,
                "max_wait_seconds": max(steady, default=0.0),
                "flag_input_bound": (waited / elapsed if elapsed > 0 else 0.0) > WAIT_FRACTION_FLAG}

    def write(self, path):
        path = Path(path); pending = path.with_suffix(path.suffix + ".pending")
        pending.write_text(json.dumps({"summary": self.summary(), "batches": self.rows}))
        os.replace(pending, path)


class Prefetcher:
    """In-order bounded pipeline: yields (spec, batch) with up to `config.prefetch` batches loading ahead.

    to_device(batch) runs in the prefetch thread (e.g. jax.device_put, or a pinned non_blocking Torch copy);
    on_consume(value) runs on the consumer thread just before yielding (e.g. make the compute stream wait on a copy
    event). Worker exceptions are re-raised at the consumer; leaving the loop early cancels pending work."""

    def __init__(self, source: BatchSource, specs: Iterable, config: LoaderConfig | None = None,
                 to_device: Callable | None = None, on_consume: Callable | None = None, telemetry: LoaderTelemetry | None = None):
        self.source = source; self.specs = list(specs); self.config = config or LoaderConfig()
        self.to_device = to_device; self.on_consume = on_consume
        self.telemetry = telemetry if telemetry is not None else (LoaderTelemetry() if self.config.record_waits else None)

    def _task(self, spec):
        began = time.perf_counter(); batch = self.source.load(spec)
        if self.to_device is not None: batch = self.to_device(batch)
        return batch, time.perf_counter() - began

    def __iter__(self) -> Iterator:
        pool = ThreadPoolExecutor(max_workers=self.config.prefetch, thread_name_prefix="prefetch")
        pending = deque(); upcoming = iter(enumerate(self.specs))
        try:
            for _ in range(self.config.prefetch):
                item = next(upcoming, None)
                if item is None: break
                pending.append((item, pool.submit(self._task, item[1])))
            while pending:
                (index, spec), future = pending.popleft()
                waited = time.perf_counter(); batch, assembly = future.result(); waited = time.perf_counter() - waited
                item = next(upcoming, None)
                if item is not None: pending.append((item, pool.submit(self._task, item[1])))
                if self.telemetry is not None:
                    self.telemetry.add({"batch": index, "wait_seconds": waited, "assembly_seconds": assembly})
                if self.on_consume is not None: batch = self.on_consume(batch)
                yield spec, batch
        finally:
            for _, future in pending: future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class BucketPolicy:
    """Residue-count size buckets and per-bucket batch sizes.

    LEGACY_PAIRED reproduces gqt_paired_pinder (_bucket_n bounds, manifest batch sizes, _plans). PRODUCTION extends the
    cap from 768 to 1536 residues so no pool structure is dropped (pool-v3 max: PINDER 1500, pKPDB 1053) and sizes
    batches by a residue budget (batch size = budget // bucket bound), so every batch carries a similar number of
    residues; the budget is provisional until a GPU memory check on each bucket's largest structure."""
    bounds: tuple = (128, 256, 384, 512, 640, 768)
    batch_sizes: tuple = (8, 8, 8, 4, 2, 2)

    def __post_init__(self):
        if len(self.bounds) != len(self.batch_sizes) or list(self.bounds) != sorted(set(self.bounds)):
            raise ValueError(f"invalid bucket policy: {self}")

    def bucket(self, n):
        for bound in self.bounds:
            if int(n) <= bound: return str(bound)
        raise ValueError(f"{n} residues exceeds the largest bucket ({self.bounds[-1]})")

    def batch_size(self, bucket):
        return self.batch_sizes[self.bounds.index(int(bucket))]

    def capacities(self, records, keys=("k", "q", "s", "sk"), quantum=32):
        """{bucket: [bound, *max(key) rounded up to `quantum`]} over the records in each bucket (as prepare() does)."""
        members = {}
        for row in records: members.setdefault(self.bucket(row["n"]), []).append(row)
        return {name: [int(name), *[int(-(-max(int(r[key]) for r in rows) // quantum) * quantum) for key in keys]]
                for name, rows in sorted(members.items(), key=lambda item: int(item[0]))}

    def plans(self, records, rng, id_key="id"):
        """Same algorithm as gqt_paired_pinder._plans: shuffle ids within each bucket, chunk, shuffle the batches."""
        groups = {}
        for row in records: groups.setdefault(self.bucket(row["n"]), []).append(row[id_key])
        plans = []
        for name, ids in groups.items():
            rng.shuffle(ids); size = self.batch_size(name)
            plans.extend(ids[i:i + size] for i in range(0, len(ids), size))
        rng.shuffle(plans); return plans

    def to_json(self):
        return {"bounds": list(self.bounds), "batch_sizes": list(self.batch_sizes)}

    @classmethod
    def from_residue_budget(cls, bounds, budget):
        return cls(tuple(bounds), tuple(max(1, int(budget) // int(bound)) for bound in bounds))

    @classmethod
    def from_json(cls, value):
        return cls(tuple(value["bounds"]), tuple(value["batch_sizes"]))


LEGACY_PAIRED = BucketPolicy()
PRODUCTION_BOUNDS = (128, 256, 384, 512, 640, 768, 1024, 1280, 1536)
PRODUCTION_RESIDUE_BUDGET = 3072  # = 8 x 384, the factorial's largest full batch; provisional pending GPU calibration
PRODUCTION = BucketPolicy.from_residue_budget(PRODUCTION_BOUNDS, PRODUCTION_RESIDUE_BUDGET)  # 24,12,8,6,4,4,3,2,2


class DeferredScalars:
    """Collect per-step device scalars (JAX arrays or Torch tensors) and convert them to floats in groups.

    `add` never synchronises; `flush` performs one transfer for everything pending (call every N steps and at the end
    of an epoch). Plain Python numbers pass straight through."""

    def __init__(self, every: int = 50):
        self.every = int(every); self.pending = []; self.values = []

    def add(self, value):
        self.pending.append(value)
        if len(self.pending) >= self.every: self.flush()

    def flush(self):
        if not self.pending: return self.values
        first = self.pending[0]
        if hasattr(first, "detach"):  # torch
            import torch
            values = torch.stack([v.detach().reshape(()) for v in self.pending]).cpu().tolist()
        elif hasattr(first, "block_until_ready") or type(first).__module__.startswith("jax"):
            import jax
            import numpy as np
            values = [float(v) for v in np.asarray(jax.device_get(self.pending), dtype=float).reshape(-1)]
        else:
            values = [float(v) for v in self.pending]
        self.values.extend(values); self.pending = []
        return self.values

    def mean(self):
        values = self.flush(); return sum(values) / len(values) if values else float("nan")

    def reset(self):
        self.flush(); values = self.values; self.values = []; return values
