"""BatchSource implementations for the JAX/GQT loaders (see pkatrain.loading).

These wrap the existing, code-hashed stores and assembly functions without modifying them:
- PairedSource: PINDER paired batches from gqt_paired_pinder.PairedMMap via _load_one (same arrays as Loader.batch);
- LegacyLoaderSource: any graph_batches.BatchLoader / CutoffBatchLoader / site_graph_data.SiteBatchLoader, with its
  fixed 4-thread assembly pool replaced by a `workers`-sized pool on the instance.
JaxStepRunner calls the engines' jitted step functions directly, keeping losses on the device (no float() per step);
it keeps the engines' finite-update check as one boolean synchronisation per step.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .gqt_paired_pinder import PairedMMap, _load_one
from .loading import LEGACY_PAIRED, BucketPolicy, LoaderConfig


class PairedSource:
    """spec: list of PINDER ids in one size bucket (gqt_paired_pinder._plans or BucketPolicy.plans).

    The bucket policy comes from manifest["bucket_policy"] when present (production), else LEGACY_PAIRED, which is
    gqt_paired_pinder's 768-residue scheme."""

    def __init__(self, manifest, mmap_path, config: LoaderConfig | None = None):
        self.manifest = manifest; self.config = config or LoaderConfig()
        self.policy = BucketPolicy.from_json(manifest["bucket_policy"]) if "bucket_policy" in manifest else LEGACY_PAIRED
        self.store = PairedMMap(Path(mmap_path), manifest["records"])
        self.by_id = {row["id"]: row for row in manifest["records"]}
        self.pool = ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="paired-assembly")

    def load(self, ids):
        rows = [self.by_id[cid] for cid in ids]; bucket = self.policy.bucket(rows[0]["n"])
        if any(self.policy.bucket(row["n"]) != bucket for row in rows): raise AssertionError("mixed bucket")
        capacity = self.manifest["capacities"][bucket]; norms = self.manifest["normalization"]
        values = list(self.pool.map(lambda row: _load_one(self.store.raw(row["id"]), row, capacity, norms), rows))
        return jax.tree.map(lambda *items: np.stack(items), *values)

    def close(self):
        self.pool.shutdown(wait=True); self.store.close()

    def provenance(self):
        return {"source": "paired-mmap", "path": str(self.store.path), "workers": self.config.workers, "buckets": self.policy.to_json()}


class LegacyLoaderSource:
    """spec: list of complex ids, or {"cids": [...], "capacities": [...]} (as BatchLoader.iterate accepts)."""

    def __init__(self, loader, config: LoaderConfig | None = None):
        self.loader = loader; self.config = config or LoaderConfig()
        pool = ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="graph-assembly")
        for holder in (loader, getattr(loader, "_base", None)):
            if holder is not None and hasattr(holder, "pool"):
                holder.pool.shutdown(wait=True); holder.pool = pool
        self.pool = pool

    def load(self, spec):
        cids, capacities = (spec["cids"], spec["capacities"]) if isinstance(spec, dict) else (spec, None)
        return self.loader.load(cids, capacities)

    def close(self):
        self.loader.close()

    def provenance(self):
        base = self.loader.provenance() if hasattr(self.loader, "provenance") else {}
        return {"source": type(self.loader).__name__, "workers": self.config.workers, **base}


def paired_to_device(batch):
    """Device-put the training arrays of a PairedSource batch; per-site metadata stays on the host."""
    graphs, targets, mask, burial, interface, metadata = batch
    return (*jax.device_put((graphs, targets, mask, burial, interface)), metadata)


def to_device(batch):
    return jax.device_put(batch)


class JaxStepRunner:
    """Engine steps without per-step float(): returns device scalars for DeferredScalars."""

    def __init__(self, engine):
        self.engine = engine

    def paired(self, params, state, batch, rate):
        """gqt_paired_pinder.PairedEngine: same computation as PairedEngine.update."""
        graphs, targets, mask, wb, wi, _ = batch; valid = jnp.ones(targets.shape[0], bool)
        params, state, loss, finite = self.engine.step(params, state, graphs, targets, mask, wb, wi, valid, rate)
        if not bool(finite): raise FloatingPointError("nonfinite paired oGQT update")
        return params, state, loss

    def joint(self, params, state, pair_batch, pk_batch, rate):
        """gqt_multitask_replay.JointEngine: same computation as JointEngine.update."""
        engine = self.engine; graphs, targets, mask, wb, wi, _ = pair_batch; valid = jnp.ones(targets.shape[0], bool)
        (pair_total, (state_loss, pair_loss)), pair_gradient = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, valid)
        pk_loss, pk_gradient = engine.pkpdb_value_grad(params, *pk_batch)
        params, state, finite = engine.apply(params, state, pair_gradient, pk_gradient, rate)
        losses = jnp.stack((pair_total, state_loss, pair_loss, pk_loss))
        if not bool(finite & jnp.all(jnp.isfinite(losses))): raise FloatingPointError("nonfinite joint oGQT update")
        return params, state, {"total": pair_total + pk_loss, "pkpdb": pk_loss, "pinder_state": state_loss, "pinder_paired": pair_loss}
