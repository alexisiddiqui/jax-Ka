import threading
import time

import numpy as np
import pytest

from pkatrain.loading import LEGACY_PAIRED, PRODUCTION, BucketPolicy, DeferredScalars, LoaderConfig, Prefetcher


class SlowSource:
    def __init__(self, fail_at=None):
        self.lock = threading.Lock(); self.active = 0; self.peak = 0; self.loaded = []; self.fail_at = fail_at
    def load(self, spec):
        with self.lock: self.active += 1; self.peak = max(self.peak, self.active); self.loaded.append(spec)
        time.sleep(0.002 * (spec % 3))
        with self.lock: self.active -= 1
        if spec == self.fail_at: raise ValueError(spec)
        return {"x": np.full(3, spec)}
    def close(self): pass
    def provenance(self): return {}


def test_prefetcher_is_in_order_and_bounded():
    source = SlowSource(); config = LoaderConfig(workers=2, prefetch=3)
    out = [(spec, int(batch["x"][0])) for spec, batch in Prefetcher(source, range(20), config)]
    assert out == [(i, i) for i in range(20)] and source.peak <= 3


def test_prefetcher_reraises_worker_errors():
    with pytest.raises(ValueError):
        for _ in Prefetcher(SlowSource(fail_at=5), range(10), LoaderConfig(workers=1, prefetch=2)): pass


def test_early_exit_stops_loading():
    source = SlowSource(); iterator = iter(Prefetcher(source, range(100), LoaderConfig(workers=1, prefetch=2)))
    next(iterator); iterator.close()
    assert len(source.loaded) <= 4


def test_device_hook_runs_in_worker_and_consume_hook_on_consumer():
    threads = {}
    def to_device(batch): threads["device"] = threading.current_thread().name; return batch
    def on_consume(batch): threads["consume"] = threading.current_thread().name; return batch
    prefetcher = Prefetcher(SlowSource(), range(3), LoaderConfig(workers=1, prefetch=2), to_device, on_consume)
    assert len(list(prefetcher)) == 3
    assert threads["device"].startswith("prefetch") and threads["consume"] == threading.current_thread().name
    summary = prefetcher.telemetry.summary()
    assert summary["batches"] == 3 and {"wait_fraction", "flag_input_bound"} <= summary.keys()


def test_deferred_scalars_numpy_and_jax():
    import jax.numpy as jnp
    values = DeferredScalars(every=2)
    for v in (1.0, 2.0, 3.0): values.add(jnp.asarray(v))
    assert values.mean() == pytest.approx(2.0) and values.reset() == [1.0, 2.0, 3.0] and values.values == []


def test_legacy_policy_reproduces_paired_plans():
    from pkatrain.gqt_paired_pinder import _plans
    rng = np.random.default_rng(3)
    records = [{"id": f"c{i}", "n": int(n)} for i, n in enumerate(rng.integers(30, 769, 300))]
    manifest = {"batch_sizes": {"128": 8, "256": 8, "384": 8, "512": 4, "640": 2, "768": 2}}
    assert LEGACY_PAIRED.plans(records, np.random.default_rng(17)) == _plans(records, np.random.default_rng(17), manifest)


def test_production_policy_covers_pools_and_rounds_capacities():
    assert PRODUCTION.bucket(1500) == "1536" and PRODUCTION.bucket(769) == "1024"
    assert PRODUCTION.batch_sizes == (24, 12, 8, 6, 4, 4, 3, 2, 2)
    with pytest.raises(ValueError): PRODUCTION.bucket(1537)
    caps = PRODUCTION.capacities([{"n": 700, "k": 33, "q": 5, "s": 70, "sk": 64}, {"n": 760, "k": 40, "q": 9, "s": 1, "sk": 1}])
    assert caps == {"768": [768, 64, 32, 96, 64]}
    with pytest.raises(ValueError): BucketPolicy((128, 64), (1, 1))


def test_packed_site_source_stream_returns_requested_rows_in_order(tmp_path):
    torch = pytest.importorskip("torch")
    from pkatrain.loading_torch import CombinedSource, PackedSiteSource
    np.save(tmp_path / "x.npy", np.arange(40, dtype=np.float32).reshape(20, 2)); np.save(tmp_path / "y.npy", np.arange(20, dtype=np.float32))
    arrays = {"x": np.load(tmp_path / "x.npy", mmap_mode="r"), "y": np.load(tmp_path / "y.npy", mmap_mode="r")}
    source = CombinedSource({"a": PackedSiteSource(arrays, device="cpu", config=LoaderConfig(workers=2, prefetch=2))})
    specs = [{"a": np.array([7, 3, 3, 19])}, {"a": np.array([0, 1])}]
    out = [batch["a"] for _, batch in Prefetcher(source, specs, LoaderConfig(workers=2, prefetch=2), source.to_device, source.on_consume)]
    assert torch.equal(out[0]["y"], torch.tensor([7., 3., 3., 19.])) and torch.equal(out[0]["x"][0], torch.tensor([14., 15.]))
    assert torch.equal(out[1]["y"], torch.tensor([0., 1.]))
