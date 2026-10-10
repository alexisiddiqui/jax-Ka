"""Production GQT batches from the compressed graph stores (pkatrain.production_graphs, store-v2), 2026-10-10.

BatchSources for pkatrain.loading.Prefetcher; a spec is a list of structure ids from one size bucket
(BucketPolicy.plans). Batches are padded to the policy's fixed batch size (repeats of the last structure with every
mask cleared and valid False, as SiteBatchLoader does), so each bucket compiles once.

- PinderSource.load -> (graphs, targets, mask, burial, interface, metadata, valid): gqt_paired_pinder._load_one on
  each record (AB and partner-free branches, burial/interface weights divided by the training normalisation), the
  format PairedEngine.step consumes (valid passed explicitly).
- PkpdbSource.load -> (graphs, targets, eligible, valid): the SiteBatchLoader format (residue graph padded as
  graph_data.pad, site graph as site_graph_data.pad_site). The store keeps every mapped site, so `eligible` is the
  store's train_mask (training) or eval_mask (evaluation); sites without a site token (query_site -1, never
  eligible) point at site 0.

Manifests (<runtime>/training/gqt-production-v1/<dataset>/manifest-v1.json, `manifest` action, built once under
scripts/sqfs_run.sh because it reads pool-v3.tsv from the dataset image):
- bucket_policy and capacities over all records, so every pool fraction uses the same compiled shapes;
- per record: split, n/k/q/s/sk, min_fraction (pool-v3; training subsets are min_fraction <= fraction);
- PINDER: per-complex means of w_burial and w_interface over its sites, so normalisation(manifest, fraction) is the
  mean complex-level weight over that fraction's training complexes (the factorial's definition).

  python -m pkatrain.production_loading manifest {pkpdb,pinder}
  python -m pkatrain.production_loading check {pkpdb,pinder} [fraction]     (content and throughput check)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest
from .loading import PRODUCTION, BucketPolicy, LoaderConfig
from .production_graphs import DIMS, PINDER, PKPDB, ProductionStore, output, read

MANIFEST = "manifest-v1.json"
RESIDUE_FIELDS = ("nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch", "query_residue", "query_group")
SITE_FIELDS = ("site_residue", "site_type", "site_mask", "site_neighbors", "site_edge", "site_edge_mask", "site_switch", "query_site")


def _pool(root, dataset):
    path = Path(root) / (PKPDB if dataset == "pkpdb" else PINDER) / "pool-v3.tsv"
    with open(path) as handle: return {row["id"]: row for row in csv.DictReader(handle, delimiter="\t")}, path


def build_manifest(root, dataset, policy=PRODUCTION, workers=8):
    root = Path(root); out = output(root, dataset); store_path = out / "store-v2"; store = ProductionStore(store_path)
    pool, pool_path = _pool(root, dataset); records = []
    def weights(cid):
        raw = store.raw(cid); return float(np.mean(raw["w_burial"])), float(np.mean(raw["w_interface"]))
    means = {}
    if dataset == "pinder":
        ids = [cid for cid in store.ids]
        with ThreadPoolExecutor(workers) as executor: means = dict(zip(ids, executor.map(weights, ids)))
    for row in read(store_path / "records.json"):
        record = {"id": row["id"], "split": row["split"], **{d: row[d] for d in DIMS}}
        if row["split"] == "train":
            if row["id"] not in pool: raise AssertionError((row["id"], "training record not in pool-v3"))
            record["min_fraction"] = float(pool[row["id"]]["min_fraction"])
        if dataset == "pinder": record["w_burial_mean"], record["w_interface_mean"] = means[row["id"]]
        else: record["train_sites"], record["eval_sites"] = row.get("train_sites"), row.get("eval_sites")
        records.append(record)
    store.close()
    manifest = {"version": "gqt-production-v1", "dataset": dataset, "store": str(store_path),
                "store_verification_sha256": digest(store_path / "verification.json"), "pool": str(pool_path),
                "pool_sha256": digest(pool_path), "bucket_policy": policy.to_json(), "capacities": policy.capacities(records),
                "records": records, "test_data_included": False}
    atomic_json(out / MANIFEST, manifest)
    return manifest


def select(manifest, split, fraction=1.0):
    """Training: the nested pool-v3 subset (min_fraction <= fraction). Validation: all validation records."""
    return [r for r in manifest["records"] if r["split"] == split and (split != "train" or r["min_fraction"] <= fraction + 1e-12)]


def normalization(manifest, fraction=1.0):
    """PINDER weight normalisation: mean complex-level site weight over the fraction's training complexes."""
    rows = select(manifest, "train", fraction)
    return {"burial": float(np.mean([r["w_burial_mean"] for r in rows])), "interface": float(np.mean([r["w_interface_mean"] for r in rows]))}


def _pad(value, shape):
    out = np.zeros(shape, value.dtype); out[tuple(slice(0, size) for size in value.shape)] = value; return out


def _stack(items, batch_size, clear):
    """Stack per-structure tuples; pad to batch_size with copies of the last item whose `clear` positions are zeroed."""
    import jax
    valid = np.arange(batch_size) < len(items); items = list(items)
    while len(items) < batch_size:
        last = list(items[-1])
        for i in clear: last[i] = jax.tree.map(np.zeros_like, last[i])
        items.append(tuple(last))
    return (*jax.tree.map(lambda *x: np.stack(x), *items), valid)


class _Source:
    def __init__(self, manifest, store_path=None, config: LoaderConfig | None = None):
        self.manifest = manifest; self.config = config or LoaderConfig()
        self.policy = BucketPolicy.from_json(manifest["bucket_policy"])
        self.store = ProductionStore(store_path or os.environ.get(f"PKATRAIN_{manifest['dataset'].upper()}_STORE") or manifest["store"])
        if digest(self.store.path / "verification.json") != manifest["store_verification_sha256"]: raise AssertionError("store differs from manifest")
        self.by_id = {row["id"]: row for row in manifest["records"]}
        self.pool = ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix=f"{manifest['dataset']}-assembly")

    def _bucket(self, ids):
        rows = [self.by_id[cid] for cid in ids]; bucket = self.policy.bucket(rows[0]["n"])
        if any(self.policy.bucket(row["n"]) != bucket for row in rows): raise AssertionError("mixed bucket")
        if len(rows) > self.policy.batch_size(bucket): raise AssertionError("batch larger than the bucket's batch size")
        return rows, bucket, self.manifest["capacities"][bucket]

    def close(self):
        self.pool.shutdown(wait=True); self.store.close()

    def provenance(self):
        return {"source": self.__class__.__name__, "store": str(self.store.path),
                "workers": self.config.workers, "buckets": self.policy.to_json()}


class PinderSource(_Source):
    def __init__(self, manifest, store_path=None, config=None, fraction=1.0, norms=None):
        super().__init__(manifest, store_path, config); self.norms = norms or normalization(manifest, fraction)

    def load(self, ids):
        from .gqt_paired_pinder import _load_one
        rows, bucket, capacity = self._bucket(ids)
        items = list(self.pool.map(lambda row: _load_one(self.store.raw(row["id"]), row, capacity, self.norms), rows))
        # clear: targets(1), mask(2), burial(3), interface(4); graphs/metadata stay as copies of a real structure
        return _stack(items, self.policy.batch_size(bucket), clear=(1, 2, 3, 4))


class PkpdbSource(_Source):
    def __init__(self, manifest, store_path=None, config=None, mask="train_mask"):
        if mask not in ("train_mask", "eval_mask"): raise ValueError(mask)
        super().__init__(manifest, store_path, config); self.mask = mask

    def _one(self, row, capacity):
        n, k, q, s, sk = capacity; raw = self.store.raw(row["id"])
        shapes = {"nodes": (n, 24), "node_mask": (n,), "neighbors": (n, k), "edge": (n, k, 20), "edge_mask": (n, k),
                  "switch": (n, k), "query_residue": (q,), "query_group": (q,), "site_residue": (s,), "site_type": (s,),
                  "site_mask": (s,), "site_neighbors": (s, sk), "site_edge": (s, sk, 34), "site_edge_mask": (s, sk),
                  "site_switch": (s, sk), "query_site": (q,)}
        graph = {name: _pad(raw[name], shapes[name]) for name in RESIDUE_FIELDS + SITE_FIELDS}
        graph["query_site"] = np.maximum(graph["query_site"], 0)
        return graph, _pad(raw["labels"], (q,)), _pad(raw[self.mask], (q,))

    def load(self, ids):
        rows, bucket, capacity = self._bucket(ids)
        items = list(self.pool.map(lambda row: self._one(row, capacity), rows))
        return _stack(items, self.policy.batch_size(bucket), clear=(1, 2))


# ---------------------------------------------------------------- checks
def check(root, dataset, fraction=1.0, batches=300):
    """Content: unpadded slices equal the store record, padding is zero, masks match. Throughput: Prefetcher over
    `batches` training batches at 8 and 32 workers."""
    from .loading import Prefetcher
    root = Path(root); manifest = read(output(root, dataset) / MANIFEST); rng = np.random.default_rng(17)
    train = select(manifest, "train", fraction); plans = BucketPolicy.from_json(manifest["bucket_policy"]).plans(train, rng)
    report = {"dataset": dataset, "fraction": fraction, "train_structures": len(train), "batches_per_epoch": len(plans),
              "buckets": {b: sum(1 for r in train if BucketPolicy.from_json(manifest["bucket_policy"]).bucket(r["n"]) == b)
                          for b in manifest["capacities"]}}
    source = PinderSource(manifest, fraction=fraction) if dataset == "pinder" else PkpdbSource(manifest)
    for ids in plans[:20]:  # content
        batch = source.load(ids); graphs = batch[0]
        for slot, cid in enumerate(ids):
            raw = source.store.raw(cid); row = source.by_id[cid]
            nodes = graphs["nodes"][slot][0] if dataset == "pinder" else graphs["nodes"][slot]
            if not (np.array_equal(nodes[:row["n"]], raw["nodes"]) and not nodes[row["n"]:].any()): raise AssertionError((cid, "nodes"))
            q = row["q"]
            if dataset == "pkpdb":
                if not (np.array_equal(batch[1][slot][:q], raw["labels"]) and np.array_equal(batch[2][slot][:q], raw["train_mask"]) and not batch[2][slot][q:].any()):
                    raise AssertionError((cid, "labels/mask"))
            else:
                if not (np.array_equal(batch[1][slot][:, :q], raw["targets"]) and batch[2][slot][:q].all() and not batch[2][slot][q:].any()):
                    raise AssertionError((cid, "targets/mask"))
        if batch[-1].sum() != len(ids) or (len(ids) < len(batch[-1]) and batch[2][len(ids):].any()): raise AssertionError("padding")
    report["content_checked_batches"] = 20; source.close()
    for workers in (8, 32):
        config = LoaderConfig(workers=workers, prefetch=4); source = PinderSource(manifest, config=config, fraction=fraction) if dataset == "pinder" else PkpdbSource(manifest, config=config)
        chosen = plans[:batches]; began = time.time(); structures = 0
        prefetcher = Prefetcher(source, chosen, config)
        for spec, batch in prefetcher: structures += len(spec)
        seconds = time.time() - began; source.close()
        report[f"workers_{workers}"] = {"batches_per_s": round(len(chosen) / seconds, 1), "structures_per_s": round(structures / seconds, 1)}
        print(json.dumps({dataset: report[f"workers_{workers}"], "workers": workers}), flush=True)
    atomic_json(output(root, dataset) / "loader-check.json", report)
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(os.environ["PKABENCH_RUNTIME"]); action, dataset = argv[0], argv[1]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    if action == "manifest":
        m = build_manifest(root, dataset, workers=workers)
        print(json.dumps({"records": len(m["records"]), "capacities": m["capacities"], "bucket_policy": m["bucket_policy"]}))
    elif action == "check":
        print(json.dumps(check(root, dataset, float(argv[2]) if len(argv) > 2 else 1.0)))
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
