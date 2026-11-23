"""Production GQT batches from the compressed graph stores (pkatrain.production_graphs, store-v2), 2026-10-10.

BatchSources for pkatrain.loading.Prefetcher; a spec is a list of structure ids from one size bucket
(BucketPolicy.plans). Batches are padded to the policy's fixed batch size (repeats of the last structure with every
mask cleared and valid False, as SiteBatchLoader does), so each bucket compiles once.

- PinderSource.load -> (graphs, targets, mask, burial, interface, metadata, valid): gqt_paired_pinder._load_one on
  each record (AB and partner-free branches, burial/interface weights divided by the training normalisation), the
  format PairedEngine.step consumes (valid passed explicitly).
- PkpdbSource.load -> (graphs, targets, eligible, valid), or with weights=True (graphs, targets, eligible, w_burial, valid), also for benchmark-val (mask="eval_mask"): the SiteBatchLoader format (residue graph padded as
  graph_data.pad, site graph as site_graph_data.pad_site). The store keeps every mapped site, so `eligible` is the
  store's train_mask (training) or eval_mask (evaluation); sites without a site token (query_site -1, never
  eligible) point at site 0.
- Padding spread (2026-10-10, spread=True, the default): the zero fill points every masked neighbour slot, every padded
  node's slots and every padded site at row 0, so the backward scatters of the gathers (Triton atomic_add of dK/dV,
  XLA scatter-add) all accumulate into one row: 60-77% of residue and 77-91% of site gather indices (padding census,
  experiment note 07). spread_padding points masked neighbour slots at their own row and padded sites at distinct real
  residues (mode "rotate": masked slot j of row i -> row (i + j + 1) mod rows). Those slots carry zero attention weight and padded tokens zero upstream gradient, so the loss is unchanged
  and gradients change only in float summation order.

Manifests (<runtime>/training/<version>/<dataset>/manifest-v1.json, `manifest` action, built once under
scripts/sqfs_run.sh because it reads pool-v3.tsv from the dataset image):
- bucket_policy and capacities over all records, so every pool fraction uses the same compiled shapes;
- per record: split, n/k/q/s/sk, min_fraction (pool-v3 or pool-v4; training subsets are min_fraction <= fraction);
  gqt-production-v2 pKPDB stores also hold the pool-v4 pKPDB validation set (split 'val', scored on train_mask);
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
from .production_graphs import DIMS, PINDER, PKPDB, POOL_V4, VERSION, ProductionStore, output, read

MANIFEST = "manifest-v1.json"
RESIDUE_FIELDS = ("nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch", "query_residue", "query_group")
SITE_FIELDS = ("site_residue", "site_type", "site_mask", "site_neighbors", "site_edge", "site_edge_mask", "site_switch", "query_site")


def _pool(root, dataset):
    """The training pool with min_fraction: pool-v3 (gqt-production-v1) or pool-v4 (gqt-production-v2)."""
    path = Path(root) / POOL_V4 / f"{dataset}.tsv" if VERSION == "gqt-production-v2" else \
        Path(root) / (PKPDB if dataset == "pkpdb" else PINDER) / "pool-v3.tsv"
    with open(path) as handle: return {row["id"]: row for row in csv.DictReader(handle, delimiter="\t")}, path


def manifest_kind(manifest):
    """Tensor capability is separate from catalogue/source identity."""
    kind = manifest.get('data_kind')
    if kind is None:
        kind = {'pinder': 'paired', 'pkpdb': 'site_only', 'benchmark-val': 'site_only'}.get(manifest['dataset'])
    if kind not in {'paired', 'site_only'}:
        raise ValueError('Source must declare paired or site_only data_kind: ' + manifest['dataset'])
    return kind


def build_manifest(root, dataset, policy=PRODUCTION, workers=8, *, pool_path=None, data_kind=None, record_provenance=None):
    root = Path(root); out = output(root, dataset); store_path = out / "store-v2"
    aware = data_kind is not None or record_provenance is not None or pool_path is not None
    kind = manifest_kind(dict(dataset=dataset, **({'data_kind': data_kind} if data_kind is not None else {})))
    if aware and (pool_path is None or record_provenance is None):
        raise ValueError('Source-aware manifests require an explicit pool and record provenance')
    store = ProductionStore(store_path)
    if aware and store.metadata.get('dataset') != dataset:
        store.close(); raise ValueError('Store catalogue identity differs from requested manifest source')
    if aware and store.metadata.get('data_kind') != kind:
        store.close(); raise ValueError('Store tensor capability differs from requested manifest')
    if pool_path is not None:
        pool_path = Path(pool_path)
        with pool_path.open() as handle: pool_rows = list(csv.DictReader(handle, delimiter='\t'))
        pool = {r['id']: r for r in pool_rows}
        if len(pool) != len(pool_rows):
            store.close(); raise ValueError('Duplicate training IDs in source-aware pool')
        train_ids = {r['id'] for r in read(store_path / 'records.json') if r['split'] == 'train'}
        if set(pool) != train_ids:
            store.close(); raise ValueError('Source-aware pool must cover exactly the training records')
        if any(not np.isfinite(float(r['min_fraction'])) or not 0 <= float(r['min_fraction']) <= 1 for r in pool_rows):
            store.close(); raise ValueError('Invalid nested training-pool fraction')
    else:
        pool, pool_path = _pool(root, dataset) if dataset != "benchmark-val" else ({}, None)
    records = []
    def weights(cid):
        raw = store.raw(cid); return float(np.mean(raw["w_burial"])), float(np.mean(raw["w_interface"]))
    means = {}
    if kind == "paired":
        ids = [cid for cid in store.ids]
        with ThreadPoolExecutor(workers) as executor: means = dict(zip(ids, executor.map(weights, ids)))
    for row in read(store_path / "records.json"):
        record = {"id": row["id"], "split": row["split"], **{d: row[d] for d in DIMS}}
        if "component_id" in row: record["component_id"] = row["component_id"]  # group-macro metrics group by component
        if row["split"] == "train":
            if row["id"] not in pool: raise AssertionError((row["id"], f"training record not in {pool_path.name}"))
            record["min_fraction"] = float(pool[row["id"]]["min_fraction"])
        if kind == "paired": record["w_burial_mean"], record["w_interface_mean"] = means[row["id"]]
        else: record["train_sites"], record["eval_sites"] = row.get("train_sites"), row.get("eval_sites")
        if aware:
            if row['id'] not in record_provenance:
                store.close(); raise ValueError('Missing source record provenance: ' + row['id'])
            provenance = record_provenance[row['id']]
            required = {'source', 'source_native_id', 'native_source_split', 'native_component', 'operational_role', 'input_scope_id'}
            if not required <= provenance.keys() or provenance['source'] != dataset:
                store.close(); raise ValueError('Missing or inconsistent source provenance')
            if dataset == 'sabdab' and {'train': 'train', 'test': 'validation'}.get(provenance['native_source_split']) != provenance['operational_role']:
                store.close(); raise ValueError('SAbDab operational role contradicts its published split')
            expected = {'train': 'train', 'validation': 'val'}.get(provenance['operational_role'])
            if expected != row['split']:
                store.close(); raise ValueError('Store split differs from operational role')
            if set(provenance) & set(record):
                store.close(); raise ValueError('Provenance must not overwrite graph dimensions, split or sampling fields')
            record.update(provenance)
        records.append(record)
    store.close()
    manifest = {"version": VERSION, "dataset": dataset, "store": str(store_path),
                "store_verification_sha256": digest(store_path / "verification.json"), "pool": str(pool_path) if pool_path else None,
                "pool_sha256": digest(pool_path) if pool_path else None, "bucket_policy": policy.to_json(), "capacities": policy.capacities(records),
                "records": records, "test_data_included": False}
    if aware:
        if set(record_provenance) != {r['id'] for r in records}:
            raise ValueError('Source provenance does not cover exactly the store records')
        manifest.update(source_manifest_schema=1, data_kind=kind)
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


def spread_padding(graph, mode="rotate"):
    """Masked neighbour slots -> spread rows (residue and site graphs); padded sites -> residue (index mod real residues).
    mode "own": slot -> its own row; "rotate": slot j of row i -> row (i + j + 1) mod rows, so one row's masked slots
    never share an address (same-address atomics within a program serialise). Works on one structure's graph or a batch
    (any leading axes)."""
    for index, mask in (("neighbors", "edge_mask"), ("site_neighbors", "site_edge_mask")):
        value = graph[index]; rows, slots = value.shape[-2:]
        target = np.arange(rows)[:, None] if mode == "own" else (np.arange(rows)[:, None] + np.arange(1, slots + 1)[None, :]) % rows
        graph[index] = np.where(graph[mask].astype(bool), value, np.broadcast_to(target, value.shape)).astype(value.dtype)
    residue = graph["site_residue"]; real = np.maximum(graph["node_mask"].astype(bool).sum(axis=-1, keepdims=True), 1)
    spread = np.broadcast_to(np.arange(residue.shape[-1]), residue.shape) % real
    graph["site_residue"] = np.where(graph["site_mask"].astype(bool), residue, spread).astype(residue.dtype)
    return graph


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
    def __init__(self, manifest, store_path=None, config: LoaderConfig | None = None, spread=True, sidechains=None):
        """sidechains: production_sidechains.SidechainTable; its per-residue features are appended to the node
        features (24 -> 152) for the side-chain model variant (2026-10-11)."""
        self.manifest = manifest; self.config = config or LoaderConfig(); self.spread = spread; self.sidechains = sidechains
        # production_gate (2026-10-19): set after construction; disulfides adds graph["disulfide"], drop_bonded_cys
        # removes bonded-Cys queries from the pKa mask (training sources only)
        self.disulfides = None; self.drop_bonded_cys = False
        # production_residue (2026-10-19): residues adds rsa_residue to PINDER graphs; sample_edges also the sampled
        # residue pairs (training source only)
        self.residues = None; self.sample_edges = False; self._visits = {}
        # decoy-rank-v1 (2026-10-29): decoy_retype adds the re-typed decoy fields to the sampled-pair graphs (production_residue.retype)
        self.decoy_retype = False
        # negatives-v1 (2026-10-29): within_sample adds the within-complex listwise fields (production_residue.within_sample);
        # strata ({id: code}, production_residue.load_strata) adds graph["stratum"] (2,) int32 (-1 when unknown)
        self.within_sample = False; self.strata = None
        # hotspot-mask-v1 (2026-10-30): hotspot = (oracle store, oracle, keep, soft) adds the unmasked-set fields (production_residue.hotspot_fields)
        self.hotspot = None
        # quartet-rank-v1 (2026-11-15): quartet_count > 0 adds the native-pairing quartet fields (production_residue.quartet_fields)
        self.quartet_count = 0
        # min-aug-v1 (2026-10-31): alt (production_alt.AltConfig) swaps the free branch for the side-store geometry per visit
        # (minimised / AlphaFold / ANM; apo-negatome-v1's apo hook renamed); keep_crystal adds the crystal free branch as xtal_*
        self.alt = None
        self.policy = BucketPolicy.from_json(manifest["bucket_policy"])
        self.store = ProductionStore(store_path or os.environ.get(f"PKATRAIN_{manifest['dataset'].upper().replace('-', '_')}_STORE") or manifest["store"])
        if digest(self.store.path / "verification.json") != manifest["store_verification_sha256"]: raise AssertionError("store differs from manifest")
        if 'data_kind' in manifest:
            if self.store.metadata.get('dataset') != manifest['dataset']:
                self.store.close(); raise ValueError('Store source identity differs from manifest')
            from .production_native import declared_store_kind
            if declared_store_kind(manifest,self.store) != manifest_kind(manifest):
                self.store.close(); raise ValueError('Store tensor capability differs from manifest')
            if manifest_kind(manifest) == 'paired':
                from .production_graphs import FIELDS
                if set(self.store.fields) != set(FIELDS['pinder']):
                    self.store.close(); raise ValueError('Paired source has incompatible graph fields')
        self.by_id = {row["id"]: row for row in manifest["records"]}
        if manifest.get('record_overrides'):
            from .production_native import PatchedStore
            self.store=PatchedStore(self.store,manifest['record_overrides'],self.by_id,manifest['dataset'])
        self.pool = ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix=f"{manifest['dataset']}-assembly")

    def _bucket(self, ids):
        rows = [self.by_id[cid] for cid in ids]; bucket = self.policy.bucket(rows[0]["n"])
        if any(self.policy.bucket(row["n"]) != bucket for row in rows): raise AssertionError("mixed bucket")
        if len(rows) > self.policy.batch_size(bucket): raise AssertionError("batch larger than the bucket's batch size")
        return rows, bucket, self.manifest["capacities"][bucket]

    def _side(self, graph, cid, n):
        if self.sidechains is None: return graph
        features = self.sidechains(cid)  # (nodes, w), or (branches, nodes, w) for branch-specific tables (production_packed)
        if features.shape[-2] != self.by_id[cid]["n"]: raise AssertionError((cid, "side-chain table does not match the store"))
        nodes = graph["nodes"]; extra = _pad(features, features.shape[:-2] + (n, features.shape[-1]))
        if extra.ndim == 3 and (nodes.ndim != 3 or extra.shape[0] != nodes.shape[0]): raise AssertionError((cid, "branch-specific side-chain table for this source"))
        if nodes.ndim == 3: extra = np.broadcast_to(extra, (nodes.shape[0],) + extra.shape[-2:])
        graph["nodes"] = np.concatenate((nodes, extra), axis=-1).astype(np.float32); return graph

    def _cys(self, graph, raw, cid, n, mask, paired):
        if self.disulfides is None: return graph, mask
        from .production_gate import bonded_queries
        bonded, own = self.disulfides.flags(cid)
        graph["disulfide"] = np.stack((_pad(bonded, (n,)), _pad(own, (n,)))).astype(np.float32) if paired else _pad(bonded, (n,)).astype(np.float32)
        if self.drop_bonded_cys: mask = mask & ~_pad(bonded_queries(raw, bonded), mask.shape)
        return graph, mask

    def _residue(self, graph, cid, n):
        if self.residues is None: return graph
        from .production_residue import attach
        if len(self.residues.row(cid)["rsa"]) != self.by_id[cid]["n"]: raise AssertionError((cid, "residue table does not match the store"))
        hotspot = getattr(self, "hotspot", None)
        quartets = getattr(self, "quartet_count", 0)
        if quartets:  # quartet-rank-v1
            graph = attach(graph, self.residues, cid, n, self.sample_edges, self._visits, self.decoy_retype, self.within_sample, hotspot, quartets=quartets)
        else:
            graph = attach(graph, self.residues, cid, n, self.sample_edges, self._visits, self.decoy_retype, self.within_sample) if hotspot is None else \
                attach(graph, self.residues, cid, n, self.sample_edges, self._visits, self.decoy_retype, self.within_sample, hotspot)
        if self.strata is not None: graph["stratum"] = np.full(2, self.strata.get(cid, -1), np.int32)
        return graph

    def _alt(self, graph, cid, capacity):
        if getattr(self, "alt", None) is None: return graph
        from .production_alt import attach_alt
        from .production_native import augmentation_guard
        return attach_alt(graph, self.alt, cid, self.by_id[cid]["n"], capacity,
            role_guard=augmentation_guard(self.manifest,self.by_id[cid]),
            biological_side=self.by_id[cid].get('residue_side') if 'augmentation_role_policy' in self.manifest else None)

    def _spread(self, graph):
        if not self.spread: return graph
        graph = spread_padding(graph)
        if getattr(self, "alt", None) is not None and self.alt.keep_crystal:
            from .production_alt import spread_crystal
            graph = spread_crystal(graph, spread_padding)
        return graph

    def close(self):
        self.pool.shutdown(wait=True); self.store.close()

    def provenance(self):
        return {"source": self.__class__.__name__, "store": str(self.store.path),
                "workers": self.config.workers, "buckets": self.policy.to_json(), "spread_padding": self.spread,
                **({"sidechains_verification_sha256": self.sidechains.verification_sha256} if self.sidechains is not None else {}),
                **({'dataset': self.manifest['dataset'], 'data_kind': manifest_kind(self.manifest)} if 'data_kind' in self.manifest else {})}


class PinderSource(_Source):
    def __init__(self, manifest, store_path=None, config=None, fraction=1.0, norms=None, spread=True, sidechains=None):
        super().__init__(manifest, store_path, config, spread, sidechains); self.norms = norms or normalization(manifest, fraction)

    def load(self, ids):
        from .gqt_paired_pinder import _load_one
        rows, bucket, capacity = self._bucket(ids)
        def one(row):
            raw = self.store.raw(row["id"]); graph, target, mask, *rest = _load_one(raw, row, capacity, self.norms); graph = self._side(graph, row["id"], capacity[0])
            graph, mask = self._cys(graph, raw, row["id"], capacity[0], mask, True); graph = self._residue(graph, row["id"], capacity[0])
            graph = self._alt(graph, row["id"], capacity)
            return (self._spread(graph), target, mask, *rest)
        items = list(self.pool.map(one, rows))
        # clear: targets(1), mask(2), burial(3), interface(4); graphs/metadata stay as copies of a real structure
        return _stack(items, self.policy.batch_size(bucket), clear=(1, 2, 3, 4))


class PairedSource(PinderSource):
    """The existing paired tensor loader with an explicit native source identity."""
    def __init__(self, manifest, *args, **kwargs):
        if manifest_kind(manifest) != 'paired':
            raise ValueError('PairedSource requires paired data')
        super().__init__(manifest, *args, **kwargs)


class AuxPinderSource(PinderSource):
    """PinderSource for the ordinal auxiliary heads (pkatrain.production_aux, 2026-10-11): same tuple, but slot 3 holds
    the free-state RSA (NaN -> -1, masked) and slot 4 the partner-contact score (production_aux.ContactTable) in place of
    the normalised w_burial / w_interface, which the auxiliary objectives do not use. `contacts` is any per-site
    production_aux.ContactTable (the rsa-bound-v1 table for Siamese burial); NaN -> -1 (masked)."""

    def __init__(self, manifest, contacts, store_path=None, config=None, fraction=1.0, norms=None, spread=True, sidechains=None):
        super().__init__(manifest, store_path, config, fraction, norms, spread, sidechains); self.contacts = contacts

    def load(self, ids):
        from .gqt_paired_pinder import _load_one
        rows, bucket, capacity = self._bucket(ids); q = capacity[2]
        def one(row):
            raw = self.store.raw(row["id"]); graph, target, mask, _, _, metadata = _load_one(raw, row, capacity, self.norms)
            graph = self._side(graph, row["id"], capacity[0]); graph, mask = self._cys(graph, raw, row["id"], capacity[0], mask, True)
            graph = self._residue(graph, row["id"], capacity[0]); graph = self._alt(graph, row["id"], capacity)
            score = self.contacts(row["id"])
            if len(score) != len(raw["rsa_free"]): raise AssertionError((row["id"], "contact table does not match the store"))
            rsa = _pad(np.nan_to_num(raw["rsa_free"].astype(np.float32), nan=-1.0), (q,))
            return (self._spread(graph), target, mask, rsa, _pad(np.nan_to_num(score.astype(np.float32), nan=-1.0), (q,)), metadata)
        items = list(self.pool.map(one, rows))
        return _stack(items, self.policy.batch_size(bucket), clear=(1, 2, 3, 4))

    def provenance(self): return {**super().provenance(), "contacts_verification_sha256": self.contacts.verification_sha256}


class PkpdbSource(_Source):
    def __init__(self, manifest, store_path=None, config=None, mask="train_mask", spread=True, weights=False, aux=False, sidechains=None):
        """aux=True (production_aux): slot 3 holds the site RSA (NaN -> -1, masked) instead of w_burial."""
        if mask not in ("train_mask", "eval_mask"): raise ValueError(mask)
        super().__init__(manifest, store_path, config, spread, sidechains); self.mask = mask; self.weights = weights or aux; self.aux = aux

    def _one(self, row, capacity):
        n, k, q, s, sk = capacity; raw = self.store.raw(row["id"])
        shapes = {"nodes": (n, 24), "node_mask": (n,), "neighbors": (n, k), "edge": (n, k, 20), "edge_mask": (n, k),
                  "switch": (n, k), "query_residue": (q,), "query_group": (q,), "site_residue": (s,), "site_type": (s,),
                  "site_mask": (s,), "site_neighbors": (s, sk), "site_edge": (s, sk, 34), "site_edge_mask": (s, sk),
                  "site_switch": (s, sk), "query_site": (q,)}
        graph = {name: _pad(raw[name], shapes[name]) for name in RESIDUE_FIELDS + SITE_FIELDS}
        graph["query_site"] = np.maximum(graph["query_site"], 0); graph = self._side(graph, row["id"], n)
        graph, eligible = self._cys(graph, raw, row["id"], n, _pad(raw[self.mask], (q,)), False)
        if self.spread: graph = spread_padding(graph)
        if self.aux: return graph, _pad(raw["labels"], (q,)), eligible, _pad(np.nan_to_num(raw["rsa"].astype(np.float32), nan=-1.0), (q,))
        if self.weights: return graph, _pad(raw["labels"], (q,)), eligible, _pad(np.nan_to_num(raw["w_burial"].astype(np.float32), nan=0.0), (q,))
        return graph, _pad(raw["labels"], (q,)), eligible

    def load(self, ids):
        rows, bucket, capacity = self._bucket(ids)
        items = list(self.pool.map(lambda row: self._one(row, capacity), rows))
        return _stack(items, self.policy.batch_size(bucket), clear=(1, 2, 3) if self.weights else (1, 2))


def source_from_manifest(manifest, **kwargs):
    """Build a source without encoding SAbDab/VHH records as PINDER entries."""
    if manifest_kind(manifest) == 'paired':
        cls = PinderSource if manifest['dataset'] == 'pinder' else PairedSource
        return cls(manifest, **kwargs)
    return PkpdbSource(manifest, **kwargs)


# ---------------------------------------------------------------- checks
def check(root, dataset, fraction=1.0, batches=300):
    """Content: unpadded slices equal the store record, padding is zero, masks match. Throughput: Prefetcher over
    `batches` training batches at 8 and 32 workers."""
    from .loading import Prefetcher
    root = Path(root); manifest = read(output(root, dataset) / MANIFEST); rng = np.random.default_rng(17)
    split = "val" if dataset == "benchmark-val" else "train"; mask = "eval_mask" if dataset == "benchmark-val" else "train_mask"
    train = select(manifest, split, fraction); plans = BucketPolicy.from_json(manifest["bucket_policy"]).plans(train, rng)
    report = {"dataset": dataset, "fraction": fraction, "split": split, "structures": len(train), "batches_per_epoch": len(plans),
              "buckets": {b: sum(1 for r in train if BucketPolicy.from_json(manifest["bucket_policy"]).bucket(r["n"]) == b)
                          for b in manifest["capacities"]}}
    source = PinderSource(manifest, fraction=fraction) if dataset == "pinder" else PkpdbSource(manifest, mask=mask)
    for ids in plans[:20]:  # content
        batch = source.load(ids); graphs = batch[0]
        for slot, cid in enumerate(ids):
            raw = source.store.raw(cid); row = source.by_id[cid]
            nodes = graphs["nodes"][slot][0] if dataset == "pinder" else graphs["nodes"][slot]
            if not (np.array_equal(nodes[:row["n"]], raw["nodes"]) and not nodes[row["n"]:].any()): raise AssertionError((cid, "nodes"))
            q = row["q"]
            if dataset != "pinder":
                if not (np.array_equal(batch[1][slot][:q], raw["labels"]) and np.array_equal(batch[2][slot][:q], raw[mask]) and not batch[2][slot][q:].any()):
                    raise AssertionError((cid, "labels/mask"))
            else:
                if not (np.array_equal(batch[1][slot][:, :q], raw["targets"]) and batch[2][slot][:q].all() and not batch[2][slot][q:].any()):
                    raise AssertionError((cid, "targets/mask"))
        if batch[-1].sum() != len(ids) or (len(ids) < len(batch[-1]) and batch[2][len(ids):].any()): raise AssertionError("padding")
    report["content_checked_batches"] = 20; source.close()
    for workers in (8, 32):
        config = LoaderConfig(workers=workers, prefetch=4); source = PinderSource(manifest, config=config, fraction=fraction) if dataset == "pinder" else PkpdbSource(manifest, config=config, mask=mask)
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
