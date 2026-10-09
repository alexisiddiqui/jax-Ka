"""Scratch-first paired oGQT pilot on the prepared PINDER/pKAI cohort.

The model always predicts a signed shift from the fixed pKPDB PK_MOD value.
The state loss supervises that shift in AB and free branches; the paired loss
supervises their difference.  Structural weights are independent switches so
the four arms form a matched 2x2 factorial.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import mmap
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from biotite.structure.io import pdbx

from jaxpropka.parameters import GROUPS
from jaxpropka.topology import load_topology
from pkanet.graph import geometry
from pkanet.model import PKPDB_PK_MOD
from pkanet.ogqt import initialize as initialize_ogqt, predict_shift as predict_ogqt_shift
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_crop_radius import EPOCHS, MIN_DELTA, PATIENCE, learning_rate
from .gqt_regularization import decay_mask
from .site_graph_data import build_site_graph
from .trainer import load_checkpoint, save_checkpoint


ARMS = ("vanilla", "burial", "interface", "both")
SEED = 17
TRAIN_COMPLEXES = 5000
VAL_COMPLEXES = 400
MAX_RESIDUES = 768
STATE_NAMES = ("AB", "free")
GROUP_ALIAS = {"NTR": "NTERM", "CTR": "CTERM"}
KEY_FIELDS = ("chain", "resnum", "icode", "group")
_ROOT = None
_SELECTED = None


def source(root): return Path(root) / "pretraining/pinder-pkai-v1"
def experiment_root(root): return Path(root) / "training/ogqt-pinder-factorial-v1"
def pretrained_root(root): return Path(root) / "pretraining/gqt-site-weighting-v1/baseline/seed-17"


def read(path): return json.loads(Path(path).read_text())


def code_hashes():
    here = Path(__file__); src = here.parents[1]
    paths = (here, src / "pkanet/model.py", src / "pkanet/site_model.py",
             src / "pkanet/ogqt.py", src / "pkanet/triton_attention.py",
             src / "pkatrain/site_graph_data.py")
    return {str(path): digest(path) for path in paths}


def _index_rows(root):
    rows = []
    for path in sorted((source(root) / "index").glob("label_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "accepted": rows.append(row)
    if len({row["id"] for row in rows}) != len(rows): raise AssertionError("duplicate PINDER id")
    return rows


def _label_maps(folder):
    labels = read(folder / "labels.json")["pkai"]
    result = {}
    for state in ("AB", "A", "B"):
        state_rows = {}
        for chain, number, insertion, group, value in labels.get(state, []):
            group = GROUP_ALIAS.get(group, group)
            key = (str(chain), int(number), str(insertion), str(group))
            if value is not None and np.isfinite(value): state_rows[key] = float(value)
        result[state] = state_rows
    return result


def _paired_rows(folder, split):
    labels = _label_maps(folder); sites = read(folder / "sites.json")
    mask_name = "train_mask" if split == "train" else "eval_mask"
    output = []
    for site in sites:
        key = tuple(site[name] for name in KEY_FIELDS)
        free_state = site["partner"]
        if not site[mask_name] or key not in labels["AB"] or key not in labels[free_state]: continue
        if site.get("w_burial") is None or site.get("w_interface") is None: continue
        output.append({**site, "key": key, "target_ab": labels["AB"][key],
                       "target_free": labels[free_state][key]})
    return output


def _rank(identifier):
    return hashlib.sha256(("ogqt-pinder-factorial-v1|" + identifier).encode()).hexdigest()


def _select(rows, root, split, count, forbidden_clusters=frozenset()):
    candidates = [row for row in rows if row["split"] == split and row["n_res"] <= MAX_RESIDUES
                  and row["cluster_id"] not in forbidden_clusters]
    representatives = {}
    for row in candidates:
        current = representatives.get(row["cluster_id"])
        if current is None or _rank(row["id"]) < _rank(current["id"]): representatives[row["cluster_id"]] = row
    accepted = []
    for row in sorted(representatives.values(), key=lambda value: _rank(value["id"])):
        folder = source(root) / "entries" / row["id"]
        paired = _paired_rows(folder, split)
        interface = sum(item["interface"] for item in paired)
        if not paired or not interface: continue
        accepted.append({**row, "q": len(paired), "q_interface": interface})
        if len(accepted) == count: break
    if len(accepted) != count: raise RuntimeError(f"Only {len(accepted)} of {count} {split} complexes available")
    return accepted


def _normalization(root, records):
    values = defaultdict(list)
    for record in records:
        rows = _paired_rows(source(root) / "entries" / record["id"], "train")
        values["burial"].append(float(np.mean([row["w_burial"] for row in rows])))
        values["interface"].append(float(np.mean([row["w_interface"] for row in rows])))
    return {name: float(np.mean(rows)) for name, rows in values.items()}


def register(root):
    root = Path(root); out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    rows = _index_rows(root)
    train = _select(rows, root, "train", TRAIN_COMPLEXES)
    train_clusters = {row["cluster_id"] for row in train}
    val = _select(rows, root, "val", VAL_COMPLEXES, train_clusters)
    if train_clusters & {row["cluster_id"] for row in val}: raise AssertionError("cluster leakage")
    norms = _normalization(root, train)
    records = []
    for row in (*train, *val):
        folder = source(root) / "entries" / row["id"]
        records.append({**row, "source_hashes": {name: digest(folder / name) for name in
            ("AB.cif.gz", "sites.json", "labels.json", "meta.json")}})
    protocol = {
        "version": "ogqt-pinder-factorial-v1", "seed": SEED,
        "cohort": {"train": TRAIN_COMPLEXES, "val": VAL_COMPLEXES,
                   "one_complex_per_pinder_cluster": True, "max_residues": MAX_RESIDUES},
        "initialization": "identical scratch oGQT parameters for all four arms",
        "follow_up": "zero-shot pretrained oGQT, then pretrained vanilla and scratch-winning weighting arm",
        "arms": {"vanilla": [False, False], "burial": [True, False],
                 "interface": [False, True], "both": [True, True]},
        "targets": {"state": "pKAI(state) - fixed pKPDB PK_MOD",
                    "paired": "state shift AB - state shift free; PK_MOD cancels"},
        "loss": "equal sum of state-shift MSE over AB/free and binding-shift MSE",
        "weight_normalization": {"definition": "mean complex-level site weight over selected training complexes",
                                 "raw_means": norms},
        "optimizer": "AdamW, weight decay 1e-4, global gradient clip 1",
        "schedule": "1e-3 through epoch 10, cosine to 1e-5; 20 epoch cap; patience 8",
        "checkpoint_selection": "minimum unweighted validation state MAE + interface binding-shift MAE",
        "test_data_included": False,
    }
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "cohort.json", {"records": records, "train": [r["id"] for r in train],
                "val": [r["id"] for r in val], "normalization": norms,
                "source": str(source(root)), "source_summary_sha256": digest(source(root) / "summary.json")})
    atomic_json(out / "registration.json", {"passed": True, "records": len(records),
                "train_sites": sum(r["q"] for r in train), "val_sites": sum(r["q"] for r in val),
                "train_interface_sites": sum(r["q_interface"] for r in train),
                "val_interface_sites": sum(r["q_interface"] for r in val),
                "role_counts": dict(Counter(r["ctype"] for r in records)), "code_hashes": code_hashes(),
                "protocol_sha256": digest(out / "protocol.json"), "cohort_sha256": digest(out / "cohort.json")})


def _read_cif_gz(path):
    with gzip.open(path, "rt") as stream: file = pdbx.CIFFile.read(stream)
    return pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True, include_bonds=True)


def _initialize_worker(root, selected):
    global _ROOT, _SELECTED
    _ROOT = Path(root); _SELECTED = selected


def _prepare_one(record):
    root = _ROOT; cid = record["id"]; split = record["split"]
    src = source(root) / "entries" / cid; out = experiment_root(root) / "graphs" / cid
    path = out / "graph.npz"; receipt_path = out / "receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if digest(path) == receipt["sha256"]: return receipt
    for name, expected in record["source_hashes"].items():
        if digest(src / name) != expected: raise AssertionError((cid, name, "source changed"))
    topology = load_topology(_read_cif_gz(src / "AB.cif.gz"), gap_policy="cap", freeze_disulfides=True)
    graph, frames_valid = geometry(topology.backbone, topology.chain_index, radius=20.0)
    # Backbone-only contract: disulfide identity is not observable from N/CA/C.
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[topology.native_index],
        np.stack((topology.nterm, topology.cterm, np.zeros(topology.n_residues, bool), frames_valid), axis=-1)), axis=-1)
    graph.update(nodes=nodes.astype(np.float32), node_mask=np.ones(topology.n_residues, bool))
    paired = _paired_rows(src, split); lookup = {(key.chain, key.number, key.insertion): i for i, key in enumerate(topology.keys)}
    queries = []; targets = []; keys = []; burial = []; interface_weight = []; interface = []; distance = []; rsa = []
    for row in paired:
        residue_key = row["key"][:3]
        if residue_key not in lookup: continue
        group = GROUPS.index(row["key"][3]); queries.append((lookup[residue_key], group))
        targets.append((row["target_ab"], row["target_free"])); keys.append(list(row["key"]))
        burial.append(row["w_burial"]); interface_weight.append(row["w_interface"])
        interface.append(row["interface"]); distance.append(row["partner_distance_A"]); rsa.append(row["rsa_free"])
    if not queries or not any(interface): raise ValueError((cid, "no graph-mapped paired interface sites"))
    query = np.asarray(queries, np.int32); graph.update(query_residue=query[:, 0], query_group=query[:, 1])
    site = build_site_graph(topology.backbone, topology.chain_index, nodes, query[:, 0], query[:, 1])
    # For GQT, removing the partner is exactly removal of cross-chain graph edges;
    # every retained node feature is local backbone information.
    residue_free = graph["edge_mask"] & (graph["edge"][..., -1] > 0.5)
    site_free = site["site_edge_mask"] & (site["site_edge"][..., 31] > 0.5)
    payload = {**graph, **site,
        "branch_edge_mask": np.stack((graph["edge_mask"], residue_free)),
        "branch_site_edge_mask": np.stack((site["site_edge_mask"], site_free)),
        "targets": np.asarray(targets, np.float32).T,
        "w_burial": np.asarray(burial, np.float32), "w_interface": np.asarray(interface_weight, np.float32),
        "interface": np.asarray(interface, bool), "partner_distance_A": np.asarray(distance, np.float32),
        "rsa_free": np.asarray(rsa, np.float32)}
    out.mkdir(parents=True, exist_ok=True); pending = out / f"graph.npz.pending-{os.getpid()}"
    with pending.open("wb") as stream: np.savez(stream, **payload)
    os.replace(pending, path)
    receipt = {"id": cid, "split": split, "cluster_id": record["cluster_id"], "ctype": record["ctype"],
        "n": topology.n_residues, "k": graph["neighbors"].shape[1], "q": len(queries),
        "s": len(site["site_type"]), "sk": site["site_neighbors"].shape[1],
        "q_interface": int(np.sum(interface)), "keys": keys, "sha256": digest(path)}
    atomic_json(receipt_path, receipt); return receipt


def _bucket_n(n):
    for bound in (128, 256, 384, 512, 640, 768):
        if n <= bound: return str(bound)
    raise ValueError(n)


def prepare(root):
    root = Path(root); cohort = read(experiment_root(root) / "cohort.json")
    records = cohort["records"]; workers = min(int(os.environ["SLURM_CPUS_PER_TASK"]), 96)
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker, initargs=(str(root), None)) as pool:
        receipts = []
        for number, receipt in enumerate(pool.map(_prepare_one, records, chunksize=1), 1):
            receipts.append(receipt)
            if number % 100 == 0: print(json.dumps({"graphs": number, "total": len(records)}), flush=True)
    capacities = {}
    for name in sorted({_bucket_n(row["n"]) for row in receipts}, key=int):
        members = [row for row in receipts if _bucket_n(row["n"]) == name]
        maxima = [max(row[key] for row in members) for key in ("k", "q", "s", "sk")]
        capacities[name] = [int(name), *[int(np.ceil(value / 32) * 32) for value in maxima]]
    manifest = {"records": receipts, "capacities": capacities, "normalization": cohort["normalization"],
        "cohort_sha256": digest(experiment_root(root) / "cohort.json"), "protocol_sha256": digest(experiment_root(root) / "protocol.json"),
        "architecture": {"width": 44, "ff": 88, "node_dim": 24}, "batch_sizes": {"128": 8, "256": 8, "384": 8, "512": 4, "640": 2, "768": 2},
        "test_data_included": False}
    atomic_json(experiment_root(root) / "manifest.json", manifest)
    atomic_json(experiment_root(root) / "preparation.json", {"passed": True, "structures": len(receipts),
        "sites": sum(row["q"] for row in receipts), "interface_sites": sum(row["q_interface"] for row in receipts),
        "capacities": capacities, "code_hashes": code_hashes(), "manifest_sha256": digest(experiment_root(root) / "manifest.json")})


BASE_FIELDS = ("nodes", "node_mask", "neighbors", "edge", "switch", "query_residue", "query_group",
               "site_residue", "site_type", "site_mask", "site_neighbors", "site_edge", "site_switch", "query_site")
RAW_FIELDS = BASE_FIELDS + ("edge_mask", "site_edge_mask", "branch_edge_mask", "branch_site_edge_mask",
    "targets", "w_burial", "w_interface", "interface", "partner_distance_A", "rsa_free")


def _pad(value, shape):
    output = np.zeros(shape, value.dtype)
    output[tuple(slice(0, size) for size in value.shape)] = value
    return output


def _raw_shape(name, row):
    n, k, q, s, sk = (int(row[key]) for key in ("n", "k", "q", "s", "sk"))
    return {"nodes": (n, 24), "node_mask": (n,), "neighbors": (n, k), "edge": (n, k, 20),
        "edge_mask": (n, k), "switch": (n, k), "query_residue": (q,), "query_group": (q,),
        "site_residue": (s,), "site_type": (s,), "site_mask": (s,), "site_neighbors": (s, sk),
        "site_edge": (s, sk, 34), "site_edge_mask": (s, sk), "site_switch": (s, sk), "query_site": (q,),
        "branch_edge_mask": (2, n, k), "branch_site_edge_mask": (2, s, sk), "targets": (2, q),
        "w_burial": (q,), "w_interface": (q,), "interface": (q,), "partner_distance_A": (q,),
        "rsa_free": (q,)}[name]


def _mmap_fingerprint(records):
    value = [[row["id"], row["sha256"], row["n"], row["k"], row["q"], row["s"], row["sk"]] for row in records]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def build_mmap(root):
    base = experiment_root(root); manifest = read(base / "manifest.json"); records = manifest["records"]
    destination = base / "mmap-v1"
    if (destination / "verification.json").exists():
        store = PairedMMap(destination, records); store.close(); return
    if destination.exists(): raise FileExistsError(destination)
    first = base / "graphs" / records[0]["id"] / "graph.npz"
    if digest(first) != records[0]["sha256"]: raise AssertionError(records[0]["id"])
    with np.load(first, allow_pickle=False) as handle:
        if set(handle.files) != set(RAW_FIELDS): raise AssertionError(handle.files)
        dtypes = {name: handle[name].dtype.str for name in RAW_FIELDS}
    offsets = np.zeros((len(records) + 1, len(RAW_FIELDS)), np.int64)
    for index, row in enumerate(records):
        offsets[index + 1] = offsets[index] + [int(np.prod(_raw_shape(name, row))) for name in RAW_FIELDS]
    pending = destination.parent / f".{destination.name}.pending-{os.getpid()}"; pending.mkdir()
    arrays = {name: np.lib.format.open_memmap(pending / f"{name}.npy", mode="w+", dtype=np.dtype(dtypes[name]),
        shape=(int(offsets[-1, field]),)) for field, name in enumerate(RAW_FIELDS)}
    for index, row in enumerate(records):
        path = base / "graphs" / row["id"] / "graph.npz"
        if digest(path) != row["sha256"]: raise AssertionError(row["id"])
        with np.load(path, allow_pickle=False) as handle:
            for field, name in enumerate(RAW_FIELDS):
                value = handle[name]; expected = _raw_shape(name, row)
                if value.shape != expected or value.dtype.str != dtypes[name]:
                    raise AssertionError((row["id"], name, value.shape, expected, value.dtype.str, dtypes[name]))
                start, stop = offsets[index, field], offsets[index + 1, field]
                arrays[name][start:stop] = value.reshape(-1)
                if not np.array_equal(arrays[name][start:stop].reshape(expected), value, equal_nan=True):
                    raise AssertionError((row["id"], name, "mmap write mismatch"))
        if (index + 1) % 100 == 0: print(json.dumps({"mmap": index + 1, "total": len(records)}), flush=True)
    for value in arrays.values(): value.flush()
    arrays.clear()
    np.savez(pending / "index.npz", complex_id=np.asarray([row["id"] for row in records]), offsets=offsets)
    metadata = {"fields": list(RAW_FIELDS), "dtypes": dtypes, "records_fingerprint": _mmap_fingerprint(records)}
    atomic_json(pending / "metadata.json", metadata)
    atomic_json(pending / "verification.json", {"passed": True, "records": len(records),
        "all_source_hashes_checked": True, "all_fields_read_back_identically": True,
        "records_fingerprint": metadata["records_fingerprint"], "code_hashes": code_hashes(),
        "bytes": sum(path.stat().st_size for path in pending.iterdir())})
    os.replace(pending, destination)


def revalidate_mmap(root):
    """Rebind an unchanged, verified store to the current reader code."""
    base = experiment_root(root); manifest = read(base / "manifest.json")
    verification = read(base / "mmap-v1/verification.json")
    if not verification["passed"] or not verification["all_source_hashes_checked"] or not verification["all_fields_read_back_identically"]:
        raise AssertionError("mmap was not fully verified")
    store = PairedMMap(base / "mmap-v1", manifest["records"]); store.close()
    verification["code_hashes"] = code_hashes()
    verification["reader_revalidated"] = True
    atomic_json(base / "mmap-v1/verification.json", verification)


class PairedMMap:
    def __init__(self, path, records):
        self.path = Path(path); verification = read(self.path / "verification.json"); metadata = read(self.path / "metadata.json")
        if not verification["passed"] or metadata["records_fingerprint"] != _mmap_fingerprint(records):
            raise AssertionError("mmap verification/fingerprint")
        with np.load(self.path / "index.npz", allow_pickle=False) as index:
            ids = index["complex_id"].astype(str); self.offsets = index["offsets"]
        self.by_id = {cid: index for index, cid in enumerate(ids)}; self.records = {row["id"]: row for row in records}
        self.arrays = {name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in RAW_FIELDS}
        # Training visits randomized records.  Sequential readahead on these
        # field-major files otherwise faults gigabytes of adjacent, unused
        # records into memory for every small batch.
        for value in self.arrays.values():
            mapped = getattr(value, "_mmap", None)
            if mapped is not None and hasattr(mapped, "madvise"):
                mapped.madvise(mmap.MADV_RANDOM)
    def raw(self, cid):
        index = self.by_id[cid]; row = self.records[cid]; output = {}
        for field, name in enumerate(RAW_FIELDS):
            value = self.arrays[name][self.offsets[index, field]:self.offsets[index + 1, field]].reshape(_raw_shape(name, row))
            if value.flags.writeable: raise AssertionError(name)
            output[name] = value
        return output
    def close(self): self.arrays.clear()


def _load_one(raw, row, capacity, norms):
    n, k, q, s, sk = capacity
    shapes = {"nodes": (n, 24), "node_mask": (n,), "neighbors": (n, k), "edge": (n, k, 20), "switch": (n, k),
        "query_residue": (q,), "query_group": (q,), "site_residue": (s,), "site_type": (s,), "site_mask": (s,),
        "site_neighbors": (s, sk), "site_edge": (s, sk, 34), "site_switch": (s, sk), "query_site": (q,)}
    graph = {name: np.stack((_pad(raw[name], shapes[name]),) * 2) for name in BASE_FIELDS}
    graph["edge_mask"] = _pad(raw["branch_edge_mask"], (2, n, k))
    graph["site_edge_mask"] = _pad(raw["branch_site_edge_mask"], (2, s, sk))
    target = _pad(raw["targets"], (2, q)); mask = np.arange(q) < row["q"]
    burial = _pad(raw["w_burial"] / norms["burial"], (q,))
    interface_weight = _pad(raw["w_interface"] / norms["interface"], (q,))
    metadata = {name: _pad(raw[name], (q,)) for name in ("interface", "partner_distance_A", "rsa_free")}
    return graph, target, mask, burial, interface_weight, metadata


class Loader:
    def __init__(self, base, manifest):
        self.base = Path(base); self.manifest = manifest; self.pool = ThreadPoolExecutor(max_workers=4)
        store = Path(os.environ.get("PKATRAIN_PAIRED_MMAP", self.base / "mmap-v1"))
        self.store = PairedMMap(store, manifest["records"])
    def batch(self, ids):
        by_id = {row["id"]: row for row in self.manifest["records"]}; rows = [by_id[cid] for cid in ids]
        bucket = _bucket_n(rows[0]["n"])
        if any(_bucket_n(row["n"]) != bucket for row in rows): raise AssertionError("mixed bucket")
        capacity = self.manifest["capacities"][bucket]
        def load(row):
            return _load_one(self.store.raw(row["id"]), row, capacity, self.manifest["normalization"])
        # _load_one performs the page-faulting copies and padding.  Parallelize
        # that work, rather than only creating the cheap mmap views.
        values = list(self.pool.map(load, rows))
        return jax.tree.map(lambda *items: np.stack(items), *values)
    def close(self): self.pool.shutdown(); self.store.close()


def _prefetched(loader, plans):
    """Load one batch ahead while the current batch runs on the GPU."""
    plans = iter(plans)
    try:
        first = next(plans)
    except StopIteration:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(loader.batch, first)
        for following in plans:
            batch = pending.result()
            current = first
            pending = pool.submit(loader.batch, following)
            first = following
            yield current, batch
        yield first, pending.result()


def _plans(records, rng, manifest):
    groups = defaultdict(list)
    for row in records: groups[_bucket_n(row["n"])].append(row["id"])
    plans = []
    for bucket, ids in groups.items():
        rng.shuffle(ids); size = manifest["batch_sizes"][bucket]
        plans.extend(ids[i:i + size] for i in range(0, len(ids), size))
    rng.shuffle(plans); return plans


class PairedEngine:
    def __init__(self, params, arm):
        burial_on = arm in ("burial", "both"); interface_on = arm in ("interface", "both")
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0),
            optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        def predictions(p, graphs):
            batch, branches = graphs["nodes"].shape[:2]
            flat = jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs)
            result = jax.vmap(predict_ogqt_shift, in_axes=(None, 0))(p, flat)
            return result.reshape(batch, branches, -1)
        def components(p, graphs, targets, mask, w_burial, w_interface, valid):
            predicted = predictions(p, graphs)
            reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32)[graphs["query_group"][:, 0]]
            expected = targets - reference[:, None, :]
            state_weight = w_burial if burial_on else jnp.ones_like(w_burial)
            pair_weight = w_interface if interface_on else jnp.ones_like(w_interface)
            state_error = jnp.square(predicted - expected) * state_weight[:, None, :] * mask[:, None, :]
            pair_error = jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1])) * pair_weight * mask
            state = jnp.sum(state_error, axis=(1, 2)) / jnp.maximum(2 * jnp.sum(mask, axis=1), 1)
            pair = jnp.sum(pair_error, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
            denom = jnp.maximum(jnp.sum(valid), 1)
            return jnp.sum(jnp.where(valid, state, 0.0)) / denom, jnp.sum(jnp.where(valid, pair, 0.0)) / denom
        def objective(p, graphs, targets, mask, wb, wi, valid):
            state, pair = components(p, graphs, targets, mask, wb, wi, valid); return state + pair
        def step(p, state, graphs, targets, mask, wb, wi, valid, rate):
            loss, gradient = jax.value_and_grad(objective)(p, graphs, targets, mask, wb, wi, valid)
            finite = jnp.isfinite(loss) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            updates = jax.tree.map(lambda value: value * rate, updates)
            return optax.apply_updates(p, updates), state, loss, finite
        self.predictions = jax.jit(predictions); self.components = jax.jit(components); self.step = jax.jit(step)
    def update(self, params, state, batch, rate):
        graphs, targets, mask, wb, wi, _ = batch; valid = np.ones(len(targets), bool)
        params, state, loss, finite = self.step(params, state, graphs, targets, mask, wb, wi, valid, rate)
        if not bool(finite): raise FloatingPointError("nonfinite paired oGQT update")
        return params, state, float(loss)


def _metrics(rows):
    def mae(subset, field):
        values = [abs(row[field]) for row in subset]; return float(np.mean(values)) if values else None
    interface = [row for row in rows if row["interface"]]
    return {"sites": len(rows), "interface_sites": len(interface),
        "state_mae": mae(rows, "state_error"), "paired_mae": mae(rows, "paired_error"),
        "interface_paired_mae": mae(interface, "paired_error"),
        "distance_bins": {name: {"sites": len(subset), "paired_mae": mae(subset, "paired_error")} for name, subset in (
            ("<=4", [r for r in rows if r["distance"] <= 4]), ("4-6", [r for r in rows if 4 < r["distance"] <= 6]),
            ("6-10", [r for r in rows if 6 < r["distance"] <= 10]), (">10", [r for r in rows if r["distance"] > 10]))}}


def evaluate(base, manifest, records, engine, params, predictions_path=None):
    loader = Loader(base, manifest); rows = []; by_id = {record["id"]: record for record in records}
    grouped = defaultdict(list)
    for record in records: grouped[_bucket_n(record["n"])].append(record["id"])
    plans = []
    for bucket in sorted(grouped, key=int):
        size = manifest["batch_sizes"][bucket]; ids = grouped[bucket]
        plans.extend(ids[index:index + size] for index in range(0, len(ids), size))
    for ids, batch in _prefetched(loader, plans):
        graphs, targets, mask, _, _, metadata = batch
        batch_predictions = np.asarray(engine.predictions(params, graphs))
        for batch_index, cid in enumerate(ids):
            record = by_id[cid]; active = mask[batch_index]
            predicted = batch_predictions[batch_index][:, active]
            groups = graphs["query_group"][batch_index, 0, active]; reference = np.asarray(PKPDB_PK_MOD)[groups]
            expected = targets[batch_index][:, active] - reference[None]
            for index, key in enumerate(record["keys"]):
                rows.append({"complex_id": cid, **dict(zip(KEY_FIELDS, key)),
                    "teacher_ab": float(targets[batch_index, 0, index]), "teacher_free": float(targets[batch_index, 1, index]),
                    "predicted_ab": float(predicted[0, index] + reference[index]),
                    "predicted_free": float(predicted[1, index] + reference[index]),
                    "state_error": float(np.mean(np.abs(predicted[:, index] - expected[:, index]))),
                    "paired_error": float((predicted[0, index] - predicted[1, index]) - (expected[0, index] - expected[1, index])),
                    "interface": bool(metadata["interface"][batch_index, index]),
                    "distance": float(metadata["partner_distance_A"][batch_index, index]),
                    "rsa_free": float(metadata["rsa_free"][batch_index, index])})
    loader.close()
    if predictions_path is not None:
        with Path(predictions_path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return _metrics(rows)


def train(root, arm, *, smoke=False, initialization="scratch", schedule="standard"):
    if arm not in ARMS: raise ValueError(arm)
    if initialization not in ("scratch", "pretrained") or schedule not in ("standard", "low"):
        raise ValueError((initialization, schedule))
    if initialization == "pretrained" and arm != "vanilla": raise ValueError((initialization, arm))
    root = Path(root); base = experiment_root(root); manifest = read(base / "manifest.json")
    prep = read(base / "preparation.json"); mmap = read(base / "mmap-v1/verification.json")
    if not prep["passed"] or not mmap["passed"] or mmap["code_hashes"] != code_hashes():
        raise AssertionError("preparation/mmap/code mismatch")
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = PairedEngine(params, arm); state = engine.optimizer.init(params); rng = np.random.default_rng(SEED)
    parent = None
    if initialization == "pretrained":
        source_run = pretrained_root(root); source_verification = read(source_run / "verification.json")
        if not source_verification["passed"] or source_verification["test_data_included"]:
            raise AssertionError("invalid pretrained checkpoint")
        source_epoch = int(source_verification["selected_epoch"])
        source_checkpoint = source_run / "checkpoints" / f"epoch-{source_epoch:03d}"
        params, _, source_metadata = load_checkpoint(source_checkpoint, (params, state))
        if int(source_metadata["parameter_count"]) != sum(value.size for value in jax.tree.leaves(params)):
            raise AssertionError("pretrained parameter count")
        state = engine.optimizer.init(params)
        parent = {"checkpoint": str(source_checkpoint), "checkpoint_sha256": digest(source_checkpoint / "state.npz"),
                  "selected_epoch": source_epoch, "verification_sha256": digest(source_run / "verification.json")}
    folder = arm if initialization == "scratch" else f"pretrained-{arm}-{schedule}"
    run = base / folder / ("smoke" if smoke else "seed-17"); run.mkdir(parents=True, exist_ok=True)
    by_split = {name: [row for row in manifest["records"] if row["split"] == name] for name in ("train", "val")}
    loader = Loader(base, manifest); provenance = {"arm": arm, "seed": SEED, "initialization": initialization,
        "schedule": schedule, "parent": parent, "manifest_sha256": digest(base / "manifest.json"),
        "code_hashes": code_hashes()}
    atomic_json(run / "run.json", provenance)
    best = None; stalled = 0; history = []; started = time.monotonic()
    if initialization == "pretrained" and not smoke:
        zero_shot = evaluate(base, manifest, by_split["val"], engine, params, run / "zero_shot_predictions.csv")
        selection = zero_shot["state_mae"] + zero_shot["interface_paired_mae"]
        best = {"epoch": 0, "selection": selection, "validation": zero_shot}
        atomic_json(run / "zero-shot.json", {"selection": selection, "validation": zero_shot,
            "predictions_sha256": digest(run / "zero_shot_predictions.csv")})
        atomic_json(run / "best.json", best)
        save_checkpoint(run / "checkpoints" / "epoch-000", params, state, {**provenance, "epoch": 0})
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        plans = _plans(by_split["train"], rng, manifest)
        plan_digest = hashlib.sha256(json.dumps(plans).encode()).hexdigest()
        if smoke: plans = plans[:1]
        losses = []; began = time.monotonic()
        for number, (_, batch) in enumerate(_prefetched(loader, plans), 1):
            rate = learning_rate(epoch, number, len(plans),
                start=(1e-3 if schedule == "standard" else 1e-4),
                end=(1e-5 if schedule == "standard" else 1e-6))
            params, state, loss = engine.update(params, state, batch, rate); losses.append(loss)
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True, "loss": float(np.mean(losses)),
                "batch_plan_digest": plan_digest, "device_memory": jax.local_devices()[0].memory_stats(), **provenance})
            loader.close(); return
        validation = evaluate(base, manifest, by_split["val"], engine, params)
        selection = validation["state_mae"] + validation["interface_paired_mae"]
        if best is None or selection < best["selection"] - MIN_DELTA:
            best = {"epoch": epoch, "selection": selection, "validation": validation}; stalled = 0
            atomic_json(run / "best.json", best)
        else: stalled += 1
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation, "selection": selection,
            "seconds": time.monotonic() - began, "batch_plan_digest": plan_digest, "best_epoch": best["epoch"], "stalled": stalled}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"arm": arm, **row}), flush=True)
        if stalled >= PATIENCE: break
    loader.close(); params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final = evaluate(base, manifest, by_split["val"], engine, params, run / "validation_predictions.csv")
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {"passed": True, "epochs_completed": len(history), "selected_epoch": best["epoch"],
        "wall_seconds": time.monotonic() - started, "device_memory": jax.local_devices()[0].memory_stats(),
        "predictions_sha256": digest(run / "validation_predictions.csv"), "test_data_included": False, **provenance})


def report(root):
    base = experiment_root(root); rows = []
    histories = []
    for arm in ARMS:
        run = base / arm / "seed-17"; verification = read(run / "verification.json"); final = read(run / "final.json")
        if not verification["passed"] or digest(run / "validation_predictions.csv") != verification["predictions_sha256"]: raise AssertionError(arm)
        histories.append(read(run / "history.json"))
        rows.append({"arm": arm, "epoch": verification["selected_epoch"], "minutes": verification["wall_seconds"] / 60,
                     "peak_vram_gib": verification["device_memory"]["peak_bytes_in_use"] / 2**30, **final})
    shortest = min(map(len, histories))
    for epoch in range(shortest):
        if len({history[epoch]["batch_plan_digest"] for history in histories}) != 1: raise AssertionError((epoch, "batch mismatch"))
    atomic_json(base / "summary.json", rows)
    lines = ["# Scratch oGQT structural-weight factorial", "",
        "One seed (17), 5,000 PINDER training complexes and 400 validation complexes, one structure per PINDER cluster. All arms start from the identical scratch parameters and use matched batches. Targets are pKAI state shifts from fixed PK_MOD and paired AB-minus-free shifts.", "",
        "| Arm | State MAE | Paired MAE | Interface paired MAE | <=4 A | 4-6 A | 6-10 A | >10 A | Epoch | Time (min) | Peak VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        bins = row["distance_bins"]
        lines.append(f"| {row['arm']} | {row['state_mae']:.4f} | {row['paired_mae']:.4f} | {row['interface_paired_mae']:.4f} | "
            f"{bins['<=4']['paired_mae']:.4f} | {bins['4-6']['paired_mae']:.4f} | {bins['6-10']['paired_mae']:.4f} | "
            f"{bins['>10']['paired_mae']:.4f} | {row['epoch']} | {row['minutes']:.1f} | {row['peak_vram_gib']:.2f} |")
    lines += ["", "Checkpoint selection used the same unweighted state-MAE plus interface-paired-MAE criterion for every arm. No test structures or labels were read.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "report-verification.json", {"passed": True, "matched_batch_plans": True,
                "report_sha256": digest(base / "report.md"), "test_data_included": False})


def transfer_report(root):
    base = experiment_root(root)
    if read(base / "pretrained-vanilla-standard/seed-17/zero-shot.json")["validation"] != \
            read(base / "pretrained-vanilla-low/seed-17/zero-shot.json")["validation"]:
        raise AssertionError("zero-shot mismatch")
    configurations = (("scratch vanilla", base / "vanilla/seed-17", None),
        ("pretrained zero-shot", base / "pretrained-vanilla-standard/seed-17", "zero-shot"),
        ("pretrained standard LR", base / "pretrained-vanilla-standard/seed-17", None),
        ("pretrained low LR", base / "pretrained-vanilla-low/seed-17", None))
    rows = []
    for name, run, mode in configurations:
        if mode == "zero-shot":
            payload = read(run / "zero-shot.json"); metrics = payload["validation"]; epoch = 0
            if digest(run / "zero_shot_predictions.csv") != payload["predictions_sha256"]: raise AssertionError(name)
        else:
            verification = read(run / "verification.json"); metrics = read(run / "final.json")
            if not verification["passed"] or verification["test_data_included"]: raise AssertionError(name)
            if digest(run / "validation_predictions.csv") != verification["predictions_sha256"]: raise AssertionError(name)
            epoch = verification["selected_epoch"]
        rows.append({"configuration": name, "epoch": epoch, **metrics})
    standard_history = read(base / "pretrained-vanilla-standard/seed-17/history.json")
    low_history = read(base / "pretrained-vanilla-low/seed-17/history.json")
    scratch_history = read(base / "vanilla/seed-17/history.json")
    common = min(len(standard_history), len(low_history), len(scratch_history))
    for epoch in range(common):
        if len({standard_history[epoch]["batch_plan_digest"], low_history[epoch]["batch_plan_digest"],
                scratch_history[epoch]["batch_plan_digest"]}) != 1: raise AssertionError((epoch, "batch mismatch"))
    lines = ["# Pretrained oGQT Siamese transfer", "",
        "All trained rows use the unweighted vanilla state-plus-paired objective on the same 5,000/400 PINDER split. The two pretrained arms start from the selected epoch-16 pKPDB oGQT checkpoint; epoch zero is eligible for selection. No test data were read.", "",
        "| Configuration | State MAE | Paired MAE | Interface paired MAE | <=4 A | 4-6 A | 6-10 A | >10 A | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        bins = row["distance_bins"]
        lines.append(f"| {row['configuration']} | {row['state_mae']:.4f} | {row['paired_mae']:.4f} | "
            f"{row['interface_paired_mae']:.4f} | {bins['<=4']['paired_mae']:.4f} | {bins['4-6']['paired_mae']:.4f} | "
            f"{bins['6-10']['paired_mae']:.4f} | {bins['>10']['paired_mae']:.4f} | {row['epoch']} |")
    destination = base / "transfer-report.md"; destination.write_text("\n".join(lines) + "\n")
    atomic_json(base / "transfer-summary.json", rows)
    atomic_json(base / "transfer-report-verification.json", {"passed": True, "matched_batch_plans": True,
        "report_sha256": digest(destination), "test_data_included": False})


def verify_smokes(root):
    base = experiment_root(root); rows = []
    for arm in ARMS:
        value = read(base / arm / "smoke/verification.json")
        if not value["passed"] or not value["finite_update"] or value["code_hashes"] != code_hashes(): raise AssertionError(arm)
        rows.append({"arm": arm, "loss": value["loss"], "batch_plan_digest": value["batch_plan_digest"]})
    if len({row["batch_plan_digest"] for row in rows}) != 1: raise AssertionError("smoke batch mismatch")
    atomic_json(base / "smoke-verification.json", {"passed": True, "rows": rows, "code_hashes": code_hashes()})


def verify_pretrained_smokes(root):
    base = experiment_root(root); rows = []
    for schedule in ("standard", "low"):
        value = read(base / f"pretrained-vanilla-{schedule}/smoke/verification.json")
        if not value["passed"] or not value["finite_update"] or value["code_hashes"] != code_hashes():
            raise AssertionError(schedule)
        if value["initialization"] != "pretrained" or value["schedule"] != schedule or value["parent"] is None:
            raise AssertionError((schedule, "provenance"))
        rows.append({"schedule": schedule, "loss": value["loss"], "batch_plan_digest": value["batch_plan_digest"]})
    if len({row["batch_plan_digest"] for row in rows}) != 1: raise AssertionError("pretrained smoke batch mismatch")
    atomic_json(base / "pretrained-smoke-verification.json", {"passed": True, "rows": rows,
        "code_hashes": code_hashes(), "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    gpu = action in ("smoke", "train", "pretrained-smoke", "pretrained")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu,
                    allow_comp1400=(gpu or action in ("mmap", "revalidate-mmap", "verify-smokes",
                        "verify-pretrained-smokes", "report", "transfer-report")))
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "prepare": prepare(root)
    elif action == "mmap": build_mmap(root)
    elif action == "revalidate-mmap": revalidate_mmap(root)
    elif action == "smoke": train(root, sys.argv[2], smoke=True)
    elif action == "verify-smokes": verify_smokes(root)
    elif action == "verify-pretrained-smokes": verify_pretrained_smokes(root)
    elif action == "train": train(root, sys.argv[2])
    elif action == "pretrained-smoke": train(root, "vanilla", smoke=True,
        initialization="pretrained", schedule=sys.argv[2])
    elif action == "pretrained": train(root, "vanilla", initialization="pretrained", schedule=sys.argv[2])
    elif action == "report": report(root)
    elif action == "transfer-report": transfer_report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
