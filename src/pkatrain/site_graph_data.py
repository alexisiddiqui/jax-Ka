"""Candidate titratable-site graphs aligned to the frozen backbone cohort."""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from jaxpropka.parameters import GROUP_AA
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_regraph import _coordinates
from .records import read


SITE_EDGE_DIM = 34
RADIUS = 20.0
FIELDS = ("site_residue", "site_type", "site_mask", "site_neighbors", "site_edge",
          "site_edge_mask", "site_switch", "query_site")
_ROOT = None
_PARENT = None


def experiment_root(root): return root / "pretraining/gqt-site-tokens-v1"
def source(root): return root / "pretraining/gqt-neighbor-cutoff-v1-triton/sources-25A/backbone"
def sidecar(root): return experiment_root(root) / "site-graphs"


def frames(backbone):
    ca = backbone[:, 1]; x = backbone[:, 2] - ca; y = backbone[:, 0] - ca
    xn = np.linalg.norm(x, axis=-1, keepdims=True); x = x / np.maximum(xn, 1e-6)
    y = y - (y * x).sum(-1, keepdims=True) * x
    yn = np.linalg.norm(y, axis=-1, keepdims=True); y = y / np.maximum(yn, 1e-6)
    valid = (xn[:, 0] > 1e-5) & (yn[:, 0] > 1e-5)
    return ca, np.stack((x, y, np.cross(x, y)), axis=-1), valid


def candidate_sites(nodes):
    native = np.argmax(nodes[:, :20], axis=1); sites = []
    aa_to_group = {int(aa): group for group, aa in enumerate(GROUP_AA)}
    for residue in range(len(nodes)):
        if int(native[residue]) in aa_to_group: sites.append((residue, aa_to_group[int(native[residue])]))
        if bool(nodes[residue, 20]): sites.append((residue, 7))
        if bool(nodes[residue, 21]): sites.append((residue, 8))
    return np.asarray(sites, np.int32)


def build_site_graph(backbone, chain, nodes, query_residue, query_group):
    sites = candidate_sites(nodes); site_residue = sites[:, 0]; site_type = sites[:, 1]
    lookup = {(int(residue), int(group)): index for index, (residue, group) in enumerate(sites)}
    if len(lookup) != len(sites): raise AssertionError("Duplicate candidate site")
    if np.any(np.asarray(query_group) == 6): raise AssertionError("pKPDB unexpectedly contains ARG targets")
    query_site = np.asarray([lookup[int(r), int(g)] for r, g in zip(query_residue, query_group)], np.int32)
    ca, frame, valid = frames(backbone); site_ca = ca[site_residue]
    rows = cKDTree(site_ca).query_ball_point(site_ca, RADIUS)
    width = max(map(len, rows)); neighbors = np.zeros((len(sites), width), np.int32)
    mask = np.zeros((len(sites), width), bool)
    for index, row in enumerate(rows):
        row = sorted(row); neighbors[index, :len(row)] = row; mask[index, :len(row)] = True
    ri = site_residue; rj = site_residue[neighbors]
    delta = ca[rj] - ca[ri, None]; distance = np.sqrt((delta * delta).sum(-1) + 1e-8)
    fi = frame[ri]; fj = frame[rj]
    direction_i = np.einsum("skc,scl->skl", delta, fi) / distance[..., None]
    direction_j = np.einsum("skc,skcl->skl", -delta, fj) / distance[..., None]
    orientation = np.matmul(np.swapaxes(fi[:, None], -1, -2), fj).reshape((len(sites), width, 9))
    pair_valid = valid[ri, None] & valid[rj]
    direction_i *= valid[ri, None, None]; direction_j *= valid[rj][..., None]
    orientation *= pair_valid[..., None]
    rbf = np.exp(-((distance[..., None] - np.linspace(0, RADIUS, 16)) / 1.5) ** 2)
    same_chain = chain[ri, None] == chain[rj]; same_residue = ri[:, None] == rj
    sequence = np.zeros(len(chain), np.int32); counters = {}
    for index, name in enumerate(chain):
        sequence[index] = counters.get(int(name), 0); counters[int(name)] = sequence[index] + 1
    separation = np.where(same_chain, np.clip((sequence[rj] - sequence[ri, None]) / 32.0, -1, 1), 0.0)
    edge = np.concatenate((rbf, direction_i, direction_j, orientation,
        same_chain[..., None], same_residue[..., None], separation[..., None]), axis=-1).astype(np.float32)
    if edge.shape[-1] != SITE_EDGE_DIM: raise AssertionError(edge.shape)
    switch = np.where(distance < RADIUS - 2, 1.0,
        0.5 * (1 + np.cos(np.pi * np.clip((distance - RADIUS + 2) / 2, 0, 1)))).astype(np.float32)
    return {"site_residue": site_residue.astype(np.int32), "site_type": site_type.astype(np.int32),
            "site_mask": np.ones(len(sites), bool), "site_neighbors": neighbors,
            "site_edge": edge, "site_edge_mask": mask, "site_switch": switch,
            "query_site": query_site}


def _initialize(runtime, parent):
    global _ROOT, _PARENT
    _ROOT = Path(runtime); _PARENT = parent


def _one(row):
    root = _ROOT; cid = row["complex_id"]; out = sidecar(root) / "data" / cid
    path = out / "sites.npz"; receipt_path = out / "receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if digest(path) == receipt["sha256"]: return receipt
    source_path = source(root) / "data" / cid / "graph.npz"
    if digest(source_path) != row["sha256"]: raise AssertionError((cid, "source graph hash"))
    with np.load(source_path, allow_pickle=False) as handle:
        nodes = handle["nodes"]; query_residue = handle["query_residue"]; query_group = handle["query_group"]
    backbone, chain = _coordinates(root, _PARENT, row)
    graph = build_site_graph(backbone, chain, nodes, query_residue, query_group)
    out.mkdir(parents=True, exist_ok=True); pending = out / f"sites.npz.pending-{os.getpid()}"
    with pending.open("wb") as stream: np.savez_compressed(stream, **graph)
    os.replace(pending, path)
    counts = np.bincount(graph["site_type"], minlength=9)
    residue_counts = np.bincount(graph["site_residue"], minlength=row["n"])
    receipt = {"complex_id": cid, "split": row["split"], "component_id": row["component_id"],
        "n": row["n"], "q": row["q"], "s": len(graph["site_type"]), "sk": graph["site_neighbors"].shape[1],
        "site_edges": int(graph["site_edge_mask"].sum()), "arg_sites": int(counts[6]),
        "multi_token_residues": int(np.sum(residue_counts > 1)), "source_graph_sha256": row["sha256"],
        "sha256": digest(path)}
    atomic_json(receipt_path, receipt); return receipt


def test(root):
    parent = read(source(root) / "manifest.json"); row = parent["records"][0]
    _initialize(str(root), parent); receipt = _one(row)
    backbone, chain = _coordinates(root, parent, row)
    with np.load(source(root) / "data" / row["complex_id"] / "graph.npz", allow_pickle=False) as handle:
        nodes = handle["nodes"]; qr = handle["query_residue"]; qg = handle["query_group"]
    original = build_site_graph(backbone, chain, nodes, qr, qg)
    rng = np.random.default_rng(91); rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0: rotation[:, 0] *= -1
    transformed = build_site_graph(backbone @ rotation + np.asarray([17.0, -9.0, 4.0]), chain, nodes, qr, qg)
    discrete = ("site_residue", "site_type", "site_mask", "site_neighbors", "site_edge_mask", "query_site")
    for name in discrete: np.testing.assert_array_equal(original[name], transformed[name])
    max_edge_difference = float(np.max(np.abs(original["site_edge"] - transformed["site_edge"])))
    max_switch_difference = float(np.max(np.abs(original["site_switch"] - transformed["site_switch"])))
    if max_edge_difference > 2e-5 or max_switch_difference > 2e-5: raise AssertionError((max_edge_difference, max_switch_difference))
    destination = experiment_root(root); destination.mkdir(parents=True, exist_ok=True)
    with (destination / "rotation-probe.npz").open("wb") as stream: np.savez_compressed(stream, **transformed)
    atomic_json(destination / "site-graph-tests.json", {"passed": True, "complex_id": row["complex_id"],
        "sites": receipt["s"], "arg_sites": receipt["arg_sites"],
        "multi_token_residues": receipt["multi_token_residues"],
        "max_rotated_edge_difference": max_edge_difference,
        "max_rotated_switch_difference": max_switch_difference,
        "all_labelled_sites_mapped": True, "arg_has_no_label": True})


def prepare(root):
    parent = read(source(root) / "manifest.json"); workers = min(int(os.environ["SLURM_CPUS_PER_TASK"]), 48)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
            initargs=(str(root), parent)) as pool:
        receipts = []
        for number, receipt in enumerate(pool.map(_one, parent["records"], chunksize=1), 1):
            receipts.append(receipt)
            if number % 100 == 0: print(json.dumps({"site_graphs": number, "total": len(parent["records"])}), flush=True)
    capacities = {}
    for limit in (384, 768, 100000):
        members = [r for r in receipts if (384 if r["n"] <= 384 else 768 if r["n"] <= 768 else 100000) == limit]
        if members: capacities[str(limit)] = [int(np.ceil(max(r[key] for r in members) / 32) * 32) for key in ("s", "sk")]
    manifest = {"source": str(source(root)), "source_manifest_sha256": digest(source(root) / "manifest.json"),
        "records": receipts, "capacities": capacities, "radius_A": RADIUS, "edge_dim": SITE_EDGE_DIM,
        "candidate_groups": ["ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG", "NTERM", "CTERM"],
        "test_data_included": False}
    atomic_json(sidecar(root) / "manifest.json", manifest)
    atomic_json(sidecar(root) / "preparation.json", {"passed": True, "structures": len(receipts),
        "sites": sum(r["s"] for r in receipts), "arg_sites": sum(r["arg_sites"] for r in receipts),
        "multi_token_residues": sum(r["multi_token_residues"] for r in receipts),
        "site_edges": sum(r["site_edges"] for r in receipts)})


def _shape(name, s, sk, q):
    return {"site_residue": (s,), "site_type": (s,), "site_mask": (s,),
        "site_neighbors": (s, sk), "site_edge": (s, sk, SITE_EDGE_DIM),
        "site_edge_mask": (s, sk), "site_switch": (s, sk), "query_site": (q,)}[name]


def _fingerprint(records):
    raw = json.dumps([[r["complex_id"], r["sha256"], r["s"], r["sk"], r["q"]] for r in records], separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def build_mmap(root):
    base = sidecar(root); manifest = read(base / "manifest.json"); records = manifest["records"]
    destination = base / "mmap-v1"
    if (destination / "verification.json").exists(): SiteMMap(destination, records).close(); return
    if destination.exists(): raise FileExistsError(destination)
    first = base / "data" / records[0]["complex_id"] / "sites.npz"
    with np.load(first, allow_pickle=False) as handle: dtypes = {name: handle[name].dtype.str for name in FIELDS}
    offsets = np.zeros((len(records) + 1, len(FIELDS)), np.int64)
    for i, row in enumerate(records):
        offsets[i + 1] = offsets[i] + [int(np.prod(_shape(name, row["s"], row["sk"], row["q"]))) for name in FIELDS]
    pending = destination.parent / f".{destination.name}.pending-{os.getpid()}"; pending.mkdir()
    arrays = {name: np.lib.format.open_memmap(pending / f"{name}.npy", mode="w+", dtype=np.dtype(dtypes[name]),
              shape=(int(offsets[-1, j]),)) for j, name in enumerate(FIELDS)}
    for i, row in enumerate(records):
        path = base / "data" / row["complex_id"] / "sites.npz"
        if digest(path) != row["sha256"]: raise AssertionError(row["complex_id"])
        with np.load(path, allow_pickle=False) as handle:
            for j, name in enumerate(FIELDS):
                value = handle[name]; expected = _shape(name, row["s"], row["sk"], row["q"])
                if value.shape != expected: raise AssertionError((row["complex_id"], name, value.shape, expected))
                arrays[name][offsets[i, j]:offsets[i + 1, j]] = value.reshape(-1)
        if (i + 1) % 200 == 0: print(json.dumps({"site_mmap": i + 1, "total": len(records)}), flush=True)
    for value in arrays.values(): value.flush()
    arrays.clear()
    np.savez(pending / "index.npz", complex_id=np.asarray([r["complex_id"] for r in records]),
             s=np.asarray([r["s"] for r in records], np.int32), sk=np.asarray([r["sk"] for r in records], np.int32),
             q=np.asarray([r["q"] for r in records], np.int32), offsets=offsets)
    metadata = {"fields": list(FIELDS), "dtypes": dtypes, "fingerprint": _fingerprint(records)}
    atomic_json(pending / "metadata.json", metadata)
    files = {path.name: digest(path) for path in pending.iterdir()}
    atomic_json(pending / "verification.json", {"passed": True, "files": files,
                "bytes": sum(path.stat().st_size for path in pending.iterdir())})
    os.replace(pending, destination)


class SiteMMap:
    def __init__(self, path, records):
        self.path = Path(path); metadata = read(self.path / "metadata.json"); verification = read(self.path / "verification.json")
        if not verification["passed"] or metadata["fingerprint"] != _fingerprint(records): raise AssertionError(path)
        with np.load(self.path / "index.npz", allow_pickle=False) as index:
            self.ids = index["complex_id"].astype(str); self.s = index["s"]; self.sk = index["sk"]; self.q = index["q"]; self.offsets = index["offsets"]
        self.by_id = {cid: i for i, cid in enumerate(self.ids)}
        self.arrays = {name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in FIELDS}
    def raw(self, cid):
        i = self.by_id[cid]; s, sk, q = int(self.s[i]), int(self.sk[i]), int(self.q[i]); output = {}
        for j, name in enumerate(FIELDS):
            output[name] = self.arrays[name][self.offsets[i, j]:self.offsets[i + 1, j]].reshape(_shape(name, s, sk, q))
            if output[name].flags.writeable: raise AssertionError(name)
        return output
    def close(self): self.arrays.clear()


def pad_site(values, capacities, q_capacity):
    s, sk = capacities; output = {}
    shapes = {name: _shape(name, s, sk, q_capacity) for name in FIELDS}
    for name, value in values.items():
        destination = np.zeros(shapes[name], value.dtype)
        destination[tuple(slice(0, size) for size in value.shape)] = value; output[name] = destination
    return output


class SiteBatchLoader:
    def __init__(self, out, manifest, batch_size, arm):
        from .gqt_neighbor_cutoff import CutoffBatchLoader
        self._base = CutoffBatchLoader(out, manifest, batch_size, backend="mmap", cutoff=20.0)
        self.out = Path(out); self.manifest = manifest; self.batch_size = batch_size; self.arm = arm
        site_root = Path(manifest["site_source"]); site_manifest = read(site_root / "manifest.json")
        self.site_records = {r["complex_id"]: r for r in site_manifest["records"]}
        self.site_capacities = site_manifest["capacities"]
        # Keep the immutable canonical store as the default, while allowing
        # Slurm jobs to stage the same verified arrays on node-local storage.
        site_location = os.environ.get("PKATRAIN_SITE_MMAP_DIR")
        self.site_store = SiteMMap(site_location or site_root / "mmap-v1", site_manifest["records"])
        self.byid = self._base.byid; self.pool = self._base.pool
    def one(self, cid, capacities=None):
        from .graph_data import bucket
        graph, targets, eligible = self._base.one(cid, capacities)
        row = self.byid[cid]; q_capacity = graph["query_group"].shape[0]
        site = pad_site(self.site_store.raw(cid), self.site_capacities[bucket(row)], q_capacity)
        if self.arm == "site-distance": site["site_edge"][..., 22:31] = 0.0
        if self.arm == "no-arg":
            keep = site["site_mask"] & (site["site_type"] != 6)
            edge_keep = keep[:, None] & keep[site["site_neighbors"]]
            site["site_mask"] = keep; site["site_edge_mask"] &= edge_keep
            site["site_switch"] *= edge_keep
        graph.update(site); return graph, targets, eligible
    def load(self, cids, capacities=None):
        # Match BatchLoader.load while dispatching through this class's one().
        from .graph_batches import tightened_capacity
        import jax
        assert 0 < len(cids) <= self.batch_size
        if capacities is None: capacities = tightened_capacity(cids, self.byid, self.manifest)
        from itertools import repeat
        items = list(self.pool.map(self.one, cids, repeat(capacities)))
        valid = np.arange(self.batch_size) < len(items)
        while len(items) < self.batch_size:
            graph, target, mask = items[-1]
            items.append((graph, np.zeros_like(target), np.zeros_like(mask)))
        graph = jax.tree.map(lambda *x: np.stack(x), *[item[0] for item in items])
        return graph, np.stack([x[1] for x in items]), np.stack([x[2] for x in items]), valid
    def iterate(self, batches):
        from concurrent.futures import ThreadPoolExecutor
        def unpack(spec): return (spec["cids"], spec["capacities"]) if isinstance(spec, dict) else (spec, None)
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            first = unpack(batches[0]) if batches else None; future = prefetch.submit(self.load, *first) if first else None
            for index, spec in enumerate(batches):
                cids, _ = unpack(spec); inputs = future.result()
                following = unpack(batches[index + 1]) if index + 1 < len(batches) else None
                future = prefetch.submit(self.load, *following) if following else None
                yield cids, inputs
    def provenance(self): return {"residue": self._base.provenance(), "site_mmap": str(self.site_store.path), "arm": self.arm}
    def close(self): self._base.close(); self.site_store.close()


if __name__ == "__main__":
    import sys
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]))
    root = Path(os.environ["PKABENCH_RUNTIME"]); action = sys.argv[1]
    if action == "test": test(root)
    elif action == "prepare": prepare(root)
    elif action == "mmap": build_mmap(root)
    else: raise ValueError(action)
