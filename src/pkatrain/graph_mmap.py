"""Exact, read-only mmap storage for the frozen GQT graph arrays."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest


FORMAT = "gqt-graph-mmap-v1"
FIELDS = (
    "nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch",
    "query_residue", "query_group", "labels",
)


def records_fingerprint(records):
    value = [
        [row["complex_id"], row["sha256"], int(row["n"]), int(row["k"]), int(row["q"])]
        for row in records
    ]
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def default_bundle(out):
    """Place one bundle beside the canonical data directory shared by all arms."""
    data = Path(out, "data").resolve()
    return data.parent / "mmap-v1"


def _shape(name, n, k, q, *, node_dim, edge_dim):
    return {
        "nodes": (n, node_dim), "node_mask": (n,), "neighbors": (n, k),
        "edge": (n, k, edge_dim), "edge_mask": (n, k), "switch": (n, k),
        "query_residue": (q,), "query_group": (q,), "labels": (q,),
    }[name]


def build_bundle(out, manifest, destination=None):
    """Convert every source graph and prove exact field equality while writing."""
    out = Path(out); destination = Path(destination or default_bundle(out))
    if (destination / "verification.json").exists():
        store = GraphMMap(destination, manifest["records"])
        store.close()
        return destination
    if destination.exists():
        raise FileExistsError(f"Incomplete mmap destination exists: {destination}")

    records = manifest["records"]
    if not records:
        raise ValueError("Cannot build an empty graph bundle")
    first_path = out / "data" / records[0]["complex_id"] / "graph.npz"
    if digest(first_path) != records[0]["sha256"]:
        raise ValueError("First source graph hash does not match the manifest")
    with np.load(first_path, allow_pickle=False) as source:
        if set(source.files) != set(FIELDS):
            raise ValueError(f"Unexpected graph fields: {source.files}")
        dtypes = {name: source[name].dtype.str for name in FIELDS}
        node_dim = int(source["nodes"].shape[1]); edge_dim = int(source["edge"].shape[2])

    offsets = np.zeros((len(records) + 1, len(FIELDS)), dtype=np.int64)
    for i, row in enumerate(records):
        n, k, q = (int(row[key]) for key in ("n", "k", "q"))
        offsets[i + 1] = offsets[i] + [
            int(np.prod(_shape(name, n, k, q, node_dim=node_dim, edge_dim=edge_dim)))
            for name in FIELDS
        ]

    pending = destination.parent / f".{destination.name}.pending-{os.getpid()}"
    pending.mkdir(parents=True, exist_ok=False)
    arrays = {
        name: np.lib.format.open_memmap(
            pending / f"{name}.npy", mode="w+", dtype=np.dtype(dtypes[name]),
            shape=(int(offsets[-1, j]),),
        )
        for j, name in enumerate(FIELDS)
    }
    for i, row in enumerate(records):
        path = out / "data" / row["complex_id"] / "graph.npz"
        if digest(path) != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {row['complex_id']}")
        n, k, q = (int(row[key]) for key in ("n", "k", "q"))
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != set(FIELDS):
                raise ValueError(f"Unexpected fields for {row['complex_id']}")
            for j, name in enumerate(FIELDS):
                value = source[name]
                expected = _shape(name, n, k, q, node_dim=node_dim, edge_dim=edge_dim)
                if value.shape != expected or value.dtype.str != dtypes[name]:
                    raise ValueError((row["complex_id"], name, value.shape, expected, value.dtype.str, dtypes[name]))
                start, stop = offsets[i, j], offsets[i + 1, j]
                arrays[name][start:stop] = value.reshape(-1)
                if not np.array_equal(arrays[name][start:stop].reshape(expected), value, equal_nan=True):
                    raise AssertionError(f"Non-identical mmap write: {row['complex_id']} {name}")
        if (i + 1) % 100 == 0:
            print(json.dumps({"mmap_converted": i + 1, "total": len(records)}), flush=True)
    for value in arrays.values():
        value.flush()
    arrays.clear()

    np.savez(pending / "index.npz",
        complex_id=np.asarray([row["complex_id"] for row in records]),
        n=np.asarray([row["n"] for row in records], np.int32),
        k=np.asarray([row["k"] for row in records], np.int32),
        q=np.asarray([row["q"] for row in records], np.int32), offsets=offsets)
    metadata = dict(format=FORMAT, fields=list(FIELDS), dtypes=dtypes,
        node_dim=node_dim, edge_dim=edge_dim, records=len(records),
        records_fingerprint=records_fingerprint(records))
    atomic_json(pending / "metadata.json", metadata)
    files = {f"{name}.npy": digest(pending / f"{name}.npy") for name in FIELDS}
    files["index.npz"] = digest(pending / "index.npz")
    files["metadata.json"] = digest(pending / "metadata.json")
    atomic_json(pending / "verification.json", dict(format=FORMAT, passed=True,
        all_records_and_fields_bit_identical=True, files=files,
        bytes=sum((pending / name).stat().st_size for name in files)))
    os.replace(pending, destination)
    return destination


class GraphMMap:
    """Read-only views of exact arrays; padding creates the mutable work arrays."""
    def __init__(self, path, records, *, verify_files=False):
        self.path = Path(path)
        self.verification_sha256 = digest(self.path / "verification.json")
        receipt = json.loads((self.path / "verification.json").read_text())
        metadata = json.loads((self.path / "metadata.json").read_text())
        if not receipt.get("passed") or metadata.get("format") != FORMAT:
            raise ValueError("Unverified graph mmap bundle")
        if digest(self.path / "metadata.json") != receipt["files"]["metadata.json"]:
            raise ValueError("Graph mmap metadata hash mismatch")
        if metadata["records_fingerprint"] != records_fingerprint(records):
            raise ValueError("Graph mmap source records do not match this manifest")
        if verify_files:
            for name, expected in receipt["files"].items():
                if digest(self.path / name) != expected:
                    raise ValueError(f"Graph mmap hash mismatch: {name}")
        with np.load(self.path / "index.npz", allow_pickle=False) as index:
            self.ids = index["complex_id"].astype(str)
            self.n = index["n"]; self.k = index["k"]; self.q = index["q"]
            self.offsets = index["offsets"]
        self.byid = {cid: i for i, cid in enumerate(self.ids)}
        self.fields = tuple(metadata["fields"]); self.node_dim = metadata["node_dim"]
        self.edge_dim = metadata["edge_dim"]
        self.arrays = {name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                       for name in self.fields}

    def raw(self, cid):
        i = self.byid[cid]; n, k, q = int(self.n[i]), int(self.k[i]), int(self.q[i])
        values = {}
        for j, name in enumerate(self.fields):
            shape = _shape(name, n, k, q, node_dim=self.node_dim, edge_dim=self.edge_dim)
            values[name] = self.arrays[name][self.offsets[i, j]:self.offsets[i + 1, j]].reshape(shape)
            if values[name].flags.writeable:
                raise AssertionError(f"mmap field unexpectedly writable: {name}")
        labels = values.pop("labels")
        return values, labels

    def close(self):
        self.arrays.clear()
