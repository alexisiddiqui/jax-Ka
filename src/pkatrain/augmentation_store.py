"""Indexed independent zstd records for coordinate releases and pilot graphs.

The release manifest is the publication marker. Partial output is resumable but
never readable as a release. No source files or native training stores are edited.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import zstandard

from pkabench.runtime import atomic_json, digest

FORMAT = "augmentation-records-v1"
MAGIC = b"KAAR1\0"
SHARD_BYTES = 2 * 1024**3
ZSTD_LEVEL = 3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def encode(metadata, arrays):
    fields = {}; buffers = []; offset = 0
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        if value.dtype.kind not in "biuf" or value.dtype.hasobject:
            raise ValueError(f"unsupported array dtype: {name}: {value.dtype}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite array: {name}")
        data = value.tobytes()
        fields[name] = dict(dtype=value.dtype.str, shape=list(value.shape), offset=offset, bytes=len(data))
        buffers.append(data); offset += len(data)
    header = canonical(dict(metadata=metadata, arrays=fields))
    raw = MAGIC + struct.pack("<Q", len(header)) + header + b"".join(buffers)
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(raw), len(raw)


def decode(data, *, max_output_size):
    if zstandard.frame_content_size(data) != max_output_size:
        raise ValueError("indexed decompressed size differs from frame")
    raw = zstandard.ZstdDecompressor().decompress(data, max_output_size=max_output_size, allow_extra_data=False)
    if len(raw) != max_output_size or raw[:len(MAGIC)] != MAGIC:
        raise ValueError("invalid record framing")
    size = struct.unpack("<Q", raw[len(MAGIC):len(MAGIC)+8])[0]
    start = len(MAGIC)+8; end = start+size
    if end > len(raw): raise ValueError("truncated header")
    header = json.loads(raw[start:end]); arrays = {}; offset = 0
    for name, field in sorted(header["arrays"].items()):
        dtype = np.dtype(field["dtype"]); shape = tuple(field["shape"])
        if dtype.kind not in "biuf" or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("invalid field layout")
        count = int(np.prod(shape, dtype=np.int64))
        if field["offset"] != offset or field["bytes"] != count*dtype.itemsize:
            raise ValueError("invalid field offset/size")
        if end+offset+field["bytes"] > len(raw): raise ValueError("truncated field")
        arrays[name] = np.frombuffer(raw, dtype=dtype, count=count, offset=end+offset).reshape(shape)
        offset += field["bytes"]
    if end+offset != len(raw): raise ValueError("trailing record bytes")
    return dict(metadata=header["metadata"], arrays=arrays)


class ReleaseWriter:
    """Single writer; independent workers must each use a distinct output root.

    A checkpoint commits at most 64 records at once. On resume only this writer's
    uncommitted tail is truncated. Finalisation verifies every compressed record.
    """
    def __init__(self, path, provenance, *, shard_bytes=SHARD_BYTES, file_limit=5000):
        self.path = Path(path); self.path.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance; self.shard_bytes = shard_bytes; self.file_limit = file_limit
        self.signature = fingerprint(dict(provenance=provenance, shard_bytes=shard_bytes, format=FORMAT))
        self.rows = []; self.handle = None; self.shard = 0; self.position = 0; self.since_checkpoint = 0
        self.lock = self.path / ".writer.lock"
        # flock survives crashes without a stale-lock recovery policy.
        import fcntl
        self.lock_handle = self.lock.open("a+b")
        try: fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_handle.close(); raise RuntimeError("release already has an active writer")
        if (self.path/"manifest.json").exists():
            self.close(); raise FileExistsError("immutable release already published")
        checkpoint = self.path/"checkpoint.json"
        if checkpoint.exists():
            old = json.loads(checkpoint.read_text())
            if old["signature"] != self.signature:
                self.close(); raise ValueError("resume provenance or layout differs")
            self.rows = old["records"]; self.shard = old["shard"]; self.position = old["position"]
            # Validate committed bytes before appending to them.
            for row in self.rows:
                with (self.path/row["shard"]).open("rb") as f:
                    f.seek(row["offset"]); data = f.read(row["bytes"])
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    self.close(); raise ValueError("committed record corrupted")
        self.ids = {r["id"] for r in self.rows}
        # A crash between opening a new shard and committing its checkpoint can
        # leave an orphan. Delete only this writer's uncommitted shard files.
        import re
        for path in self.path.glob("records-*.bin"):
            match = re.fullmatch(r"records-(\d{5})\.bin", path.name)
            if match and int(match[1]) > self.shard:
                path.unlink()
        self._open()

    def _name(self): return f"records-{self.shard:05d}.bin"

    def _open(self):
        path = self.path/self._name()
        self.handle = path.open("r+b" if path.exists() else "w+b")
        self.handle.truncate(self.position); self.handle.seek(self.position)

    def checkpoint(self):
        self.handle.flush(); os.fsync(self.handle.fileno())
        atomic_json(self.path/"checkpoint.json", dict(signature=self.signature, records=self.rows,
            shard=self.shard, position=self.position))
        self.since_checkpoint = 0

    def add(self, parent, metadata, arrays, *, summary=None):
        if parent in self.ids: raise ValueError(f"duplicate parent: {parent}")
        frame, raw_bytes = encode(dict(metadata, parent=parent), arrays)
        if len(frame) > self.shard_bytes: raise ValueError("record exceeds shard limit")
        if self.position and self.position+len(frame) > self.shard_bytes:
            self.checkpoint(); self.handle.close(); self.shard += 1; self.position = 0
            if self.shard+8 > self.file_limit: raise ValueError("release file budget exceeded")
            self._open(); self.checkpoint()
        row = dict(id=parent, shard=self._name(), offset=self.position, bytes=len(frame),
                   raw_bytes=raw_bytes, sha256=hashlib.sha256(frame).hexdigest(), summary=summary or {})
        self.handle.write(frame); self.position += len(frame); self.rows.append(row); self.ids.add(parent)
        self.since_checkpoint += 1
        if self.since_checkpoint >= 64: self.checkpoint()
        return row

    def finish(self, *, expected_ids=None):
        self.checkpoint()
        if expected_ids is not None and set(expected_ids) != self.ids:
            raise ValueError("release differs from expected inventory")
        index = dict(format=FORMAT, records=self.rows)
        atomic_json(self.path/"index.json", index)
        report = verify_records(self.path, self.rows)
        atomic_json(self.path/"verification.json", report)
        files = dict(report["file_digests"])
        for p in self.path.glob("records-*.bin"):
            if p.name not in files:
                if p.stat().st_size: raise ValueError("unindexed shard")
                files[p.name] = hashlib.sha256(b"").hexdigest()
        for name in ("index.json", "verification.json"): files[name] = digest(self.path/name)
        manifest = dict(format=FORMAT, zstd_level=ZSTD_LEVEL, shard_bytes=self.shard_bytes,
                        provenance=self.provenance, signature=self.signature, files=files,
                        records=len(self.rows), compressed_bytes=sum(r["bytes"] for r in self.rows),
                        raw_bytes=sum(r["raw_bytes"] for r in self.rows), verified=True)
        atomic_json(self.path/"manifest.json", manifest)
        self.close(); return manifest

    def close(self):
        if self.handle is not None: self.handle.close(); self.handle = None
        if getattr(self, "lock_handle", None) is not None:
            self.lock_handle.close(); self.lock_handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def verify_records(path, rows, *, check_record=None, expected_files=None):
    """One streaming pass for file SHA, frame SHA, exact layout and decoding.

    check_record supplies domain checks without a second read/decompression pass.
    Sorting within each shard also works for adopted/interleaved worker indexes.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows: groups[row["shard"]].append(row)
    checked = 0; raw_bytes = 0; files = {}
    for shard, members in sorted(groups.items()):
        relative = Path(shard)
        if relative.is_absolute() or ".." in relative.parts or not (Path(path)/shard).resolve().is_relative_to(Path(path).resolve()):
            raise ValueError("invalid shard path")
        whole = hashlib.sha256(); position = 0
        with (Path(path)/shard).open("rb") as f:
            for row in sorted(members, key=lambda r: r["offset"]):
                if row["offset"] != position or row["bytes"] <= 0: raise ValueError("indexed record gap or overlap")
                data = f.read(row["bytes"])
                if hashlib.sha256(data).hexdigest() != row["sha256"]: raise ValueError("record digest mismatch")
                whole.update(data); position += len(data)
                record = decode(data, max_output_size=row["raw_bytes"])
                if record["metadata"]["parent"] != row["id"]: raise ValueError("record identity mismatch")
                for value in record["arrays"].values():
                    if value.dtype.kind == "f" and not np.isfinite(value).all(): raise ValueError("nonfinite coordinates")
                if check_record is not None: check_record(record, row)
                checked += 1; raw_bytes += row["raw_bytes"]
            if f.read(1): raise ValueError("unindexed shard tail")
        files[shard] = whole.hexdigest()
        if expected_files is not None and files[shard] != expected_files[shard]:
            raise ValueError("release file digest mismatch")
    if expected_files is not None:
        for shard, expected in expected_files.items():
            if shard.endswith(".bin") and shard not in files:
                if (Path(path)/shard).stat().st_size or expected != hashlib.sha256(b"").hexdigest():
                    raise ValueError("unindexed shard")
    return dict(passed=True, records_checked=checked, raw_bytes=raw_bytes, file_digests=files)


class AugmentationStore:
    """Thread-safe pread; one record read/decompression per parent, read-only views."""
    def __init__(self, path, *, verify_reads=True):
        self.path = Path(path); self.verify_reads = verify_reads
        self.manifest = json.loads((self.path/"manifest.json").read_text())
        if self.manifest.get("format") != FORMAT or not self.manifest.get("verified"):
            raise ValueError("not a verified augmentation release")
        for name in ("index.json", "verification.json"):
            if digest(self.path/name) != self.manifest["files"][name]: raise ValueError("release metadata changed")
        index = json.loads((self.path/"index.json").read_text())
        self.rows = index["records"]; self.by_id = {r["id"]: r for r in self.rows}
        if len(self.rows) != len(self.by_id) or len(self.rows) != self.manifest["records"]:
            raise ValueError("duplicate or missing indexed records")
        self.fds = {}
        try:
            for name in {r["shard"] for r in self.rows}:
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or name not in self.manifest["files"]:
                    raise ValueError("invalid shard path")
                if not (self.path/name).resolve().is_relative_to(self.path.resolve()):
                    raise ValueError("shard escapes release root")
                self.fds[name] = os.open(self.path/name, os.O_RDONLY)
        except Exception:
            self.close(); raise

    def raw(self, parent):
        row = self.by_id[parent]
        data = os.pread(self.fds[row["shard"]], row["bytes"], row["offset"])
        if self.verify_reads and hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("record digest mismatch")
        record = decode(data, max_output_size=row["raw_bytes"])
        if record["metadata"]["parent"] != parent: raise ValueError("record identity mismatch")
        return record

    def conformer(self, parent, candidate):
        record = self.raw(parent); ids = record["metadata"]["selected"]
        return record["arrays"]["coordinates"][ids.index(candidate)]

    def verify_files(self):
        for name, expected in self.manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not (self.path/name).resolve().is_relative_to(self.path.resolve()):
                raise ValueError("invalid release file path")
            if digest(self.path/name) != expected: raise ValueError(f"release file digest mismatch: {name}")
        return True

    def close(self):
        for fd in self.fds.values(): os.close(fd)
        self.fds.clear()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def finalize_workers(path, workers, provenance, *, expected_ids, file_limit=5000):
    """Adopt verified worker releases beneath the destination; no payload copying.

    Workers write under OUTPUT/workers/NNNN from the start. Only small final
    indexes and manifests are created here. Worker releases remain immutable.
    """
    path = Path(path).resolve(); path.mkdir(parents=True, exist_ok=True)
    if (path/"manifest.json").exists(): raise FileExistsError("release already published")
    rows = []; files = {}; ids = set(); manifests = {}
    for worker in workers:
        worker = Path(worker).resolve()
        if not worker.is_relative_to(path) or worker == path:
            raise ValueError("workers must already reside inside release root")
        prefix = str(worker.relative_to(path))
        with AugmentationStore(worker) as store:
            verify_records(worker, store.rows, expected_files=store.manifest["files"])
            manifests[prefix] = digest(worker/"manifest.json")
            for row in store.rows:
                if row["id"] in ids: raise ValueError("duplicate parent across workers")
                ids.add(row["id"])
                name = prefix+"/"+row["shard"]
                rows.append(dict(row, shard=name))
                files[name] = store.manifest["files"][row["shard"]]
    if ids != set(expected_ids): raise ValueError("workers differ from expected inventory")
    # Include worker manifests/checkpoints/indexes/log allowances in the file budget.
    if len(files)+len(workers)*8+5 > file_limit: raise ValueError("release file budget exceeded")
    order = {key: i for i, key in enumerate(expected_ids)}; rows.sort(key=lambda row: order[row["id"]])
    atomic_json(path/"index.json", dict(format=FORMAT, records=rows))
    atomic_json(path/"verification.json", verify_records(path, rows))
    for name in ("index.json", "verification.json"): files[name] = digest(path/name)
    result = dict(format=FORMAT, verified=True, zstd_level=ZSTD_LEVEL, shard_bytes=SHARD_BYTES,
                  records=len(rows), files=files, provenance=provenance, worker_manifests=manifests,
                  raw_bytes=sum(r["raw_bytes"] for r in rows), compressed_bytes=sum(r["bytes"] for r in rows))
    atomic_json(path/"manifest.json", result)
    return result
