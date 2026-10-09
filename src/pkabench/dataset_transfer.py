"""Per-dataset export/import bundles and idempotent node-local staging (2026-10-09).

All operations are idempotent and safe to re-run or run concurrently:
- install_dir: copy a file set into a directory under an flock; a matching `.installed.json` marker (key + file sizes)
  makes it a no-op; copies go to `<dest>.pending-<host>-<pid>` and are verified (size, sha256 when known) before an
  atomic rename; pending directories of dead processes are removed.
- stage: node-local copy of a verified store (mmap-v1 etc.), keyed by sha256 of the store's verification.json and file
  sizes, so a changed source gets a new directory and an unchanged one is reused.
- export: deterministic shards (<= SHARD_BYTES) per dataset part, written by parallel processes with per-file and
  per-shard sha256 in `<dataset>-bundle.json`; shards that already verify are skipped on re-run.
- import: verify shard hash, extract to a pending directory, verify every file against the manifest, then install
  files; a per-shard marker under `<root>/.imports/` makes re-runs no-ops and interrupted imports resume.
Paths inside bundles are relative to PKABENCH_RUNTIME, so code that uses runtime-relative paths works after import.

Usage (compute node):
  python -m pkabench.dataset_transfer export {pinder,pkpdb,validation} OUT [--graphs] [--scope pool|all]
  python -m pkabench.dataset_transfer import OUT/<dataset>-bundle.json ROOT [--parts core,graphs]
  python -m pkabench.dataset_transfer verify OUT/<dataset>-bundle.json ROOT
  python -m pkabench.dataset_transfer stage SOURCE_STORE --local /tmp/$USER-stores
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import tarfile
import time
from pathlib import Path

FORMAT = "pkabench-dataset-bundle-v1"
SHARD_BYTES = 2 << 30
CHUNK = 8 << 20
PINDER = "pretraining/pinder-pkai-v1"
PKPDB = "pretraining/pkpdb-full-v1"
PKPDB_STRUCTURES = "pretraining/pkpdb-v1/structures"
PINDER_VALIDATION = "training/ogqt-pinder-factorial-v1/cohort.json"
PKAI_VALIDATION = "pretraining/pkpdb-val-pkai-v1"
PINDER_TOP = ("pool-v3.tsv", "pool-v3.json", "pinder_heldout_exclusions_v3.tsv", "exclusions_v3.json",
              "summary_clean_v3.json", "summary.json")
PINDER_SKIP = {"labels_noterm.json"}
PKPDB_TOP = ("pilot.json", "protocol.json", "verification.json", "references.json", "references.fasta",
             "labels.json", "labels.sqlite", "pool-v3.tsv", "pool-v3.json")
PKPDB_ENTRY = ("sites.json", "environment.json", "defects.json", "conformers.json", "receipt.json", "removed_components.json")


# ---------------------------------------------------------------- hashing and locking
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""): h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path); pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    pending.write_text(json.dumps(value, indent=1, sort_keys=True)); os.replace(pending, path)


@contextlib.contextmanager
def locked(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def _alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def _pending_name(dest):
    return dest.parent / f"{dest.name}.pending-{socket.gethostname()}-{os.getpid()}"


def _clean_stale(dest):
    host = socket.gethostname()
    for path in dest.parent.glob(f"{dest.name}.pending-*"):
        parts = path.name.rsplit("-", 2)
        if len(parts) == 3 and parts[1] == host and parts[2].isdigit() and not _alive(int(parts[2])):
            shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- idempotent directory install / staging
def _marker_ok(dest, key, files):
    marker = dest / ".installed.json"
    if not marker.exists(): return False
    try: value = json.loads(marker.read_text())
    except (OSError, ValueError): return False
    if value.get("key") != key: return False
    return all((dest / rel).is_file() and (dest / rel).stat().st_size == size for rel, (_, size, _) in files.items())


def install_dir(files, dest, key):
    """files: {relative path: (source path, size, sha256 or None)}. Returns True if a copy was made."""
    dest = Path(dest)
    if _marker_ok(dest, key, files): return False
    with locked(dest.parent / f".{dest.name}.lock"):
        if _marker_ok(dest, key, files): return False
        _clean_stale(dest); pending = _pending_name(dest)
        if pending.exists(): shutil.rmtree(pending)
        pending.mkdir(parents=True)
        for rel, (source, size, digest) in sorted(files.items()):
            target = pending / rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if target.stat().st_size != size: raise IOError(f"size mismatch after copy: {rel}")
            if digest is not None and sha256_file(target) != digest: raise IOError(f"sha256 mismatch after copy: {rel}")
        _atomic_json(pending / ".installed.json", {"key": key, "files": len(files), "installed": time.time()})
        if dest.exists():
            retired = dest.parent / f".{dest.name}.retired-{os.getpid()}"; os.replace(dest, retired); shutil.rmtree(retired)
        os.replace(pending, dest)
    return True


def stage(source, local_root):
    """Node-local copy of a verified store; returns its path. Unchanged source -> no copy; changed -> new directory."""
    source = Path(source).resolve(); verification = source / "verification.json"
    if not verification.is_file(): raise FileNotFoundError(f"store has no verification.json: {source}")
    recorded = json.loads(verification.read_text()).get("files") or {}
    files = {}
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = str(path.relative_to(source)); files[rel] = (path, path.stat().st_size, recorded.get(rel))
    key = hashlib.sha256(verification.read_bytes() + json.dumps(sorted((r, v[1]) for r, v in files.items())).encode()).hexdigest()
    dest = Path(local_root) / f"{source.parent.name}-{source.name}-{key[:16]}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    install_dir(files, dest, key)
    return dest


# ---------------------------------------------------------------- dataset file lists
def _read_tsv(path):
    with open(path) as handle: return list(csv.DictReader(handle, delimiter="\t"))


def dataset_files(root, dataset, scope="pool"):
    """{part: [relative paths]} for one dataset, in deterministic order."""
    root = Path(root); parts = {"core": [], "graphs": []}
    if dataset == "pinder":
        ids = {r["id"] for r in _read_tsv(root / PINDER / "pool-v3.tsv")}
        ids |= {r["id"] for r in json.loads((root / PINDER_VALIDATION).read_text())["records"] if r["split"] == "val"}
        if scope == "all":
            for path in sorted((root / PINDER / "index").glob("label_*.jsonl")):
                ids |= {json.loads(l)["id"] for l in path.read_text().splitlines() if json.loads(l)["status"] == "accepted"}
        parts["core"] += [f"{PINDER}/{name}" for name in PINDER_TOP] + [PINDER_VALIDATION]
        parts["core"] += [f"{PINDER}/index/{p.name}" for p in sorted((root / PINDER / "index").glob("label_*.jsonl"))]
        for cid in sorted(ids):
            folder = root / PINDER / "entries" / cid
            parts["core"] += [f"{PINDER}/entries/{cid}/{p.name}" for p in sorted(folder.iterdir()) if p.name not in PINDER_SKIP]
    elif dataset == "pkpdb":
        ids = [r["pdb_id"] for r in json.loads((root / PKPDB / "pilot.json").read_text())["records"]] if scope == "all" \
            else [r["id"] for r in _read_tsv(root / PKPDB / "pool-v3.tsv")]
        parts["core"] += [f"{PKPDB}/{name}" for name in PKPDB_TOP]
        for cid in sorted(ids):
            parts["core"] += [f"{PKPDB}/entries/{cid}/{name}" for name in PKPDB_ENTRY]
            parts["core"] += [f"{PKPDB_STRUCTURES}/{cid[1:3]}/{cid}.cif.gz", f"{PKPDB_STRUCTURES}/{cid[1:3]}/{cid}.json"]
            parts["graphs"].append(f"{PKPDB}/entries/{cid}/graph.npz")
    elif dataset == "validation":
        folder = root / PKAI_VALIDATION
        if not (folder / "manifest.json").exists(): raise FileNotFoundError(f"build it first: {folder} (build-validation)")
        parts["core"] += [f"{PKAI_VALIDATION}/{p.name}" for p in sorted(folder.iterdir()) if p.is_file()]
    else:
        raise ValueError(dataset)
    missing = [rel for rels in parts.values() for rel in rels if not (root / rel).is_file()]
    if missing: raise FileNotFoundError(f"{len(missing)} missing files, e.g. {missing[:5]}")
    return {part: rels for part, rels in parts.items() if rels}


def expected_hashes(root, dataset):
    """Hashes already recorded by the builds, checked while exporting (pKPDB graph.npz and sites.json)."""
    if dataset != "pkpdb": return {}
    out = {}
    for r in json.loads((Path(root) / PKPDB / "pilot.json").read_text())["records"]:
        out[f"{PKPDB}/entries/{r['pdb_id']}/graph.npz"] = r["sha256"]; out[f"{PKPDB}/entries/{r['pdb_id']}/sites.json"] = r["sites_sha256"]
    return out


def shard_plan(root, rels, limit=None):
    """Consecutive groups of files (sorted order) whose total size stays under `limit` (a larger file is alone)."""
    limit = SHARD_BYTES if limit is None else limit; shards, current, size = [], [], 0
    for rel in rels:
        n = (Path(root) / rel).stat().st_size
        if current and size + n > limit: shards.append(current); current, size = [], 0
        current.append(rel); size += n
    if current: shards.append(current)
    return shards


# ---------------------------------------------------------------- export
class _HashingWriter(io.RawIOBase):
    def __init__(self, handle): self.handle = handle; self.hash = hashlib.sha256(); self.size = 0
    def writable(self): return True
    def tell(self): return self.size  # tarfile asks for the stream position; the stream is append-only
    def write(self, data):
        self.hash.update(data); self.size += len(data); return self.handle.write(data)


class _HashingReader(io.RawIOBase):
    def __init__(self, handle): self.handle = handle; self.hash = hashlib.sha256()
    def readable(self): return True
    def readinto(self, buffer):
        n = self.handle.readinto(buffer); self.hash.update(memoryview(buffer)[:n]); return n


def _write_shard(task):
    root, out, name, rels, compress, expected = task
    root = Path(root); out = Path(out); record_path = out / ".shards" / f"{name}.json"; target = out / name
    listing = hashlib.sha256(json.dumps(rels).encode()).hexdigest()
    if record_path.exists() and target.exists():
        record = json.loads(record_path.read_text())
        if record.get("listing") == listing and record["size"] == target.stat().st_size and sha256_file(target) == record["sha256"]:
            return record  # already written and verified
    pending = out / f".{name}.pending-{os.getpid()}"; files = {}
    with open(pending, "wb") as raw:
        writer = _HashingWriter(raw)
        with tarfile.open(fileobj=writer, mode="w:gz" if compress else "w", **({"compresslevel": 1} if compress else {})) as tar:
            for rel in rels:
                path = root / rel; info = tarfile.TarInfo(rel); stat = path.stat()
                info.size = stat.st_size; info.mtime = int(stat.st_mtime); info.mode = 0o644
                with open(path, "rb") as handle:
                    reader = _HashingReader(handle); tar.addfile(info, io.BufferedReader(reader, CHUNK))
                digest = reader.hash.hexdigest()
                if rel in expected and expected[rel] != digest: raise IOError(f"{rel}: sha256 differs from the build record")
                files[rel] = {"size": stat.st_size, "sha256": digest}
    os.replace(pending, target)
    record = {"name": name, "listing": listing, "size": target.stat().st_size, "sha256": writer.hash.hexdigest(), "files": files}
    _atomic_json(record_path, record)
    return record


def _git_commit():
    try:
        return subprocess.run(["git", "-C", os.environ.get("PKABENCH_SOURCE", "."), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return None


def export(root, dataset, out, *, graphs=False, scope="pool", workers=4):
    root = Path(root); out = Path(out); (out / ".shards").mkdir(parents=True, exist_ok=True)
    lists = dataset_files(root, dataset, scope); expected = expected_hashes(root, dataset)
    if not graphs: lists.pop("graphs", None)
    tasks = []
    for part, rels in lists.items():
        compress = part != "graphs"
        for i, group in enumerate(shard_plan(root, rels)):
            tasks.append((str(root), str(out), f"{dataset}-{part}-{i:03d}.tar{'.gz' if compress else ''}", group, compress, expected))
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(_write_shard, tasks))
    shards = {part: [] for part in lists}
    for task, record in zip(tasks, records):
        shards[task[2].split("-")[1]].append({k: record[k] for k in ("name", "size", "sha256")})
    sources = {rel: sha256_file(root / rel) for rel in
               {"pinder": [f"{PINDER}/pool-v3.tsv", f"{PINDER}/pinder_heldout_exclusions_v3.tsv", PINDER_VALIDATION],
                "pkpdb": [f"{PKPDB}/pool-v3.tsv", f"{PKPDB}/pilot.json", f"{PKPDB}/protocol.json"],
                "validation": [f"{PKAI_VALIDATION}/manifest.json"]}[dataset]}
    manifest = {"format": FORMAT, "dataset": dataset, "scope": scope, "parts": shards,
                "files": {rel: {**meta, "shard": record["name"]} for record in records for rel, meta in record["files"].items()},
                "sources": sources, "git_commit": _git_commit(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _atomic_json(out / f"{dataset}-bundle.json", manifest)
    return manifest


# ---------------------------------------------------------------- import / verify
def _safe_members(tar, expected):
    for member in tar.getmembers():
        name = member.name
        if not member.isfile() or name.startswith("/") or ".." in Path(name).parts or name not in expected:
            raise IOError(f"unexpected member in shard: {name}")
        yield member


def import_bundle(manifest_path, root, parts=None):
    """Install a bundle under `root`. Returns {shard: 'installed'|'present'}."""
    manifest_path = Path(manifest_path); bundle = manifest_path.parent; root = Path(root)
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != FORMAT: raise ValueError(manifest["format"])
    dataset = manifest["dataset"]; marks = root / ".imports" / dataset; marks.mkdir(parents=True, exist_ok=True)
    by_shard = {}
    for rel, meta in manifest["files"].items(): by_shard.setdefault(meta["shard"], {})[rel] = meta
    result = {}
    for part, shards in manifest["parts"].items():
        if parts is not None and part not in parts: continue
        for shard in shards:
            name = shard["name"]; marker = marks / f"{name}.json"; files = by_shard[name]
            def present():
                if not marker.exists() or json.loads(marker.read_text()).get("sha256") != shard["sha256"]: return False
                return all((root / rel).is_file() and (root / rel).stat().st_size == meta["size"] for rel, meta in files.items())
            if present(): result[name] = "present"; continue
            with locked(marks / f".{name}.lock"):
                if present(): result[name] = "present"; continue
                path = bundle / name
                if sha256_file(path) != shard["sha256"]: raise IOError(f"shard sha256 mismatch: {path}")
                pending = marks / f"{name}.pending-{socket.gethostname()}-{os.getpid()}"
                if pending.exists(): shutil.rmtree(pending)
                pending.mkdir()
                with tarfile.open(path, "r:*") as tar:
                    for member in _safe_members(tar, files):
                        target = pending / member.name; target.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src, open(target, "wb") as dst: shutil.copyfileobj(src, dst, CHUNK)
                for rel, meta in files.items():
                    target = pending / rel
                    if not target.is_file() or target.stat().st_size != meta["size"] or sha256_file(target) != meta["sha256"]:
                        raise IOError(f"extracted file does not match manifest: {rel}")
                for rel, meta in files.items():
                    dest = root / rel; dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.is_file() and dest.stat().st_size == meta["size"] and sha256_file(dest) == meta["sha256"]: continue
                    if dest.exists() and not dest.is_file(): raise IOError(f"not a file: {dest}")
                    os.replace(pending / rel, dest)
                shutil.rmtree(pending)
                _atomic_json(marker, {"sha256": shard["sha256"], "files": len(files), "installed": time.time()})
                result[name] = "installed"
    return result


def _check(task):
    root, rel, size, digest = task; path = Path(root) / rel
    if not path.is_file(): return rel, "missing"
    if path.stat().st_size != size: return rel, "size"
    return rel, None if sha256_file(path) == digest else "sha256"


def verify(manifest_path, root, workers=4, parts=None):
    manifest = json.loads(Path(manifest_path).read_text())
    wanted = {s["name"] for part, shards in manifest["parts"].items() if parts is None or part in parts for s in shards}
    tasks = [(str(root), rel, m["size"], m["sha256"]) for rel, m in manifest["files"].items() if m["shard"] in wanted]
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        bad = [(rel, why) for rel, why in pool.map(_check, tasks, chunksize=256) if why]
    return {"files": len(tasks), "bad": bad, "passed": not bad}


# ---------------------------------------------------------------- compact pKAI validation package
def build_pkai_validation(root):
    """Validation rows of the frozen 5k pKAI pilot (the set experiment 48 scores), copied into a compact package with
    provenance, so the validation bundle does not carry the 2 x 7.7 GB pilot feature files."""
    import numpy as np
    root = Path(root); pilot = root / "pretraining/pkpdb-5k-comparison-v1/pkai-packed"
    backbone = root / "pretraining/pkai-backbone-ablation-v1/backbone-features.npy"
    groups = {"ASP", "CYS", "TYR", "GLU", "HIS", "LYS"}  # pkai_joint_scale.SIDECHAIN_GROUPS
    rows = json.loads((pilot / "rows.json").read_text())
    ids = np.asarray([i for i, r in enumerate(rows) if r["split"] == "val" and r["group"] in groups], np.int64)
    out = root / PKAI_VALIDATION
    pending = out.parent / f".{out.name}.pending-{os.getpid()}"
    if pending.exists(): shutil.rmtree(pending)
    pending.mkdir(parents=True)
    full = np.load(pilot / "features.npy", mmap_mode="r"); bb = np.load(backbone, mmap_mode="r")
    np.save(pending / "full.npy", np.ascontiguousarray(full[ids])); np.save(pending / "backbone.npy", np.ascontiguousarray(bb[ids]))
    np.save(pending / "target.npy", np.asarray([rows[i]["pka"] - rows[i]["model_pka"] for i in ids], np.float32))
    np.save(pending / "source_rows.npy", ids)
    (pending / "rows.json").write_text(json.dumps([rows[i] for i in ids]))
    _atomic_json(pending / "manifest.json", {
        "description": "frozen 5k pKAI pilot validation rows (split == val, side-chain groups) as scored by experiment 48",
        "rows": int(len(ids)), "sources": {str(p.relative_to(root)): sha256_file(p) for p in (pilot / "rows.json", pilot / "features.npy", backbone)},
        "files": {p.name: sha256_file(p) for p in sorted(pending.iterdir()) if p.name != "manifest.json"}})
    with locked(out.parent / f".{out.name}.lock"):
        if out.exists(): shutil.rmtree(out)
        os.replace(pending, out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pkabench.dataset_transfer"); sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("export"); p.add_argument("dataset", choices=("pinder", "pkpdb", "validation")); p.add_argument("out")
    p.add_argument("--graphs", action="store_true"); p.add_argument("--scope", choices=("pool", "all"), default="pool")
    p = sub.add_parser("import"); p.add_argument("manifest"); p.add_argument("root"); p.add_argument("--parts")
    p = sub.add_parser("verify"); p.add_argument("manifest"); p.add_argument("root"); p.add_argument("--parts")
    p = sub.add_parser("stage"); p.add_argument("source"); p.add_argument("--local", required=True)
    sub.add_parser("build-validation")
    args = parser.parse_args(argv)
    runtime = Path(os.environ.get("PKABENCH_RUNTIME", ".")); workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    parts = set(args.parts.split(",")) if getattr(args, "parts", None) else None
    if args.action == "export":
        m = export(runtime, args.dataset, args.out, graphs=args.graphs, scope=args.scope, workers=workers)
        print(json.dumps({part: {"shards": len(s), "bytes": sum(x["size"] for x in s)} for part, s in m["parts"].items()}))
    elif args.action == "import": print(json.dumps(import_bundle(args.manifest, args.root, parts)))
    elif args.action == "verify":
        report = verify(args.manifest, args.root, workers, parts); print(json.dumps({k: v for k, v in report.items() if k != "bad"} | {"bad": report["bad"][:20]}))
        if not report["passed"]: raise SystemExit(1)
    elif args.action == "stage": print(stage(args.source, args.local))
    elif args.action == "build-validation": print(build_pkai_validation(runtime))


if __name__ == "__main__":
    main()
