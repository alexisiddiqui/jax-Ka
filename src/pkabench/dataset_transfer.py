"""Per-dataset export/import bundles and idempotent node-local staging (2026-10-09).

All operations are idempotent and safe to re-run or run concurrently:
- install_dir: copy a file set into a directory under an flock; a matching `.installed.json` marker (key + file sizes)
  makes it a no-op; copies go to `<dest>.pending-<host>-<pid>` and are verified (size, sha256 when known) before an
  atomic rename; pending directories of dead processes are removed.
- stage: node-local copy of a verified store (mmap-v1 etc.), keyed by sha256 of the store's verification.json and file
  sizes, so a changed source gets a new directory and an unchanged one is reused.
- export: deterministic shards (<= SHARD_BYTES) per dataset part, written by parallel processes with per-file and
  per-shard sha256 in `<dataset>-bundle.json`; shards that already verify are skipped on re-run.
- import: one streaming pass per shard (shards in parallel) hashes the shard and every file while extracting to a
  pending directory; files are installed only if all match the manifest; a per-shard marker under `<root>/.imports/` makes re-runs no-ops and interrupted imports resume.
- squash (file-count-limited file systems, e.g. Isambard-AI scratch at 1,024,000 inodes): instead of importing, build
  one squashfs image per large bundle directory (root/images/<name>.sqfs) in one streaming pass that checks every file
  against the manifest, install the few remaining files loose, and make root/<directory> a symlink to the node-local
  mount point. scripts/sqfs_run.sh mounts the images (squashfuse) in a private mount namespace for one command.
Paths inside bundles are relative to PKABENCH_RUNTIME, so code that uses runtime-relative paths works after import.

Usage (compute node):
  python -m pkabench.dataset_transfer export {pinder,pkpdb,validation} OUT [--graphs] [--scope pool|all]
  python -m pkabench.dataset_transfer import OUT/<dataset>-bundle.json ROOT [--parts core,graphs]
  python -m pkabench.dataset_transfer verify OUT/<dataset>-bundle.json ROOT
  python -m pkabench.dataset_transfer squash OUT/<dataset>-bundle.json ROOT [--mksquashfs PATH] [--replace]
  scripts/sqfs_run.sh python -m pkabench.dataset_transfer verify OUT/<dataset>-bundle.json ROOT
  python -m pkabench.dataset_transfer squash-dir PREFIX NAME            (a verified directory -> images/NAME.sqfs)
  scripts/sqfs_run.sh python -m pkabench.dataset_transfer verify-dir /tmp/$USER-sqfs/NAME [--against ROOT/PREFIX]
  python -m pkabench.dataset_transfer link-dir PREFIX NAME              (PREFIX -> the mount point)
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
import zlib
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
def _member_ok(member, expected):
    name = member.name
    return member.isfile() and not name.startswith("/") and ".." not in Path(name).parts and name in expected


def _extract_verified(path, pending, files, seen, shard, name):
    """Stream the shard once, hashing it and each member while extracting into `pending`; raise on any mismatch."""
    with open(path, "rb") as raw:
        reader = _HashingReader(raw)
        with tarfile.open(fileobj=io.BufferedReader(reader, CHUNK), mode="r|*") as tar:
            for member in tar:
                if not _member_ok(member, files) or member.name in seen: raise IOError(f"unexpected member in shard {name}: {member.name}")
                target = pending / member.name; target.parent.mkdir(parents=True, exist_ok=True); digest = hashlib.sha256()
                with tar.extractfile(member) as src, open(target, "wb") as dst:
                    for block in iter(lambda: src.read(CHUNK), b""): digest.update(block); dst.write(block)
                meta = files[member.name]
                if member.size != meta["size"] or digest.hexdigest() != meta["sha256"]: raise IOError(f"file does not match manifest: {member.name}")
                seen.add(member.name)
        while reader.readinto(bytearray(CHUNK)): pass  # hash any trailing bytes
    if reader.hash.hexdigest() != shard["sha256"]: raise IOError(f"shard sha256 mismatch: {path}")
    if seen != set(files): raise IOError(f"shard {name} is missing {len(set(files) - seen)} files")


def _import_shard(task):
    """One shard, one streaming pass: the shard hash and every file hash are computed while extracting into a pending
    directory; nothing is installed unless all of them match the manifest."""
    bundle, root, marks, shard, files = task
    bundle, root, marks = Path(bundle), Path(root), Path(marks); name = shard["name"]; marker = marks / f"{name}.json"
    def present():
        if not marker.exists() or json.loads(marker.read_text()).get("sha256") != shard["sha256"]: return False
        return all((root / rel).is_file() and (root / rel).stat().st_size == meta["size"] for rel, meta in files.items())
    if present(): return name, "present"
    with locked(marks / f".{name}.lock"):
        if present(): return name, "present"
        _clean_stale(marks / name); pending = marks / f"{name}.pending-{socket.gethostname()}-{os.getpid()}"
        if pending.exists(): shutil.rmtree(pending)
        pending.mkdir(); seen = set()
        try:
            _extract_verified(bundle / name, pending, files, seen, shard, name)
        except (tarfile.TarError, EOFError, zlib.error) as error:
            shutil.rmtree(pending, ignore_errors=True); raise IOError(f"unreadable shard {bundle / name}: {error}") from error
        except Exception:
            shutil.rmtree(pending, ignore_errors=True); raise
        for rel, meta in files.items():
            dest = root / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not dest.is_file(): raise IOError(f"not a file: {dest}")
            if dest.is_file() and dest.stat().st_size == meta["size"] and sha256_file(dest) == meta["sha256"]: continue
            os.replace(pending / rel, dest)
        shutil.rmtree(pending)
        _atomic_json(marker, {"sha256": shard["sha256"], "files": len(files), "installed": time.time()})
    return name, "installed"


def import_bundle(manifest_path, root, parts=None, workers=4):
    """Install a bundle under `root`, shards in parallel processes. Returns {shard: 'installed'|'present'}."""
    manifest_path = Path(manifest_path); bundle = manifest_path.parent; root = Path(root)
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != FORMAT: raise ValueError(manifest["format"])
    marks = root / ".imports" / manifest["dataset"]; marks.mkdir(parents=True, exist_ok=True)
    by_shard = {}
    for rel, meta in manifest["files"].items(): by_shard.setdefault(meta["shard"], {})[rel] = meta
    tasks = [(str(bundle), str(root), str(marks), shard, by_shard[shard["name"]])
             for part, shards in manifest["parts"].items() if parts is None or part in parts for shard in shards]
    if workers <= 1 or len(tasks) <= 1: return dict(map(_import_shard, tasks))
    with concurrent.futures.ProcessPoolExecutor(min(workers, len(tasks)), mp_context=multiprocessing.get_context("spawn")) as pool:
        return dict(pool.map(_import_shard, tasks))


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


# ---------------------------------------------------------------- squashfs images (inode-limited file systems)
IMAGE_MIN_FILES = 1000


def mount_root():
    """Node-local mount point root; scripts/sqfs_run.sh mounts <root>/images/<name>.sqfs at <mount_root>/<name>."""
    return os.environ.get("PKABENCH_SQFS_MOUNT") or f"/tmp/{os.environ.get('USER') or os.getuid()}-sqfs"


def image_plan(manifest):
    """{runtime-relative directory: [files]} for the directories that become images (two path components, at least
    IMAGE_MIN_FILES files, e.g. pretraining/pinder-pkai-v1), and the remaining files under "" (installed loose)."""
    groups = {}
    for rel in manifest["files"]:
        parts = rel.split("/"); groups.setdefault("/".join(parts[:2]) if len(parts) > 2 else "", []).append(rel)
    loose = groups.pop("", [])
    for prefix in [p for p, rels in groups.items() if len(rels) < IMAGE_MIN_FILES]: loose += groups.pop(prefix)
    return groups, sorted(loose)


def _image_key(manifest, rels):
    return hashlib.sha256(json.dumps(sorted((rel, manifest["files"][rel]["sha256"]) for rel in rels)).encode()).hexdigest()


def _image_present(images, prefix, key):
    name = Path(prefix).name; image = images / f"{name}.sqfs"; marker = images / f"{name}.sqfs.json"
    if not (image.is_file() and marker.is_file()): return False
    record = json.loads(marker.read_text())
    return record.get("key") == key and record.get("size") == image.stat().st_size


def _link(root, prefix, replace, name=None):
    """root/prefix -> <mount_root>/<name> (dangling outside sqfs_run.sh, so a missing mount fails loudly)."""
    link = Path(root) / prefix; target = f"{mount_root()}/{name or Path(prefix).name}"
    if link.is_symlink() and os.readlink(link) == target: return
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            if not replace: raise IOError(f"{link} is a directory; pass --replace to retire it in favour of the image")
            retired = link.parent / f".{link.name}.retired-{os.getpid()}"; os.replace(link, retired); shutil.rmtree(retired)
        else: link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True); pending = link.parent / f".{link.name}.link-{os.getpid()}"
    os.symlink(target, pending); os.replace(pending, link)


def squash(manifest_path, root, *, mksquashfs="mksquashfs", processors=4, replace=False):
    """Build one squashfs image per large directory of a bundle in a single streaming pass over its shards (every file
    is checked against the manifest on the way), install the remaining files loose, and point root/<directory> at the
    node-local mount point. Images go to root/images/<name>.sqfs with a .json marker; an image whose marker matches the
    manifest is kept, so re-runs are no-ops."""
    manifest_path = Path(manifest_path); bundle = manifest_path.parent; root = Path(root)
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != FORMAT: raise ValueError(manifest["format"])
    groups, loose = image_plan(manifest); images = root / "images"; images.mkdir(parents=True, exist_ok=True)
    keys = {prefix: _image_key(manifest, rels) for prefix, rels in groups.items()}
    blocked = [str(root / p) for p in groups if (root / p).is_dir() and not (root / p).is_symlink()]
    if blocked and not replace: raise IOError(f"directories in the way of images (pass --replace to retire them): {blocked}")
    result = {}
    with locked(images / f".{manifest['dataset']}.lock"):
        todo = [p for p in groups if not _image_present(images, p, keys[p])]
        loose_todo = [rel for rel in loose if not ((root / rel).is_file() and (root / rel).stat().st_size == manifest["files"][rel]["size"]
                                                   and sha256_file(root / rel) == manifest["files"][rel]["sha256"])]
        result.update({Path(p).name: "present" for p in groups if p not in todo})
        if todo or loose_todo:
            result.update(_squash_pass(manifest, bundle, root, images, todo, set(loose_todo), keys, mksquashfs, processors))
        for prefix in groups: _link(root, prefix, replace)
    return result


def _squash_pass(manifest, bundle, root, images, todo, loose_todo, keys, mksquashfs, processors):
    files = manifest["files"]; host = socket.gethostname(); procs, tars, pendings = {}, {}, {}
    loose_pending = images / f".loose-{manifest['dataset']}.pending-{host}-{os.getpid()}"
    for prefix in todo:
        name = Path(prefix).name; pendings[prefix] = images / f".{name}.sqfs.pending-{host}-{os.getpid()}"
        procs[prefix] = subprocess.Popen([mksquashfs, "-", str(pendings[prefix]), "-tar", "-noappend", "-no-xattrs", "-all-root",
                                          "-default-mode", "0755", "-comp", "zstd", "-processors", str(processors),
                                          "-mem", os.environ.get("PKABENCH_MKSQUASHFS_MEM", "4G"),
                                          "-quiet", "-no-progress"], stdin=subprocess.PIPE)
        tars[prefix] = tarfile.open(fileobj=procs[prefix].stdin, mode="w|")
    seen = set(); counts = {prefix: 0 for prefix in todo}
    try:
        for part, shards in manifest["parts"].items():
            for shard in shards:
                with open(bundle / shard["name"], "rb") as raw:
                    reader = _HashingReader(raw)
                    with tarfile.open(fileobj=io.BufferedReader(reader, CHUNK), mode="r|*") as tar:
                        for member in tar:
                            if not _member_ok(member, files) or member.name in seen or files[member.name]["shard"] != shard["name"]:
                                raise IOError(f"unexpected member in shard {shard['name']}: {member.name}")
                            data = tar.extractfile(member).read(); meta = files[member.name]
                            if len(data) != meta["size"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
                                raise IOError(f"file does not match manifest: {member.name}")
                            seen.add(member.name); prefix = "/".join(member.name.split("/")[:2])
                            if prefix in tars:
                                info = tarfile.TarInfo(member.name[len(prefix) + 1:]); info.size = len(data)
                                info.mtime = member.mtime; info.mode = 0o644
                                tars[prefix].addfile(info, io.BytesIO(data)); counts[prefix] += 1
                            elif member.name in loose_todo:
                                target = loose_pending / member.name; target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(data)
                    while reader.readinto(bytearray(CHUNK)): pass
                if reader.hash.hexdigest() != shard["sha256"]: raise IOError(f"shard sha256 mismatch: {shard['name']}")
        if seen != set(files): raise IOError(f"bundle is missing {len(set(files) - seen)} files")
        for prefix in todo:
            tars[prefix].close(); procs[prefix].stdin.close()
            if procs[prefix].wait() != 0: raise IOError(f"mksquashfs failed for {prefix} (exit {procs[prefix].returncode})")
    except BaseException:
        for tar in tars.values():
            with contextlib.suppress(Exception): tar.fileobj = io.BytesIO(); tar.close()
        for proc in procs.values():
            with contextlib.suppress(Exception): proc.stdin.close()
            proc.kill(); proc.wait()
        for path in list(pendings.values()) + [loose_pending]:
            if path.is_dir(): shutil.rmtree(path, ignore_errors=True)
            elif path.exists(): path.unlink()
        raise
    out = {}
    for prefix in todo:
        name = Path(prefix).name; image = images / f"{name}.sqfs"; pending = pendings[prefix]
        record = {"key": keys[prefix], "prefix": prefix, "files": counts[prefix], "size": pending.stat().st_size,
                  "sha256": sha256_file(pending), "dataset": manifest["dataset"], "bundle_git_commit": manifest.get("git_commit"),
                  "mount": f"{mount_root()}/{name}", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        os.replace(pending, image); _atomic_json(images / f"{name}.sqfs.json", record); out[name] = "built"
    for rel in sorted(loose_todo):
        dest = root / rel; dest.parent.mkdir(parents=True, exist_ok=True); os.replace(loose_pending / rel, dest); out[rel] = "installed"
    if loose_pending.exists(): shutil.rmtree(loose_pending)
    return out


# ---------------------------------------------------------------- squashfs image of a verified directory (e.g. a packed store)
def squash_dir(root, prefix, name, *, mksquashfs="mksquashfs", processors=4):
    """Build root/images/<name>.sqfs (zstd) from the directory root/<prefix>, which must hold a passing
    verification.json with per-file sha256 ("files"). Keyed by that verification.json, so re-runs are no-ops.
    The directory is left in place: check the image through the mount (verify-dir under sqfs_run.sh), then link-dir."""
    root = Path(root); source = root / prefix; images = root / "images"; images.mkdir(parents=True, exist_ok=True)
    verification = json.loads((source / "verification.json").read_text())
    if not verification.get("passed") or not verification.get("files"): raise IOError(f"{source} has no passing per-file verification")
    key = hashlib.sha256((source / "verification.json").read_bytes()).hexdigest()
    image = images / f"{name}.sqfs"; marker = images / f"{name}.sqfs.json"
    with locked(images / f".{name}.lock"):
        if image.is_file() and marker.is_file() and json.loads(marker.read_text()).get("key") == key: return "present"
        pending = images / f".{name}.sqfs.pending-{socket.gethostname()}-{os.getpid()}"
        try:
            subprocess.run([mksquashfs, str(source), str(pending), "-noappend", "-no-xattrs", "-all-root", "-comp", "zstd",
                            "-processors", str(processors), "-mem", os.environ.get("PKABENCH_MKSQUASHFS_MEM", "8G"),
                            "-quiet", "-no-progress"], check=True)
        except BaseException:
            if pending.exists(): pending.unlink()
            raise
        record = {"key": key, "prefix": prefix, "files": len(verification["files"]) + 1, "size": pending.stat().st_size,
                  "source_bytes": sum(p.stat().st_size for p in source.iterdir() if p.is_file()), "sha256": sha256_file(pending),
                  "mount": f"{mount_root()}/{name}", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        os.replace(pending, image); _atomic_json(marker, record)
    return "built"


def verify_dir(path, workers=4):
    """Re-hash every file listed in path/verification.json (run on the mounted image)."""
    path = Path(path); files = json.loads((path / "verification.json").read_text())["files"]
    tasks = [(str(path), rel, (path / rel).stat().st_size if (path / rel).is_file() else -1, digest) for rel, digest in files.items()]
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        bad = [(rel, why) for rel, why in pool.map(_check, tasks) if why]
    return {"files": len(tasks), "bad": bad, "passed": not bad}


RANGE_BYTES = 256 << 20


def _compare_range(task):
    a, b, start, length = task
    with open(a, "rb") as x, open(b, "rb") as y:
        x.seek(start); y.seek(start); remaining = length
        while remaining:
            n = min(CHUNK, remaining); u = x.read(n); v = y.read(n)
            if u != v or len(u) != n: return f"{Path(a).name}@{start + length - remaining}"
            remaining -= n
    return None


def compare_dir(mounted, source, workers=4):
    """Byte-compare every file of `source` (a directory whose verification.json passed) with the mounted image,
    in RANGE_BYTES ranges across `workers` processes, so one huge file is still read in parallel (squashfuse
    decompresses concurrent reads on several threads)."""
    mounted = Path(mounted); source = Path(source)
    names = sorted(p.name for p in source.iterdir() if p.is_file())
    if sorted(p.name for p in mounted.iterdir()) != names: return {"passed": False, "bad": ["file list differs"]}
    tasks = []
    for name in names:
        size = (source / name).stat().st_size
        if (mounted / name).stat().st_size != size: return {"passed": False, "bad": [f"{name}: size"]}
        tasks += [(str(mounted / name), str(source / name), start, min(RANGE_BYTES, size - start)) for start in range(0, size, RANGE_BYTES)]
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        bad = [r for r in pool.map(_compare_range, tasks) if r]
    return {"files": len(names), "ranges": len(tasks), "bytes": sum(t[3] for t in tasks), "bad": bad, "passed": not bad}


def link_dir(root, prefix, name):
    """Replace root/<prefix> by a symlink to the mount point of images/<name>.sqfs (after verify-dir passed)."""
    root = Path(root); marker = root / "images" / f"{name}.sqfs.json"
    record = json.loads(marker.read_text())
    if record["prefix"] != prefix: raise IOError(f"{marker} was built from {record['prefix']}, not {prefix}")
    _link(root, prefix, True, name)


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
    p = sub.add_parser("squash"); p.add_argument("manifest"); p.add_argument("root")
    p.add_argument("--mksquashfs", default=os.environ.get("PKABENCH_MKSQUASHFS", "mksquashfs")); p.add_argument("--replace", action="store_true")
    p = sub.add_parser("squash-dir"); p.add_argument("prefix"); p.add_argument("name")
    p.add_argument("--mksquashfs", default=os.environ.get("PKABENCH_MKSQUASHFS", "mksquashfs"))
    p = sub.add_parser("verify-dir"); p.add_argument("path"); p.add_argument("--against")
    p = sub.add_parser("link-dir"); p.add_argument("prefix"); p.add_argument("name")
    p = sub.add_parser("stage"); p.add_argument("source"); p.add_argument("--local", required=True)
    sub.add_parser("build-validation")
    args = parser.parse_args(argv)
    runtime = Path(os.environ.get("PKABENCH_RUNTIME", ".")); workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    parts = set(args.parts.split(",")) if getattr(args, "parts", None) else None
    if args.action == "export":
        m = export(runtime, args.dataset, args.out, graphs=args.graphs, scope=args.scope, workers=workers)
        print(json.dumps({part: {"shards": len(s), "bytes": sum(x["size"] for x in s)} for part, s in m["parts"].items()}))
    elif args.action == "import": print(json.dumps(import_bundle(args.manifest, args.root, parts, workers)))
    elif args.action == "verify":
        report = verify(args.manifest, args.root, workers, parts); print(json.dumps({k: v for k, v in report.items() if k != "bad"} | {"bad": report["bad"][:20]}))
        if not report["passed"]: raise SystemExit(1)
    elif args.action == "squash":
        print(json.dumps(squash(args.manifest, args.root, mksquashfs=args.mksquashfs, processors=workers, replace=args.replace)))
    elif args.action == "squash-dir":
        print(squash_dir(runtime, args.prefix, args.name, mksquashfs=args.mksquashfs, processors=workers))
    elif args.action == "verify-dir":
        report = compare_dir(args.path, args.against, workers) if args.against else verify_dir(args.path, workers)
        print(json.dumps({**report, "bad": report["bad"][:20]}))
        if not report["passed"]: raise SystemExit(1)
    elif args.action == "link-dir": link_dir(runtime, args.prefix, args.name); print("linked")
    elif args.action == "stage": print(stage(args.source, args.local))
    elif args.action == "build-validation": print(build_pkai_validation(runtime))


if __name__ == "__main__":
    main()
