import json
import multiprocessing
import tarfile

import pytest

from pkabench import dataset_transfer as dt


def _store(path, payload=b"abc"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "a.npy").write_bytes(payload); (path / "sub").mkdir(exist_ok=True); (path / "sub" / "b.json").write_text("{}")
    (path / "verification.json").write_text(json.dumps({"passed": True, "payload": len(payload)}))
    return path


def _stage_worker(args):
    source, local = args
    return str(dt.stage(source, local))


def test_stage_is_idempotent_and_keyed_on_source(tmp_path):
    source = _store(tmp_path / "exp" / "mmap-v1"); local = tmp_path / "local"
    first = dt.stage(source, local); marker = (first / ".installed.json").stat().st_mtime_ns
    assert dt.stage(source, local) == first and (first / ".installed.json").stat().st_mtime_ns == marker
    assert (first / "sub" / "b.json").read_text() == "{}"
    _store(source, b"changed!")
    second = dt.stage(source, local)
    assert second != first and (second / "a.npy").read_bytes() == b"changed!"


def test_concurrent_stage_copies_once(tmp_path):
    source = _store(tmp_path / "exp" / "mmap-v1"); local = tmp_path / "local"
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        paths = set(pool.map(_stage_worker, [(source, local)] * 8))
    assert len(paths) == 1 and not list(local.glob("*.pending-*"))


def test_install_dir_detects_bad_digest(tmp_path):
    source = tmp_path / "f"; source.write_bytes(b"xyz")
    with pytest.raises(IOError): dt.install_dir({"f": (source, 3, "0" * 64)}, tmp_path / "dest", "k")
    assert not (tmp_path / "dest").exists()


def _fake_pkpdb(root):
    ids = ["1abc", "2abd", "3abe"]; base = root / dt.PKPDB; records = []
    for i, cid in enumerate(ids):
        entry = base / "entries" / cid; entry.mkdir(parents=True)
        for name in dt.PKPDB_ENTRY: (entry / name).write_text(json.dumps({"id": cid, "name": name, "pad": "x" * 200 * i}))
        (entry / "graph.npz").write_bytes(bytes([i]) * 1000)
        s = root / dt.PKPDB_STRUCTURES / cid[1:3]; s.mkdir(parents=True, exist_ok=True)
        (s / f"{cid}.cif.gz").write_bytes(b"cif" * 100); (s / f"{cid}.json").write_text("{}")
        records.append({"pdb_id": cid, "sha256": dt.sha256_file(entry / "graph.npz"), "sites_sha256": dt.sha256_file(entry / "sites.json")})
    for name in dt.PKPDB_TOP: (base / name).write_text("{}")
    (base / "pilot.json").write_text(json.dumps({"records": records}))
    (base / "pool-v3.tsv").write_text("id\tgroup\n" + "".join(f"{c}\tg\n" for c in ids))
    return ids


def test_export_import_roundtrip_is_idempotent(tmp_path, monkeypatch):
    source = tmp_path / "src"; _fake_pkpdb(source); monkeypatch.setattr(dt, "SHARD_BYTES", 1500)
    out = tmp_path / "bundle"
    manifest = dt.export(source, "pkpdb", out, graphs=True, workers=2)
    assert len(manifest["parts"]["core"]) > 1 and manifest["parts"]["graphs"]
    again = dt.export(source, "pkpdb", out, graphs=True, workers=2)
    assert again["parts"] == manifest["parts"]  # deterministic, verified shards reused
    target = tmp_path / "dst"
    first = dt.import_bundle(out / "pkpdb-bundle.json", target, parts={"core"})
    assert set(first.values()) == {"installed"} and not (target / dt.PKPDB / "entries/1abc/graph.npz").exists()
    second = dt.import_bundle(out / "pkpdb-bundle.json", target)  # resumes with the graphs part
    assert {v for k, v in second.items() if "-core-" in k} == {"present"} and {v for k, v in second.items() if "-graphs-" in k} == {"installed"}
    assert dt.verify(out / "pkpdb-bundle.json", target, workers=2)["passed"]
    for rel in manifest["files"]: assert (target / rel).read_bytes() == (source / rel).read_bytes()


def test_corrupted_shard_and_tampered_source_are_rejected(tmp_path, monkeypatch):
    source = tmp_path / "src"; ids = _fake_pkpdb(source); out = tmp_path / "bundle"
    dt.export(source, "pkpdb", out, workers=1)
    shard = next(out.glob("pkpdb-core-*.tar.gz")); data = bytearray(shard.read_bytes()); data[-10] ^= 0xFF; shard.write_bytes(bytes(data))
    with pytest.raises(IOError): dt.import_bundle(out / "pkpdb-bundle.json", tmp_path / "dst")
    (source / dt.PKPDB / "entries" / ids[0] / "graph.npz").write_bytes(b"tampered")
    with pytest.raises(IOError): dt.export(source, "pkpdb", tmp_path / "bundle2", graphs=True, workers=1)


def test_import_rejects_unexpected_members(tmp_path):
    source = tmp_path / "src"; _fake_pkpdb(source); out = tmp_path / "bundle"; dt.export(source, "pkpdb", out, workers=1)
    manifest = json.loads((out / "pkpdb-bundle.json").read_text()); shard = manifest["parts"]["core"][0]["name"]
    evil = tmp_path / "evil.txt"; evil.write_text("x")
    with tarfile.open(out / shard, "w:gz") as tar: tar.add(evil, arcname="../escape.txt")
    manifest["parts"]["core"][0]["sha256"] = dt.sha256_file(out / shard); (out / "pkpdb-bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(IOError): dt.import_bundle(out / "pkpdb-bundle.json", tmp_path / "dst")
    assert not (tmp_path / "escape.txt").exists()


def _mksquashfs():
    import os, shutil, subprocess
    tool = os.environ.get("PKABENCH_MKSQUASHFS") or shutil.which("mksquashfs")
    if not tool: return None
    help_text = subprocess.run([tool, "-help-option", "tar"], capture_output=True, text=True).stdout
    unsquashfs = os.path.join(os.path.dirname(tool), "unsquashfs")
    return (tool, unsquashfs) if "-tar" in help_text and os.path.exists(unsquashfs) else None


def test_squash_builds_images_installs_loose_files_and_is_idempotent(tmp_path, monkeypatch):
    import os, subprocess
    tools = _mksquashfs()
    if tools is None: pytest.skip("mksquashfs with -tar (squashfs-tools >= 4.6) not available")
    source = tmp_path / "src"; _fake_pkpdb(source); monkeypatch.setattr(dt, "SHARD_BYTES", 1500)
    monkeypatch.setattr(dt, "IMAGE_MIN_FILES", 5); monkeypatch.setenv("PKABENCH_SQFS_MOUNT", str(tmp_path / "mnt"))
    out = tmp_path / "bundle"; manifest = dt.export(source, "pkpdb", out, workers=1)
    target = tmp_path / "dst"; (target / dt.PKPDB).mkdir(parents=True)  # an earlier loose import is refused without --replace
    with pytest.raises(IOError): dt.squash(out / "pkpdb-bundle.json", target, mksquashfs=tools[0], processors=1)
    first = dt.squash(out / "pkpdb-bundle.json", target, mksquashfs=tools[0], processors=1, replace=True)
    assert first == {"pkpdb-full-v1": "built", "pkpdb-v1": "built"}
    assert os.readlink(target / dt.PKPDB) == str(tmp_path / "mnt" / "pkpdb-full-v1")
    assert dt.squash(out / "pkpdb-bundle.json", target, mksquashfs=tools[0]) == {"pkpdb-full-v1": "present", "pkpdb-v1": "present"}
    for name, prefix in (("pkpdb-full-v1", dt.PKPDB), ("pkpdb-v1", "pretraining/pkpdb-v1")):
        record = json.loads((target / "images" / f"{name}.sqfs.json").read_text())
        assert record["sha256"] == dt.sha256_file(target / "images" / f"{name}.sqfs")
        (tmp_path / "x").mkdir(exist_ok=True)
        subprocess.run([tools[1], "-q", "-n", "-d", str(tmp_path / "x" / name), str(target / "images" / f"{name}.sqfs")], check=True)
        rels = [rel for rel in manifest["files"] if rel.startswith(prefix + "/")]
        assert record["files"] == len(rels)
        for rel in rels: assert (tmp_path / "x" / name / rel[len(prefix) + 1:]).read_bytes() == (source / rel).read_bytes()
    assert not list((target / "images").glob(".*pending*"))


def test_squash_rejects_corrupted_shard(tmp_path, monkeypatch):
    tools = _mksquashfs()
    if tools is None: pytest.skip("mksquashfs with -tar (squashfs-tools >= 4.6) not available")
    source = tmp_path / "src"; _fake_pkpdb(source); monkeypatch.setattr(dt, "IMAGE_MIN_FILES", 5)
    out = tmp_path / "bundle"; dt.export(source, "pkpdb", out, workers=1)
    shard = next(out.glob("pkpdb-core-*.tar.gz")); data = bytearray(shard.read_bytes()); data[-10] ^= 0xFF; shard.write_bytes(bytes(data))
    with pytest.raises(Exception): dt.squash(out / "pkpdb-bundle.json", tmp_path / "dst", mksquashfs=tools[0], processors=1)
    assert not list((tmp_path / "dst" / "images").glob("*.sqfs")) and not list((tmp_path / "dst" / "images").glob(".*pending*"))


def test_squash_dir_verify_and_link(tmp_path, monkeypatch):
    import os, subprocess
    tools = _mksquashfs()
    if tools is None: pytest.skip("mksquashfs with -tar (squashfs-tools >= 4.6) not available")
    monkeypatch.setenv("PKABENCH_SQFS_MOUNT", str(tmp_path / "mnt"))
    store = tmp_path / "root" / "training" / "x" / "store-v1"; store.mkdir(parents=True)
    (store / "a.npy").write_bytes(b"\0" * 5000); (store / "b.json").write_text("{}")
    with pytest.raises(Exception): dt.squash_dir(tmp_path / "root", "training/x/store-v1", "x-store", mksquashfs=tools[0])
    (store / "verification.json").write_text(json.dumps({"passed": True, "files": {n: dt.sha256_file(store / n) for n in ("a.npy", "b.json")}}))
    assert dt.squash_dir(tmp_path / "root", "training/x/store-v1", "x-store", mksquashfs=tools[0], processors=1) == "built"
    assert dt.squash_dir(tmp_path / "root", "training/x/store-v1", "x-store", mksquashfs=tools[0]) == "present"
    out = tmp_path / "extracted"
    subprocess.run([tools[1], "-q", "-n", "-d", str(out), str(tmp_path / "root" / "images" / "x-store.sqfs")], check=True)
    assert dt.verify_dir(out, workers=1)["passed"]
    (out / "a.npy").write_bytes(b"\1" * 5000)
    assert not dt.verify_dir(out, workers=1)["passed"]
    dt.link_dir(tmp_path / "root", "training/x/store-v1", "x-store")
    assert os.readlink(store) == str(tmp_path / "mnt" / "x-store")
