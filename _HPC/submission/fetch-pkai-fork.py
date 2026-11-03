#!/usr/bin/env python3
"""Fetch the immutable pKAI fork source archive on a compute node."""

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

revision = "49ab002684450c55f1e2364dd1b1e9a6e09d6071"
url = f"https://github.com/alexisiddiqui/pKAI/archive/{revision}.tar.gz"
destination = Path("/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/sources/pKAI-fork-backbone")
if destination.exists():
    raise SystemExit(f"Refusing to replace existing destination: {destination}")
with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
    archive = Path(temporary) / "source.tar.gz"
    urllib.request.urlretrieve(url, archive)
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    unpacked = Path(temporary) / "unpacked"
    unpacked.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(unpacked, filter="data")
    roots = list(unpacked.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError("unexpected GitHub archive layout")
    shutil.move(roots[0], destination)
(destination / ".source-revision").write_text(revision + "\n")
(destination / ".archive-sha256").write_text(sha256 + "\n")
print(revision, sha256)
