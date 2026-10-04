"""Compute guards and atomic, content-addressed artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("pkabench workloads require a Slurm compute allocation")
    if os.environ["SLURMD_NODENAME"].split(".")[0] == "comp1400":
        raise RuntimeError("comp1400 is excluded")
    if os.environ.get("SLURM_MEM_PER_CPU") != "2048":
        raise RuntimeError("pkabench requires --mem-per-cpu=2G")
    # Extra requested CPUs provide memory headroom. Scientific workers use one
    # allocated CPU, including native libraries that ignore OMP thread settings.
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
