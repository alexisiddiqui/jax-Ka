"""Compute guards and atomic, content-addressed artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def require_compute(*, threads=1, gpu_benchmark=False, allow_comp1400=False):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("pkabench workloads require a Slurm compute allocation")
    if gpu_benchmark and not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')):
        raise RuntimeError('GPU benchmark exception requires an allocated GPU')
    if os.environ["SLURMD_NODENAME"].split(".")[0] == "comp1400" and not (gpu_benchmark or allow_comp1400):
        raise RuntimeError("comp1400 is excluded")
    if os.environ.get("SLURM_MEM_PER_CPU") != "2048":
        raise RuntimeError("pkabench requires --mem-per-cpu=2G")
    if not isinstance(threads,int) or not 1<=threads<=int(os.environ.get('SLURM_CPUS_PER_TASK','1')):
        raise ValueError('threads must fit the allocated Slurm CPUs')
    # Existing workers stay single-threaded. Explicit training experiments can
    # retain more allocated CPUs without changing unrelated benchmark jobs.
    if hasattr(os,"sched_getaffinity"):
        available=sorted(os.sched_getaffinity(0))
        if threads>len(available):raise RuntimeError('CPU affinity is narrower than requested threads')
        os.sched_setaffinity(0,set(available[:threads]))


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
