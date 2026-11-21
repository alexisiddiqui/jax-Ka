"""jaxpb.compile_budget -- XLA compile counting, the persistent-cache tarball, and the compile budget.

CompileCounter: counts XLA backend compiles (persistent-cache misses) through jax.monitoring's
'/jax/core/compile/backend_compile_duration' event, persistent-cache hits through '/jax/compilation_cache/cache_hits',
and names the compiled programs from JAX's compile log (jax_log_compiles). Listeners are registered once per process
and gated by the active counter (jax.monitoring has no public unregister).

cache_open / cache_close: the compilation cache lives in a node-local directory (default /tmp/jaxpb-cc-<jobid>); it is
untarred from ONE tarball on $SCRATCH at job start and re-tarred atomically at the end (1 inode on $SCRATCH).

budget(): the number of executables the M1 PB path may compile for a set of size classes (see BUDGET_TABLE).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
from typing import Dict, Iterable, List, Optional

_ACTIVE: List["CompileCounter"] = []
_REGISTERED = {"done": False}
_NAME_RE = [re.compile(r"Finished XLA compilation of (\S+)"), re.compile(r"Compiling (\S+) with global shapes"),
            re.compile(r"Compiling (\S+)"), re.compile(r"Persistent compilation cache hit for '([^']+)'")]


class _LogHandler(logging.Handler):
    def emit(self, record):
        if not _ACTIVE:
            return
        try:
            msg = record.getMessage()
        except Exception:
            return
        for c in _ACTIVE:
            c._log(msg)


def _register():
    if _REGISTERED["done"]:
        return
    import jax
    import jax.monitoring as mon

    def dur(event, duration, **kw):
        for c in _ACTIVE:
            c._dur(event, duration)

    def ev(event, **kw):
        for c in _ACTIVE:
            c._ev(event)

    mon.register_event_duration_secs_listener(dur)
    mon.register_event_listener(ev)
    h = _LogHandler(); h.setLevel(logging.DEBUG)
    for name in ("jax", "jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.compiler", "jax._src.compilation_cache"):
        logging.getLogger(name).addHandler(h)
    jax.config.update("jax_log_compiles", True)
    _REGISTERED["done"] = True


class CompileCounter:
    """with CompileCounter() as cc: ...;  cc.n_compiled (real XLA compiles), cc.n_cache_hits, cc.names, cc.seconds."""

    def __init__(self, quiet: bool = True):
        self.n_backend = 0
        self.n_cache_hits = 0
        self.n_cache_misses = 0
        self.seconds = 0.0
        self.names: List[str] = []
        self.hit_names: List[str] = []
        self.events: Dict[str, int] = {}
        self.quiet = quiet

    def _dur(self, event, duration):
        self.events[event] = self.events.get(event, 0) + 1
        if event.endswith("backend_compile_duration"):
            self.n_backend += 1
            self.seconds += float(duration)

    def _ev(self, event):
        self.events[event] = self.events.get(event, 0) + 1
        if event.endswith("cache_hits"):
            self.n_cache_hits += 1
        elif event.endswith("cache_misses"):
            self.n_cache_misses += 1

    def _log(self, msg):
        m = _NAME_RE[3].search(msg)
        if m:
            self.hit_names.append(m.group(1)); return
        m = _NAME_RE[0].search(msg)
        if m:
            self.names.append(m.group(1))

    def __enter__(self):
        _register()
        if self.quiet:
            for name in ("jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.compiler"):
                logging.getLogger(name).propagate = False
        _ACTIVE.append(self)
        return self

    def __exit__(self, *a):
        _ACTIVE.remove(self)

    @property
    def n_compiled(self) -> int:
        """XLA compiles that were not served by the persistent cache (some JAX versions emit backend_compile_duration
        for cache hits too: it then times the retrieval)."""
        return self.n_backend - self.n_cache_hits

    def summary(self) -> dict:
        return dict(n_compiled=self.n_compiled, n_backend=self.n_backend, n_cache_hits=self.n_cache_hits, n_cache_misses=self.n_cache_misses,
                    compile_s=round(self.seconds, 3), names=self.names, hit_names=self.hit_names)


# ------------------------------------------------------------------------------------------------ cache tarball
def cache_open(tar_path: Optional[str], local_dir: Optional[str] = None, min_compile_s: float = 0.0) -> dict:
    """Point JAX's persistent compilation cache at a node-local dir, filled from `tar_path` if it exists.
    min_compile_s: jax_persistent_cache_min_compile_time_secs (default 0: every executable is cached; the cache is one
    tarball on $SCRATCH, so entry count costs no inodes there)."""
    import jax
    local_dir = local_dir or os.path.join("/tmp", f"jaxpb-cc-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    os.makedirs(local_dir, exist_ok=True)
    info = {"local_dir": local_dir, "tar": tar_path, "untar_entries": 0, "untar_s": 0.0}
    if tar_path and os.path.exists(tar_path):
        t = time.perf_counter()
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.isfile() and "/" not in m.name.strip("./") and ".." not in m.name]
            tf.extractall(local_dir, members=members)
        info["untar_entries"] = len(members); info["untar_s"] = time.perf_counter() - t
    jax.config.update("jax_compilation_cache_dir", local_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_s))
    try:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except Exception:
        pass
    try:   # JAX caches the cache object at first use; make sure the new dir is used
        from jax._src import compilation_cache as _cc
        _cc.reset_cache()
    except Exception:
        pass
    return info


def cache_close(tar_path: str, local_dir: str) -> dict:
    """Re-tar the local cache dir into `tar_path` atomically (write tmp in the same dir, then rename)."""
    t = time.perf_counter()
    names = sorted(n for n in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, n)))
    d = os.path.dirname(os.path.abspath(tar_path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".cc-", suffix=".tar", dir=d)
    os.close(fd)
    with tarfile.open(tmp, "w") as tf:
        for n in names:
            tf.add(os.path.join(local_dir, n), arcname=n)
    os.replace(tmp, tar_path)
    return {"entries": len(names), "bytes": os.path.getsize(tar_path), "tar_s": time.perf_counter() - t}


def cache_clear_local(local_dir: str):
    shutil.rmtree(local_dir, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ budget
# Executables of the M1 PB path (names as in the compile log).
# Per size class c touched (array shapes NA_C[c] / NP_C[c] / NATOM_ALL_C[c]): geometry _cell_hash, geo_pairs,
# geo_vertex, geo_arc, geo_pack; maps_batch (coarse + focused + model maps of B sites); post (site potentials at all
# class-c atoms). Class independent: relfac_terms (M2 spectral radius), geometry_sites (model probes), setup1, setup2, and the lane pool: sor_block,
# lanes_init, lanes_store per lane width (3 widths), lanes_take per width transition (<= 3).
# An XL retry adds maps_batch + geometry_sites for the XL Sizes (once per process).
# SPEED (2026-11-20): geo_vertex + geo_arc + geo_pack (host chunk loop) -> ONE geo_probes per class (device loops); a
# geometry cell-occupancy overflow (K = GEO_K 16) retries the class with XL_GEO_K = 64 (one more geo_probes per class).
# M5: phase 2 pads the coefficient table to E = 2B, so sor_block compiles once per lane width (M1 compiled it per
# (width, E) = 6x). The Pallas backend replaces the whole lane pool by ONE executable (sor_solve: fixed W = 2*B*T_MAX
# lanes, E = 2B operators, the Triton kernel inside).
BUDGET_TABLE = {
    "per_class": {"_cell_hash": 1, "geo_pairs": 1, "geo_probes": 1, "maps_batch": 1, "post": 1},
    "fixed": {"relfac_terms": 1, "geometry_sites": 1, "setup1": 1, "setup2": 1},
    "sor": {"xla": {"sor_block": 3, "lanes_init": 3, "lanes_store": 3, "lanes_take": 3}, "pallas": {"sor_solve": 1}},
    "per_xl_variant": {"maps_batch": 1, "geometry_sites": 1},
    "per_geo_k_retry_class": {"geo_probes": 1},
}


def budget(classes: Iterable[int], xl_variants: int = 0, backend: str = "xla", geo_k_retry_classes: int = 0) -> int:
    """max executables for a run touching `classes` (+ XL variants compiled after overflow retries) on SOR `backend`."""
    nc = len(set(classes))
    return (sum(BUDGET_TABLE["per_class"].values()) * nc + sum(BUDGET_TABLE["fixed"].values())
            + sum(BUDGET_TABLE["sor"][backend].values()) + xl_variants * sum(BUDGET_TABLE["per_xl_variant"].values())
            + geo_k_retry_classes * sum(BUDGET_TABLE["per_geo_k_retry_class"].values()))
