"""jaxpb.api -- public API and the production driver (M6).

  energies(state_batch, cfg) -> [records.Energies]     PB (geometry -> maps -> SOR -> post) + JAX assembly per state
  titrate(energies_batch, cfg, chain_ids) -> [dict]    GPU MC in Mp buckets (curves [73,n], pKa [n], method, Mp, flags)
  label(records, cfg) -> Labels                          both, in memory
  run(items, out_dir, cfg, cache=...)                    the driver: StateRecord shards -> label shards (resumable)

Precision: importing this module turns process-wide x64 ON (DESIGN 6.9, energy-track request 8): assembly accumulates
e_back / pKint / g in float64; the PB path runs its float32 programs inside an x64-off context (pb.solve_state ->
geometry.x32) and its probe generation inside x64-on, so PB outputs do not depend on the flag (verified bitwise in M6).

MC RNG: chain c of a state at pH index j uses titrate.chain_keys(cfg.seed, chain_id * 73 + j) with chain_id =
chain_id_of(state alias) (28-bit sha256 of the alias). A state's labels therefore depend only on its own record, cfg
and alias -- not on batch composition, processing order, shard boundaries or restarts.

Driver (run):
  plan    items (input shard, state_id, alias) in the given order, cut into output shards of cfg.shard_size; saved
          to <out>/plan.json; a restart must reproduce the same plan and cfg (else it raises).
  shard   records of the shard loaded once (with_ref=False), PB + assembly in size-class order (executables stay warm),
          Energies queued per Mp bucket and titrated when a queue reaches cfg.s_flush (= 32) and at the end of the
          shard; output written to <out>/labels_<k>.npz via a temp file + os.replace (atomic).
  resume  a shard whose output exists, loads and carries this cfg_hash + code_sha is skipped (idempotent); stale temp
          files from a killed run are removed.
  overflow PB: maps overflow -> XL sizes and geometry overflow -> next class inside pb.solve_state (pb flags);
          any OverflowError that escapes -> one driver retry at the next size class; still failing -> state status
          'failed_pb' with the reason (written in the shard, counted in the run log). Assembly promotes to the next
          class (diag promoted), titration to the next Mp bucket (promoted) or the cluster-MF fallback (method 'cmf').
          Every event is logged in <out>/run.log.jsonl and in the shard's per-state diag JSON. Never silent.

Label shard layout (np.savez, ragged arrays concatenated along axis 0 with `<key>__off` offsets per state):
  state_id [S] U, alias [S] U, input [S] U, status [S] U ('ok' | 'failed_pb' | 'failed_titration')
  g [M] f32, occ [M] i8, W_ut [M(M-1)/2] f32 (np.triu_indices(M, 1) of the symmetric W; same-site entries 0),
  start/nst [n] i32, pairs [P,2] i32, pkint [M] f64, e_back [M] f64,
  curves [n,73] f64 (protonated fraction per site, i.e. curves[73,n] transposed), pka [n] f64 (nan = no midpoint),
  esolvM/esolvS/esolvM0 [T] f32, sweeps/maxchange [T,3], rho [T,2] f32, sitpot_sha [S] U (sha256 of sitpotM bytes),
  diag_json [S] U (pb flags, size class, promotions, Mp, method), provenance_json (cfg, cfg_hash, code_sha, jax,
  device, resolved backends, input shard sha256s; no timestamps), timing_json (wall times, compiles; excluded from
  bitwise comparisons).
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from . import records as RC          # noqa: E402
from . import titrate as T           # noqa: E402
from . import assemble as A          # noqa: E402

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_FORMAT = "jaxpb.labels/1"
RAGGED = ("g", "occ", "W_ut", "start", "nst", "pairs", "pkint", "e_back", "curves", "pka",
          "esolvM", "esolvS", "esolvM0", "sweeps", "maxchange", "rho")
DTYPES = dict(g=np.float32, occ=np.int8, W_ut=np.float32, start=np.int32, nst=np.int32, pairs=np.int32,
              pkint=np.float64, e_back=np.float64, curves=np.float64, pka=np.float64, esolvM=np.float32,
              esolvS=np.float32, esolvM0=np.float32, sweeps=np.int32, maxchange=np.float32, rho=np.float32)
TAIL = dict(pairs=(2,), curves=(RC.NPH,), sweeps=(3,), maxchange=(3,), rho=(2,))


# ------------------------------------------------------------------------------------------------ config / provenance
@dataclass(frozen=True)
class Config:
    pb_mode: str = "delphi"              # only "delphi" has ground truth (M7 adds "converged")
    backend: str = "auto"                # PB auto/Pallas/XLA; auto MC uses CPU chains or supported GPU kernels; explicit XLA keeps ref MC
    mc_sweeps: int = 200_000
    ph_grid: tuple = tuple(T.PH)         # PyPKA's 73 points -2..16 (fixed: titrate's pH table)
    seed: int = 1
    specrad: str = "estimate"            # focused/model Gauss-Seidel spectral radius: DelPhi relfac port (M2)
    B: int = RC.B                        # sites per PB batch
    s_flush: int = RC.S_FLUSH            # states per titration launch
    shard_size: int = 256                # states per output shard
    acc: str = "f64"                     # assembly accumulation
    geo_reuse: bool = True               # SPEED lever 3: A / B states of a triple reuse the AB geometry (identical probes)

    def check(self):
        if self.pb_mode != "delphi":
            raise NotImplementedError(f"pb_mode {self.pb_mode!r} (only 'delphi' until M7)")
        if tuple(self.ph_grid) != tuple(T.PH):
            raise ValueError("ph_grid must be PyPKA's 73-point grid (-2..16 step 0.25)")
        if self.backend not in ("auto", "pallas", "xla"):
            raise ValueError(self.backend)
        assert self.mc_sweeps % T.BLOCK == 0


def cfg_dict(cfg: Config) -> dict:
    d = dataclasses.asdict(cfg)
    d["ph_grid"] = list(d["ph_grid"])
    return d


def cfg_hash(cfg: Config) -> str:
    return RC.cfg_hash(cfg_dict(cfg))


def code_sha() -> str:
    """sha256 over the package sources (sorted file names + bytes), 16 hex."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(SRC_DIR, "*.py"))):
        h.update(os.path.basename(p).encode()); h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def pb_backend(cfg: Config) -> str:
    from . import pb as PB
    return PB.resolve_backend(cfg.backend)


def mc_backend(cfg: Config) -> str:
    if cfg.backend == "auto":
        return T.resolve_backend()
    return "pallas" if cfg.backend == "pallas" else "ref"


def chain_id_of(alias: str) -> int:
    """stable 28-bit MC chain id of a state alias (chain_id * 73 + 72 < 2^35, exact in the uint64 key hash)."""
    return int(hashlib.sha256(alias.encode()).hexdigest()[:7], 16)


def gpu_mem():
    """(bytes_in_use, peak_bytes_in_use, bytes_limit) of device 0 (-1 when the backend reports nothing)."""
    try:
        st = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001
        st = {}
    return int(st.get("bytes_in_use", -1)), int(st.get("peak_bytes_in_use", -1)), int(st.get("bytes_limit", -1))


def host_rss_bytes() -> int:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) * (1 if sys.platform == "darwin" else 1024)


class MemTrack:
    """GPU memory per stage from the allocator's process-wide high-water mark (JAX exposes no peak reset).
    mark(stage) after each stage: if peak_bytes_in_use rose since the previous mark, the rise belongs to that stage
    (true stage peak = new value); otherwise the stage stayed below the running peak (recorded as an upper bound).
    ctx = the size class 'c<k>' (PB, assembly) or the bucket 'Mp<k>' (titration). In-use bytes after the stage are
    kept too (live buffers that persist: probes, maps, cached executables)."""

    def __init__(self):
        self.ctx = "-"
        self.last = -1                 # initialize lazily at the first mark, never initialize JAX on import
        self.rows: Dict[tuple, dict] = {}

    def mark(self, stage, ctx=None):
        use, peak, _ = gpu_mem()
        key = (stage, ctx or self.ctx)
        d = self.rows.setdefault(key, dict(stage=stage, ctx=key[1], peak_raised_to=0, n_raised=0, n=0,
                                           upper_bound=0, in_use_after_max=0))
        d["n"] += 1
        d["in_use_after_max"] = max(d["in_use_after_max"], use)
        if peak > self.last:
            d["peak_raised_to"] = max(d["peak_raised_to"], peak); d["n_raised"] += 1
        d["upper_bound"] = max(d["upper_bound"], peak)
        self.last = max(self.last, peak)

    def table(self):
        return sorted(self.rows.values(), key=lambda d: (d["stage"], d["ctx"]))


MEMTRACK = MemTrack()


def sitpot_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


# ------------------------------------------------------------------------------------------------ API
@dataclass
class Labels:
    """label() output. Per state (ragged, lists): curves [73,n] f64, pKa [n] f64 (nan = PyPKA's None), energies
    (records.Energies with diag pkint/e_back/pb), titration info (method, Mp, promoted, flag)."""
    state_ids: List[str]
    curves: List[np.ndarray]
    pKa: List[np.ndarray]
    energies: List[RC.Energies]
    titration: List[dict] = field(default_factory=list)


class PBFailure(Exception):
    pass


def _pbcfg(cfg: Config):
    from . import pb as PB
    return PB.PBConfig(pb_mode=cfg.pb_mode, backend=cfg.backend, B=cfg.B, specrad=cfg.specrad)


def triple_role(state_id: str):
    """(cid, role) of a state id '<cid>_<AB|A|B>' (role None otherwise)."""
    cid, _, role = state_id.rpartition("_")
    return (cid, role) if role in ("AB", "A", "B") and cid else (state_id, None)


def solve_pb(rec: RC.StateRecord, cfg: Config, log: Optional[Callable] = None, geo_cache: Optional[dict] = None):
    """PB of one state with the driver-level overflow retry (one more size class). Returns (PBOut, timing).
    geo_cache (lever 3, cfg.geo_reuse): dict cid -> AB GeoReuse; an AB state stores its own, an A / B state uses it."""
    from . import pb as PB
    pbc = _pbcfg(cfg)
    c = rec.size_class
    tries = []
    cid, role = triple_role(rec.state_id)
    use = geo_cache is not None and cfg.geo_reuse and role is not None
    parent = geo_cache.get(cid) if use and role != "AB" else None
    while True:
        tim: Dict[str, Any] = {"memtrack": MEMTRACK}
        MEMTRACK.ctx = f"c{c}"
        gout = {} if use and role == "AB" else None
        try:
            out = PB.solve_state(rec, pbc, timing=tim, c=c, geo_parent=parent, geo_out=gout)
            if gout is not None and gout.get("reuse") is not None:
                geo_cache[cid] = gout["reuse"]
            out.flags["driver_retries"] = len(tries)
            if tries:
                out.flags["driver_retry_reasons"] = tries
            return out, tim
        except OverflowError as e:
            tries.append(dict(size_class=c, error=str(e)))
            if log:
                log(dict(event="pb_overflow", state_id=rec.state_id, size_class=c, error=str(e)))
            if len(tries) > 1 or c + 1 >= len(RC.NA_C):
                raise PBFailure(json.dumps(tries))
            c += 1


def energies(state_batch: Sequence[RC.StateRecord], cfg: Config = Config(), log: Optional[Callable] = None,
             timing: Optional[list] = None, geo_cache: Optional[dict] = None) -> List[RC.Energies]:
    """PB + assembly for each StateRecord -> records.Energies (diag: pkint, e_back, size_class, promoted, npairs,
    occ_matches_record, pb = {esolvM, esolvS, esolvM0, sweeps, maxchange, rho, flags, sitpot_sha}).
    Lever 3: AB states before their A / B states reuse geometry (geo_cache; a fresh dict per call if None)."""
    cfg.check()
    out = []
    geo_cache = {} if geo_cache is None else geo_cache
    for rec in state_batch:
        t0 = time.perf_counter()
        pb, tim = solve_pb(rec, cfg, log, geo_cache)
        t1 = time.perf_counter()
        e = A.energies([rec], [pb], acc=cfg.acc)[0]
        t2 = time.perf_counter()
        MEMTRACK.mark("assembly", f"c{e.diag['size_class']}")
        e.diag["pb"] = dict(esolvM=pb.esolvM, esolvS=pb.esolvS, esolvM0=pb.esolvM0, sweeps=pb.sweeps,
                            maxchange=pb.maxchange, rho=pb.rho, flags=pb.flags, sitpot_sha=sitpot_sha(pb.sitpotM),
                            backend=pb.backend, cfg_hash=pb.cfg_hash)
        if timing is not None:
            timing.append(dict(state_id=rec.state_id, t_pb=t1 - t0, t_asm=t2 - t1,
                               **{k: v for k, v in tim.items() if isinstance(v, (int, float))}))
        out.append(e)
    return out


def titrate(energies_batch: Sequence[RC.Energies], cfg: Config = Config(), chain_ids=None, on_batch=None,
            mc_sweeps: Optional[int] = None) -> List[dict]:
    """GPU MC on a batch of Energies (any mix of sizes: titrate buckets them, S = cfg.s_flush per launch).
    chain_ids: per-state ints (default chain_id_of(str(k))). Returns titrate.titrate dicts (curves [73,n], pka [n])."""
    cfg.check()
    if chain_ids is None:
        chain_ids = [chain_id_of(str(k)) for k in range(len(energies_batch))]
    return T.titrate(list(energies_batch), mc_sweeps=mc_sweeps or cfg.mc_sweeps, seed=cfg.seed,
                     chain_ids=np.asarray(chain_ids, np.int64), backend=mc_backend(cfg), S=cfg.s_flush,
                     on_batch=on_batch)


def label(records: Sequence[RC.StateRecord], cfg: Config = Config(), aliases: Optional[List[str]] = None) -> Labels:
    """PB -> assembly -> titration for records in memory (no shards, no resume)."""
    aliases = aliases or [r.state_id for r in records]
    E = energies(records, cfg)
    R = titrate(E, cfg, chain_ids=[chain_id_of(a) for a in aliases])
    return Labels(state_ids=[r.state_id for r in records], curves=[r.get("curves") for r in R],
                  pKa=[r.get("pka") for r in R], energies=E, titration=R)


# ------------------------------------------------------------------------------------------------ label shards
def _state_arrays(rec_or_n, e: Optional[RC.Energies], tr: Optional[dict]) -> Dict[str, np.ndarray]:
    """one state's ragged arrays (shapes per module docstring)."""
    n = rec_or_n
    d = {k: np.zeros((0,) + TAIL.get(k, ()), DTYPES[k]) for k in RAGGED}
    d["curves"] = np.full((n, RC.NPH), np.nan); d["pka"] = np.full(n, np.nan)
    if e is not None:
        M = len(e.g)
        if not np.array_equal(e.W, e.W.T):
            raise ValueError("W not exactly symmetric: upper-triangle storage would lose information")
        d.update(g=e.g, occ=e.occ, W_ut=e.W[np.triu_indices(M, 1)], start=e.start, nst=e.nst,
                 pairs=np.asarray(e.pairs).reshape(-1, 2), pkint=e.diag["pkint"], e_back=e.diag["e_back"])
        p = e.diag["pb"]
        d.update(esolvM=p["esolvM"], esolvS=p["esolvS"], esolvM0=p["esolvM0"], sweeps=p["sweeps"],
                 maxchange=p["maxchange"], rho=p["rho"])
    if tr is not None and tr.get("curves") is not None:
        d["curves"] = np.asarray(tr["curves"], np.float64).T; d["pka"] = np.asarray(tr["pka"], np.float64)
    return {k: np.asarray(v, DTYPES[k]).reshape((-1,) + TAIL.get(k, ())) for k, v in d.items()}


def write_shard(path: str, rows: List[dict], provenance: dict, timing: dict) -> None:
    """rows: dicts with state_id, alias, input, status, arrays (from _state_arrays), diag (json-able)."""
    out = {"format": np.array(LABEL_FORMAT), "n_states": np.array(len(rows)),
           "state_id": np.array([r["state_id"] for r in rows]), "alias": np.array([r["alias"] for r in rows]),
           "input": np.array([r["input"] for r in rows]), "status": np.array([r["status"] for r in rows]),
           "sitpot_sha": np.array([r.get("sitpot_sha", "") for r in rows]),
           "diag_json": np.array([json.dumps(r["diag"], sort_keys=True, default=_jdefault) for r in rows]),
           "provenance_json": np.array(json.dumps(provenance, sort_keys=True, default=_jdefault)),
           "timing_json": np.array(json.dumps(timing, sort_keys=True, default=_jdefault))}
    for k in RAGGED:
        arrs = [r["arrays"][k] for r in rows]
        off = np.zeros(len(arrs) + 1, np.int64); off[1:] = np.cumsum([len(a) for a in arrs])
        out[k] = np.concatenate(arrs) if arrs else np.zeros((0,) + TAIL.get(k, ()), DTYPES[k])
        out[k + "__off"] = off
    tmp = f"{path}.tmp-{os.getpid()}.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **out)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def load_shard(path: str, full_W: bool = True) -> List[dict]:
    """label shard -> list of per-state dicts (W rebuilt as the full symmetric [M,M] when full_W)."""
    Z = np.load(path, allow_pickle=False)
    n = int(Z["n_states"])
    arr = {k: (Z[k], Z[k + "__off"]) for k in RAGGED}
    out = []
    for i in range(n):
        d = dict(state_id=str(Z["state_id"][i]), alias=str(Z["alias"][i]), input=str(Z["input"][i]),
                 status=str(Z["status"][i]), sitpot_sha=str(Z["sitpot_sha"][i]), diag=json.loads(str(Z["diag_json"][i])))
        for k, (a, off) in arr.items():
            d[k] = a[off[i]:off[i + 1]]
        if full_W:
            M = len(d["g"]); W = np.zeros((M, M), np.float32)
            iu = np.triu_indices(M, 1); W[iu] = d["W_ut"]; W[(iu[1], iu[0])] = d["W_ut"]
            d["W"] = W
        d["curves"] = d["curves"].T          # -> [73, n]
        out.append(d)
    return out


def shard_provenance(path: str) -> dict:
    Z = np.load(path, allow_pickle=False)
    return json.loads(str(Z["provenance_json"]))


def to_energies(d: dict) -> RC.Energies:
    return RC.Energies(g=d["g"], occ=d["occ"], W=d["W"], start=d["start"], nst=d["nst"], pairs=d["pairs"],
                       diag=dict(pkint=d["pkint"], e_back=d["e_back"]))


# ------------------------------------------------------------------------------------------------ driver
def items_from_shards(paths: Sequence[str]) -> List[dict]:
    return [dict(input=os.path.abspath(p), state_id=sid, alias=sid) for p in paths for sid in RC.list_states(p)]


def make_plan(items: List[dict], cfg: Config) -> dict:
    shards = [items[i:i + cfg.shard_size] for i in range(0, len(items), cfg.shard_size)]
    return dict(cfg=cfg_dict(cfg), cfg_hash=cfg_hash(cfg), n_items=len(items), shards=shards)


class _Log:
    def __init__(self, path: str, echo: bool = True):
        self.path = path; self.echo = echo
        self.counts: Dict[str, int] = {}

    def __call__(self, ev: dict):
        ev = dict(ev, t=round(time.time(), 3), pid=os.getpid())
        self.counts[ev["event"]] = self.counts.get(ev["event"], 0) + 1
        line = json.dumps(ev, sort_keys=True, default=_jdefault)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


def _shard_done(path: str, chash: str, csha: str, plan_ids: List[str]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        Z = np.load(path, allow_pickle=False)
        prov = json.loads(str(Z["provenance_json"]))
        ids = [str(x) for x in Z["alias"]]
    except Exception:            # truncated / unreadable -> recompute (os.replace makes this unlikely)
        return False
    if prov.get("cfg_hash") != chash:
        raise RuntimeError(f"{path}: written with cfg_hash {prov.get('cfg_hash')} != {chash}")
    if prov.get("code_sha") != csha:
        raise RuntimeError(f"{path}: written with code_sha {prov.get('code_sha')} != {csha} (refusing to mix code)")
    if ids != plan_ids:
        raise RuntimeError(f"{path}: state list differs from the plan")
    return True


def run(items: List[dict], out_dir: str, cfg: Config = Config(), cache: Optional[str] = None,
        max_shards: Optional[int] = None, echo: bool = True) -> dict:
    """Label every item into <out_dir>/labels_<k>.npz (resumable). Returns the run summary (also written to
    <out_dir>/run_<pid>.json)."""
    cfg.check()
    os.makedirs(out_dir, exist_ok=True)
    from .compile_budget import CompileCounter, cache_open, cache_close
    cinfo = cache_open(cache) if cache else None
    log = _Log(os.path.join(out_dir, "run.log.jsonl"), echo)
    plan = make_plan(items, cfg)
    pp = os.path.join(out_dir, "plan.json")
    if os.path.exists(pp):
        old = json.load(open(pp))
        if old != json.loads(json.dumps(plan)):
            raise RuntimeError(f"{pp} differs from this run's plan/cfg: refusing to resume into it")
    else:
        tmp = pp + f".tmp-{os.getpid()}"
        json.dump(plan, open(tmp, "w"), indent=0); os.replace(tmp, pp)
    for stale in glob.glob(os.path.join(out_dir, "labels_*.tmp-*")):
        log(dict(event="remove_stale_tmp", path=stale)); os.remove(stale)
    chash, csha = cfg_hash(cfg), code_sha()
    inputs = sorted({it["input"] for it in items})
    prov = dict(format=LABEL_FORMAT, cfg=cfg_dict(cfg), cfg_hash=chash, code_sha=csha, jax=jax.__version__,
                device=jax.devices()[0].device_kind, platform=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
                pb_backend=pb_backend(cfg), mc_backend=mc_backend(cfg),
                inputs={p: RC.sha256_file(p) for p in inputs})
    log(dict(event="start", out=out_dir, n_items=len(items), n_shards=len(plan["shards"]), cfg_hash=chash, code_sha=csha,
             cache=cinfo, pb_backend=prov["pb_backend"], mc_backend=prov["mc_backend"]))
    summary = dict(out=out_dir, cfg_hash=chash, code_sha=csha, cache_open=cinfo, shards_skipped=[], shards_written=[],
                   state_timing=dict(count=0, failed=0, totals={}, maxima={}),
                   titration_batches=[], compiles_after_shard=[], shard_bytes=[])
    T0 = time.perf_counter()
    with CompileCounter() as cc:
        for k, shard in enumerate(plan["shards"]):
            path = os.path.join(out_dir, f"labels_{k:05d}.npz")
            if _shard_done(path, chash, csha, [it["alias"] for it in shard]):
                summary["shards_skipped"].append(k); log(dict(event="shard_skip", shard=k))
                continue
            if max_shards is not None and len(summary["shards_written"]) >= max_shards:
                break
            _run_shard(k, shard, path, cfg, prov, log, cc, summary)
            summary["shards_written"].append(k)
            summary["compiles_after_shard"].append(dict(shard=k, states_done=sum(len(plan["shards"][j]) for j in range(k + 1)),
                                                        n_compiled=cc.n_compiled, n_cache_hits=cc.n_cache_hits))
        summary["compiles"] = cc.summary()
    summary["wall_s"] = time.perf_counter() - T0
    if cache:
        summary["cache_close"] = cache_close(cache, cinfo["local_dir"])
    summary["events"] = log.counts
    use, peak, lim = gpu_mem()
    summary["memory"] = dict(gpu_peak_bytes=peak, gpu_in_use_end=use, gpu_bytes_limit=lim, host_maxrss_bytes=host_rss_bytes(),
                             stages=MEMTRACK.table(),
                             env={k: os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                                                                  "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")})
    log(dict(event="end", wall_s=summary["wall_s"], n_compiled=summary["compiles"]["n_compiled"],
             gpu_peak_bytes=peak, host_maxrss_bytes=summary["memory"]["host_maxrss_bytes"],
             written=summary["shards_written"], skipped=summary["shards_skipped"]))
    json.dump(summary, open(os.path.join(out_dir, f"run_{os.getpid()}.json"), "w"), indent=1, default=_jdefault)
    return summary


def _run_shard(k, shard, path, cfg, prov, log, cc, summary):
    t_shard = time.perf_counter()
    by_input: Dict[str, List[str]] = {}
    for it in shard:
        by_input.setdefault(it["input"], []).append(it["state_id"])
    recs: Dict[tuple, RC.StateRecord] = {}
    for p, ids in by_input.items():
        for r in RC.load_states(p, only=list(dict.fromkeys(ids)), with_ref=False):
            recs[(p, r.state_id)] = r
    t_load = time.perf_counter() - t_shard
    cls_of = lambda i: recs[(shard[i]["input"], shard[i]["state_id"])].size_class
    order = sorted(range(len(shard)), key=lambda i: (cls_of(i), i))
    grp: Dict[int, tuple] = {}
    if cfg.geo_reuse:         # lever 3: each triple's AB, then its A and B, consecutively (outputs do not depend on order)
        tri: Dict[tuple, Dict[str, List[int]]] = {}
        for i, it in enumerate(shard):
            cid, role = triple_role(it["state_id"])
            if role is not None:
                tri.setdefault((it["input"], cid), {}).setdefault(role, []).append(i)
        key = {}
        for g, d in tri.items():
            if "AB" in d and ("A" in d or "B" in d):
                mem = d.get("AB", []) + d.get("A", []) + d.get("B", [])
                for i in mem:
                    key[i] = (cls_of(d["AB"][0]), min(mem), {"AB": 0, "A": 1, "B": 2}[triple_role(shard[i]["state_id"])[1]], i)
                    grp[i] = g
        order = sorted(range(len(shard)), key=lambda i: key.get(i, (cls_of(i), i, 0, i)))
    left = {}
    for i, g in grp.items():
        left[g] = left.get(g, 0) + 1
    geo_cache: Dict[str, Any] = {}
    rows: List[Optional[dict]] = [None] * len(shard)
    E: Dict[int, RC.Energies] = {}
    queues: Dict[Any, List[int]] = {}
    st_tim: Dict[int, dict] = {}

    def flush(key):
        idx = queues.pop(key, [])
        if not idx:
            return
        c0 = cc.n_compiled; t0 = time.perf_counter(); bt = []

        def onb(tb):
            bt.append(tb); MEMTRACK.mark("titration", f"Mp{tb['Mp']}")
        res = titrate([E[i] for i in idx], cfg, chain_ids=[chain_id_of(shard[i]["alias"]) for i in idx],
                      on_batch=onb)
        dt = time.perf_counter() - t0
        summary["titration_batches"].append(dict(shard=k, Mp=key, states=len(idx), seconds=dt,
                                                 compiles=cc.n_compiled - c0, launches=bt))
        for i, r in zip(idx, res):
            st_tim[i]["t_titr"] = dt / len(idx); st_tim[i]["titr_compiles"] = cc.n_compiled - c0
            _finish(i, r)

    def _finish(i, r):
        row = rows[i]
        ok = r.get("curves") is not None          # MC result, or the cluster-MF fallback (method 'cmf', logged)
        row["status"] = "ok" if ok else "failed_titration"
        row["diag"]["titration"] = dict(method=r.get("method"), Mp=r.get("Mp"), promoted=bool(r.get("promoted")),
                                        flag=r.get("flag"))
        if r.get("promoted") or r.get("method") == "cmf" or not ok:
            log(dict(event="titration_" + ("promoted" if ok else "failed"), shard=k, alias=row["alias"],
                     Mp=r.get("Mp"), method=r.get("method"), flag=r.get("flag")))
        row["arrays"] = _state_arrays(rows[i]["n_sites"], E[i], r)

    for i in order:
        it = shard[i]; rec = recs[(it["input"], it["state_id"])]
        rows[i] = dict(state_id=it["state_id"], alias=it["alias"], input=it["input"], n_sites=rec.n_sites,
                       status="pending", diag={})
        c0 = cc.n_compiled; tl: list = []
        g = grp.get(i)
        try:
            e = energies([rec], cfg, log=log, timing=tl, geo_cache=geo_cache if g is not None else None)[0]
        except PBFailure as ex:
            if g is not None:
                left[g] -= 1
                if left[g] == 0:
                    geo_cache.pop(g[1], None)
            rows[i].update(status="failed_pb", diag=dict(pb_failure=json.loads(str(ex))))
            rows[i]["arrays"] = _state_arrays(rec.n_sites, None, None)
            log(dict(event="pb_failed", shard=k, alias=it["alias"], reason=str(ex)))
            st_tim[i] = dict(state_id=it["state_id"], failed=True)
            continue
        if g is not None:
            left[g] -= 1
            if left[g] == 0:
                geo_cache.pop(g[1], None)         # the triple is done: free the AB geometry
        st_tim[i] = dict(tl[0], alias=it["alias"], size_class=rec.size_class, n_taus=rec.n_taus,
                         compiles=cc.n_compiled - c0)
        pbf = e.diag["pb"]["flags"]
        rows[i]["sitpot_sha"] = e.diag["pb"]["sitpot_sha"]
        rows[i]["diag"] = dict(pb_flags=pbf, pb_backend=e.diag["pb"]["backend"], size_class=e.diag["size_class"],
                               asm_promoted=bool(e.diag["promoted"]), npairs=int(e.diag["npairs"]),
                               occ_matches_record=bool(e.diag["occ_matches_record"]))
        if pbf.get("xl_retries") or pbf.get("class_bumps") or pbf.get("driver_retries"):
            log(dict(event="pb_retry", shard=k, alias=it["alias"], xl_retries=pbf.get("xl_retries"),
                     class_bumps=pbf.get("class_bumps"), driver_retries=pbf.get("driver_retries")))
        if e.diag["promoted"]:
            log(dict(event="asm_promoted", shard=k, alias=it["alias"], size_class=e.diag["size_class"]))
        E[i] = e
        Mp, _ = T.fit_bucket(len(e.g), len(e.nst), len(e.pairs))
        key = Mp if Mp is not None else "none"
        queues.setdefault(key, []).append(i)
        if len(queues[key]) >= cfg.s_flush:
            flush(key)
    for key in sorted(queues, key=str):
        flush(key)
    assert all(r is not None and r["status"] != "pending" for r in rows)
    timing = dict(shard=k, t_load=t_load, t_shard=time.perf_counter() - t_shard,
                  states=[st_tim[i] for i in range(len(shard))], host_maxrss_bytes=host_rss_bytes(),
                  gpu_mem=dict(zip(("in_use", "peak", "limit"), gpu_mem())))
    write_shard(path, rows, prov, timing)
    summary["shard_bytes"].append(dict(shard=k, states=len(shard), bytes=os.path.getsize(path)))
    agg = summary["state_timing"]
    for row in st_tim.values():
        agg["count"] += 1
        agg["failed"] += int(row.get("failed", False))
        for name, value in row.items():
            if name.startswith("t_") and isinstance(value, (int, float)):
                agg["totals"][name] = agg["totals"].get(name, 0.0) + value
                agg["maxima"][name] = max(agg["maxima"].get(name, 0.0), value)
    log(dict(event="shard_written", shard=k, path=path, n=len(shard), t_shard=timing["t_shard"],
             status={s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
             n_compiled=cc.n_compiled))


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="jaxpb label driver: StateRecord shards -> label shards (resumable)")
    ap.add_argument("out")
    ap.add_argument("--inputs", nargs="*", default=[], help="StateRecord shards (all states, in file order)")
    ap.add_argument("--items", default=None, help="JSON list of {input, state_id, alias} (overrides --inputs)")
    ap.add_argument("--cache", default=None, help="compile-cache tarball (opened at start, re-tarred at the end)")
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--mc-sweeps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--max-shards", type=int, default=None)
    a = ap.parse_args(argv)
    items = json.load(open(a.items)) if a.items else items_from_shards(a.inputs)
    cfg = Config(shard_size=a.shard_size, mc_sweeps=a.mc_sweeps, seed=a.seed, backend=a.backend)
    s = run(items, a.out, cfg, cache=a.cache, max_shards=a.max_shards)
    print(json.dumps({k: s[k] for k in ("shards_written", "shards_skipped", "wall_s")} | {"compiles": s["compiles"]["n_compiled"]},
                     default=_jdefault))


if __name__ == "__main__":
    main()
