"""jobs/bench_pypka.py -- arm A of BUILD-BENCH (2026-11-21): native PyPKA (DelPhi PB + Cython MC) on one state, timed.

PyPKA env (python3.10, psutil). Exactly the campaign run (request.json config, ncpus=N, pH -2..16 step 0.25, 200k sweeps,
seed 1234567, couple_min 2.0, save_mc_energies), with in-process monkeypatches that only add timing / memory / pinning:
  - Titration.preprocessing + processDelPhiParams -> stage 'prep'; DelPhiLaunch (PB + pKint) -> 'pb';
    calcSiteInteractionsParallel -> 'assembly'; run_mc -> 'mc'
  - pool initializer pins each worker to its own allowed core (pkabench pinning note): worker identity k -> core k mod n
  - psutil sampler (50 ms): RSS summed over the process tree (main + pool workers), peak per stage
CPU seconds per stage = getrusage(SELF) + getrusage(CHILDREN) deltas (pool workers are reaped at pool.join).
usage: python bench_pypka.py <campaign_state_dir> <work_dir> <out.json> --ncpus N   (<work_dir>/input.pdb must exist)
"""
import argparse
import json
import multiprocessing as mp
import os
import resource
import threading
import time
from pathlib import Path

import psutil

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("work"); ap.add_argument("out")
ap.add_argument("--ncpus", type=int, default=1)
A = ap.parse_args()
SRC = Path(A.src).resolve(); WORK = Path(A.work).resolve(); WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
T0 = time.perf_counter()
CORES = sorted(os.sched_getaffinity(0))

import pypka.concurrency as conc  # noqa: E402
import pypka.main as pmain  # noqa: E402
from pypka import Titration  # noqa: E402

_refresh = conc.configRefresh


def pinned_refresh(*a):
    k = mp.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {CORES[k % len(CORES)]})
    return _refresh(*a)


conc.configRefresh = pinned_refresh

STAGE = {"cur": "import"}
PEAK = {}
PINS = set()
STOP = threading.Event()
ME = psutil.Process()


def sampler():
    while not STOP.is_set():
        try:
            procs = [ME] + ME.children(recursive=True)
            tot = 0
            for p in procs:
                try:
                    tot += p.memory_info().rss
                    if p.pid != ME.pid:
                        PINS.add(tuple(sorted(p.cpu_affinity())))
                except psutil.Error:
                    pass
            s = STAGE["cur"]
            PEAK[s] = max(PEAK.get(s, 0), tot)
            PEAK["_n_procs_max"] = max(PEAK.get("_n_procs_max", 0), len(procs))
        except psutil.Error:
            pass
        STOP.wait(0.05)


threading.Thread(target=sampler, daemon=True).start()


def cpu_s():
    a = resource.getrusage(resource.RUSAGE_SELF); b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


TIM = {}


def timed(name, fn):
    def w(self, *a, **kw):
        prev = STAGE["cur"]; STAGE["cur"] = name
        t, c = time.perf_counter(), cpu_s()
        try:
            return fn(self, *a, **kw)
        finally:
            d = TIM.setdefault(name, dict(wall=0.0, cpu=0.0))
            d["wall"] += time.perf_counter() - t; d["cpu"] += cpu_s() - c
            STAGE["cur"] = prev
    return w


pmain.Titration.preprocessing = timed("prep", pmain.Titration.preprocessing)
pmain.Titration.processDelPhiParams = timed("prep", pmain.Titration.processDelPhiParams)
pmain.Titration.DelPhiLaunch = timed("pb", pmain.Titration.DelPhiLaunch)
pmain.Titration.calcSiteInteractionsParallel = timed("assembly", pmain.Titration.calcSiteInteractionsParallel)
pmain.Titration.run_mc = timed("mc", pmain.Titration.run_mc)

request = json.loads((SRC / "request.json").read_text())
params = dict(request["config"])
params.update(structure=str(WORK / "input.pdb"), ncpus=A.ncpus, pH="-2,16", pHstep=.25,
              save_mc_energies="mc-energies.json", output="midpoints.txt", titration_output="curves.txt")
t_start, c_start = time.perf_counter(), cpu_s()
STAGE["cur"] = "other"
model = Titration(params)
t_run, c_run = time.perf_counter() - t_start, cpu_s() - c_start
STAGE["cur"] = "post"
rows = []
for site in model:
    rows.append({"chain": site.molecule.chain, "resnum": site.getResNumber(),
                 "group": {"NTR": "NTERM", "CTR": "CTERM"}.get(site.res_name, site.res_name), "pka": site.getpK()})
STOP.set(); time.sleep(0.1)
saved = {(r["chain"], r["resnum"], r["group"]): r["pka"] for r in json.load(open(SRC / "result.json"))["rows"]}
d = []; miss = 0
for r in rows:
    s = saved.get((r["chain"], r["resnum"], r["group"]), "absent")
    if s == "absent" or (s is None) != (r["pka"] is None):
        miss += 1
    elif s is not None:
        d.append(abs(s - r["pka"]))
mce_equal = open("mc-energies.json").read() == open(SRC / "mc-energies.json").read()
res = dict(src=str(SRC), ncpus=A.ncpus, cores=CORES, worker_pins=sorted(PINS), host=os.uname().nodename,
           wall_total_incl_import=time.perf_counter() - T0, wall_titration=t_run, cpu_titration=c_run,
           stages=TIM, peak_tree_rss_bytes={k: v for k, v in PEAK.items()}, peak_tree_rss_max=max(v for k, v in PEAK.items()
                                                                                                  if not k.startswith("_")),
           main_ru_maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           children_ru_maxrss_kb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
           n_sites=len(rows), mc_energies_bitwise_vs_campaign=mce_equal,
           pka_vs_campaign=dict(n=len(d), mae=(sum(d) / len(d)) if d else None, max=max(d) if d else None, mismatch=miss),
           rows=rows)
json.dump(res, open(A.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
