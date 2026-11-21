"""jaxpb.prep_pypka -- CPU producer: run PyPKA's preprocessing for one state and write a StateRecord (records.py).

Runs in the PyPKA 2.10 env on Coulson (python3.10 + numpy; no jax), exactly as the pkabench campaign worker
(same params, ncpus=1), with in-process monkeypatches only (the env is never edited):
  - startPoolProcesses runs the ncpus=1 job list serially in-process (no numerical effect);
  - DelPhi4py.runDelPhi is wrapped (pbrec.py:90 hook of jaxpb-pilot-v1).

modes
  record  DelPhi runs normally; every call's inputs are checked for the per-site invariants the GPU path relies on,
          outputs + log statistics (spectral radius, sweeps, max change, boundary points, estimated iterations) are
          stored as the golden reference (StateRecord.ref), PyPKA continues through assembly + MC and the run is gated
          against the saved campaign mc-energies.json / result.json.
  prep    DelPhi is NOT called (zeros returned); the run stops after the PB phase. Only inputs are stored.

usage: python -m jaxpb.prep_pypka <record|prep> <campaign_state_dir> <work_dir> <out.npz> --state-id <cid_S> [--role r]
       (<work_dir>/input.pdb must exist: regenerated from the request CIF by jobs/m0_inputpdb.py)
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from jaxpb import records as RC

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["record", "prep"])
ap.add_argument("src"); ap.add_argument("work"); ap.add_argument("out")
ap.add_argument("--state-id", required=True); ap.add_argument("--role", default=""); ap.add_argument("--split", default="")
ap.add_argument("--nlogtext", type=int, default=2, help="full DelPhi log texts kept for the first n tautomers")
A = ap.parse_args()
MODE = A.mode
SRC = Path(A.src).resolve(); WORK = Path(A.work).resolve(); WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
T0 = time.perf_counter()

import pypka.concurrency as conc  # noqa: E402
import pypka.main as pmain  # noqa: E402
import pypka.mc.run_mc as prunmc  # noqa: E402
from pypka.config import Config  # noqa: E402
from delphi4py import DelPhi4py  # noqa: E402


class _V:
    def __init__(self, v): self.value = v


def serialPool(targetFunction, iterable_job_arguments_list, ncpus, assign="distributed", merged_results=False):
    assert ncpus == 1, ncpus
    n = len(list(iterable_job_arguments_list))
    Config.parallel_params.total_jobs = n
    Config.parallel_params.pb_time = []
    Config.parallel_params.njobs = _V(n)
    results = [targetFunction(job) for job in [list(range(n))]]
    out = []
    for r in results:
        if merged_results: out += r
        else: out.append(r)
    return out


pmain.startPoolProcesses = serialPool
prunmc.startPoolProcesses = serialPool

CTX = {"tau": None, "k": 0}
_calcPotential = conc.calcPotential


def calcPotential(taut):
    CTX["tau"] = Config.parallel_params.all_tautomers_order[taut]; CTX["k"] = 0
    return _calcPotential(taut)


conc.calcPotential = calcPotential

LOGKEYS = {"specrad": r"gauss-seidel spectral radius is\s+(\S+)", "loops": r"# loops\s*:\s*(\S+)",
           "maxchange": r"mean,max change \(kT/e\)\s*:\s*\S+\s+(\S+)", "nbound": r"number of dielectric boundary points\s+(\S+)",
           "est_iter": r"estimated iterations to convergence\s+(\S+)", "corr_rf": r"corrected reaction field energy:\s*(\S+)"}


def parse_log(txt):
    out = {}
    for k, pat in LOGKEYS.items():
        m = re.findall(pat, txt)
        try:
            out[k] = float(m[-1].replace("D", "E")) if m else np.nan
        except ValueError:
            out[k] = np.nan
    return out


def _arr(c, n, w=None):
    a = np.ctypeslib.as_array(c)
    a = a.reshape(-1, w)[:n] if w else a[:n]
    return np.array(a, dtype=np.float32)


CALLS = {}      # id(tautomer) -> {k: dict}
BASE = {}       # first whole-molecule pos/rad (all M calls must equal it)
CHK = {"pos_maxdiff": 0.0, "rad_maxdiff": 0.0, "site_inv_maxdiff": 0.0, "mq_outside_site": 0}
LOGTEXT = []
_orig = DelPhi4py.runDelPhi


def patched(self, in_frc="", out_phi=False, filename=None, outputfile="/dev/null", **kw):
    tau, k = CTX["tau"], CTX["k"]; CTX["k"] += 1
    n = self.natoms
    pos = _arr(self.p_atpos, n, 3); rad = _arr(self.p_rad3, n); chg = _arr(self.p_chrgv4, n)
    r = dict(k=k)
    if k < 2:
        if not BASE:
            BASE.update(pos=pos.copy(), rad=rad.copy())
        CHK["pos_maxdiff"] = max(CHK["pos_maxdiff"], float(np.max(np.abs(pos - BASE["pos"]))))
        CHK["rad_maxdiff"] = max(CHK["rad_maxdiff"], float(np.max(np.abs(rad - BASE["rad"]))))
        nz = np.nonzero(chg)[0]
        r["mq_idx"] = nz; r["mq_val"] = chg[nz]
    else:
        r["pos"] = pos; r["rad"] = rad; r["chg"] = chg
    if MODE == "prep":
        for key, value in kw.items():
            self[key] = value
        open(outputfile, "a").close()
        self.esolvation = 0.0; self.p_sitpot_list = [0.0] * n
        r.update(esolv=0.0, wall=0.0, log={})
    else:
        t = time.perf_counter()
        _orig(self, in_frc=in_frc, out_phi=out_phi, filename=filename, outputfile=outputfile, **kw)
        r["wall"] = time.perf_counter() - t
        r["esolv"] = float(self.esolvation)
        if k == 1:
            r["sitpot"] = np.array(self.p_sitpot_list, dtype=np.float32)
        r["log"] = {}
        if outputfile != "/dev/null" and os.path.exists(outputfile):
            txt = open(outputfile, errors="replace").read()
            r["log"] = parse_log(txt)
            if len(LOGTEXT) < 3 * A.nlogtext:
                LOGTEXT.append(json.dumps({"tau": tau.name, "site": int(tau.site.res_number), "k": k, "log": txt}))
    r.update(acent=np.array(self.acent, dtype=np.float64), scale=float(self.scale), igrid=int(self.igrid),
             ibctyp=int(self.ibctyp), epsin=float(self.epsin), epsout=float(self.epsout), conc=float(self.conc),
             radprb=float(self.radprb), res2=float(self.res2), nlit=int(self.nlit))
    assert r["acent"].shape == (3,), r["acent"]
    CALLS.setdefault(id(tau), {})[k] = r


DelPhi4py.runDelPhi = patched


class StopAfterPB(Exception):
    pass


TIM = {}
_launch = pmain.Titration.DelPhiLaunch


def DelPhiLaunch(self):
    CTX["model"] = self
    t = time.perf_counter(); _launch(self); TIM["pb_phase"] = time.perf_counter() - t
    if MODE == "prep":
        raise StopAfterPB()


pmain.Titration.DelPhiLaunch = DelPhiLaunch

# --------------------------------------------------------------------------------- the run
from pypka import Titration  # noqa: E402

request = json.loads((SRC / "request.json").read_text())
params = dict(request["config"])
params.update(structure=str(WORK / "input.pdb"), ncpus=1, pH="-2,16", pHstep=.25,
              save_mc_energies="mc-energies.json", output="midpoints.txt", titration_output="curves.txt")
rows = None
model = None
try:
    model = Titration(params)
except StopAfterPB:
    pass
TIM["total"] = time.perf_counter() - T0

if MODE == "record":
    rows = []
    for site in model:
        curve = site.getTitrationCurve()
        rows.append({"chain": site.molecule.chain, "resnum": site.getResNumber(),
                     "group": {"NTR": "NTERM", "CTR": "CTERM"}.get(site.res_name, site.res_name),
                     "pka": site.getpK(), "curve": [float(curve[round(-2 + i * .25, 2)]) for i in range(73)]})
    json.dump({"rows": rows}, open("result.json", "w"))


def gate():
    new = json.load(open("mc-energies.json")); old = json.load(open(SRC / "mc-energies.json"))
    g = {"keys_equal": sorted(new) == sorted(old), "all_sites_equal": new["all_sites"] == old["all_sites"],
         "bitwise_equal": open("mc-energies.json").read() == open(SRC / "mc-energies.json").read()}

    def flat(x):
        return np.array([np.nan if v is None else v for row in x for v in row], float)
    for key in ("possible_states_g", "interactions"):
        fa, fb = flat(new[key]), flat(old[key])
        if fa.shape != fb.shape:
            g[key + "_shape_equal"] = False; continue
        m = np.isfinite(fa) & np.isfinite(fb) & (np.abs(fa) < 1e5) & (np.abs(fb) < 1e5)
        g[key + "_maxabs"] = float(np.max(np.abs(fa[m] - fb[m]))) if m.any() else 0.0
    saved = {(r["chain"], r["resnum"], r["group"]): r for r in json.load(open(SRC / "result.json"))["rows"]}
    d = []; nmiss = 0
    for r in rows:
        s = saved.get((r["chain"], r["resnum"], r["group"]))
        if s is None or (r["pka"] is None) != (s["pka"] is None):
            nmiss += 1; continue
        if r["pka"] is not None:
            d.append(abs(r["pka"] - s["pka"]))
    g.update(pka_maxabs=float(max(d)) if d else None, pka_n=len(d), pka_mismatch=nmiss, nsites=len(rows))
    return g


# --------------------------------------------------------------------------------- pack the StateRecord
dm = Config.delphi_params["delphimol"]
N = dm.natoms
pos = _arr(Config.delphi_params.p_atpos, N, 3); rad = _arr(Config.delphi_params.p_rad3, N)
q_bg = _arr(Config.delphi_params.p_chrgv4, N)
lookup = Config.delphi_params.lookup_atoms
atom_id = np.array([lookup[i][0] if i in lookup else -1 for i in range(N)], np.int64)
assert CHK["pos_maxdiff"] == 0.0 and CHK["rad_maxdiff"] == 0.0, CHK
delphi_order = {id(t): i for i, t in enumerate(Config.parallel_params.all_tautomers_order)}
if MODE == "record":
    sites = list(Config.parallel_params.all_sites)
else:
    sites = list(CTX["model"].get_all_sites(get_list=True))   # the order calcSiteInteractionsParallel would use
taus = []; site_tau_off = [0]
for s in sites:
    taus += list(s.iterOrderedTautomersWithoutRef()) + [s.ref_tautomer]
    site_tau_off.append(len(taus))
T = len(taus)
assert len(CALLS) == T and all(len(CALLS[id(t)]) == 3 for t in taus), (len(CALLS), T)
id2pos = {}
for i in range(N):
    if atom_id[i] >= 0:
        id2pos.setdefault(int(atom_id[i]), []).append(i)
site_atoms = []; site_atoms_off = [0]; model_pos = []; model_rad = []; model_off = [0]
acM = []; acS = []
tau_q = []; tau_q_off = [0]; tau_mq = []; tau_mq_off = [0]
inv = 0.0
for si, s in enumerate(sites):
    ids = set(s.getAtomNumbersList())
    idx = np.array(sorted(i for i in range(N) if atom_id[i] in ids), np.int64)
    site_atoms.append(idx); site_atoms_off.append(site_atoms_off[-1] + len(idx))
    st = taus[site_tau_off[si]:site_tau_off[si + 1]]
    c0 = CALLS[id(st[0])]
    acM.append(c0[0]["acent"]); acS.append(c0[2]["acent"])
    model_pos.append(c0[2]["pos"]); model_rad.append(c0[2]["rad"]); model_off.append(model_off[-1] + len(c0[2]["rad"]))
    pin = {int(a): j for j, a in enumerate(idx)}
    for t in st:
        c = CALLS[id(t)]
        # per-site invariants: grid centres, model atoms identical across the site's tautomers; M charges on site atoms
        inv = max(inv, float(np.max(np.abs(c[0]["acent"] - acM[-1]))), float(np.max(np.abs(c[1]["acent"] - acM[-1]))),
                  float(np.max(np.abs(c[2]["acent"] - acS[-1]))))
        assert np.array_equal(c[2]["pos"], model_pos[-1]) and np.array_equal(c[2]["rad"], model_rad[-1])
        assert np.array_equal(c[0]["mq_idx"], c[1]["mq_idx"]) and np.array_equal(c[0]["mq_val"], c[1]["mq_val"])
        q = np.zeros(len(idx), np.float32)
        for a, v in zip(c[1]["mq_idx"], c[1]["mq_val"]):
            if int(a) in pin:
                q[pin[int(a)]] = v
            else:
                CHK["mq_outside_site"] += 1
        tau_q.append(q); tau_q_off.append(tau_q_off[-1] + len(q))
        tau_mq.append(c[2]["chg"]); tau_mq_off.append(tau_mq_off[-1] + len(c[2]["chg"]))
CHK["site_inv_maxdiff"] = inv
assert CHK["mq_outside_site"] == 0, CHK
c00 = CALLS[id(taus[0])]
pp = Config.pypka_params
params_out = dict(epsin=c00[0]["epsin"], epsout=c00[0]["epsout"], conc=c00[0]["conc"], gsize=c00[0]["igrid"],
                  scale_c=c00[0]["scale"], scale_f=c00[1]["scale"], radprb=c00[0]["radprb"], res2=c00[0]["res2"],
                  nlit=c00[0]["nlit"], relpar_c=RC.RELPAR_COARSE, temp_pypka=float(pp["temp"]), kboltz=RC.KBOLTZ,
                  cutoff_back=float(pp["cutoff"]), couple_min=float(Config.mc_params["couple_min"]))
for k in range(3):
    for key in ("scale", "igrid", "epsin", "epsout", "conc", "radprb", "res2", "nlit"):
        vals = {CALLS[id(t)][k][key] for t in taus}
        assert len(vals) == 1, (k, key, vals)
typ = np.array([s.type for s in sites])
rec = RC.StateRecord(
    state_id=A.state_id, cid=A.state_id.rsplit("_", 1)[0], state=A.state_id.rsplit("_", 1)[1], role=A.role, split=A.split,
    source=str(SRC), producer=MODE, config_json=json.dumps(request["config"], sort_keys=True),
    pos=pos, rad=rad, q_bg=q_bg, atom_id=atom_id,
    site_chain=np.array([s.molecule.chain for s in sites]), site_resnum=np.array([s.res_number for s in sites], np.int64),
    site_resname=np.array([s.res_name for s in sites]), site_type=typ,
    site_chargediff=np.where(typ == "a", -1, 1).astype(np.int8),
    site_refprot=np.array([s.getRefProtState() for s in sites], np.int8),
    site_center_h=np.array([np.asarray(s.getCenterH(), np.float64) for s in sites]),
    site_acent_M=np.array(acM), site_acent_S=np.array(acS),
    site_atoms=np.concatenate(site_atoms), site_atoms_off=np.array(site_atoms_off, np.int64),
    site_tau_off=np.array(site_tau_off, np.int64),
    model_pos=np.concatenate(model_pos), model_rad=np.concatenate(model_rad), model_off=np.array(model_off, np.int64),
    tau_name=np.array([t.name for t in taus]), tau_site=np.repeat(np.arange(len(sites)), np.diff(site_tau_off)),
    tau_isref=np.array([t.isRefTautomer() for t in taus]),
    tau_pkmod=np.array([np.nan if t.isRefTautomer() else float(getattr(t, "pKmod", np.nan)) for t in taus]),
    tau_occ=np.array([(0 if s.type == "c" else 1) if not t.isRefTautomer() else (1 if s.type == "c" else 0)
                      for s in sites for t in list(s.iterOrderedTautomersWithoutRef()) + [s.ref_tautomer]], np.int8),
    tau_delphi=np.array([delphi_order[id(t)] for t in taus], np.int64),
    tau_q=np.concatenate(tau_q), tau_q_off=np.array(tau_q_off, np.int64),
    tau_mq=np.concatenate(tau_mq), tau_mq_off=np.array(tau_mq_off, np.int64),
    params=params_out,
)
rec.validate()
if MODE == "record":
    G = gate()
    G.update(CHK); G["timing"] = TIM
    G["pb_wall_sum"] = float(sum(CALLS[id(t)][k]["wall"] for t in taus for k in range(3)))
    sidx = {id(s): i for i, s in enumerate(sites)}
    pairs = np.array([[sidx[id(a)], sidx[id(b)]] for a, b in Config.parallel_params.site_interactions], np.int64).reshape(-1, 2)

    def lg(key):
        return np.array([[CALLS[id(t)][k]["log"].get(key, np.nan) for t in taus] for k in range(3)], np.float64)

    def attr(name, default=np.nan):
        return np.array([float(getattr(t, name, default)) if not t.isRefTautomer() else np.nan for t in taus])

    rec.ref = dict(
        esolvM=np.array([CALLS[id(t)][1]["esolv"] for t in taus]), esolvS=np.array([CALLS[id(t)][2]["esolv"] for t in taus]),
        esolvM0=np.array([CALLS[id(t)][0]["esolv"] for t in taus]),
        sitpotM=np.stack([CALLS[id(t)][1]["sitpot"] for t in taus]).astype(np.float32),
        loops=lg("loops"), specrad=lg("specrad"), maxchange=lg("maxchange"), nbound=lg("nbound"), est_iter=lg("est_iter"),
        wall=np.array([[CALLS[id(t)][k]["wall"] for t in taus] for k in range(3)]),
        pkint=attr("pKint"), eback=np.array([float(t.e_back) for t in taus]), dg=attr("dg"),
        dGsolvM=attr("dG_solvationM"), dGsolvS=attr("dG_solvationS"), dGback=attr("dG_back"),
        mc_energies=open("mc-energies.json").read(), result=open("result.json").read(), gate=json.dumps(G), pairs=pairs,
    )
    extra = {"logtext": np.array(LOGTEXT)}
else:
    G = dict(CHK); G["timing"] = TIM
    extra = None
sha = RC.save_states(A.out, [rec], extra=extra)
G["out_sha256"] = sha
json.dump(G, open("gate.json", "w"), indent=1, default=str)
print(json.dumps(G, indent=1, default=str))
