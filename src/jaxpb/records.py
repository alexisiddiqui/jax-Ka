"""jaxpb.records -- the data interface of the jaxpb pipeline (owned by the PB track; stable from M0).

Pure numpy + dataclasses: importable in the PyPKA producer env on Coulson (no jax), on the Mac and on Isambard.
Every other module takes its static shapes from the tables in this file; nothing else may invent a shape.

Precision: process-wide float64 may be ON (api.py enables jax_enable_x64 at import; assembly accumulates e_back/pKint/g
in float64). The PB path (geometry/maps/pb/sor) is precision-explicit: it runs its float32 programs inside an x64-off
context and its float64 probe generation inside an x64-on context, so its results do not depend on the global flag.
Producer content (energy-track requests 1-3): tau_pkmod is filled for every non-reference tautomer in both producer
modes (tautomer.pKmod); params['cutoff_back'] = PyPKA's pypka_params['cutoff'] (-1 = all atoms); M0 golden shards keep
ref['mc_energies'], ref['pkint'], ref['eback'] (+ dg, dGsolvM/S, dGback, pairs).

Units: lengths A; charges e; potentials kT/e at DelPhi's temperature (EPKT = 561.99 kT*A/e^2, T = 297.33 K);
solvation energies kT (DelPhi); PyPKA free energies (dg, mc-energies) in kT at PyPKA's temp (298.15 K), as PyPKA writes them.
Index conventions: atom indices are PyPKA/DelPhi atom positions 0..Na-1 (delphimol order); tautomer arrays are in
**MC order**: for each site in PyPKA's `all_sites` order, the non-reference tautomers in `iterOrderedTautomersWithoutRef()`
order, then the reference tautomer last (= the column order of mc-energies.json `possible_states_g` / `interactions`).
Ragged per-site / per-tautomer lists are stored flat with `*_off` offset arrays of length n+1 (CSR).

==================================================================================================================
1. StateRecord (host numpy, one per protonation-state problem = one PyPKA run; producer output, PB input)
==================================================================================================================
meta                    str      state_id "<cid>_<S>", cid, state ("AB"/"A"/"B"), role ("general"/"antibody_antigen"),
                                 split ("val"/...), source (campaign dir), producer ("record"|"prep"), config_json
atoms (Na = all DelPhi atoms, incl. radius-0 H)
  pos        [Na,3]  f32  A    DelPhi atom coordinates (delphimol.p_atpos)
  rad        [Na]    f32  A    DelPhi radii (p_rad3); 0 for most H (excluded from the surface)
  q_bg       [Na]    f32  e    delphimol.p_chrgv4 at the end of the PB phase (PyPKA's background charges: the
                               charges used by calcBackEnergy / calcInteractionWith as "charge_ref")
  atom_id    [Na]    i64       PyPKA atom number (lookup_atoms), -1 if none
sites (S, MC order = PyPKA Config.parallel_params.all_sites)
  site_chain [S] U, site_resnum [S] i64, site_resname [S] U (PyPKA res_name, e.g. NTR/CTR/LYS)
  site_type  [S] U1   'a' anionic / 'c' cationic;  site_chargediff [S] i8 = -1 ('a') / +1 ('c') (calcpKint)
  site_refprot [S] i8 = getRefProtState(): +1 ('c'), -1 ('a') (calcInteractionWith sign)
  site_center_h [S,3] f64 A    site.getCenterH() (calcBackEnergy cutoff centre)
  site_acent_M [S,3] f64 A     grid centre of the whole-molecule calls (coarse + focused); equal for all tautomers
  site_acent_S [S,3] f64 A     grid centre of the model-compound call
  site_atoms [sum] i64, site_atoms_off [S+1]   atom indices of the site (site.getAtomNumbersList(), incl. all protons)
  site_tau_off [S+1]          tautomers of site s are rows site_tau_off[s]:site_tau_off[s+1] (ref last)
  model atoms (model-compound DelPhi call; same atoms for all tautomers of a site, checked by the producer):
  model_pos [sum,3] f32, model_rad [sum] f32, model_off [S+1]
tautomers (T = sum of npossible_states, MC order)
  tau_name [T] U, tau_site [T] i64 (site index), tau_isref [T] bool
  tau_pkmod [T] f64 pK      tautomer.pKmod (ref: nan)
  tau_occ   [T] i8          possible_states_occ (protonated = 1)
  tau_delphi [T] i64        index in PyPKA's all_tautomers_order (DelPhi call order)
  tau_q [sum] f32 e, tau_q_off [T+1]   charges of the tautomer on its site's atoms, aligned with site_atoms of
                                       its site (= the whole-molecule DelPhi call charges; all other atoms 0)
  tau_mq [sum] f32 e, tau_mq_off [T+1] charges of the model-compound call, aligned with model_pos of its site
params (scalars, from the recorded DelPhi calls / PyPKA config)
  epsin 15, epsout 80, conc 0.1 M, gsize 81, scale_c 1, scale_f 4, radprb 1.4, res2 0.01, nlit 500,
  relpar_c 0.75, temp_pypka 298.15 K, kboltz 5.98435e-6, cutoff_back (PyPKA cutoff, nm; -1 = all), couple_min 2.0
golden reference (StateRecord.ref, only for producer="record"; dict of arrays, all indexed by MC-order tau)
  esolvM [T] f64 kT, esolvS [T] f64 kT, esolvM0 [T] f64 kT (coarse call, unused downstream)
  sitpotM [T,Na] f32 kT/e     focused site potentials filled with coarse values outside the fine box (PyPKA sitpotM)
  loops [3,T] f64 (coarse, focused, model SOR sweeps from the DelPhi log), specrad [3,T] f64 (logged Gauss-Seidel
  spectral radius), maxchange [3,T], nbound [3,T], est_iter [3,T], wall [3,T] s
  pkint [T], eback [T], dg [T], dGsolvM [T], dGsolvS [T], dGback [T] (PyPKA tautomer attributes; ref rows nan/0)
  mc_energies (json str: the saved mc-energies.json), result (json str), gate (json str)
  pairs [P,2] i64             site pairs PyPKA computed interactions for (site_interactions, site indices)

==================================================================================================================
2. PB outputs (PBOut, host numpy, per state; input of assemble.py)
==================================================================================================================
  esolvM  [T] f32 kT    focused whole-molecule corrected reaction-field energy (DelPhi convention, no rescale)
  esolvS  [T] f32 kT    model-compound reaction-field energy
  esolvM0 [T] f32 kT    coarse reaction-field energy (diagnostic)
  sitpotM [T,Na] f32 kT/e  potential at every atom: focused value inside the fine grid, coarse value elsewhere
                           (0 outside the coarse grid), exactly as PyPKA's sitpotM
  sweeps  [T,3] i32     SOR sweeps (coarse, focused, model);  maxchange [T,3] f32 kT/e at the stop check
  rho     [T,2] f32     Gauss-Seidel spectral radius used for the focused / model SOR (logged or estimated)
  flags   dict[str,int] every overflow / retry counter (0 = clean); pb_mode, backend, cfg_hash str

==================================================================================================================
3. Size classes and static shapes (the ONLY source of static shapes)
==================================================================================================================
State context (class c = 0..4):  NA_C[c] = 2048 * 2**c  surface atoms (radius > 0), padded at FAR with radius 0
                                 NP_C[c] = 192 * NA_C[c] whole-molecule probe centres, padded at FAR
Site batch:  B sites (fixed), T_MAX tautomers per site, QM charged atoms per tautomer (whole-molecule call),
             MA model-compound atoms, MP_MODEL model probes, FINE_PROBES probes gathered per site for the fine grid
Lanes:       SOR lane widths LANE_WIDTHS (lanes compacted to the smallest width that holds the live lanes)
Assembly (energy track):   MC_C[c] = NATOM_ALL_C[c] // 8 microstates, NC_C = MC_C // 2 sites, PC_C = MC_C // 4 pairs
Titration (energy track):  MP_BUCKETS microstate buckets; NP = Mp // 2 sites; PP = Mp // 16 pairs; S_FLUSH states
Overflow:    every static size has a counter; any counter > size sets a flag and the caller re-runs in the XL variant
             (XL_* sizes) or the next class; never silent.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ------------------------------------------------------------------------------------------------ physical constants
EPKT = 0.702489257 * 800.0            # kT*A/e^2 at DelPhi's 297.33 K (DelPhi coulombic BC, = 561.99)
DEBYE_I1 = 9.70593166 * np.sqrt(0.1)  # DelPhi Debye length (A) * sqrt(I/M)
EXRAD = 2.0                           # ion-exclusion radius (A)
RADPRB = 1.4                          # probe radius (A)
KBOLTZ = 5.98435e-6                   # PyPKA constants.py (e^2/(A K))
LOG10 = 2.302585092994046
FAR = 1.0e4                           # padding coordinate (A)

# ------------------------------------------------------------------------------------------------ DelPhi run parameters
GSIZE = 81
SCALE_COARSE = 1.0
SCALE_FINE = 4.0
RELPAR_COARSE = 0.75                  # fixed Gauss-Seidel spectral radius of the coarse run (PyPKA config relfac/relpar)
RES2 = 0.01                           # max-change stop (kT/e)
NLIT = 500                            # sweep cap
CHECK = 10                            # stop test every CHECK sweeps
NVERT = 1500                          # Fibonacci vertices per SAS sphere (probe centres)
ARC_STEP = 0.03                       # A between probe centres on SAS circles

# ------------------------------------------------------------------------------------------------ size classes
NA_C = tuple(2048 * 2 ** c for c in range(5))          # surface atoms per class
NP_PER_ATOM = 192
NP_C = tuple(NP_PER_ATOM * n for n in NA_C)            # whole-molecule probe centres per class
NATOM_ALL_C = tuple(2 * n for n in NA_C)               # all atoms (incl. radius-0 H) per class (site potentials)

B = 16            # sites per batch
T_MAX = 8         # tautomers per site (incl. reference)
QM = 24           # charged atoms per tautomer in the whole-molecule call
MA = 32           # model-compound atoms per site
MP_MODEL = 8192   # model-compound probe centres per site
FINE_PROBES = 131072   # probes gathered per site for the fine-grid probe region

# cell-table slots and compaction sizes (raised margins vs the prototypes: Ka 16 -> 24, Kp 576 -> 768)
KA = 24
KP = 768
# per grid kind: NS shell midpoints (per axis), NB boundary points; v2 values (verified on the 5 pilot states)
SZ = {"c": dict(NS=120000, NB=48000), "f": dict(NS=330000, NB=40000), "s": dict(NS=32000, NB=4000)}
# XL variants (automatic retry after an overflow flag)
XL_KA = 48
XL_KP = 1536
XL_SZ = {"c": dict(NS=240000, NB=96000), "f": dict(NS=531441 // 1 // 1, NB=80000), "s": dict(NS=64000, NB=8000)}
XL_FINE_PROBES = 262144
XL_MP_MODEL = 16384

# probe-centre construction (geometry_state): pairs per class, burial candidates
PAIRS_PER_ATOM = 64                   # overlapping SAS-sphere pairs (i<j) per surface atom
GEO_CHUNK = 1 << 17                   # probe candidates per geometry chunk
GEO_K = 16                            # atoms per hashed cell (burial candidates; overflow -> XL_GEO_K retry, same class). SPEED lever 2: 32 -> 16 (max measured occupancy 10)
XL_GEO_K = 64

LANE_WIDTHS = (16, 64, 128, 256)           # GPU SOR lane-pool widths
CPU_LANE_WIDTHS = (4, 8, 16)             # CPU working-set tiles; independent lanes

# assembly microstate class (energy track, request 6): per state-size class c
MC_C = tuple(n // 8 for n in NATOM_ALL_C)      # microstates (= tautomers incl. references) per state
NC_C = tuple(m // 2 for m in MC_C)             # sites
PC_C = tuple(m // 4 for m in MC_C)             # coupled pairs

# titration buckets (energy track)
MP_BUCKETS = (64, 128, 256, 512, 1024, 2048)
S_FLUSH = 32
NPH = 73                              # pH grid -2..16 step 0.25


def np_bucket(mp: int) -> int:
    """sites per titration bucket: n_sites <= M/2 <= Mp/2 always (every site has >= 2 states), so Mp//2 never overflows
    (energy-track request 4; was Mp//4)."""
    return mp // 2


def pp_bucket(mp: int) -> int:
    """coupled pairs per titration bucket (tight: Mp 256 -> max 14 of 16 on the 426 v2 states; titrate promotes on overflow)."""
    return mp // 16


def mp_bucket(m: int) -> int:
    for b in MP_BUCKETS:
        if m <= b:
            return b
    raise ValueError(f"{m} microstates exceed the largest Mp bucket {MP_BUCKETS[-1]}")


def size_class(n_surface_atoms: int, n_all_atoms: Optional[int] = None) -> int:
    """Smallest class c with n_surface_atoms <= NA_C[c] and n_all_atoms <= NATOM_ALL_C[c]."""
    for c, n in enumerate(NA_C):
        if n_surface_atoms <= n and (n_all_atoms is None or n_all_atoms <= NATOM_ALL_C[c]):
            return c
    raise ValueError(f"state with {n_surface_atoms} surface atoms exceeds class {len(NA_C) - 1}")


# ------------------------------------------------------------------------------------------------ dataclasses
REF_KEYS = ("esolvM", "esolvS", "esolvM0", "sitpotM", "loops", "specrad", "maxchange", "nbound", "est_iter", "wall",
            "pkint", "eback", "dg", "dGsolvM", "dGsolvS", "dGback", "mc_energies", "result", "gate", "pairs")

ARRAY_FIELDS = ("pos", "rad", "q_bg", "atom_id",
                "site_chain", "site_resnum", "site_resname", "site_type", "site_chargediff", "site_refprot",
                "site_center_h", "site_acent_M", "site_acent_S", "site_atoms", "site_atoms_off", "site_tau_off",
                "model_pos", "model_rad", "model_off",
                "tau_name", "tau_site", "tau_isref", "tau_pkmod", "tau_occ", "tau_delphi",
                "tau_q", "tau_q_off", "tau_mq", "tau_mq_off")
META_FIELDS = ("state_id", "cid", "state", "role", "split", "source", "producer", "config_json")
PARAM_KEYS = ("epsin", "epsout", "conc", "gsize", "scale_c", "scale_f", "radprb", "res2", "nlit", "relpar_c",
              "temp_pypka", "kboltz", "cutoff_back", "couple_min")


@dataclass
class StateRecord:
    """One protonation-state problem. See the module docstring for every field's shape, dtype and units."""
    state_id: str
    cid: str
    state: str
    role: str
    split: str
    source: str
    producer: str
    config_json: str
    pos: np.ndarray
    rad: np.ndarray
    q_bg: np.ndarray
    atom_id: np.ndarray
    site_chain: np.ndarray
    site_resnum: np.ndarray
    site_resname: np.ndarray
    site_type: np.ndarray
    site_chargediff: np.ndarray
    site_refprot: np.ndarray
    site_center_h: np.ndarray
    site_acent_M: np.ndarray
    site_acent_S: np.ndarray
    site_atoms: np.ndarray
    site_atoms_off: np.ndarray
    site_tau_off: np.ndarray
    model_pos: np.ndarray
    model_rad: np.ndarray
    model_off: np.ndarray
    tau_name: np.ndarray
    tau_site: np.ndarray
    tau_isref: np.ndarray
    tau_pkmod: np.ndarray
    tau_occ: np.ndarray
    tau_delphi: np.ndarray
    tau_q: np.ndarray
    tau_q_off: np.ndarray
    tau_mq: np.ndarray
    tau_mq_off: np.ndarray
    params: Dict[str, float] = field(default_factory=dict)
    ref: Dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------- sizes
    @property
    def n_atoms(self) -> int:
        return int(len(self.pos))

    @property
    def n_surface_atoms(self) -> int:
        return int(np.sum(self.rad > 0))

    @property
    def n_sites(self) -> int:
        return int(len(self.site_type))

    @property
    def n_taus(self) -> int:
        return int(len(self.tau_name))

    @property
    def size_class(self) -> int:
        return size_class(self.n_surface_atoms, self.n_atoms)

    def site_atom_idx(self, s: int) -> np.ndarray:
        return self.site_atoms[self.site_atoms_off[s]:self.site_atoms_off[s + 1]]

    def site_taus(self, s: int) -> np.ndarray:
        return np.arange(self.site_tau_off[s], self.site_tau_off[s + 1])

    def tau_charges(self, t: int) -> np.ndarray:
        return self.tau_q[self.tau_q_off[t]:self.tau_q_off[t + 1]]

    def model_atoms(self, s: int):
        a, b = self.model_off[s], self.model_off[s + 1]
        return self.model_pos[a:b], self.model_rad[a:b]

    def tau_model_charges(self, t: int) -> np.ndarray:
        return self.tau_mq[self.tau_mq_off[t]:self.tau_mq_off[t + 1]]

    def validate(self) -> None:
        S, T, Na = self.n_sites, self.n_taus, self.n_atoms
        assert self.pos.shape == (Na, 3) and self.rad.shape == (Na,) and self.q_bg.shape == (Na,)
        for k in ("site_atoms_off", "site_tau_off", "model_off"):
            assert getattr(self, k).shape == (S + 1,), k
        for k in ("tau_q_off", "tau_mq_off"):
            assert getattr(self, k).shape == (T + 1,), k
        assert self.site_tau_off[-1] == T
        for s in range(S):
            taus = self.site_taus(s)
            assert np.all(self.tau_site[taus] == s) and self.tau_isref[taus[-1]] and not self.tau_isref[taus[:-1]].any()
            na = self.site_atoms_off[s + 1] - self.site_atoms_off[s]
            nm = self.model_off[s + 1] - self.model_off[s]
            for t in taus:
                assert self.tau_q_off[t + 1] - self.tau_q_off[t] == na
                assert self.tau_mq_off[t + 1] - self.tau_mq_off[t] == nm


@dataclass
class PBOut:
    """PB outputs of one state (module docstring section 2)."""
    esolvM: np.ndarray
    esolvS: np.ndarray
    esolvM0: np.ndarray
    sitpotM: np.ndarray
    sweeps: np.ndarray
    maxchange: np.ndarray
    flags: Dict[str, int] = field(default_factory=dict)
    pb_mode: str = "delphi"
    backend: str = "xla"
    cfg_hash: str = ""
    rho: Optional[np.ndarray] = None   # [T,2] f32 Gauss-Seidel spectral radius used (focused, model)


@dataclass
class Energies:
    """Assembled MC energies of one state (energy track output; PyPKA mc-energies conventions, kT at temp_pypka).
    M = sum_s npossible_states[s] (= T of the StateRecord), MC order.
      g     [M] f32 kT     possible_states_g (reference state of each site = 0)
      occ   [M] i8         possible_states_occ (1 protonated)
      W     [M,M] f32 kT   interactions (symmetric, 0 within a site and for pairs PyPKA skipped)
      start [S] i32, nst [S] i32   first microstate and number of states of each site
      pairs [P,2] i32      coupled site pairs (couple_min rule)
      diag  dict           pKint [M] f64, dG terms, flags"""
    g: np.ndarray
    occ: np.ndarray
    W: np.ndarray
    start: np.ndarray
    nst: np.ndarray
    pairs: np.ndarray
    diag: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnergiesBatch:
    """Up to S_FLUSH states padded into one Mp bucket = the titrate input (energy track request 7; padding inert).
      g     [S,Mp] f32 kT      possible_states_g (padding 0)
      occ   [S,Mp] i8          possible_states_occ
      W     [S,Mp,Mp] f32 kT   interactions
      start [S,Np] i32, nst [S,Np] i32   per-site first microstate / state count (Np = np_bucket(Mp))
      pa, pb [S,Pp] i32        coupled site pairs (Pp = pp_bucket(Mp))
      meta  [S,3] i32          (n_sites, n_pairs, mc_blocks); -1 marks a padding state
      keys  [S,NPH] u32x2      MC PRNG keys per pH point
    (the pH-dependent energies u are formed inside the titration jit from the 73-value pH table)"""
    Mp: int
    g: np.ndarray
    occ: np.ndarray
    W: np.ndarray
    start: np.ndarray
    nst: np.ndarray
    pa: np.ndarray
    pb: np.ndarray
    meta: np.ndarray
    keys: np.ndarray
    state_ids: List[str]


# ------------------------------------------------------------------------------------------------ (de)serialisation
def _ragged_concat(arrs):
    off = np.zeros(len(arrs) + 1, np.int64)
    off[1:] = np.cumsum([len(a) for a in arrs])
    return (np.concatenate(arrs) if len(arrs) else np.zeros(0)), off


def save_states(path: str, recs: List[StateRecord], extra: Optional[Dict[str, np.ndarray]] = None) -> str:
    """Write a shard (one compressed npz) of StateRecords. Per-state arrays are concatenated along axis 0 with
    `<field>__off` offsets; sitpotM (2-D, [T,Na]) is stored flat. Returns the sha256 of the written file."""
    out: Dict[str, np.ndarray] = {"n_states": np.array(len(recs)), "format": np.array("jaxpb.records/1")}
    for k in META_FIELDS:
        out["meta_" + k] = np.array([getattr(r, k) for r in recs])
    for k in ARRAY_FIELDS:
        a, off = _ragged_concat([np.asarray(getattr(r, k)) for r in recs])
        out[k] = a; out[k + "__off"] = off
    out["params_json"] = np.array([json.dumps(r.params, sort_keys=True) for r in recs])
    refk = sorted({k for r in recs for k in r.ref})
    for k in refk:
        vals = [r.ref.get(k) for r in recs]
        if any(v is None for v in vals):
            raise ValueError(f"ref key {k} missing for some states")
        if isinstance(vals[0], str):
            out["ref_" + k] = np.array(vals)
            continue
        arrs = [np.asarray(v) for v in vals]
        if k in ("loops", "specrad", "maxchange", "nbound", "est_iter", "wall"):   # [3,T] -> stored [T,3]
            arrs = [a.T for a in arrs]
        shp = [a.shape for a in arrs]
        arrs = [a.reshape(-1) if a.ndim > 1 and k == "sitpotM" else a for a in arrs]
        a, off = _ragged_concat(arrs)
        out["ref_" + k] = a; out["ref_" + k + "__off"] = off
        if k == "sitpotM":
            out["ref_sitpotM__shape"] = np.array([s for s in shp])
    if extra:
        out.update(extra)
    np.savez_compressed(path, **out)
    return sha256_file(path)


def load_states(path: str, only: Optional[List[str]] = None, with_ref: bool = True) -> List[StateRecord]:
    Z = np.load(path, allow_pickle=False)
    n = int(Z["n_states"])
    ids = [str(x) for x in Z["meta_state_id"]]
    sel = [i for i in range(n) if only is None or ids[i] in only]
    refk = sorted({k[4:].split("__")[0] for k in Z.files if k.startswith("ref_")})
    cache = {}

    def arr(k):
        if k not in cache:
            cache[k] = Z[k]
        return cache[k]

    recs = []
    for i in sel:
        kw = {k: str(Z["meta_" + k][i]) for k in META_FIELDS}
        for k in ARRAY_FIELDS:
            off = arr(k + "__off")
            kw[k] = arr(k)[off[i]:off[i + 1]]
        kw["params"] = json.loads(str(Z["params_json"][i]))
        ref = {}
        if with_ref:
            for k in refk:
                a = arr("ref_" + k)
                if a.dtype.kind == "U":
                    ref[k] = str(a[i]); continue
                off = arr("ref_" + k + "__off")
                v = a[off[i]:off[i + 1]]
                if k == "sitpotM":
                    v = v.reshape(tuple(arr("ref_sitpotM__shape")[i]))
                elif k in ("loops", "specrad", "maxchange", "nbound", "est_iter", "wall"):
                    v = v.T
                ref[k] = v
        kw["ref"] = ref
        recs.append(StateRecord(**kw))
    return recs


def list_states(path: str) -> List[str]:
    Z = np.load(path, allow_pickle=False)
    return [str(x) for x in Z["meta_state_id"]]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_hash(cfg: Any) -> str:
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dev(x, dtype):
    """host -> device without an XLA convert program (cast in numpy first; jax imported lazily)."""
    import jax
    return jax.device_put(np.asarray(x, dtype))


# ------------------------------------------------------------------------------------------------ prototype adapter
def from_pilot_rec(path: str, state_id: str = "", role: str = "", mc_energies: Optional[str] = None) -> StateRecord:
    """Build a StateRecord from a jaxpb-pilot-v1 rec.npz (tautomers in DelPhi order there; reordered to MC order by
    site appearance then tautomer name with the reference last -- PyPKA's iterOrderedTautomersWithoutRef sorts the
    tautomer dict keys, which are the names). pKmod/occ are not in pilot records (nan / derived from type)."""
    R = np.load(path)
    nt = len(R["tau_name"])
    key = [(str(c), int(s)) for c, s in zip(R["tau_chain"], R["tau_site"])]
    sites = list(dict.fromkeys(key))
    order = []
    site_tau_off = [0]
    for s in sites:
        taus = [i for i in range(nt) if key[i] == s]
        nonref = sorted([i for i in taus if not R["tau_isref"][i]], key=lambda i: str(R["tau_name"][i]))
        ref = [i for i in taus if R["tau_isref"][i]]
        assert len(ref) == 1
        order += nonref + ref
        site_tau_off.append(len(order))
    order = np.array(order)
    sa, so = R["site_atoms"], R["site_atoms_off"]
    mq_idx, mq_val, mq_off = R["mq_idx"], R["mq_val"], R["mq_off"]
    s_off = R["s_off"]
    site_atoms = []; site_atoms_off = [0]; model_pos = []; model_rad = []; model_off = [0]
    for si, s in enumerate(sites):
        t0 = order[site_tau_off[si]]
        idx = sa[so[t0]:so[t0 + 1]]
        site_atoms.append(idx); site_atoms_off.append(site_atoms_off[-1] + len(idx))
        model_pos.append(R["s_pos"][s_off[t0]:s_off[t0 + 1]]); model_rad.append(R["s_rad"][s_off[t0]:s_off[t0 + 1]])
        model_off.append(model_off[-1] + s_off[t0 + 1] - s_off[t0])
    tau_q = []; tau_mq = []
    tau_site = np.zeros(nt, np.int64)
    for si in range(len(sites)):
        idx = site_atoms[si]
        pos_in = {a: k for k, a in enumerate(idx)}
        for t in order[site_tau_off[si]:site_tau_off[si + 1]]:
            q = np.zeros(len(idx), np.float32)
            for a, v in zip(mq_idx[mq_off[t]:mq_off[t + 1]], mq_val[mq_off[t]:mq_off[t + 1]]):
                q[pos_in[int(a)]] = v
            tau_q.append(q)
            tau_mq.append(R["s_chg"][s_off[t]:s_off[t + 1]].astype(np.float32))
    for si in range(len(sites)):
        tau_site[site_tau_off[si]:site_tau_off[si + 1]] = si
    typ = np.array([str(R["tau_type"][order[site_tau_off[si]]]) for si in range(len(sites))])
    tq, tqo = _ragged_concat(tau_q); tmq, tmqo = _ragged_concat(tau_mq)
    sa_c, _ = _ragged_concat(site_atoms)
    mp, _ = _ragged_concat(model_pos); mr, _ = _ragged_concat(model_rad)
    isref = R["tau_isref"][order]
    occ = np.where(np.repeat(typ, np.diff(site_tau_off)) == "a", np.where(isref, 0, 1), np.where(isref, 1, 0))
    rec = StateRecord(
        state_id=state_id, cid=state_id.split("_")[0], state=state_id.split("_")[-1], role=role, split="val",
        source=str(path), producer="pilot-v1", config_json="{}",
        pos=R["base_pos"].astype(np.float32), rad=R["base_rad"].astype(np.float32), q_bg=R["base_chg"].astype(np.float32),
        atom_id=R["atom_id"].astype(np.int64),
        site_chain=np.array([s[0] for s in sites]), site_resnum=np.array([s[1] for s in sites], np.int64),
        site_resname=np.array([str(R["tau_name"][order[site_tau_off[si]]])[:2] for si in range(len(sites))]),
        site_type=typ, site_chargediff=np.where(typ == "a", -1, 1).astype(np.int8),
        site_refprot=np.where(typ == "a", -1, 1).astype(np.int8),
        site_center_h=np.full((len(sites), 3), np.nan),
        site_acent_M=R["acentM"][order[np.array(site_tau_off[:-1])]], site_acent_S=R["acentS"][order[np.array(site_tau_off[:-1])]],
        site_atoms=sa_c.astype(np.int64), site_atoms_off=np.array(site_atoms_off, np.int64),
        site_tau_off=np.array(site_tau_off, np.int64),
        model_pos=mp.astype(np.float32), model_rad=mr.astype(np.float32), model_off=np.array(model_off, np.int64),
        tau_name=R["tau_name"][order], tau_site=tau_site, tau_isref=isref, tau_pkmod=np.full(nt, np.nan),
        tau_occ=occ.astype(np.int8), tau_delphi=order.astype(np.int64),
        tau_q=tq.astype(np.float32), tau_q_off=tqo, tau_mq=tmq.astype(np.float32), tau_mq_off=tmqo,
        params=dict(epsin=float(R["par_epsin"][0, 0]), epsout=float(R["par_epsout"][0, 0]), conc=float(R["par_conc"][0, 0]),
                    gsize=int(R["par_igrid"][0, 0]), scale_c=float(R["par_scale"][0, 0]), scale_f=float(R["par_scale"][1, 0]),
                    radprb=float(R["par_radprb"][0, 0]), res2=float(R["par_res2"][0, 0]), nlit=int(R["par_nlit"][0, 0]),
                    relpar_c=RELPAR_COARSE, temp_pypka=298.15, kboltz=KBOLTZ, cutoff_back=float("nan"), couple_min=2.0),
        ref=dict(esolvM=R["esolvM"][order], esolvS=R["esolvS"][order], esolvM0=R["esolvM0"][order],
                 sitpotM=R["sitpotM"][order], loops=R["log_loops"][:, order], specrad=R["log_specrad"][:, order],
                 maxchange=R["log_maxchange"][:, order], nbound=R["log_nbound"][:, order],
                 est_iter=np.full((3, nt), np.nan), wall=np.stack([R["wallM0"], R["wallM1"], R["wallS"]])[:, order],
                 pkint=R["tau_pkint"][order], eback=R["tau_eback"][order]),
    )
    if mc_energies is not None:
        rec.ref["mc_energies"] = mc_energies
    return rec
