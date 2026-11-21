"""M6 local checks of the driver logic (CPU, any JAX; seconds): synthetic StateRecords, PB replaced by a deterministic
stand-in (api.solve_pb), real JAX assembly and the XLA reference MC at 2,000 sweeps.
  1. full run -> label shards; W round trip (upper triangle), curves/pKa shapes, provenance, label() == shard values
  2. resume: run stopped after 1 shard (+ a stale temp file from a "killed" writer) then restarted -> every array bitwise
     equal to the uninterrupted run (timing_json excluded); restart recomputes nothing already written
  3. shard-size / order independence: same states, shard_size 2 and reversed item order -> per-state arrays bitwise equal
  4. overflow: an OverflowError on the first class -> driver retry at the next class (logged); on every class ->
     status 'failed_pb' (logged, written, other states unaffected)
  5. a changed cfg refuses to resume into the same directory
pytest runs the same checks in a temporary directory with api.solve_pb patched for the test only.
usage: python tests/test_jaxpb_api.py [workdir]"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from jaxpb import api, records as RC

F32 = np.float32


def synth(sid, seed):
    rng = np.random.default_rng(seed)
    ns = int(rng.integers(3, 7)); Na = 3 * ns + 30
    pos = rng.normal(0, 8, (Na, 3)).astype(F32); rad = np.full(Na, 1.5, F32); q_bg = rng.normal(0, 0.3, Na).astype(F32)
    typ = np.array(["a" if s % 2 == 0 else "c" for s in range(ns)])
    nst = rng.integers(2, 5, ns)
    site_atoms = np.arange(3 * ns, dtype=np.int64)
    tau_q, tau_site, isref, pkmod, occ = [], [], [], [], []
    for s in range(ns):
        idx = site_atoms[3 * s:3 * s + 3]
        for j in range(nst[s]):
            ref = j == nst[s] - 1
            tau_q.append(q_bg[idx] if ref else (q_bg[idx] + rng.normal(0, 0.5, 3)).astype(F32))
            tau_site.append(s); isref.append(ref); pkmod.append(np.nan if ref else rng.uniform(3, 10))
            occ.append((0 if ref else 1) if typ[s] == "a" else (1 if ref else 0))
    T_ = int(nst.sum())
    off = lambda k, m: np.arange(0, k * m + 1, m, dtype=np.int64)
    return RC.StateRecord(
        state_id=sid, cid=sid.split("_")[0], state=sid.split("_")[1], role="general", split="val", source="synth",
        producer="synth", config_json="{}", pos=pos, rad=rad, q_bg=q_bg, atom_id=np.arange(Na, dtype=np.int64),
        site_chain=np.array(["A"] * ns), site_resnum=np.arange(ns, dtype=np.int64), site_resname=np.array(["GLU"] * ns),
        site_type=typ, site_chargediff=np.where(typ == "a", -1, 1).astype(np.int8),
        site_refprot=np.where(typ == "a", -1, 1).astype(np.int8), site_center_h=np.zeros((ns, 3)),
        site_acent_M=np.zeros((ns, 3)), site_acent_S=np.zeros((ns, 3)), site_atoms=site_atoms, site_atoms_off=off(ns, 3),
        site_tau_off=np.concatenate([[0], np.cumsum(nst)]).astype(np.int64),
        model_pos=pos[site_atoms], model_rad=rad[site_atoms], model_off=off(ns, 3),
        tau_name=np.array([f"T{k}" for k in range(T_)]), tau_site=np.array(tau_site, np.int64), tau_isref=np.array(isref),
        tau_pkmod=np.array(pkmod), tau_occ=np.array(occ, np.int8), tau_delphi=np.arange(T_, dtype=np.int64),
        tau_q=np.concatenate(tau_q).astype(F32), tau_q_off=off(T_, 3), tau_mq=np.concatenate(tau_q).astype(F32),
        tau_mq_off=off(T_, 3), params=dict(epsin=15.0, epsout=80.0, conc=0.1, cutoff_back=-1.0), ref={})


FAIL = {"first_class": set(), "always": set()}


def fake_pb(rec, cfg, log=None, geo_cache=None):
    """deterministic PB stand-in with the driver's overflow-retry contract (pb.solve_state raises OverflowError)."""
    tries = []
    for c in (rec.size_class, rec.size_class + 1):
        if rec.state_id in FAIL["always"] or (rec.state_id in FAIL["first_class"] and c == rec.size_class):
            tries.append(dict(size_class=c, error="synthetic overflow"))
            if log:
                log(dict(event="pb_overflow", state_id=rec.state_id, size_class=c, error="synthetic overflow"))
            continue
        rng = np.random.default_rng(int(api.chain_id_of(rec.state_id)))
        Tn, Na = rec.n_taus, rec.n_atoms
        out = RC.PBOut(esolvM=(-5 + rng.normal(0, 1, Tn)).astype(F32), esolvS=(-4 + rng.normal(0, 1, Tn)).astype(F32),
                       esolvM0=rng.normal(0, 1, Tn).astype(F32), sitpotM=rng.normal(0, 2, (Tn, Na)).astype(F32),
                       sweeps=np.full((Tn, 3), 100, np.int32), maxchange=np.full((Tn, 3), 0.005, F32),
                       flags={"xl_retries": 0, "class_bumps": 0, "size_class": c, "driver_retries": len(tries)},
                       rho=np.full((Tn, 2), 0.99, F32), backend="fake")
        if tries:
            out.flags["driver_retry_reasons"] = tries
        return out, {"t_total": 0.0}
    raise api.PBFailure(json.dumps(tries))



def shard_arrays(d):
    out = {}
    for p in sorted(Path(d).glob("labels_*.npz")):
        Z = np.load(p)
        for k in Z.files:
            if k != "timing_json":
                out[(p.name, k)] = Z[k]
    return out


def per_state(d):
    out = {}
    for p in sorted(Path(d).glob("labels_*.npz")):
        for s in api.load_shard(str(p)):
            out[s["alias"]] = s
    return out


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def main(wd=None):
    wd = Path(wd) if wd else Path(tempfile.mkdtemp(prefix="jaxpb-m6-"))
    shutil.rmtree(wd, ignore_errors=True); wd.mkdir(parents=True)
    recs = [synth(f"c{i:02d}_{s}", 10 * i + k) for i in range(4) for k, s in enumerate(("AB", "A", "B"))]
    inp = str(wd / "in.npz"); RC.save_states(inp, recs)
    cfg = api.Config(backend="xla", mc_sweeps=2000, shard_size=5, s_flush=4)
    items = api.items_from_shards([inp])
    res = {}

    # 1. full run
    s1 = api.run(items, str(wd / "full"), cfg, echo=False)
    full = per_state(wd / "full")
    assert len(full) == len(recs) and all(v["status"] == "ok" for v in full.values())
    lab = api.label(recs[:3], cfg)
    for r, cur, pk, e in zip(recs[:3], lab.curves, lab.pKa, lab.energies):
        f = full[r.state_id]
        assert same(f["W"], e.W) and same(f["g"], e.g) and same(f["curves"], cur) and same(f["pka"], pk)
        assert f["curves"].shape == (RC.NPH, r.n_sites) and f["W"].shape == (r.n_taus,) * 2
    prov = api.shard_provenance(str(wd / "full" / "labels_00000.npz"))
    res["full"] = dict(shards=s1["shards_written"], provenance_keys=sorted(prov), x64=prov["x64"],
                       flushes=len(s1["titration_batches"]), label_api_equal=True)

    # 2. resume after 1 shard + stale temp
    s2a = api.run(items, str(wd / "resume"), cfg, max_shards=1, echo=False)
    (wd / "resume" / "labels_00001.npz.tmp-999.npz").write_bytes(b"garbage")
    s2b = api.run(items, str(wd / "resume"), cfg, echo=False)
    A_, B_ = shard_arrays(wd / "full"), shard_arrays(wd / "resume")
    assert A_.keys() == B_.keys() and all(same(A_[k], B_[k]) for k in A_), "resume not bitwise"
    assert s2b["shards_skipped"] == [0] and not list((wd / "resume").glob("*.tmp-*"))
    s2c = api.run(items, str(wd / "resume"), cfg, echo=False)
    assert s2c["shards_written"] == [] and len(s2c["shards_skipped"]) == len(s1["shards_written"])
    res["resume"] = dict(first=s2a["shards_written"], restart_written=s2b["shards_written"], skipped=s2b["shards_skipped"],
                         bitwise=True, third_run_written=s2c["shards_written"])

    # 3. other shard size and reversed order
    cfg2 = api.Config(backend="xla", mc_sweeps=2000, shard_size=2, s_flush=4)
    api.run(items[::-1], str(wd / "rev"), cfg2, echo=False)
    rev = per_state(wd / "rev")
    keys = [k for k in full[recs[0].state_id] if k not in ("diag",)]
    bad = [(a, k) for a in full for k in keys if not same(full[a][k], rev[a][k]) if not isinstance(full[a][k], str)]
    assert not bad, bad[:5]
    res["order_independent"] = True

    # 4. overflow paths
    FAIL["first_class"].add(recs[1].state_id); FAIL["always"].add(recs[2].state_id)
    s4 = api.run(items, str(wd / "ovf"), cfg, echo=False)
    ov = per_state(wd / "ovf")
    assert ov[recs[1].state_id]["status"] == "ok" and ov[recs[1].state_id]["diag"]["pb_flags"]["driver_retries"] == 1
    assert ov[recs[2].state_id]["status"] == "failed_pb" and len(ov[recs[2].state_id]["diag"]["pb_failure"]) == 2
    for a in full:
        if a not in (recs[1].state_id, recs[2].state_id):
            assert same(full[a]["W"], ov[a]["W"]) and same(full[a]["curves"], ov[a]["curves"])
    assert same(full[recs[1].state_id]["curves"], ov[recs[1].state_id]["curves"])
    res["overflow"] = dict(events=s4["events"])
    FAIL["first_class"].clear(); FAIL["always"].clear()

    # 5. cfg change refuses to resume
    try:
        api.run(items, str(wd / "full"), api.Config(backend="xla", mc_sweeps=3000, shard_size=5, s_flush=4), echo=False)
        raise AssertionError("resumed with a different cfg")
    except RuntimeError as e:
        res["cfg_change_refused"] = str(e)[:80]
    res["pass"] = True
    print(json.dumps(res, indent=1))
    return res


def test_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "solve_pb", fake_pb)
    FAIL["first_class"].clear(); FAIL["always"].clear()
    assert main(tmp_path / "w")["pass"]


if __name__ == "__main__":
    api.solve_pb = fake_pb
    main(sys.argv[1] if len(sys.argv) > 1 else None)
