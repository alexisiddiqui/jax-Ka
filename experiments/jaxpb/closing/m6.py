"""M6 closing tests (GPU job on the A40; driver runs are `python -m jaxpb.api`, this script prepares and scores them).
phases
  prep    <out> --goldens pilot5.npz --pilotjson DIR --sts DIR
          pilot StateRecords + tau_pkmod (first line of <sts>/<RES>tau<k+1>.st, as the producer does) + ref
          mc_energies/result from the saved pilot json -> <out>/pilot5_m6.npz
  items   <out.json> --inputs shards... [--cycles K] [--smallest n]
          driver items: all states (cycle 0 in file order); cycles 1..K-1 permuted (rng seed = cycle) with alias
          "<sid>@<cycle>" (new MC chain ids); --smallest n keeps the n states with fewest atoms (smoke)
  compare <out.json> DIR_A DIR_B [DIR_C ...]   every array of every label shard bitwise (timing_json excluded)
  l3      <out.json> --labels DIR --inputs shards... [--det-ref det_ref.npz] [--mfs mf-surrogate-v1 dir]
          L3 vs the saved PyPKA teacher: pKint, W, pKa (200k from the driver; 50k re-titrated), binding shift on paired
          interface sites; MC floor (G on the SAVED energies, same chain ids, 200k/50k; R1 from mf-surrogate v1);
          localisation by common random numbers (our g + saved W, saved g + our W); x64 bitwise check vs det_ref.npz
          (M5 PB outputs, x64 off); throughput and compiles from the run summaries
  scale   <out.json> --labels DIR [--ref-labels DIR]   compile counts per shard (flatness), warm throughput, every
          cycle's energies bitwise equal per record (and to the reference run)"""
import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["prep", "items", "compare", "l3", "scale"]); ap.add_argument("out")
ap.add_argument("dirs", nargs="*")
ap.add_argument("--goldens"); ap.add_argument("--pilotjson"); ap.add_argument("--sts")
ap.add_argument("--inputs", nargs="*", default=[]); ap.add_argument("--cycles", type=int, default=1)
ap.add_argument("--smallest", type=int, default=None)
ap.add_argument("--labels"); ap.add_argument("--ref-labels"); ap.add_argument("--det-ref")
ap.add_argument("--mfs", default="/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/mf-surrogate-v1")
A = ap.parse_args()

from jaxpb import records as RC  # noqa: E402


def dump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
              (o.item() if isinstance(o, np.generic) else str(o)))
    print(json.dumps(obj, indent=1, default=str)[:6000], flush=True)


# ------------------------------------------------------------------------------------------------ prep
def prep():
    recs = RC.load_states(A.goldens)
    sts = Path(A.sts); pj = Path(A.pilotjson)
    info = {}
    for r in recs:
        mce = (pj / f"{r.state_id}.mc-energies.json").read_text(); res = (pj / f"{r.state_id}.result.json").read_text()
        e = json.loads(mce)
        toks = e["all_sites"]
        assert [f"{c}_{n}" for c, n in zip(r.site_chain, r.site_resnum)] == [t.split("_")[0] + "_" + t.split("_")[2] for t in toks]
        assert list(np.diff(r.site_tau_off)) == e["npossible_states"]
        pk = np.full(r.n_taus, np.nan)
        for s, tok in enumerate(toks):
            resn = tok.split("_")[1]
            for j, t in enumerate(r.site_taus(s)):
                if not r.tau_isref[t]:
                    pk[t] = float((sts / f"{resn}tau{j + 1}.st").read_text().splitlines()[0])
        r.tau_pkmod = pk
        r.site_resname = np.array([t.split("_")[1] for t in toks])
        r.params["cutoff_back"] = -1.0
        r.ref["mc_energies"] = mce; r.ref["result"] = res
        occ = np.concatenate([np.array(e["possible_states_occ"][i][:n], np.int8) for i, n in enumerate(e["npossible_states"])])
        assert np.array_equal(occ, r.tau_occ), r.state_id
        info[r.state_id] = dict(n_taus=r.n_taus, pkmod_filled=int(np.isfinite(pk).sum()))
    os.makedirs(A.out, exist_ok=True)
    p = os.path.join(A.out, "pilot5_m6.npz")
    sha = RC.save_states(p, recs)
    dump(dict(path=p, sha256=sha, states=info), os.path.join(A.out, "pilot5_m6.json"))


# ------------------------------------------------------------------------------------------------ items
def items():
    base = []
    for p in A.inputs:
        Z = np.load(p)
        ids = [str(x) for x in Z["meta_state_id"]]; off = Z["pos__off"]
        base += [dict(input=os.path.abspath(p), state_id=s, alias=s, n_atoms=int(off[i + 1] - off[i])) for i, s in enumerate(ids)]
    if A.smallest:
        base = sorted(base, key=lambda d: d["n_atoms"])[:A.smallest]
    out = [dict(input=d["input"], state_id=d["state_id"], alias=d["alias"]) for d in base]
    for c in range(1, A.cycles):
        perm = np.random.default_rng(c).permutation(len(base))
        out += [dict(input=base[i]["input"], state_id=base[i]["state_id"], alias=f"{base[i]['state_id']}@{c}") for i in perm]
    json.dump(out, open(A.out, "w"), indent=0)
    print(len(out), "items ->", A.out)


# ------------------------------------------------------------------------------------------------ compare
def shard_arrays(d):
    out = {}
    for p in sorted(Path(d).glob("labels_*.npz")):
        Z = np.load(p)
        for k in Z.files:
            if k != "timing_json":
                out[(p.name, k)] = Z[k]
    return out


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def compare():
    ref = shard_arrays(A.dirs[0])
    res = dict(reference=A.dirs[0], n_shards=len({k[0] for k in ref}), n_arrays=len(ref), others={})
    for d in A.dirs[1:]:
        o = shard_arrays(d)
        diff = sorted(f"{k[0]}:{k[1]}" for k in set(ref) | set(o) if k not in ref or k not in o or not same(ref[k], o[k]))
        res["others"][d] = dict(n_arrays=len(o), bitwise=not diff, differing=diff[:50])
    res["pass"] = all(v["bitwise"] for v in res["others"].values())
    dump(res, A.out)


# ------------------------------------------------------------------------------------------------ helpers
def load_labels(d):
    from jaxpb import api
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "labels_*.npz"))):
        for s in api.load_shard(p):
            out[s["alias"]] = s
    return out


def run_summaries(d):
    """per-process run summaries of a label dir (several when the run was killed and restarted)."""
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "run_*.json")))]


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], np.float64)
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mae=float(x.mean()), p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)),
                max=float(x.max()))


def pka_err(p, t):
    """sites with both midpoints: |p - t|; mismatches = exactly one missing."""
    p = np.asarray(p, np.float64); t = np.asarray(t, np.float64)
    both = np.isfinite(p) & np.isfinite(t)
    return np.abs(p[both] - t[both]), int((np.isfinite(p) != np.isfinite(t)).sum())


def saved_energies(mce):
    from jaxpb import titrate as T
    e = json.loads(mce)
    nst = np.array(e["npossible_states"], np.int32); n = len(nst)
    start = np.array([e["interactions_look"][i][0] for i in range(n)], np.int32)
    g = np.concatenate([np.array(e["possible_states_g"][i][:nst[i]], np.float64) for i in range(n)]).astype(np.float32)
    occ = np.concatenate([np.array(e["possible_states_occ"][i][:nst[i]], np.int8) for i in range(n)])
    W = np.array(e["interactions"], np.float64)
    so = np.repeat(np.arange(n), nst)
    W[so[:, None] == so[None, :]] = 0.0
    W = W.astype(np.float32)
    return RC.Energies(g=g, occ=occ, W=W, start=start, nst=nst, pairs=T.select_pairs_np(W, start, nst)), e["all_sites"]


def teacher_pka(res_json, toks):
    sys.path.insert(0, os.path.join(A.mfs, "code"))
    import common as C
    rows = {(r["chain"], r["resnum"], r["group"]): r for r in json.loads(res_json)["rows"]}
    pk, cur = [], []
    for t in toks:
        r = rows[C.token_key(t)]
        pk.append(np.nan if r["pka"] is None else r["pka"]); cur.append(r["curve"])
    return np.array(pk, np.float64), np.array(cur, np.float64).T


# ------------------------------------------------------------------------------------------------ l3
def l3():
    from jaxpb import api, titrate as T
    sys.path.insert(0, os.path.join(A.mfs, "code"))
    import common as C
    lab = load_labels(A.labels)
    cfg = api.Config()
    S = {}
    for p in A.inputs:
        ids = [i for i in RC.list_states(p) if i in lab]
        if not ids:
            continue
        for r in RC.load_states(p, only=ids):
            sav, toks = saved_energies(r.ref["mc_energies"])
            assert len(toks) == r.n_sites and np.array_equal(sav.occ, r.tau_occ)
            tp, tc = teacher_pka(r.ref["result"], toks)
            S[r.state_id] = dict(saved=sav, toks=toks, t_pka=tp, t_curve=tc, pkint_ref=np.asarray(r.ref["pkint"], np.float64),
                                 isref=np.asarray(r.tau_isref), cid=r.cid, state=r.state)
    ids = [i for i in lab if i in S]
    res = dict(labels=A.labels, n_states=len(ids), n_sites=int(sum(len(S[i]["toks"]) for i in ids)),
               status={s: sum(1 for i in lab if lab[i]["status"] == s) for s in {lab[i]["status"] for i in lab}})
    # --- energies: pKint, g, W, pairs
    pk_err, g_err, w_err, w_rel, pair_same, pair_n = [], [], [], [], 0, 0
    per_state = {}
    for i in ids:
        L, s = lab[i], S[i]
        nr = ~s["isref"]
        d = np.abs(L["pkint"][nr] - s["pkint_ref"][nr]); pk_err.append(d)
        g_err.append(np.abs(L["g"].astype(np.float64) - s["saved"].g))
        so = np.repeat(np.arange(len(L["nst"])), L["nst"])
        iu = np.triu_indices(len(L["g"]), 1); inter = so[iu[0]] != so[iu[1]]
        dw = np.abs(L["W"][iu][inter].astype(np.float64) - s["saved"].W[iu][inter]); w_err.append(dw)
        pa = {tuple(x) for x in L["pairs"].tolist()}; pb = {tuple(x) for x in s["saved"].pairs.tolist()}
        pair_same += int(pa == pb); pair_n += 1
        per_state[i] = dict(pkint_max=float(d.max()) if len(d) else 0.0, W_max=float(dw.max()) if len(dw) else 0.0,
                            pairs_equal=pa == pb, n_pairs=len(pa))
    pk_all = np.concatenate(pk_err); w_all = np.concatenate(w_err)
    res["pkint_vs_teacher"] = dict(stats(pk_all), criterion="MAE <= 0.05 pK", passed=bool(pk_all.mean() <= 0.05))
    res["g_vs_saved_kT"] = stats(np.concatenate(g_err))
    res["W_vs_saved_kT"] = dict(stats(w_all), n_gt_0p05=int((w_all > 0.05).sum()), criterion="p99 <= 0.05 kT",
                                passed=bool(np.percentile(w_all, 99) <= 0.05))
    res["pairs_equal_states"] = [pair_same, pair_n]
    # --- titrations (same chain ids as the driver: chain_id_of(state_id); common random numbers)
    cid = [api.chain_id_of(i) for i in ids]
    ours = [api.to_energies(lab[i]) for i in ids]
    saved = [S[i]["saved"] for i in ids]
    hyb_g = [RC.Energies(g=o.g, occ=s.occ, W=s.W, start=s.start, nst=s.nst, pairs=s.pairs) for o, s in zip(ours, saved)]
    hyb_w = [RC.Energies(g=s.g, occ=s.occ, W=o.W, start=o.start, nst=o.nst, pairs=o.pairs) for o, s in zip(ours, saved)]
    arms = {"pipeline_200k": [lab[i]["pka"] for i in ids]}
    import time
    tt = {}
    mem_titr = []      # this process runs no PB: per-bucket titration peaks are clean (buckets run in ascending Mp)
    on_b = lambda tb: mem_titr.append(dict(arm=cur_arm[0], Mp=tb["Mp"], states=tb["states"],
                                           **dict(zip(("in_use", "peak", "limit"), api.gpu_mem()))))
    cur_arm = [None]
    for name, E, sw in (("pipeline_50k", ours, 50_000), ("saved_200k", saved, 200_000), ("saved_50k", saved, 50_000),
                        ("ourg_savedW_200k", hyb_g, 200_000), ("savedg_ourW_200k", hyb_w, 200_000)):
        t0 = time.time(); cur_arm[0] = name
        R = api.titrate(E, cfg, chain_ids=cid, mc_sweeps=sw, on_batch=on_b)
        tt[name] = time.time() - t0
        arms[name] = [r["pka"] for r in R]
    # R1 (PyPKA MC, seed 7654321, mf-surrogate v1) on the saved energies
    curves = pickle.load(open(os.path.join(A.mfs, "work", "curves.pkl"), "rb"))
    r1, r1_teacher_curve_mismatch = [], []
    for i in ids:
        c = curves.get((S[i]["cid"], S[i]["state"], "R1"))
        tch = curves.get((S[i]["cid"], S[i]["state"], "teacher"))
        if c is None or tch is None or np.shape(tch) != S[i]["t_curve"].shape or np.abs(np.asarray(tch, np.float64) - S[i]["t_curve"]).max() > 1e-6:
            # curves.pkl stores float32 curves: alignment check at 1e-6
            r1.append(None); r1_teacher_curve_mismatch.append(i); continue
        r1.append(T.curves_to_pka(c))
    arms["R1_pypka_seed2"] = r1
    res["r1_unavailable_or_misaligned"] = r1_teacher_curve_mismatch
    res["titration_seconds"] = tt
    first = [m for m in mem_titr if m["arm"] == "pipeline_50k"]
    res["gpu_mem_titration_per_bucket"] = dict(
        note="fresh process without PB; peak_bytes_in_use after each launch (S = 32 states padded), buckets ascending",
        rows=first, host_maxrss_bytes_l3=api.host_rss_bytes(),
        W_bytes_per_launch={str(m["Mp"]): 32 * m["Mp"] ** 2 * 4 for m in first})
    res["disk"] = disk(A.labels, lab)
    teach = [S[i]["t_pka"] for i in ids]
    res["pka_vs_teacher"] = {}
    for name, pk in arms.items():
        e, mm = [], 0
        for p, t in zip(pk, teach):
            if p is None:
                continue
            d, m = pka_err(p, t); e.append(d); mm += m
        res["pka_vs_teacher"][name] = dict(stats(np.concatenate(e) if e else []), midpoint_mismatches=mm,
                                           states=sum(p is not None for p in pk))
    pp = res["pka_vs_teacher"]
    for k in ("pipeline_200k", "pipeline_50k"):
        pp[k]["criterion"] = "MAE <= 0.05"; pp[k]["passed"] = bool(pp[k].get("mae", np.inf) <= 0.05)
    # CRN decomposition vs the same-RNG titration of the saved energies
    res["crn_vs_saved_200k"] = {}
    for name in ("pipeline_200k", "ourg_savedW_200k", "savedg_ourW_200k"):
        e, mm = [], 0
        for p, q in zip(arms[name], arms["saved_200k"]):
            d, m = pka_err(p, q); e.append(d); mm += m
        res["crn_vs_saved_200k"][name] = dict(stats(np.concatenate(e)), midpoint_mismatches=mm)
    # --- binding shifts on paired interface sites (hybrid-mc-v1 flags via mf-surrogate-v1 load_state)
    flags = {}
    for i in ids:
        st = C.load_state(S[i]["cid"], S[i]["state"])
        assert st["n"] == len(S[i]["toks"]) and st["energies"]["all_sites"] == S[i]["toks"]
        flags[i] = st["rows"]
    pos = {i: k for k, i in enumerate(ids)}
    cids = sorted({S[i]["cid"] for i in ids if all(f"{S[i]['cid']}_{x}" in pos for x in ("AB", "A", "B"))})
    res["triples"] = len(cids)
    shifts = {name: [] for name in list(arms) + ["teacher"]}
    nsites_pair = 0
    for c in cids:
        ab = f"{c}_AB"; rab = {C.sitekey(r): k for k, r in enumerate(flags[ab])}
        for fs in ("A", "B"):
            fr = f"{c}_{fs}"
            for kf, row in enumerate(flags[fr]):
                ka = rab.get(C.sitekey(row))
                if ka is None:
                    continue
                if not (flags[ab][ka]["interface"] and flags[ab][ka]["supervision_eligible"] and row["supervision_eligible"]):
                    continue
                nsites_pair += 1
                for name in shifts:
                    if name == "teacher":
                        a, f = S[ab]["t_pka"][ka], S[fr]["t_pka"][kf]
                    else:
                        if arms[name][pos[ab]] is None or arms[name][pos[fr]] is None:
                            a = f = np.nan
                        else:
                            a, f = arms[name][pos[ab]][ka], arms[name][pos[fr]][kf]
                    shifts[name].append(a - f if np.isfinite(a) and np.isfinite(f) else np.nan)
    t = np.array(shifts["teacher"])
    res["shift_paired_interface"] = dict(n_sites=nsites_pair, teacher_mean_abs_shift=float(np.nanmean(np.abs(t))) if len(t) else None)
    for name in arms:
        d, mm = pka_err(np.array(shifts[name]), t)
        res["shift_paired_interface"][name] = dict(stats(d), midpoint_mismatches=mm)
    for k in ("pipeline_200k", "pipeline_50k"):
        q = res["shift_paired_interface"][k]; q["criterion"] = "MAE <= 0.05"; q["passed"] = bool(q.get("mae", np.inf) <= 0.05)
    res["worst_states_pkint"] = sorted(((v["pkint_max"], k) for k, v in per_state.items()), reverse=True)[:8]
    res["worst_states_W"] = sorted(((v["W_max"], k) for k, v in per_state.items()), reverse=True)[:8]
    res["per_state"] = per_state
    # --- x64 check vs M5 det_ref (PB, x64 off)
    if A.det_ref and os.path.exists(A.det_ref):
        R = np.load(A.det_ref)
        sids = sorted({k.split("__")[0] for k in R.files} & set(ids))
        bad = []
        for i in sids:
            L = lab[i]
            for k in ("esolvM", "esolvS", "esolvM0", "sweeps", "maxchange", "rho"):
                if not same(L[k], R[f"{i}__{k}"]):
                    bad.append(f"{i}:{k}")
            if L["sitpot_sha"] != api.sitpot_sha(R[f"{i}__sitpotM"]):
                bad.append(f"{i}:sitpotM")
        res["x64_vs_m5_det_ref"] = dict(states=sids, bitwise=not bad, differing=bad, ref=A.det_ref)
    # --- run summaries: compiles + throughput
    res["runs"] = [throughput(s) for s in run_summaries(A.labels)]
    dump(res, A.out)


def disk(d, lab):
    files = sorted(glob.glob(os.path.join(d, "labels_*.npz")))
    tot = sum(os.path.getsize(f) for f in files); n = len(lab)
    per_key = {}
    for f in files:
        Z = np.load(f)
        for k in Z.files:
            per_key[k] = per_key.get(k, 0) + Z[k].nbytes
    M = np.array([len(v["g"]) for v in lab.values()], np.float64)
    pts = np.array([sum(v[k].nbytes for k in ("g", "occ", "W_ut", "start", "nst", "pairs", "pkint", "e_back", "curves", "pka",
                                                "esolvM", "esolvS", "esolvM0", "sweeps", "maxchange", "rho")) for v in lab.values()])
    return dict(files=len(files), states=n, bytes_total=tot, bytes_per_state_mean=tot / max(n, 1),
                bytes_per_state_median=float(np.median(pts)), bytes_per_state_max=int(pts.max()), M_median=float(np.median(M)),
                M_max=int(M.max()), per_key_bytes_per_state={k: v / max(n, 1) for k, v in sorted(per_key.items(), key=lambda x: -x[1])[:10]},
                W_fraction=per_key.get("W_ut", 0) / max(sum(per_key.values()), 1),
                projected_30k_states_GB=tot / max(n, 1) * 30000 / 1e9, format="np.savez (uncompressed)")


def throughput(s):
    ps = [p for p in s["per_state"] if not p.get("failed")]
    warm = [p for p in ps if p.get("compiles", 1) == 0]
    tb = [b for b in s["titration_batches"] if b["compiles"] == 0]
    out = dict(pid_summary_out=s["out"], states=len(ps), warm_states=len(warm), compiles=s["compiles"]["n_compiled"],
               cache_hits=s["compiles"]["n_cache_hits"], compiles_after_shard=s["compiles_after_shard"],
               shards_written=s["shards_written"], shards_skipped=s["shards_skipped"], wall_s=s["wall_s"],
               compile_names=s["compiles"]["names"], memory=s.get("memory"), shard_bytes=s.get("shard_bytes"))
    if warm:
        tpb = sum(p["t_pb"] for p in warm); tasm = sum(p["t_asm"] for p in warm)
        sub = {k: sum(p.get(k, 0.0) for p in warm) for k in ("t_geometry", "t_maps", "t_solves", "t_sor", "t_post")}
        nt = sum(b["states"] for b in tb); tt = sum(b["seconds"] for b in tb)
        out["warm"] = dict(pb_s_per_state=tpb / len(warm), asm_s_per_state=tasm / len(warm),
                           titr_s_per_state=(tt / nt) if nt else None, titr_states=nt,
                           pb_breakdown_s_per_state={k: v / len(warm) for k, v in sub.items()},
                           states_per_a40_hour=dict(pb=3600 * len(warm) / tpb, assembly=3600 * len(warm) / tasm,
                                                    titration=(3600 * nt / tt) if tt else None,
                                                    total=3600 / (tpb / len(warm) + tasm / len(warm) + ((tt / nt) if nt else 0))))
    return out


# ------------------------------------------------------------------------------------------------ scale
def scale():
    lab = load_labels(A.labels)
    keys = ("g", "occ", "W_ut", "start", "nst", "pairs", "pkint", "e_back", "esolvM", "esolvS", "esolvM0", "sweeps",
            "maxchange", "rho")
    base = {}
    bad, ncmp = [], 0
    for a, d in lab.items():
        sid = d["state_id"]
        if sid not in base:
            base[sid] = d; continue
        ncmp += 1
        for k in keys:
            if not same(d[k], base[sid][k]):
                bad.append(f"{a}:{k}")
        if d["sitpot_sha"] != base[sid]["sitpot_sha"]:
            bad.append(f"{a}:sitpot")
    res = dict(labels=A.labels, n_states=len(lab), distinct_records=len(base), repeats_compared=ncmp,
               energies_bitwise_across_cycles=not bad, differing=bad[:50],
               status={s: sum(1 for i in lab if lab[i]["status"] == s) for s in {lab[i]["status"] for i in lab}})
    if A.ref_labels:
        ref = load_labels(A.ref_labels); bad2 = []
        for sid, d in base.items():
            if sid in ref:
                for k in keys + ("curves", "pka"):
                    if not same(d[k], ref[sid][k]):
                        bad2.append(f"{sid}:{k}")
        res["cycle0_vs_ref_run_bitwise"] = not bad2; res["cycle0_vs_ref_differing"] = bad2[:50]
    res["runs"] = [throughput(s) for s in run_summaries(A.labels)]
    for r in res["runs"]:
        c = r["compiles_after_shard"]
        r["compile_flat"] = dict(after_first_shard=c[0] if c else None, after_last_shard=c[-1] if c else None,
                                 flat=bool(c) and c[0]["n_compiled"] == c[-1]["n_compiled"])
    dump(res, A.out)


{"prep": prep, "items": items, "compare": compare, "l3": l3, "scale": scale}[A.phase]()
