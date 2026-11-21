"""abdata-v5 pool-v5 ledger (2026-11-16): one row per PINDER Ab/Ag dimer (index.parquet contains_antibody & contains_antigen)
with the stage at which it left the PINDER -> pool-v5 pipeline and why, or 'admitted'; plus a cluster-level funnel.

  PKABENCH_RUNTIME=<root> POOL_OUT=<pool_v5 out> python ledger.py OUT_DIR   (needs POOL_OUT/state.pkl, cdr-identity.tsv, pool-v5/pinder.tsv)
Stages, in pipeline order (each dimer gets the FIRST one it fails):
  01 size            prefilter: length1 + length2 > 1500
  02 shortlist       prefilter: cluster rank >= SHORT=10 (X-ray first, best resolution, distinct PDB entries first)
  03 sequence        prefilter: no RCSB entity sequence
  04 prefilter_leak  prefilter held-out leak screen (Ab/Ag: antigen 70% / CDR 70%, cdr_untransferred, antigen_unidentified)
  05 prep_not_run    survivor never prepared (keep_rank >= KEEP=5 and the cluster had an accepted first-pass dimer, or
                     beyond the 5 fallbacks of the fallback round)
  06 prep            preparation (pinder-prep-v2 prep_v7, first pass pf_* / fallback fb_*) not accepted: status + code
  07 heldout_excl    pinder_heldout_exclusions_v3 (rules)
  08 no_label_iface  no training-mask interface site with AB + own-free pKAI labels and both weights (reason from sites/labels)
  09 exact_ref       a chain exactly equal to a held-out benchmark chain or pKPDB reference sequence
  10 validation      the pool-v4 PINDER validation structure itself / 11 val_cluster: in a validation group
  12 v4_screen       pool-v4 whole-chain screen (>= 70% / 80% or exact to a validation chain)  [informational for v5]
  13 v5_screen       pool-v5 rule (antibody CDRs / antigen whole chain); admitted = in pool-v5 pinder.tsv
Columns: id, cluster_id, pdb_id, split, method, resolution, n_res, stage, reason, in_pool_v4, in_pool_v5, plus details.
"""
import csv, glob, json, os, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

R = Path(os.environ.get("PKABENCH_RUNTIME", "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench"))  # runtime root (or the extracted input bundle)
POOL = Path(os.environ.get("POOL_OUT", "/home/coulson/oc/lina4225/_runtime/abdata-v5/pool"))       # pool_v5.py output dir
IX = R / "audits/pinder-index-v1"; PF = R / "audits/pinder-prefilter-v1"; PREP = R / "audits/pinder-prep-v2/results_v7"
LAB = R / "pretraining/pinder-pkai-v1"; SHORT, KEEP = 10, 5
GROUP_ALIAS = {"NTR": "NTERM", "CTR": "CTERM"}


def label_reason(cid):
    """(labelled training interface sites, reason) with training_pools._pinder_complex's rule, and why it is 0."""
    folder = LAB / "entries" / cid
    sites = json.loads((folder / "sites.json").read_text()); labels = json.loads((folder / "labels.json").read_text())["pkai"]
    have = {st: {(str(c), int(n), str(i), GROUP_ALIAS.get(g, g)) for c, n, i, g, v in rows if v is not None} for st, rows in labels.items()}
    iface = [s for s in sites if s["interface"]]; tm = [s for s in iface if s["train_mask"]]
    key = lambda s: (s["chain"], s["resnum"], s["icode"], s["group"])
    ab = [s for s in tm if key(s) in have.get("AB", ())]; free = [s for s in ab if key(s) in have.get(s["partner"], ())]
    w = [s for s in free if s.get("w_burial") is not None and s.get("w_interface") is not None]
    if w: return len(w), ""
    if not iface: return 0, "no interface sites"
    if not tm:
        why = Counter(("gap:" + str(s.get("gap_rule"))) if s.get("gap_rule") not in (None, "usable") else
                      ("tier:" + str(s.get("natural_tier"))) if s.get("natural_tier") not in ("clean", "uncertain") else "component/defect radius" for s in iface)
        return 0, f"all {len(iface)} interface sites outside the training mask ({', '.join(f'{k} {v}' for k, v in why.most_common(3))})"
    if not ab: return 0, f"{len(tm)} training-mask interface sites, none with an AB pKAI label"
    if not free: return 0, f"{len(ab)} AB-labelled training interface sites, none with the own-free pKAI label"
    return 0, f"{len(free)} labelled training interface sites without site weights"


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    i = pd.read_parquet(IX / "index.parquet", columns=["id", "split", "pdb_id", "cluster_id", "contains_antibody", "contains_antigen"])
    m = pd.read_parquet(IX / "metadata.parquet", columns=["id", "length1", "length2", "method", "resolution"])
    x = i.merge(m, on="id", how="left"); x["n"] = x.length1.astype(int) + x.length2.astype(int); x["resolution"] = x.resolution.astype(float)
    # the prefilter's shortlist rank over ALL dimers <= 1500 (Ab/Ag clusters can hold non-Ab/Ag dimers)
    y = x[x.n <= 1500].copy(); y["xray"] = (y.method == "X-RAY DIFFRACTION").astype(int)
    y = y.sort_values(["cluster_id", "xray", "resolution"], ascending=[True, False, True], na_position="last")
    y["pdb_rank"] = y.groupby(["cluster_id", "pdb_id"]).cumcount()
    y = y.sort_values(["cluster_id", "pdb_rank", "xray", "resolution"], ascending=[True, True, False, True], na_position="last")
    y["rank"] = y.groupby("cluster_id").cumcount(); rank = dict(zip(y.id, y["rank"]))
    ab = x[x.contains_antibody & x.contains_antigen].copy()
    s = pd.read_parquet(PF / "shortlist_v1.parquet", columns=["id", "leak"]); leak = dict(zip(s.id, s.leak))
    k = pd.read_parquet(PF / "prefiltered_v1.parquet", columns=["id", "cluster_id", "keep_rank", "prep_first"]); keep = dict(zip(k.id, k.keep_rank)); first = dict(zip(k.id, k.prep_first))
    prep = {}; timing = []
    for f in sorted(glob.glob(str(PREP / "pf_*.jsonl"))) + sorted(glob.glob(str(PREP / "fb_*.jsonl"))):
        rnd = "first" if "/pf_" in f else "fallback"
        for line in open(f):
            o = json.loads(line); prep[o["id"]] = (rnd, o["status"], o.get("code", ""), (o.get("detail") or "")[:120])
            if o.get("kind") == "Ab/Ag": timing.append((o.get("prep_s") or 0.0) + (o.get("parse_s") or 0.0))
    pf_ok = defaultdict(bool)
    for cid, (rnd, st, _, _) in prep.items():
        if rnd == "first" and st == "accepted": pf_ok[cid] = True
    cl_of = dict(zip(k.id, k.cluster_id)); first_ok_cluster = defaultdict(bool)
    for cid, ok in pf_ok.items():
        if ok and cid in cl_of: first_ok_cluster[cl_of[cid]] = True
    excl = {r["id"]: r["rules"] for r in csv.DictReader(open(LAB / "pinder_heldout_exclusions_v3.tsv"), delimiter="\t")}
    labelled = set()
    for f in glob.glob(str(LAB / "index/label_*.jsonl")):
        for line in open(f): labelled.add(json.loads(line)["id"])
    v4 = {r["id"] for r in csv.DictReader(open(R / "training/pool-v4/pinder.tsv"), delimiter="\t")}
    have_pool = (POOL / "state.pkl").exists() and (POOL / "pool-v5/pinder.tsv").exists()  # stages 09-13 need the pool_v5 run
    if have_pool:
        with open(POOL / "state.pkl", "rb") as f: state = pickle.load(f)
        cands = {r["id"]: r for r in state["candidates"]["pinder"][0]}; val = {r["id"] for r in state["validation"]["pinder"]}; vgroups = state["val_groups"]["pinder"]
        v5 = {r["id"] for r in csv.DictReader(open(POOL / "pool-v5/pinder.tsv"), delimiter="\t")}
        cdr = {r["id"]: r for r in csv.DictReader(open(POOL / "cdr-identity.tsv"), delimiter="\t")}
    else: cands, val, vgroups, v5, cdr = {}, set(), set(), set(), {}
    rows = []
    for r in ab.itertuples(index=False):
        cid = r.id; stage, reason, extra = "", "", {}
        if r.n > 1500: stage, reason = "01_size", f"n {r.n} > 1500"
        elif rank.get(cid, 99) >= SHORT: stage, reason = "02_shortlist", f"cluster rank {rank.get(cid)} >= {SHORT}"
        elif cid not in leak: stage, reason = "03_sequence", "no RCSB entity sequence"
        elif leak[cid]: stage, reason = "04_prefilter_leak", leak[cid]
        elif cid not in prep:
            stage = "05_prep_not_run"
            reason = (f"fallback keep_rank {keep.get(cid)}; cluster already had an accepted first-pass dimer" if first_ok_cluster[r.cluster_id]
                      else f"fallback keep_rank {keep.get(cid)} beyond the {KEEP} fallbacks of the fallback round")
        elif prep[cid][1] != "accepted": stage, reason = "06_prep", f"{prep[cid][0]} pass: {prep[cid][1]} {prep[cid][2]} {prep[cid][3]}".strip()
        elif cid not in labelled: stage, reason = "06_prep", "accepted but not in pinder-pkai-v1 (unlabelled)"
        elif cid in excl: stage, reason = "07_heldout_excl", excl[cid]
        else:
            n_lab, why = label_reason(cid); extra["labelled_interface_sites"] = n_lab
            if not n_lab: stage, reason = "08_no_label_iface", why
            elif not have_pool: stage, reason = "09+_pending_pool", "labelled; stages 09-13 need the pool_v5 run" + ("; in pool-v4" if cid in v4 else "")
            elif cid not in cands: stage, reason = "09_exact_ref", "chain exactly equal to a held-out benchmark / pKPDB reference sequence"
            elif cid in val: stage, reason = "10_validation", "pool-v4 PINDER validation structure"
            elif r.cluster_id in vgroups: stage, reason = "11_val_cluster", "in a pool-v4 validation group (sibling)"
            elif cid not in v5:
                c = cdr.get(cid, {}); stage = "13_v5_screen"
                reason = f"{c.get('rule', '?')}: antigen leak {c.get('ag_leak')}, cdr3 {c.get('cdr3')}, cdrs {c.get('cdrs')}, whole-chain hit {c.get('ab_whole_chain_hit')}"
            else: stage, reason = "admitted", ""
        rows.append({"id": cid, "cluster_id": r.cluster_id, "pdb_id": r.pdb_id, "split": r.split, "method": r.method, "resolution": r.resolution,
                     "n_res": int(r.n), "shortlist_rank": rank.get(cid), "keep_rank": keep.get(cid), "prep_round": prep.get(cid, ("",))[0],
                     "stage": stage, "reason": reason, "in_pool_v4": int(cid in v4), "in_pool_v5": int(cid in v5),
                     "v4_screen_removed": int(cid in cands and cid not in val and r.cluster_id not in vgroups and cid not in v4) if have_pool else None,
                     "labelled_interface_sites": extra.get("labelled_interface_sites")})
    df = pd.DataFrame(rows).sort_values(["cluster_id", "shortlist_rank", "id"], na_position="last")
    df.to_parquet(out / "ledger.parquet", index=False); df.to_csv(out / "ledger.tsv", sep="\t", index=False)
    order = sorted(df.stage.unique(), key=lambda s: (s == "admitted", s))
    dim = Counter(df.stage); funnel = []; alive = set(df.cluster_id); alive_d = len(df)
    funnel.append({"stage": "00_all Ab/Ag dimers", "dimers_left_here": 0, "dimers_surviving": alive_d, "clusters_surviving": len(alive)})
    for st in order:
        if st == "admitted": continue
        after = df[df.stage.map(lambda z: (z == "admitted") or (z > st))]
        funnel.append({"stage": st, "dimers_left_here": dim[st], "dimers_surviving": len(after), "clusters_surviving": after.cluster_id.nunique()})
    ad = df[df.stage == "admitted"]
    summary = {"dimers": len(df), "clusters": df.cluster_id.nunique(), "by_stage": dict(sorted(dim.items())), "funnel": funnel,
               "admitted": {"dimers": len(ad), "clusters": ad.cluster_id.nunique(), "in_pool_v4": int(ad.in_pool_v4.sum())},
               "pool_v4_abag": {"dimers": int(df.in_pool_v4.sum()), "clusters": df[df.in_pool_v4 == 1].cluster_id.nunique()},
               "top_reasons": {st: Counter(df[df.stage == st].reason.map(lambda z: z.split(" (")[0].split(":")[0] if st in ("06_prep", "08_no_label_iface", "07_heldout_excl") else z.split(";")[0][:60])).most_common(8)
                               for st in order if st not in ("admitted", "01_size", "02_shortlist")}}
    # cluster terminal stage: the furthest stage any of its dimers reached
    best = df.groupby("cluster_id").stage.agg(lambda z: max(z, key=lambda s: (s == "admitted", s)))
    summary["clusters_by_furthest_stage"] = dict(sorted(Counter(best).items()))
    # fixable clusters (scope extension 3): furthest stage is a pipeline loss another dimer of the same cluster could avoid
    fixable_stages = ("02_shortlist", "05_prep_not_run", "06_prep", "08_no_label_iface")
    fix = best[best.isin(fixable_stages)].index; plan = []
    for c in fix:
        d = df[df.cluster_id == c]
        tried = d[d.stage.isin(("06_prep", "08_no_label_iface"))]
        ready = d[d.stage == "05_prep_not_run"].sort_values("keep_rank"); fresh = d[(d.stage == "02_shortlist")].sort_values("shortlist_rank")
        take = pd.concat([ready, fresh]).head(KEEP)
        plan += [{"cluster_id": c, "id": z.id, "pdb_id": z.pdb_id, "from": z.stage, "furthest": best[c], "tried_before": len(tried)} for z in take.itertuples()]
    plan = pd.DataFrame(plan); plan.to_csv(out / "fixable-plan.tsv", sep="\t", index=False)
    import statistics
    med = statistics.median(timing) if timing else None; mean = sum(timing) / len(timing) if timing else None
    summary["fixable"] = {"stages": fixable_stages, "clusters": int(len(fix)), "clusters_by_furthest_stage": dict(Counter(best[fix])),
                          "clusters_with_untried_dimers": int(plan.cluster_id.nunique()) if len(plan) else 0,
                          "dimers_to_prepare_at_keep5": int(len(plan)), "of_which_prefiltered": int((plan["from"] == "05_prep_not_run").sum()) if len(plan) else 0,
                          "of_which_need_prefilter_seq_and_leak_screen": int((plan["from"] == "02_shortlist").sum()) if len(plan) else 0,
                          "pdb_entries_to_download": int(plan.pdb_id.nunique()) if len(plan) else 0,
                          "abag_prep_seconds_per_dimer": {"median": med, "mean": mean, "n": len(timing)},
                          "prep_cpu_hours_estimate": (len(plan) * mean / 3600.0) if len(plan) and mean else 0.0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_reasons"}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
