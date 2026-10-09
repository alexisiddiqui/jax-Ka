"""Held-out antibodies without SAbDab CDR annotations: the antibody path (pinder-prefilter-v1 Ab/Ag path, PINDER exclusions
v2, pKPDB ANTIBODY_PATH) compares CDRs only with held-out complexes that carry SAbDab CDRs. Here every held-out chain
(pinder-prefilter-v1/ref_heldout_all.fasta: test + val + reserved + set-2) that aligns to a SAbDab-annotated V domain gets
CDRs by alignment transfer (as the prefilter), and pool antibody chains are re-checked at >= 70% CDR identity (same chain
type) against this complete held-out CDR set. Query chains: PINDER pool-v1 Ab/Ag dimers' antibody chains; pKPDB pool-v1
entries released by the antibody path. Writes heldout_cdr_{H,L}.fasta (complete held-out CDR sets), flagged_pinder.tsv, flagged_pkpdb.tsv and summary.json."""
import json, os, subprocess, tempfile, hashlib, csv, collections
import pandas as pd
R = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/"; P = R + "audits/pinder-prefilter-v1/"; MM = R + "audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs"
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR")); TH = os.environ.get("SLURM_CPUS_PER_TASK", "4")
def mm(*c): subprocess.run([MM, *c, "--threads", TH], check=True, stdout=subprocess.DEVNULL)
def rfa(p):
    d = {}; k = None
    for l in open(p):
        l = l.strip()
        if l.startswith(">"): k = l[1:]; d[k] = ""
        elif k: d[k] += l
    return d
def wfa(d, p):
    with open(p, "w") as f:
        for k, v in d.items(): f.write(f">{k}\n{v}\n")
def rd(p, n):
    try: return pd.read_csv(p, sep="\t", names=n)
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=n)
refcdr = {}
for c in json.load(open(R + "universe/combined-split-v1/index.json"))["candidates"]:
    sab = c.get("sabdab")
    if not sab: continue
    for chn, V, names in (("H", "VH", ("CDR-H1", "CDR-H2", "CDR-H3")), ("L", "VL", ("CDR-L1", "CDR-L2", "CDR-L3"))):
        v = (sab.get(V) or "").strip().upper(); cd = [(sab.get(n) or "").strip().upper() for n in names]
        if not v or not all(cd): continue
        pos = []; p0 = 0
        for q in cd:
            j = v.find(q, p0)
            if j < 0: pos = None; break
            pos.append((j, j + len(q))); p0 = j + len(q)
        if pos is not None: refcdr[f"{c['complex_id']}|{chn}"] = pos
def cdrs(name, seqs):
    wfa(seqs, name + ".fasta")
    mm("easy-search", name + ".fasta", P + "ref_V.fasta", name + "_align.tsv", tmp + "/" + name, "--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "1",
       "--format-output", "query,target,fident,bits,qstart,qend,tstart,tend,qaln,taln")
    al = rd(name + "_align.tsv", ["q", "t", "f", "bits", "qs", "qe", "ts", "te", "qa", "ta"]).sort_values("bits", ascending=False).drop_duplicates("q")
    out = {}
    for r in al.itertuples():
        pos = refcdr.get(r.t)
        if pos is None: continue
        qi = r.qs - 1; ti = r.ts - 1; mp = {}
        for a_, b_ in zip(r.qa, r.ta):
            if a_ != "-" and b_ != "-": mp[ti] = qi
            if a_ != "-": qi += 1
            if b_ != "-": ti += 1
        if all(p in mp and e - 1 in mp for p, e in pos): out[r.q] = (r.t.split("|")[1], "".join(seqs[r.q][mp[p]:mp[e - 1] + 1] for p, e in pos))
    return set(al.q), out
held_ab, held_cdr = cdrs("heldout", rfa(P + "ref_heldout_all.fasta"))
old = {"H": set(rfa(P + "cdr_r_H.fasta").values()), "L": set(rfa(P + "cdr_r_L.fasta").values())}
new_sets = {c: {s for k, (cc, s) in held_cdr.items() if cc == c} for c in "HL"}
for c in "HL": wfa({f"h{i}": s for i, s in enumerate(sorted(new_sets[c]))}, f"heldout_cdr_{c}.fasta")  # complete held-out CDR sets
print("held-out chains aligning to V domains", len(held_ab), "| with CDRs", len(held_cdr), "| CDR sets complete H/L", {c: len(new_sets[c]) for c in "HL"},
      "| previously H/L", {c: len(old[c]) for c in "HL"}, "| new H/L", {c: len(new_sets[c] - old[c]) for c in "HL"})
# query chains
uniq = rfa(P + "uniq.fasta")
pf = pd.read_parquet(P + "prefiltered_v1.parquet", columns=["id", "hR", "hL"]).set_index("id")
pin = [r for r in csv.DictReader(open(R + "pretraining/pinder-pkai-v1/pool-v1.tsv"), delimiter="\t") if r["stratum"] == "Ab/Ag"]
qp = {h: uniq[h] for r in pin for h in (pf.loc[r["id"]].hR, pf.loc[r["id"]].hL)}
st = pd.read_csv(R + "audits/pkpdb-ab-path-v1/pkpdb_ab_path.tsv", sep="\t"); released = set(st[st.status.str.startswith("released")].pdb)
pool_k = [r for r in csv.DictReader(open(R + "pretraining/pkpdb-full-v1/pool-v1.tsv"), delimiter="\t") if r["id"] in released]
kch = {r["id"]: [s["sequence"] for s in json.load(open(R + f"pretraining/pkpdb-full-v1/entries/{r['id']}/defects.json"))["sequences"]] for r in pool_k}
qk = {"k" + hashlib.sha256(s.encode()).hexdigest()[:20]: s for v in kch.values() for s in v}
q_ab, q_cdr = cdrs("query", {**qp, **qk})
bad = {}
for c in "HL":
    q = {k: s for k, (cc, s) in q_cdr.items() if cc == c}; ref = {f"r{i}": s for i, s in enumerate(sorted(new_sets[c]))}
    wfa(q, f"cq_{c}.fasta"); wfa(ref, f"cr_{c}.fasta")
    for k, s in q.items():
        if s in new_sets[c]: bad[k] = 1.0
    mm("easy-search", f"cq_{c}.fasta", f"cr_{c}.fasta", f"cdr_hits_{c}.tsv", tmp + f"/c{c}", "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
       "--alignment-mode", "3", "--max-seqs", "10000", "-s", "7.5", "--format-output", "query,target,fident")
    h = rd(f"cdr_hits_{c}.tsv", ["q", "t", "f"])
    for r in h[h.f >= 0.7].itertuples(): bad[r.q] = max(bad.get(r.q, 0), r.f)
fp = [dict(id=r["id"], group=r["group"], labelled_interface_sites=r["labelled_interface_sites"], best_cdr_identity=max(bad.get(h, 0) for h in (pf.loc[r["id"]].hR, pf.loc[r["id"]].hL)))
      for r in pin if any(h in bad for h in (pf.loc[r["id"]].hR, pf.loc[r["id"]].hL))]
inv = {s: k for k, s in qk.items()}
fk = [dict(id=r["id"], group=r["group"], labelled_sites=r["labelled_sites"], best_cdr_identity=max(bad.get(inv[s], 0) for s in kch[r["id"]]))
      for r in pool_k if any(inv[s] in bad for s in kch[r["id"]])]
pd.DataFrame(fp).to_csv("flagged_pinder.tsv", sep="\t", index=False); pd.DataFrame(fk).to_csv("flagged_pkpdb.tsv", sep="\t", index=False)
summary = dict(heldout_v_chains=len(held_ab), heldout_with_cdrs=len(held_cdr), cdr_sets_complete={c: len(new_sets[c]) for c in "HL"}, cdr_sets_previous={c: len(old[c]) for c in "HL"},
               pinder_abag_pool=len(pin), pinder_flagged=len(fp), pinder_flagged_clusters=len({r["group"] for r in fp}),
               pinder_flagged_interface_sites=sum(int(r["labelled_interface_sites"]) for r in fp),
               pkpdb_released_in_pool=len(pool_k), pkpdb_flagged=len(fk), pkpdb_flagged_sites=sum(int(r["labelled_sites"]) for r in fk))
json.dump(summary, open("summary.json", "w"), indent=1); print(json.dumps(summary, indent=1))
