"""Antibody path for the extra pKPDB rules, as in pinder-prefilter-v1/prefilter.py: for Ab/Ag dimers the antibody chain is
checked by CDRs, not whole-chain identity (whole-chain rules on antibodies match any Fab via the conserved framework and
constant domains). Here, for flagged Ab/Ag dimers:
  antigen chain : experimental_30 and fragment_90 whole-chain, as in exclusions.py
  antibody chain: concatenated CDRs (transferred by alignment to SAbDab-annotated V domains, prefilter.transfer) >= 70%
                  identity, same chain type, vs the CDRs of experimental antibody references (cdr_exp_70); fragment_90 is
                  dropped for antibody chains (prefilter already CDR-checked them against the held-out benchmark)
  reserved_pdb_id unchanged.
Antibody-containing dimers that are not Ab/Ag keep the whole-chain rules (prefilter does the same).
Reports how many flagged Ab/Ag complexes would remain excluded. Does not modify the v1 list."""
import json, os, subprocess, tempfile, collections
import pandas as pd
R = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/"; MM = R + "audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs"; P = R + "audits/pinder-prefilter-v1/"
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
def mm(*c): subprocess.run([MM, *c, "--threads", os.environ.get("SLURM_CPUS_PER_TASK", "4")], check=True, stdout=subprocess.DEVNULL)
def rd(p, n):
    try: return pd.read_csv(p, sep="\t", names=n)
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=n)
def wfa(d, p):
    with open(p, "w") as f:
        for k, v in d.items(): f.write(f">{k}\n{v}\n")
# SAbDab CDR positions on reference V domains (same construction as prefilter.py)
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
def cdrs(aln, seqs):
    al = rd(aln, ["q", "t", "f", "bits", "qs", "qe", "ts", "te", "qa", "ta"]).sort_values("bits", ascending=False).drop_duplicates("q")
    out = {}
    for r in al.itertuples():
        pos = refcdr.get(r.t)
        if pos is None: continue
        qi = r.qs - 1; ti = r.ts - 1; mp = {}
        for a_, b_ in zip(r.qa, r.ta):
            if a_ != "-" and b_ != "-": mp[ti] = qi
            if a_ != "-": qi += 1
            if b_ != "-": ti += 1
        if all(p in mp and e - 1 in mp for p, e in pos):
            out[r.q] = (r.t.split("|")[1], "".join(seqs[r.q][mp[p]:mp[e - 1] + 1] for p, e in pos))
    return set(al.q), out
# experimental antibody references -> CDRs
exp = {k: v for k, v in (l.split("\n")[:2] for l in open("exp.fasta").read().split(">")[1:])}
mm("easy-search", "exp.fasta", P + "ref_V.fasta", "exp_ab_align.tsv", tmp + "/x", "--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "1",
   "--format-output", "query,target,fident,bits,qstart,qend,tstart,tend,qaln,taln")
exp_isab, exp_cdr = cdrs("exp_ab_align.tsv", exp)
print("experimental references aligned to V domains:", len(exp_isab), "| with CDRs:", {k: v for k, v in exp_cdr.items()})
# PINDER antibody chains -> CDRs (prefilter's alignment)
uniq = {k: v for k, v in (l.split("\n")[:2] for l in open(P + "uniq.fasta").read().split(">")[1:])}
pab_isab, pab_cdr = cdrs(P + "ab_align.tsv", uniq)
bad_cdr = set()
for chn in ("H", "L"):
    q = {k: s for k, (c, s) in pab_cdr.items() if c == chn}; r = {k: s for k, (c, s) in exp_cdr.items() if c == chn}
    if not q or not r: continue
    wfa(q, f"cdrq_{chn}.fasta"); wfa(r, f"cdrr_{chn}.fasta"); bad_cdr |= {k for k, s in q.items() if s in r.values()}
    mm("easy-search", f"cdrq_{chn}.fasta", f"cdrr_{chn}.fasta", f"cdrexp_hits_{chn}.tsv", tmp + f"/c{chn}", "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
       "--alignment-mode", "3", "--max-seqs", "10000", "-s", "7.5", "--format-output", "query,target,fident")
    h = rd(f"cdrexp_hits_{chn}.tsv", ["q", "t", "f"]); bad_cdr |= set(h[h.f >= 0.7].q)
exp_bad = {l.split()[0] for l in open("exp_hits.tsv") if float(l.split()[2]) >= .3 and min(map(float, l.split()[3:5])) >= .8}
frag_bad = {l.split()[0] for l in open("frag_hits.tsv") if float(l.split()[2]) >= .9 and max(map(float, l.split()[3:5])) >= .8}
pf = pd.read_parquet(R + "audits/pinder-prefilter-v1/prefiltered_v1.parquet", columns=["id", "hR", "hL", "contains_antibody", "contains_antigen"]).set_index("id")
v1 = pd.read_csv("pinder_heldout_exclusions_v1.tsv", sep="\t").fillna("")
res = collections.Counter(); keep = []; dec = []
for r in v1.itertuples():
    p = pf.loc[r.id]
    if not (p.contains_antibody and p.contains_antigen): res["not_abag|" + ("excluded")] += 1; continue
    c_ = [(a, g) for a, g in ((p.hR, p.hL), (p.hL, p.hR)) if a in pab_isab and g not in pab_isab]
    if len(c_) != 1: res["abag_unresolved|excluded"] += 1; continue
    a, g = c_[0]
    rules = [k for k, v in (("antigen_experimental_30", g in exp_bad), ("antigen_fragment_90", g in frag_bad),
                            ("cdr_exp_70", a in bad_cdr), ("reserved_pdb_id", "reserved_pdb_id" in r.rules)) if v]
    res[("abag|excluded:" + ",".join(rules)) if rules else "abag|released"] += 1
    if not rules: keep.append(r.id)
    dec.append(dict(id=r.id, ab_path_rules=",".join(rules)))
print(json.dumps(dict(sorted(res.items())), indent=1))
pd.Series(keep, name="id").to_csv("ab_rule_released.tsv", sep="\t", index=False)
pd.DataFrame(dec).to_csv("ab_rule_decisions.tsv", sep="\t", index=False)
