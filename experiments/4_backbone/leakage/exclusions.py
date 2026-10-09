"""PINDER held-out exclusions v1: the pKPDB leakage rules PINDER's 70%/80%-both prefilter did not apply, applied to
every accepted labelled PINDER complex (from check.py's MMseqs2 hits):
  experimental_30 : a chain >= 30% identity, >= 80% coverage of both, vs an experimental pKa reference (pkpdb references.json)
  fragment_90     : a chain >= 90% identity over >= 80% of the shorter sequence vs a benchmark reference
  reserved_pdb_id : the complex's PDB ID is a reserved PDB ID
Writes pinder_heldout_exclusions_v1.tsv (one row per flagged complex: rules and best-hit evidence),
factorial_cohort_flagged.tsv (flagged complexes in training/ogqt-pinder-factorial-v1/cohort.json; read only) and
exclusions_v1.json (rule parameters, counts, sha256)."""
import json, glob, hashlib, collections
import pandas as pd
R = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/"; O = R + "pretraining/pinder-pkai-v1/"
refs = json.load(open(R + "pretraining/pkpdb-full-v1/references.json"))
reserved = {p.lower() for p in refs["reserved_pdb_ids"]}
def best(path, keep):
    out = {}
    for l in open(path):
        q, t, fi, qc, tc = l.split(); fi, qc, tc = float(fi), float(qc), float(tc)
        if keep(fi, qc, tc) and (q not in out or fi > out[q][1]): out[q] = (t, fi, qc, tc)
    return out
exp = best("exp_hits.tsv", lambda fi, qc, tc: fi >= .3 and min(qc, tc) >= .8)
frag = best("frag_hits.tsv", lambda fi, qc, tc: fi >= .9 and max(qc, tc) >= .8)
pf = pd.read_parquet(R + "audits/pinder-prefilter-v1/prefiltered_v1.parquet", columns=["id", "pdb_id", "cluster_id", "hR", "hL"]).set_index("id")
fmt = lambda h, hit: f"{h}>{hit[0]}:{hit[1]:.3f}/{hit[2]:.2f}/{hit[3]:.2f}"
rows = []; n = 0
for f in sorted(glob.glob(O + "index/label_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["status"] != "accepted": continue
        n += 1; p = pf.loc[r["id"]]; hs = sorted({p.hR, p.hL})
        e = [fmt(h, exp[h]) for h in hs if h in exp]; fr = [fmt(h, frag[h]) for h in hs if h in frag]
        rules = [k for k, v in (("experimental_30", e), ("fragment_90", fr), ("reserved_pdb_id", p.pdb_id.lower() in reserved)) if v]
        if rules: rows.append(dict(id=r["id"], pdb_id=p.pdb_id, cluster_id=p.cluster_id, split=r["split"], kind=r["kind"], ctype=r["ctype"],
                                   rules=",".join(rules), experimental_hits=";".join(e), fragment_hits=";".join(fr)))
df = pd.DataFrame(rows).sort_values("id"); df.to_csv("pinder_heldout_exclusions_v1.tsv", sep="\t", index=False)
coh = json.load(open(R + "training/ogqt-pinder-factorial-v1/cohort.json")); bad = df.set_index("id")
fc = pd.DataFrame([dict(id=c["id"], cohort_split=c["split"], rules=bad.loc[c["id"], "rules"]) for c in coh["records"] if c["id"] in bad.index])
fc.to_csv("factorial_cohort_flagged.tsv", sep="\t", index=False)
sha = hashlib.sha256(open("pinder_heldout_exclusions_v1.tsv", "rb").read()).hexdigest()
meta = dict(version="pinder-heldout-exclusions-v1", sha256=sha, complexes_checked=n, complexes_excluded=len(df), clusters_touched=int(df.cluster_id.nunique()),
            by_rule={k: int(df.rules.str.contains(k).sum()) for k in ("experimental_30", "fragment_90", "reserved_pdb_id")},
            rules=dict(experimental_30=">=0.30 identity, >=0.80 coverage of both, vs 'experimental' references",
                       fragment_90=">=0.90 identity, >=0.80 coverage of the shorter sequence, vs 'benchmark' references",
                       reserved_pdb_id="exact PDB ID in reserved_pdb_ids"),
            references=R + "pretraining/pkpdb-full-v1/references.json",
            references_sha256=hashlib.sha256(open(R + "pretraining/pkpdb-full-v1/references.json", "rb").read()).hexdigest(),
            factorial_cohort_flagged=fc.cohort_split.value_counts().to_dict() if len(fc) else {})
json.dump(meta, open("exclusions_v1.json", "w"), indent=1); print(json.dumps(meta, indent=1))
