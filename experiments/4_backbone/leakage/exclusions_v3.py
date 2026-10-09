"""PINDER held-out exclusions v3 = v2 plus Ab/Ag dimers whose antibody chain CDRs are >= 70% identical (same chain type) to
the complete held-out CDR sets (audits/ab-cdr-gap-v1/heldout_cdr_{H,L}.fasta: CDRs transferred onto every held-out chain
that aligns to a SAbDab-annotated V domain, including held-out antibodies without SAbDab CDR annotations, which the
prefilter's CDR check missed). Checked over every accepted Ab/Ag dimer, as identified by the prefilter. Writes
pinder_heldout_exclusions_v3.tsv and exclusions_v3.json."""
import json, os, subprocess, tempfile, hashlib, glob
import pandas as pd
R = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/"; P = R + "audits/pinder-prefilter-v1/"; G = R + "audits/ab-cdr-gap-v1/"
MM = R + "audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs"; tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
def mm(*c): subprocess.run([MM, *c, "--threads", os.environ.get("SLURM_CPUS_PER_TASK", "4")], check=True, stdout=subprocess.DEVNULL)
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
uniq = rfa(P + "uniq.fasta")
al = pd.read_csv(P + "ab_align.tsv", sep="\t", names=["q", "t", "f", "bits", "qs", "qe", "ts", "te", "qa", "ta"]).sort_values("bits", ascending=False).drop_duplicates("q")
isab = set(al.q); cdr = {}
for r in al.itertuples():
    pos = refcdr.get(r.t)
    if pos is None: continue
    qi = r.qs - 1; ti = r.ts - 1; mp = {}
    for a_, b_ in zip(r.qa, r.ta):
        if a_ != "-" and b_ != "-": mp[ti] = qi
        if a_ != "-": qi += 1
        if b_ != "-": ti += 1
    if all(p in mp and e - 1 in mp for p, e in pos): cdr[r.q] = (r.t.split("|")[1], "".join(uniq[r.q][mp[p]:mp[e - 1] + 1] for p, e in pos))
bad = {}
for c in "HL":
    held = rfa(G + f"heldout_cdr_{c}.fasta"); q = {k: s for k, (cc, s) in cdr.items() if cc == c}
    for k, s in q.items():
        if s in held.values(): bad[k] = 1.0
    wfa(q, f"v3_q_{c}.fasta")
    mm("easy-search", f"v3_q_{c}.fasta", G + f"heldout_cdr_{c}.fasta", f"v3_hits_{c}.tsv", tmp + f"/{c}", "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
       "--alignment-mode", "3", "--max-seqs", "10000", "-s", "7.5", "--format-output", "query,target,fident")
    for l in open(f"v3_hits_{c}.tsv"):
        k, t, f = l.split(); f = float(f)
        if f >= .7: bad[k] = max(bad.get(k, 0), f)
pf = pd.read_parquet(P + "prefiltered_v1.parquet", columns=["id", "pdb_id", "cluster_id", "hR", "hL", "contains_antibody", "contains_antigen"]).set_index("id")
v2 = pd.read_csv("pinder_heldout_exclusions_v2.tsv", sep="\t", dtype=str).fillna("")
add = []
for f in sorted(glob.glob(R + "pretraining/pinder-pkai-v1/index/label_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["status"] != "accepted" or r["id"] in set(v2.id): continue
        p = pf.loc[r["id"]]
        if not (p.contains_antibody and p.contains_antigen): continue
        hits = [(h, bad[h]) for h in (p.hR, p.hL) if h in isab and h in bad]
        if hits: add.append(dict(id=r["id"], pdb_id=p.pdb_id, cluster_id=p.cluster_id, split=r["split"], kind=r["kind"], ctype=r["ctype"], rules="cdr_heldout_70",
                                 experimental_hits="", fragment_hits="", ab_path_rules="cdr_heldout_70:" + ";".join(f"{h}:{v:.3f}" for h, v in hits)))
v3 = pd.concat([v2, pd.DataFrame(add)]).sort_values("id"); v3.to_csv("pinder_heldout_exclusions_v3.tsv", sep="\t", index=False)
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
meta = dict(version="pinder-heldout-exclusions-v3", sha256=h("pinder_heldout_exclusions_v3.tsv"), supersedes="pinder-heldout-exclusions-v2", v2_sha256=h("pinder_heldout_exclusions_v2.tsv"),
            complexes_excluded=len(v3), added_cdr_heldout_70=len(add), clusters_touched=int(v3.cluster_id.nunique()), abag_excluded=int((v3.kind == "Ab/Ag").sum()),
            heldout_cdr_sets={c: dict(path=G + f"heldout_cdr_{c}.fasta", sha256=h(G + f"heldout_cdr_{c}.fasta")) for c in "HL"},
            rule="v2 + Ab/Ag dimers whose antibody chain concatenated CDRs are >= 0.70 identical (same chain type) to the complete held-out CDR sets")
json.dump(meta, open("exclusions_v3.json", "w"), indent=1); print(json.dumps(meta, indent=1))
