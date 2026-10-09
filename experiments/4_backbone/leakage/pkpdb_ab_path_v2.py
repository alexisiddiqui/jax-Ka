"""v2: held-out CDRs from the complete sets (audits/ab-cdr-gap-v1/heldout_cdr_{H,L}.fasta), which add held-out antibodies
without SAbDab CDR annotations; overlap entries from revisions/00/audit.json. Writes pkpdb_ab_path_v2.tsv; entries released here but not by v1 (alignment-borderline CDRs) were then set to withheld_unstable, so v2 releases only entries both runs release.
pKPDB with the antibody path (PINDER pinder-exp-leak-v1 v2): which pkpdb-full-v1 entries rejected for sequence_overlap
would be released if antibody chains were checked by CDRs instead of whole-chain identity.
Offending chains per entry, from the build's own three sources: (a) sequence/batch-*/excluded.json (benchmark 90% / 80%
of shorter; experimental 30% / 80% both), (b) heldout_70 re-read from batch hits.tsv (owners by exact sequence via
seq-overlap-v1/pkpdb_entities.tsv), (c) seq-overlap-v1 pkpdb_heldout_exclusions_70 (pkpdb_vs_heldout.tsv).
Antibody chain = aligns to a SAbDab-annotated V domain with CDRs transferable (as pinder-prefilter-v1). It passes when its
concatenated CDRs are < 70% identical (same chain type) to every held-out CDR set (prefilter cdr_r_*.fasta) and every
experimental antibody CDR set (pinder-exp-leak-v1 cdrr_*.fasta). An entry is released iff every offending chain is an
antibody chain that passes. Variant 'with_antigen': additionally require a non-antibody chain in the entry (Ab/Ag-like)."""
import json, glob, os, hashlib, subprocess, tempfile, collections, sys
import pandas as pd
R = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/"; B = R + "pretraining/pkpdb-full-v1/"; P = R + "audits/pinder-prefilter-v1/"
X = R + "audits/pinder-exp-leak-v1/"; S = R + "audits/seq-overlap-v1/"; MM = R + "audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs"
sys.path.insert(0, "/home/coulson/oc/lina4225/jax-Ka/src"); from pkabench.runtime import config_hash
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
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
def rd(p, n):
    try: return pd.read_csv(p, sep="\t", names=n)
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=n)
refs = json.load(open(B + "references.json"))["references"]
audit = json.load(open(B + "revisions/00/audit.json")); overlap = {a["pdb_id"] for a in audit if a.get("reason") == "sequence_overlap"}
pk = pd.read_csv(S + "pkpdb_entities.tsv", sep="\t", names=["pdb", "entity", "seq"], dtype=str)
ent = pk.groupby("pdb").seq.apply(set).to_dict(); owners = pk.groupby("seq").pdb.apply(set).to_dict()
off = collections.defaultdict(set); unmapped = set()   # pdb -> offending sequences
for d in sorted(glob.glob(B + "sequence/batch-*")):
    q = rfa(d + "/queries.fasta")
    for pdb, hs in json.load(open(d + "/excluded.json")).items():
        for h in hs: off[pdb].add(q[h["query"]])
    for l in open(d + "/hits.tsv"):
        qq, t, i, qc, tc = l.split()
        if "benchmark" in refs[t]["kinds"] and float(i) >= .7 and min(float(qc), float(tc)) >= .8:
            for pdb in owners.get(q[qq], ()): off[pdb].add(q[qq])
            if q[qq] not in owners: unmapped.add(qq)
sh = {"q" + hashlib.sha1(s.encode()).hexdigest()[:16]: s for s in owners}
for h in set(rd(S + "pkpdb_vs_heldout.tsv", ["h", "t", "f"]).h):
    for pdb in owners.get(sh.get(h), ()): off[pdb].add(sh[h])
missing = overlap - set(off)
print(f"sequence_overlap entries {len(overlap):,}; with offending chains found {len(overlap & set(off)):,}; none found {len(missing)}; unmapped 70% queries {len(unmapped)}")
# antibody chains among offending sequences and all chains of those entries
allseq = {s for p in overlap for s in ent.get(p, ())} | {s for p in overlap for s in off.get(p, ())}
key = {"k" + config_hash(s)[:20]: s for s in allseq}; wfa(key, "v2_chains.fasta")
mm("easy-search", "v2_chains.fasta", P + "ref_V.fasta", "v2_ab_align.tsv", tmp + "/al", "--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "1",
   "--format-output", "query,target,fident,bits,qstart,qend,tstart,tend,qaln,taln")
refcdr = {}
for c in json.load(open(R + "universe/combined-split-v1/index.json"))["candidates"]:
    sab = c.get("sabdab")
    if not sab: continue
    for chn, V, names in (("H", "VH", ("CDR-H1", "CDR-H2", "CDR-H3")), ("L", "VL", ("CDR-L1", "CDR-L2", "CDR-L3"))):
        v = (sab.get(V) or "").strip().upper(); cd = [(sab.get(n) or "").strip().upper() for n in names]
        if not v or not all(cd): continue
        pos = []; p0 = 0
        for qs in cd:
            j = v.find(qs, p0)
            if j < 0: pos = None; break
            pos.append((j, j + len(qs))); p0 = j + len(qs)
        if pos is not None: refcdr[f"{c['complex_id']}|{chn}"] = pos
al = rd("v2_ab_align.tsv", ["q", "t", "f", "bits", "qs", "qe", "ts", "te", "qa", "ta"]).sort_values("bits", ascending=False).drop_duplicates("q")
isab = set(al.q); cdr = {}
for r in al.itertuples():
    pos = refcdr.get(r.t)
    if pos is None: continue
    qi = r.qs - 1; ti = r.ts - 1; mp = {}
    for a_, b_ in zip(r.qa, r.ta):
        if a_ != "-" and b_ != "-": mp[ti] = qi
        if a_ != "-": qi += 1
        if b_ != "-": ti += 1
    if all(p in mp and e - 1 in mp for p, e in pos): cdr[r.q] = (r.t.split("|")[1], "".join(key[r.q][mp[p]:mp[e - 1] + 1] for p, e in pos))
bad = set()
for chn in ("H", "L"):
    q = {k: s for k, (c, s) in cdr.items() if c == chn}
    ref = {**{"h" + k: v for k, v in rfa(R + f"audits/ab-cdr-gap-v1/heldout_cdr_{chn}.fasta").items()}, **{"e" + k: v for k, v in rfa(X + f"cdrr_{chn}.fasta").items()}}
    wfa(q, f"v2_cdrq_{chn}.fasta"); wfa(ref, f"v2_cdrref_{chn}.fasta"); bad |= {k for k, s in q.items() if s in ref.values()}
    mm("easy-search", f"v2_cdrq_{chn}.fasta", f"v2_cdrref_{chn}.fasta", f"v2_cdr_hits_{chn}.tsv", tmp + f"/c{chn}", "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
       "--alignment-mode", "3", "--max-seqs", "10000", "-s", "7.5", "--format-output", "query,target,fident")
    h = rd(f"v2_cdr_hits_{chn}.tsv", ["q", "t", "f"]); bad |= set(h[h.f >= 0.7].q)
inv = {s: k for k, s in key.items()}
status = {}
for p in sorted(overlap):
    o = off.get(p)
    if not o: status[p] = "no_offender_found"; continue
    ks = [inv[s] for s in o]; nonab = [k for k in ks if k not in isab]; untr = [k for k in ks if k in isab and k not in cdr]
    if nonab: status[p] = "nonantibody_chain"
    elif untr: status[p] = "cdr_untransferred"
    elif any(k in bad for k in ks): status[p] = "cdr_similar"
    else:
        antigen = any(inv[s] not in isab for s in ent.get(p, ()))
        status[p] = "released" if antigen else "released_antibody_only"
st = pd.Series(status, name="status"); st.index.name = "pdb"; st.to_csv("pkpdb_ab_path_v2.tsv", sep="\t")
ab_entries = {p for p in overlap if any(inv[s] in isab for s in off.get(p, ()))}
print(json.dumps(dict(antibody_chains=len(isab), with_cdrs=len(cdr), cdr_similar_chains=len(bad), overlap_entries_with_antibody_offender=len(ab_entries),
                      status=st.value_counts().to_dict()), indent=1))
