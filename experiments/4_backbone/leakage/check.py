"""Would PINDER pass pKPDB's extra leakage rules? (1) experimental reservations: >= 30% identity, >= 80% coverage of both
vs the 337 experimental reference sequences in pkpdb references.json; (2) exact reserved PDB IDs (2,157); (3) the
fragment rule: >= 90% identity over >= 80% of the shorter sequence vs benchmark references. Reports affected labelled
PINDER complexes, clusters and usable labelled interface complexes."""
import json, glob, subprocess, os, tempfile, collections
R = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/"; MM = R + "audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs"
refs = json.load(open(R + "pretraining/pkpdb-full-v1/references.json"))
def wfa(d, p):
    with open(p, "w") as f:
        for k, v in d.items(): f.write(f">{k}\n{v}\n")
exp = {k: v["sequence"] for k, v in refs["references"].items() if "experimental" in v["kinds"]}
bench = {k: v["sequence"] for k, v in refs["references"].items() if "benchmark" in v["kinds"]}
wfa(exp, "exp.fasta"); wfa(bench, "bench.fasta"); tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
q = R + "audits/seq-overlap-v1/pinder.fasta"
if not os.path.exists("exp_hits.tsv"): subprocess.run([MM, "easy-search", q, "exp.fasta", "exp_hits.tsv", tmp + "/e", "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "-e", "1000000",
                "--alignment-mode", "3", "--exhaustive-search", "1", "--threads", "4", "--format-output", "query,target,fident,qcov,tcov"], check=True, stdout=subprocess.DEVNULL)
if not os.path.exists("frag_hits.tsv"): subprocess.run([MM, "easy-search", q, "bench.fasta", "frag_hits.tsv", tmp + "/b", "--min-seq-id", "0.9", "-c", "0", "-e", "1000000",
                "--alignment-mode", "3", "--threads", "4", "--format-output", "query,target,fident,qcov,tcov"], check=True, stdout=subprocess.DEVNULL)
exp_bad = {l.split()[0] for l in open("exp_hits.tsv")}
frag_bad = {l.split()[0] for l in open("frag_hits.tsv") if max(float(l.split()[3]), float(l.split()[4])) >= .8}
reserved = {p.lower() for p in refs["reserved_pdb_ids"]}
import pandas as pd
pf = pd.read_parquet(R + "audits/pinder-prefilter-v1/prefiltered_v1.parquet", columns=["id", "pdb_id", "cluster_id", "hR", "hL"]).set_index("id")
O = R + "pretraining/pinder-pkai-v1/"; c = collections.Counter(); cl = collections.defaultdict(set)
for f in glob.glob(O + "index/label_*.jsonl"):
    for l in open(f):
        r = json.loads(l); row = pf.loc[r["id"]]; hs = {row.hR, row.hL}
        flags = dict(experimental_30=bool(hs & exp_bad), reserved_pdb_id=row.pdb_id.lower() in reserved, fragment_90=bool(hs & frag_bad))
        flags["any"] = any(flags.values()); c["complexes"] += 1
        usable = any(s["train_mask"] and s["interface"] for s in json.load(open(O + "entries/" + r["id"] + "/sites.json"))) if flags["any"] else None
        for k, v in flags.items():
            if v: c[k] += 1; cl[k].add(row.cluster_id); c[k + "_with_usable_interface"] += bool(usable)
print(dict(c)); print({k: len(v) for k, v in cl.items()})
coh = json.load(open(R + "training/ogqt-pinder-factorial-v1/cohort.json"))
hit = collections.Counter()
for rec in coh["records"]:
    row = pf.loc[rec["id"]]; hs = {row.hR, row.hL}
    if hs & exp_bad or row.pdb_id.lower() in reserved or hs & frag_bad: hit[rec["split"]] += 1
print("factorial cohort complexes flagged:", dict(hit), "of", collections.Counter(r["split"] for r in coh["records"]))
