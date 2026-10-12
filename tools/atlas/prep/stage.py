import os,json,csv,subprocess,hashlib,random,collections
from pathlib import Path
R=Path(os.environ['PKABENCH_RUNTIME']); A=R/'audits/atlas-v1'; S=R/'pretraining/atlas-high-confidence-v1'; O=R/'pretraining/atlas-prepared-v1'; O.mkdir(exist_ok=True)
assert os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_CPUS_PER_TASK'])==72

def fasta(p):
 d={};k=None
 for l in p.read_text().splitlines():
  if l.startswith('>'):k=l[1:];d[k]=''
  elif k:d[k]+=l.strip()
 return d

def writefa(p,d):p.write_text(''.join(f'>{k}\n{s}\n' for k,s in d.items()))
seq=fasta(A/'references.fasta'); members=json.loads((A/'reference_membership.json').read_text()); keys=set()
for d,rows in members.items():
 if d in ('PINDER_val','pKPDB_val','benchmark_val','benchmark_test','heldout_all','benchmark_reserved'):
  keys.update(k for rr in rows.values() for k in rr)
writefa(O/'heldout.fasta',{k:seq[k] for k in sorted(keys)})
v={}
for r in json.loads((A/'references/benchmark-index.json').read_text())['candidates']:
 sab=r.get('sabdab') or {}
 for t in ('VH','VL'):
  s=(sab.get(t) or '').strip().upper()
  if s:v[r['complex_id']+'|'+t]=s
assert v
writefa(O/'antibody-domains.fasta',v)
mm=A/'tools/mmseqs/bin/mmseqs'
for name,target,opts in [('fragment',O/'heldout.fasta',['--min-seq-id','0.9','-c','0']),('antibody',O/'antibody-domains.fasta',['--min-seq-id','0.4','-c','0.8','--cov-mode','1'])]:
 hits=O/f'{name}-hits.tsv'
 if not hits.exists():
  cmd=[str(mm),'easy-search',str(S/'metadata/chains.fasta'),str(target),str(hits),os.environ['TMPDIR']+'/'+name,'--threads','72','--alignment-mode','3','--max-seqs','100000','-s','7.5','-e','1000000','--format-output','query,target,fident,qcov,tcov',*opts]
  with (O/f'{name}.log').open('w') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
frag=set(); ab=set()
for l in (O/'fragment-hits.tsv').open():
 q,t,i,qc,tc=l.split()
 if float(i)>=.9 and max(float(qc),float(tc))>=.8:frag.add(q)
for l in (O/'antibody-hits.tsv').open():ab.add(l.split()[0])
qc=set(json.loads((S/'dataset.json').read_text()).get('source_qc_excluded_ids',[])); rows=json.loads((S/'metadata/atlas-high-confidence.json').read_text()); screening={r['id']:r for r in csv.DictReader((S/'metadata/atlas-screening.tsv').open(),delimiter='\t')}
elig=[]; decisions=[]
for r in rows:
 sc=screening[r['id']]; reasons=[]
 if sc['preliminary_holdout_flag'] in ('True','true','1'):reasons.append('whole_chain_holdout')
 if sum(r['lengths'])>1500:reasons.append('length')
 if r['id'] in qc:reasons.append('source_qc')
 if set(r['chains']) & frag:reasons.append('fragment_90')
 if set(r['chains']) & ab:reasons.append('antibody_pending_cdr')
 decisions.append(dict(id=r['id'],reasons=reasons))
 if not reasons:elig.append(r)
(O/'candidates.json').write_text(json.dumps(elig));(O/'decisions.json').write_text(json.dumps(decisions))
# Uniform reproducible random sample: keeps source/length proportions in expectation.
sample=random.Random(20261010).sample(elig,min(288,len(elig)))
(O/'benchmark-sample.json').write_text(json.dumps(sample))
meta=dict(staged_separate=True,training_enabled=False,source=str(S),raw_count=len(rows),candidates=len(elig),reasons=dict(collections.Counter(x for r in decisions for x in r['reasons'])),antibody_policy='Conservative quarantine of V-domain matches; no antibody release before CDR adjudication',fragment_policy='90% identity and 80% coverage of either sequence against frozen/current heldout union',benchmark_sample=len(sample),sample_seed=20261010)
(O/'stage.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
