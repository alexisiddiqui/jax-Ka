import os,json,csv,hashlib,collections,subprocess,random,time
from pathlib import Path
R=Path(os.environ['PKABENCH_RUNTIME']);A=R/'audits/atlas-v1';O=R/'audits/atlas-split-v1';S=R/'pretraining/atlas-prepared-v1';MM=A/'tools/mmseqs/bin/mmseqs'
assert os.environ.get('SLURM_JOB_ID')
def read(p):return json.loads(p.read_text())
def fa(p):
 d={};k=None
 for l in p.read_text().splitlines():
  if l.startswith('>'):k=l[1:];d[k]=''
  elif k:d[k]+=l.strip()
 return d
def wfa(p,d):p.write_text(''.join(f'>{k}\n{s}\n' for k,s in sorted(d.items())))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def search(q,t,name):
 hit=O/(name+'.tsv');receipt=O/(name+'.search.json');command=[str(MM),'easy-search',str(q),str(t),str(hit),os.environ['TMPDIR']+'/'+name,'--threads','72','--min-seq-id','0.7','-c','0','--alignment-mode','3','--max-seqs','100000','-s','7.5','-e','0.001','--format-output','query,target,fident,qcov,tcov']
 signature=dict(query=sha(q),target=sha(t),command=command)
 if hit.exists() and receipt.exists():
  old=read(receipt);assert old['query']==signature['query'] and old['target']==signature['target'];return hit
 with (O/(name+'.log')).open('w') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
 receipt.write_text(json.dumps(signature,indent=2));return hit
def disallowed(i,q,t):return (i>=.7 and min(q,t)>=.8) or (i>=.9 and max(q,t)>=.8)
labelled=read(O/'labelled-index.json');rows={r['id']:r for r in read(S/'candidates.json') if r['id'] in labelled};assert len(rows)==155677
seq=fa(A/'atlas.fasta');refseq=fa(A/'references.fasta');refs=read(A/'reference_membership.json')
keys={k for r in rows.values() for k in r['chains']};seq={k:seq[k] for k in keys};wfa(O/'atlas-labelled.fasta',seq)
trainkeys={k for name in ('PINDER_train','pKPDB_train','benchmark_train') for ks in refs[name].values() for k in ks};trainseq={k:refseq[k] for k in trainkeys}
parent={k:k for k in keys}
def find(k):
 while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
 return k
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[max(a,b)]=min(a,b)
for r in rows.values():union(*r['chains'])
hits=search(O/'atlas-labelled.fasta',O/'atlas-labelled.fasta','atlas-self')
edgecount=0
for l in hits.open():
 q,t,i,qc,tc=l.split();i,qc,tc=map(float,(i,qc,tc))
 if q!=t and disallowed(i,qc,tc):union(q,t);edgecount+=1
protected=set(keys & trainkeys)
for filename,reverse in [('atlas_vs_references.tsv',False),('references_vs_atlas.tsv',True)]:
 for l in (A/filename).open():
  ss=l.split();q,t=ss[:2];i,qc,tc=map(float,ss[2:5])
  if reverse:q,t=t,q
  if q in keys and t in trainkeys and disallowed(i,qc,tc):protected.add(q)
# Screen the complete candidate pool against existing training before sampling.
wfa(O/'existing-training.fasta',trainseq)
for reverse in (False,True):
 q,t=O/'atlas-labelled.fasta',O/'existing-training.fasta'
 if reverse:q,t=t,q
 hit=search(q,t,'all-existing-'+('reverse' if reverse else 'forward'))
 for l in hit.open():
  qk,tk,i,qc,tc=l.split();i,qc,tc=map(float,(i,qc,tc))
  if reverse:qk,tk=tk,qk
  if disallowed(i,qc,tc):protected.add(qk)
print('Existing training screened; protected chains',len(protected),flush=True)
groups=collections.defaultdict(list)
for cid,r in rows.items():groups[find(r['chains'][0])].append(cid)
protected_groups={find(k) for k in protected};strata={cid:'|'.join([str(r['class']),str(r['type']),'<500' if sum(r['lengths'])<500 else '500-999' if sum(r['lengths'])<1000 else '1000-1500','pLDDT<80' if float(r['pLDDT'])<80 else 'pLDDT>=80']) for cid,r in rows.items()}

def select():
 eligible={g:ids for g,ids in groups.items() if g not in protected_groups and len(ids)<=100};eligible_ids=[cid for ids in eligible.values() for cid in ids]
 hist=collections.Counter(strata[c] for c in eligible_ids);target={s:2000*n/len(eligible_ids) for s,n in hist.items()};buckets=collections.defaultdict(list)
 for g,ids in eligible.items():buckets[collections.Counter(strata[c] for c in ids).most_common(1)[0][0]].append(g)
 for s in buckets:buckets[s].sort(key=lambda g:hashlib.sha256(('atlas-val-20261011|'+g).encode()).hexdigest(),reverse=True)
 selected=set();count=collections.Counter()
 while len(selected)<2000 and any(buckets.values()):
  s=max((s for s,b in buckets.items() if b),key=lambda s:(target[s]-count[s])/max(target[s],1))
  g=buckets[s].pop();ids=groups[g]
  if len(selected)+len(ids)>2000:continue
  selected.update(ids);count.update(strata[c] for c in ids)
 assert len(selected)>=1900,len(selected)
 return selected,eligible_ids
checks=[]
for iteration in range(15):
 val,eligible_ids=select();vkeys={k for cid in val for k in rows[cid]['chains']};otherkeys={k for cid,r in rows.items() if cid not in val for k in r['chains']}
 wfa(O/f'validation-{iteration}.fasta',{k:seq[k] for k in vkeys});wfa(O/f'training-{iteration}.fasta',{**trainseq,**{k:seq[k] for k in otherkeys}})
 bad=set(vkeys & (otherkeys|trainkeys));newedges=0
 for reverse in (False,True):
  q,t=(O/f'validation-{iteration}.fasta',O/f'training-{iteration}.fasta')
  if reverse:q,t=t,q
  hit=search(q,t,f'verify-fullscreen-{iteration}-'+('reverse' if reverse else 'forward'))
  for l in hit.open():
   qk,tk,i,qc,tc=l.split();i,qc,tc=map(float,(i,qc,tc))
   if reverse:qk,tk=tk,qk
   if disallowed(i,qc,tc):bad.add(qk)
 checks.append(dict(iteration=iteration,validation=len(val),flagged_chains=len(bad)));(O/'clearance-progress.json').write_text(json.dumps(checks,indent=2));print('CLEARANCE',checks[-1],flush=True)
 if not bad:break
 protected_groups.update(find(k) for k in bad)
else:raise RuntimeError('Validation clearance failed after 15 selections')
assert not ({find(rows[c]['chains'][0]) for c in val} & {find(r['chains'][0]) for c,r in rows.items() if c not in val})
manifest=[]
for cid,r in sorted(rows.items()):manifest.append(dict(id=cid,split='val' if cid in val else 'train',sequence_component=find(r['chains'][0]),stratum=strata[cid],**{k:r[k] for k in ('chains','lengths','class','type','pLDDT','ipTM','LIS','LIA')},archive=labelled[cid]))
# Freeze a planned split; never mutate pool-v4 or admit records for training.
release=S/'split-v1';release.mkdir(exist_ok=True)
content=json.dumps(manifest,separators=(',',':'))
if (release/'manifest.json').exists():assert (release/'manifest.json').read_text()==content,'Frozen split differs'
else:(release/'manifest.json').write_text(content)
for split in ('train','val'):
 with (release/f'{split}.tsv').open('w') as f:
  wr=csv.writer(f,delimiter='\t');wr.writerow(['id','sequence_component','stratum'])
  wr.writerows((r['id'],r['sequence_component'],r['stratum']) for r in manifest if r['split']==split)
summary=dict(status='frozen_membership_pending_training_release_QC',train=len(rows)-len(val),val=len(val),labelled=len(rows),components=len(groups),validation_components=len({find(rows[c]['chains'][0]) for c in val}),validation_component_cap=100,largest_validation_component=max(collections.Counter(find(rows[c]['chains'][0]) for c in val).values()),largest_components=sorted(map(len,groups.values()),reverse=True)[:20],train_protected_components=len(protected_groups),eligible_validation_population=len(eligible_ids),sequence_edges=edgecount,verification=checks,thresholds='>=70% identity and >=80% both-chain coverage OR >=90% identity and >=80% either-chain coverage; either matched partner connects complexes',validation_existing_training='PINDER_train + pKPDB_train + frozen benchmark_train',seed='atlas-val-20261011',strata=['source class','homo/heterodimer','length bins','pLDDT bins'],histograms={s:dict(collections.Counter(strata[c] for c in rows if (c in val)==(s=='val'))) for s in ('train','val')},training_enabled=False,excluded_unlabelled=dict(rejected=1953,pipeline_errors=143),source_hashes={'candidates':sha(S/'candidates.json'),'labelled_index':sha(O/'labelled-index.json'),'reference_membership':sha(A/'reference_membership.json'),'split_script':sha(Path(__file__))},manifest_sha256=sha(release/'manifest.json'),limitations=['MMseqs2 sensitive search is not a proof of absence of remote homology','Existing training membership unchanged','Final site masks, weights and graph QC remain a release gate; failures must be dropped without moving records between splits'])
(release/'summary.json').write_text(json.dumps(summary,indent=2));(O/'split-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='histograms'},indent=2),flush=True)
