"""Read frozen jax-Ka memberships and canonical sequences without modifying datasets."""
import os,json,csv,hashlib,collections,concurrent.futures,gzip
from pathlib import Path
import pyarrow.parquet as pq
R=Path(os.environ['PKABENCH_RUNTIME']);O=R/'audits/atlas-v1';I=O/'references'
seqs={}; datasets=collections.defaultdict(dict); provenance={};missing=[]
def key(s):
 s=''.join(s.split()).upper(); k='s'+hashlib.sha256(s.encode()).hexdigest()[:24];seqs[k]=s;return k
def fasta(p):
 out={};k=None
 for line in p.read_text().splitlines():
  if line.startswith('>'):k=line[1:];out[k]=''
  elif k is not None:out[k]+=line.strip()
 return out
def add(d,c,ss):
 if not ss:missing.append((d,c));return
 datasets[d][c]=[key(s) for s in ss]
def source(rel):
 p=R/rel
 if not p.exists():p=R/'overlay'/rel
 return p
pf=pq.read_table(I/'prefiltered_v1.parquet',columns=['id','hR','hL']).to_pylist(); f=fasta(I/'uniq.fasta'); chains={r['id']:[f[r['hR']],f[r['hL']]] for r in pf}
for path in sorted((R/'pretraining/pinder-pkai-v1/index').glob('label_*.jsonl')):
 for line in path.open():
  r=json.loads(line)
  if r.get('status')=='accepted':add('PINDER_accepted',r['id'],chains.get(r['id'],[]))
for ds in ['pinder','pkpdb']:
 for suffix,split in [('', 'train'),('-val','val')]:
  rows=list(csv.DictReader((R/f'training/pool-v4/{ds}{suffix}.tsv').open(),delimiter='\t'))
  if ds=='pinder':
   for r in rows:add('PINDER_'+split,r['id'],chains[r['id']])
  else:
   def load(r):
    j=json.loads(source(f'pretraining/pkpdb-full-v1/entries/{r["id"]}/defects.json').read_text())
    return r,[s['sequence'] for s in j['sequences']]
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    for r,ss in ex.map(load,rows):
     add('pKPDB_'+split,r['id'],ss)
     if r.get('stratum')!='monomer':add('pKPDB_complex_'+split,r['id'],ss)
for r in json.loads((R/'training/ogqt-pinder-factorial-v1/cohort.json').read_text())['records']:
 if r['split']=='val':add('PINDER_retired_val400',r['id'],chains[r['id']])
# Full pKPDB source is comparison only, not the selected training-chain definition.
entities=collections.defaultdict(list)
with (I/'pkpdb_entities.tsv').open() as f:
 for row in csv.reader(f,delimiter='\t'):
  if len(row)>=3:entities[row[0]].append(row[2])
for r in json.loads((R/'pretraining/pkpdb-full-v1/pilot.json').read_text())['records']:
 add('pKPDB_accepted_source_entities',r['pdb_id'],entities.get(r['pdb_id'],[]))
assign={r['complex_id']:r for r in pq.read_table(I/'assignments.parquet').to_pylist()}
idx=json.loads((I/'benchmark-index.json').read_text());bm={r['complex_id']:r for r in idx['candidates']}
for cid,r in bm.items():
 ss=[c['sequence'] for c in r['chains']];add('benchmark_all',cid,ss)
 a=assign.get(cid,{})
 if a.get('split'):add('benchmark_'+a['split'],cid,ss)
 if a.get('reference_reserved') or a.get('set2_protein_g_fc_reserved'):add('benchmark_reserved',cid,ss)
for path in (R/'training/shared-v4-float32/records').glob('*.json'):
 r=json.loads(path.read_text());cid=r['complex_id']
 if r.get('split')=='val':add('benchmark_active_val142',cid,[c['sequence'] for c in bm[cid]['chains']])
held=collections.defaultdict(list)
for k,s in fasta(I/'ref_heldout_all.fasta').items():held[k.split('|')[0]].append(s)
for cid,ss in held.items():add('heldout_all',cid,ss)
refs=json.loads((R/'pretraining/pkpdb-full-v1/references.json').read_text())
(I/'pkpdb-references.json').write_text(json.dumps(refs))
for rid,r in refs['references'].items():
 for kind in r['kinds']:add('reference_'+kind,rid,[r['sequence']])
# Save CDRs separately for transparent conservative whole-chain results.
summary={d:dict(structures=len(rows),unique_sequences=len({k for ks in rows.values() for k in ks}),chain_instances=sum(map(len,rows.values()))) for d,rows in datasets.items()}
(O/'reference_membership.json').write_text(json.dumps(datasets))
(O/'references.fasta').write_text(''.join('>'+k+'\n'+v+'\n' for k,v in sorted(seqs.items())))
(O/'reference_summary.json').write_text(json.dumps(dict(datasets=summary,missing=missing,unique_sequences=len(seqs)),indent=2))
print(json.dumps(summary,indent=2),flush=True)
assert not missing,missing[:5]
