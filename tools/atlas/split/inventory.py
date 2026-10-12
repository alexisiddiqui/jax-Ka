import os,json,collections,hashlib,csv
from pathlib import Path
R=Path(os.environ['PKABENCH_RUNTIME']);A=R/'audits/atlas-v1';O=R/'audits/atlas-split-v1';O.mkdir(exist_ok=True)
B=R/'pretraining/atlas-prepared-v1/bulk-v1';members={};errors=[];reject=[]
for p in sorted((B/'receipts').glob('*.json')):
 r=json.loads(p.read_text())
 for m in r['members']:members[m['id']]={**m,'shard':str(B/'shards'/f'{r["batch"]:04d}.tar')}
 errors.extend(x for x in r['preparation']+r['labelling'] if x['status']=='error');reject.extend(x for x in r['preparation'] if x['status']=='rejected')
(O/'labelled-index.json').write_text(json.dumps(members));(O/'errors.json').write_text(json.dumps(errors,indent=2))
meta=dict(labelled=len(members),errors=dict(collections.Counter(x.get('reason','') for x in errors)),rejected=dict(collections.Counter(x.get('reason','') for x in reject)),error_examples=errors[:5],images=[p.name for p in (R/'images').glob('*')],benchmark_files=[str(p) for p in (R/'training/shared-v4-float32/records').glob('*.json')][:2])
for p in meta['benchmark_files']:meta.setdefault('benchmark_examples',[]).append(json.loads(Path(p).read_text()))
# Assert that the earlier reference membership still describes the active pool-v4 manifests.
refs=json.loads((A/'reference_membership.json').read_text());checks={}
for name,file in [('PINDER_train','pinder.tsv'),('PINDER_val','pinder-val.tsv'),('pKPDB_train','pkpdb.tsv'),('pKPDB_val','pkpdb-val.tsv')]:
 p=R/'training/pool-v4'/file;ids={r['id'] for r in csv.DictReader(p.open(),delimiter='\t')};checks[name]=dict(same=ids==set(refs[name]),count=len(ids),sha256=hashlib.sha256(p.read_bytes()).hexdigest());assert checks[name]['same'],name
meta['pool_checks']=checks
(O/'inventory.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ('benchmark_examples','error_examples')},indent=2))
