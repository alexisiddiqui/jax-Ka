"""Freeze the full input inventory and a stratified graph-build pilot on compute."""
import os,json,csv,collections,hashlib,random
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
R=Path(os.environ['PKABENCH_RUNTIME']);P=R/'pretraining/atlas-prepared-v1';O=P/'graph-v1';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta={x['id']:x for x in json.loads((P/'candidates.json').read_text())};rows=[]
for split in ['train','val']:
 for x in csv.DictReader((P/'split-v1'/f'{split}.tsv').open(),delimiter='\t'):
  rows.append(dict(x,split=split,source_metadata=meta[x['id']]))
assert len(rows)==155677 and len({r['id'] for r in rows})==len(rows)
index={}
for file in sorted((P/'bulk-v1/receipts').glob('*.json')):
 receipt=json.loads(file.read_text())
 for m in receipt['members']:index[m['id']]=dict(m,archive=str(P/'bulk-v1/shards'/f"{receipt['batch']:04d}.tar"))
assert all(x['id'] in index for x in rows)
rows=[dict(x,archive=index[x['id']]) for x in rows]
path=O/'source-inventory.json';data=json.dumps(rows)
if path.exists():assert path.read_text()==data
else:path.write_text(data)
groups=collections.defaultdict(list)
for r in rows:groups[r['split'],r['stratum']].append(r)
pilot=[]
for group in groups.values():
 ranked=sorted(group,key=lambda r:hashlib.sha256(('atlas-graphs-pilot|'+r['id']).encode()).hexdigest())
 pilot.extend(ranked[:min(8,len(ranked))])
(O/'pilot-inventory.json').write_text(json.dumps(pilot))
code=Path(__file__).parent
sources={str(P/'split-v1'/name):sha(P/'split-v1'/name) for name in ['train.tsv','val.tsv','manifest.json']}
report=dict(source_count=len(rows),train=sum(r['split']=='train' for r in rows),validation=sum(r['split']=='val' for r in rows),cohorts=dict(collections.Counter(r['source_metadata']['type'] for r in rows)),pilot=len(pilot),free_bytes=os.statvfs(O).f_bavail*os.statvfs(O).f_frsize,source_hashes=sources,code_hashes={p.name:sha(p) for p in code.glob('*.py')},training_started=False)
(O/'inventory-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
