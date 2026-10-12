"""Pack the admitted full ATLAS graphs; preserve published portion membership."""
import os,json,csv,hashlib,collections,shutil
from pathlib import Path
from pkatrain import production_graphs as pg
from pkatrain.loading import BucketPolicy,PRODUCTION_BOUNDS
from pkabench.runtime import atomic_json,digest
assert os.environ.get('SLURM_JOB_ID')
R=Path(os.environ['PKABENCH_RUNTIME']);O=R/'pretraining/atlas-prepared-v1/graph-v1';PORTIONS=R/'audits/atlas-full-portions-v1';out=O/'release-v1';out.mkdir(exist_ok=True)
def tsvread(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
results=json.loads((O/'build-results.json').read_text());assert len(results)==155677 and not any(r['status']=='error' for r in results)
accepted={r['id']:dict(r,split=r['source_split']) for r in results if r['status']=='accepted'}
source=tsvread(PORTIONS/'train-100.tsv');valsource=tsvread(PORTIONS/'validation.tsv')
assert len(source)==153677 and len(valsource)==2000
# Reproduce the previously published nested metadata ordering.
seed='atlas-full-portions-v1|20261012';rank=lambda r:hashlib.sha256((seed+'|'+r['id']).encode()).hexdigest()
strata=collections.defaultdict(list)
for r in source:strata[r['stratum']].append(r)
ranked=[]
for group in strata.values():
 for i,r in enumerate(sorted(group,key=rank)):ranked.append(((i+.5)/len(group),rank(r),r))
ordered=[r for _,__,r in sorted(ranked)]
train=[]
for i,r in enumerate(ordered,1):
 if r['id'] in accepted:
  row=accepted[r['id']];assert row['split']=='train';row.update(min_fraction=i/len(ordered),training_rank=i);train.append(row)
val=[accepted[r['id']] for r in valsource if r['id'] in accepted]
assert all(r['split']=='val' for r in val)
assert not {r['component_id'] for r in train}&{r['component_id'] for r in val}
allrows=train+val;assert len(allrows)==len(accepted)
for p in [10,50,75]:assert [r['id'] for r in tsvread(PORTIONS/f'train-{p:03d}.tsv')]==[r['id'] for r in ordered[:len(ordered)*p//100]]
fractions={}
for p in [10,50,75,100]:
 parent=PORTIONS/f'train-{p:03d}.tsv';shutil.copyfile(parent,out/parent.name);requested=tsvread(parent);ids={r['id'] for r in requested};active=[r for r in train if r['id'] in ids]
 with (out/f'train-{p:03d}-active.tsv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['id','component_id','stratum','min_fraction'],delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(active)
 assert {r['id'] for r in train if r['min_fraction']<=p/100+1e-12}=={r['id'] for r in active}
 fractions[str(p)]=dict(requested=len(requested),active=len(active),inactive=len(requested)-len(active),membership_sha256=digest(out/parent.name),active_sha256=digest(out/f'train-{p:03d}-active.tsv'))
shutil.copyfile(PORTIONS/'validation.tsv',out/'validation.tsv')
with (out/'validation-active.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['id','component_id','stratum'],delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(val)
atomic_json(out/'inactive-records.json',[r for r in results if r['status']!='accepted'])
atomic_json(out/'ids.json',dict(ids=[[r['id'],r['split']] for r in allrows]))
pg.FIELDS['atlas']=pg.FIELDS['pinder'];fields=pg.FIELDS['atlas'];dtypes=allrows[0]['dtypes'];assert all(r['dtypes']==dtypes for r in allrows)
def blobs():
 for r in allrows:
  with open(r['shard_path'],'rb') as f:f.seek(r['blob'][0]);b=f.read(r['blob'][1])
  assert hashlib.sha256(b).hexdigest()==r['blob_sha256'];yield b
if not (out/'store-v2/verification.json').exists():pg._write_store(out,'atlas',allrows,dtypes,blobs(),{r['id']:r for r in results if r['status']!='accepted'},8)
# Smoke uses batch 2; production batching can be chosen after this technical gate.
policy=BucketPolicy(PRODUCTION_BOUNDS,(2,)*len(PRODUCTION_BOUNDS))
manifest=dict(version='atlas-graph-v1',dataset='atlas',store=str(out/'store-v2'),store_verification_sha256=digest(out/'store-v2/verification.json'),bucket_policy=policy.to_json(),capacities=policy.capacities(allrows),records=allrows,weighting='uniform',test_data_included=False,training_enabled=False)
atomic_json(out/'manifest.json',manifest)
summary=dict(scope='original full ATLAS frozen split, all source cohorts',source_count=len(results),accepted=len(allrows),inactive=len(results)-len(allrows),inactive_reasons=dict(collections.Counter(r.get('reason') for r in results if r['status']!='accepted')),training_requested=len(source),training_active=len(train),validation_requested=len(valsource),validation_active=len(val),fractions=fractions,shared_validation=True,nested=True,component_overlap=0,eligible_sites=sum(r['q'] for r in allrows),site_masks=dict(sum((collections.Counter(r['masked']) for r in results),collections.Counter())),cohorts=dict(collections.Counter(r['stratum'].split('|')[1] for r in allrows)),reused_graphs=sum(bool(r.get('reused_verified_record')) for r in allrows),source_inventory_sha256=digest(O/'source-inventory.json'),graph_policy_sha256=digest(O/'graph-policy.json'),training_started=False)
for p in [10,50,75,100]:
 atomic_json(out/f'config-{p:03d}.json',dict(dataset='atlas-graph-v1',manifest=str(out/'manifest.json'),training_fraction=p/100,training_membership=str(out/f'train-{p:03d}.tsv'),active_training_ids=str(out/f'train-{p:03d}-active.tsv'),validation_membership=str(out/'validation.tsv'),active_validation_ids=str(out/'validation-active.tsv'),requested_training_complexes=fractions[str(p)]['requested'],active_training_complexes=fractions[str(p)]['active'],source_adapter='pkatrain.production_loading.PinderSource',norms={'burial':1.,'interface':1.},weighting='uniform',training_started=False))
atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
