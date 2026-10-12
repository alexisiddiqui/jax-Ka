"""Real production loader plus existing paired oGQT optimizer, full ATLAS smoke."""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
import jax
from pkatrain.production_loading import PinderSource,select
from pkatrain.loading import BucketPolicy,LoaderConfig
from pkatrain.gqt_paired_pinder import PairedEngine
from pkanet.ogqt import initialize
from pkabench.runtime import atomic_json,digest
assert os.environ.get('SLURM_JOB_ID') and jax.default_backend()=='gpu'
O=Path(os.environ['PKABENCH_RUNTIME'])/'pretraining/atlas-prepared-v1/graph-v1/release-v1';m=json.loads((O/'manifest.json').read_text());policy=BucketPolicy.from_json(m['bucket_policy'])
src=PinderSource(m,config=LoaderConfig(workers=8),norms={'burial':1.,'interface':1.});train=select(m,'train');val=select(m,'val');rng=np.random.default_rng(20261012)
# All occupied buckets: actual production padding, original-array equality, index bounds, padded example masks.
checks=[]
for bucket in m['capacities']:
 rows=[r for r in m['records'] if policy.bucket(r['n'])==bucket];r=max(rows,key=lambda r:r['n']*r['k']+r['s']*r['sk']);batch=src.load([r['id']]);g,t,mask,wb,wi,meta,valid=batch;raw=src.store.raw(r['id'])
 assert valid.tolist()==[True,False] and not mask[1].any() and not t[1].any() and not wb[1].any() and not wi[1].any()
 np.testing.assert_array_equal(t[0,:,:r['q']],raw['targets']);assert mask[0].sum()==r['q'];assert not mask[0,r['q']:].any()
 for field,limit in [('neighbors',g['nodes'].shape[-2]),('site_neighbors',g['site_type'].shape[-1]),('site_residue',g['nodes'].shape[-2]),('query_residue',g['nodes'].shape[-2]),('query_site',g['site_type'].shape[-1])]:assert ((g[field]>=0)&(g[field]<limit)).all()
 for x in jax.tree.leaves((g,t,mask,wb,wi)):assert np.isfinite(x).all()
 checks.append(dict(bucket=bucket,id=r['id'],passed=True))
params=initialize(jax.random.PRNGKey(17),width=44,ff=88);engine=PairedEngine(params,'vanilla');state=engine.optimizer.init(params)
# Three optimizer steps in a middle-sized bucket, then every active fixed validation record.
groups={b:[r for r in train if policy.bucket(r['n'])==b] for b in m['capacities']};bucket=min((b for b in groups if groups[b]),key=lambda b:abs(int(b)-512));ids=[r['id'] for r in sorted(groups[bucket],key=lambda r:r['id'])[:2]];batch=src.load(ids);g,t,mask,wb,wi,meta,valid=batch
losses=[];timings=[]
for step in range(3):
 tick=time.time();params,state,loss,finite=engine.step(params,state,g,t,mask,wb,wi,valid,1e-3);assert bool(finite);losses.append(float(loss));timings.append(time.time()-tick)
 print(json.dumps(dict(stage='smoke_train',step=step,loss=losses[-1],seconds=timings[-1])),flush=True)
assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))
large_bucket=max(groups,key=int);large=max(groups[large_bucket],key=lambda r:r['n']*r['k']+r['s']*r['sk']);batch=src.load([large['id']]);g,t,mask,wb,wi,meta,valid=batch
large_start=time.time();_,_,large_loss,large_finite=engine.step(params,state,g,t,mask,wb,wi,valid,1e-3);assert bool(large_finite)
large_check=dict(id=large['id'],bucket=large_bucket,loss=float(large_loss),seconds=time.time()-large_start,finite_gradients=True)
print(json.dumps(dict(stage='largest_bucket_gradient',**large_check)),flush=True)
validation=[]
for batch_number,ids in enumerate(policy.plans(val,rng),1):
 batch=src.load(ids);g,t,mask,wb,wi,meta,valid=batch;tick=time.time();components=engine.components(params,g,t,mask,wb,wi,valid);v=[float(x) for x in components];assert np.isfinite(v).all();validation.append(dict(ids=ids,state_mse=v[0],paired_mse=v[1],seconds=time.time()-tick));
 if batch_number%100==0:print(json.dumps(dict(stage='smoke_validation',batches=batch_number,complexes=sum(len(x['ids']) for x in validation))),flush=True)
# One valid example with/without a masked batch member must produce the same objective.
batch=src.load(ids[:1]);g,t,mask,wb,wi,meta,valid=batch
full=engine.components(params,g,t,mask,wb,wi,valid);single=engine.components(params,jax.tree.map(lambda x:x[:1],g),t[:1],mask[:1],wb[:1],wi[:1],valid[:1]);np.testing.assert_allclose(np.asarray(full),np.asarray(single),rtol=2e-5,atol=2e-5)
report=dict(passed=True,device=str(jax.devices()[0]),loader='pkatrain.production_loading.PinderSource',optimizer='pkatrain.gqt_paired_pinder.PairedEngine vanilla',architecture=dict(width=44,ff=88),loader_buckets=checks,optimizer_steps=3,largest_bucket_gradient=large_check,losses=losses,step_seconds=timings,validation=validation,validation_complexes=len(val),padding_loss_invariance=True,manifest_sha256=digest(O/'manifest.json'),full_training_started=False,limitation='Short technical smoke only; not convergence, accuracy or robust model-selection evidence')
report['code_hashes']={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}
report['jax_version']=jax.__version__
report['numpy_version']=np.__version__
assert sum(len(x['ids']) for x in validation)==len(val)
report['fraction_counts']={str(p):len(select(m,'train',p/100)) for p in [10,50,75,100]}
atomic_json(O/'smoke.json',report)
atomic_json(O/'readiness.json',dict(technical_checks_passed=True,scope='full ATLAS uniform-weight paired backbone oGQT',smoke_sha256=digest(O/'smoke.json'),summary_sha256=digest(O/'summary.json'),full_training_started=False,active_dataset_modified=False,frozen_validation_membership=2000,active_validation=len(val),limitation='Technical batch-2 smoke, uniform weighting; no convergence or production-batch memory claim'))
src.close();print(json.dumps(report,indent=2),flush=True)
