"""Registered three-scale experiment built on the shared trainer and solver."""
import json
import os
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jaxpropka.parameters import ModelConfig
from jaxpropka.optx_solver import SolverConfig
from pkabench.runtime import atomic_json,digest
from .records import read,load
from .adapters.jaxka import local_terms,physical_scales
from .trainer import Engine,save_checkpoint,load_checkpoint,sample_epoch


def make_engine(out=None):
    settings=read(out/'manifest.json')['config'] if out is not None else {}
    return Engine(local_terms,lambda theta:1e-3*jnp.mean(jnp.log(jnp.stack(physical_scales(theta)))**2),
        ModelConfig(steps=1024),SolverConfig(max_steps=settings.get('lm_steps',512)),
        seed_steps=settings.get('seed_steps'),seed_dtype=settings.get('seed_dtype'),dtype=np.dtype(settings.get('dtype','float64')))


def code_hashes():
    base=Path(__file__).parent
    files=list(base.rglob('*.py'))+list((base.parent/'jaxpropka').glob('*.py'))
    return {str(p):digest(p) for p in sorted(files)}


def train(out,seed,smoke=False):
    manifest=read(out/'manifest.json'); config=manifest['config']
    if not smoke:
        gates=('release.json','profiles/verification.json') if config.get('dtype')=='float32' else ('real_gate.json','smoke/verification.json','profiles/verification.json')
        for gate in gates:
            assert read(out/gate)['passed'],gate
        if config.get('dtype')=='float32':
            release=read(out/'release.json')
            assert release['manifest_sha256']==digest(out/'manifest.json')
            assert release['buckets_sha256']==digest(out/'buckets.json')
    from .buckets import ordered_epoch,microbatches
    from .minibatch import pack_records
    plan=read(out/'buckets.json') if (out/'buckets.json').exists() else None
    ids=manifest['smoke'] if smoke else manifest['train']
    records=[read(out/'records'/f'{c}.json') for c in ids]
    dest=out/('smoke' if smoke else f'seed-{seed}'); dest.mkdir(exist_ok=True)
    engine=make_engine(out); params=jnp.zeros(3,np.dtype(config['dtype'])); state=engine.optimizer.init(params)
    rng=np.random.default_rng(seed); step=0; start_epoch=0; offset=0; order=[]; missing=total=0
    hashes=code_hashes(); manifest_hash=digest(out/'manifest.json')
    latest=dest/'checkpoints/latest.json'
    if latest.exists():
        params,state,meta=load_checkpoint(latest.parent/read(latest)['checkpoint'],(params,state))
        assert meta['code_sha256']==hashes and meta['manifest_sha256']==manifest_hash
        rng.bit_generator.state=meta['rng']; step=meta['step']; start_epoch=meta['epoch']; offset=meta['offset']; order=meta['order']; missing=meta['missing']; total=meta['total']
    (dest/'resume_required.json').unlink(missing_ok=True)
    initial=np.asarray(params).tolist(); began=time.monotonic()
    # Smoke is exactly 20 updates, with replacement from eight registered complexes.
    epochs=1 if smoke else config['epochs']
    wall_budget=float(os.environ.get('PKATRAIN_WALL_BUDGET_SECONDS','39600'))
    for epoch in range(start_epoch,epochs):
        if not order:
            order=sample_epoch(records,rng)
            if plan is not None:order=ordered_epoch(order,plan,rng)
            if smoke: order=[str(rng.choice(ids)) for _ in range(20*config['accumulate'])]
            offset=0; missing=total=0
        weight=1. if smoke else min(1.,epoch/config['paired_ramp_epochs'])
        while offset<len(order):
            gradients=[]; logs=[]
            selected=order[offset:offset+config['accumulate']]
            if plan is not None:
                for ids,bucket in microbatches(selected,plan):
                    inputs,reference,eligible,_=pack_records(out,ids,np.dtype(config['dtype']),bucket['capacities'])
                    grad,log,_=engine.audited_batch_gradient(params,inputs,reference,eligible,weight)
                    # Each microbatch gradient is a mean; weight by its actual
                    # number of complexes before combining unequal microbatches.
                    gradients.extend([grad]*len(ids))
                    logs.extend(dict(complex_id=cid,loss=log['loss'],**cov) for cid,cov in zip(ids,log['per_complex']))
            else:
                for cid in selected:
                    inputs,reference,eligible,_=load(out,cid,dtype=np.dtype(config['dtype']))
                    grad,log=engine.audited_gradient(params,inputs,reference,eligible,weight)
                    gradients.append(grad); logs.append(dict(complex_id=cid,**log))
            batch_missing=sum(r['missing'] for r in logs); batch_total=sum(r['total'] for r in logs)
            missing+=batch_missing; total+=batch_total
            if missing/max(total,1)>.01: raise RuntimeError(f'Epoch missing-supervision threshold exceeded: {missing}/{total}')
            params,state=engine.update(params,state,gradients); step+=1; offset+=len(logs)
            metadata={'step':step,'epoch':epoch,'offset':offset,'order':order,'rng':rng.bit_generator.state,
                'missing':missing,'total':total,'manifest_sha256':manifest_hash,'code_sha256':hashes}
            save_checkpoint(dest/'checkpoints'/f'{step:06d}',params,state,metadata)
            record={'step':step,'epoch':epoch,'paired_weight':weight,'scales':list(map(float,physical_scales(params))),
                'mean_loss':float(np.mean([r['loss'] for r in logs])),'seconds':time.monotonic()-began,'observations':logs}
            with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k!='observations'}),flush=True)
            if time.monotonic()-began>wall_budget:
                atomic_json(dest/'resume_required.json',{'step':step,'checkpoint':str(latest),'reason':'wall budget reached after atomic checkpoint'})
                return
        if not smoke:
            evaluate(out,params,dest/f'epoch-{epoch+1:02d}',manifest['val'],engine=engine,diagnostics=True)
        order=[]; offset=0
    final=np.asarray(params); atomic_json(dest/'parameters.json',{'theta':final.tolist(),'scales':list(map(float,physical_scales(params))),'step':step})
    if smoke:
        assert np.isfinite(final).all() and np.any(final!=0)
    atomic_json(dest/'verification.json',{'passed':True,'steps':step,'epochs':epochs,'seed':seed,'smoke':smoke,
        'manifest_sha256':manifest_hash,'code_sha256':hashes,'test_data_included':False,'seconds':time.monotonic()-began,
        'checkpoint_selection':'fixed final epoch','seeds_measure':'sampling order only'})


def evaluate(out,params,dest,ids,engine=None,diagnostics=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from jaxpropka.model import _grid_pka_result
    from pkabench.frozen_score import measures,aggregate
    identity=dict(parameters=np.asarray(params).tolist(),complex_ids=list(ids),manifest_sha256=digest(out/'manifest.json'))
    if (dest/'verification.json').exists():
        prior=read(dest/'verification.json')
        assert prior.get('identity')==identity and prior['passed']
        assert digest(dest/'predictions.parquet')==prior['predictions_sha256']
        return
    engine=engine or make_engine(out); dest.mkdir(parents=True,exist_ok=True)
    rows=[]; metrics=[]; diag=[]; missing=total=0
    from jaxpropka.parameters import GROUPS
    from jaxpropka.cache import StructureCache
    plan=read(out/'buckets.json') if (out/'buckets.json').exists() else None
    for cid in ids:
        inputs,reference,eligible,receipt=load(out,cid,dtype=np.dtype(read(out/'manifest.json')['config']['dtype'])); r=read(out/'records'/f'{cid}.json')
        teacher=np.asarray(inputs.pop('teacher_pka'))
        if plan is not None:
            from .minibatch import pack_records
            bucket=next(b for b in plan['buckets'] if b['name']==plan['assignment'][cid])
            packed,rr,ee,_=pack_records(out,[cid],np.dtype(read(out/'manifest.json')['config']['dtype']),bucket['capacities'])
            inputs=jax.tree.map(lambda x:x[0],packed);reference=rr[0];eligible=ee[0]
        curves=engine.forward(params,inputs,jnp.ones((2,73),bool))
        h=np.asarray(curves.protonated); valid=np.asarray(curves.converged)
        from .losses import coverage,curve_loss
        cov=coverage(eligible,valid); missing+=cov['missing']; total+=cov['total']
        _,parts=curve_loss(h,reference,eligible,valid)
        mid=[]; midvalid=[]
        for branch in (0,1):
            c=jax.tree.map(lambda a:a[branch],curves); arrays={k:v[branch] for k,v in inputs['arrays'].items()}
            result=_grid_pka_result(arrays,c,engine.ph,engine.config)
            mid.append(np.asarray(result.value)); midvalid.append(np.asarray(result.valid))
        # Keep the original teacher readout; never redefine PypKa's target.
        teacher_valid=np.isfinite(teacher)
        cache=StructureCache.load(out/'prepared'/cid/'AB.npz')
        refs=[]; preds=[]
        for i,g in zip(*np.nonzero(eligible)):
            key=cache.keys[i]; ok=all(v[i,g] for v in midvalid) and bool(teacher_valid[:,i,g].all())
            ref=float(teacher[0][i,g]-teacher[1][i,g]) if all(v[i,g] for v in teacher_valid) else None
            pred=float(mid[0][i,g]-mid[1][i,g]) if all(v[i,g] for v in midvalid) else None
            rows.append(dict(complex_id=cid,component_id=r['component_id'],role=r['role'],chain=key.chain,resnum=key.number,icode=key.insertion,group=GROUPS[g],
                prediction=pred,target=ref,valid=bool(ok),curve_error=float(np.mean(abs(h[:,:,i,g]-reference[:,:,i,g])))))
            if ok:refs.append(ref);preds.append(pred)
        metric=dict(complex_id=cid,component_id=r['component_id'],role=r['role'],n=len(refs),**measures(refs,preds))
        metric.update(curve_mse=float(parts['absolute']),paired_curve_mse=float(parts['paired']),missing_fraction=cov['fraction'])
        metrics.append(metric)
        if diagnostics:
            up=engine.forward(params,inputs,jnp.ones((2,73),bool),initialization='up')
            down=engine.forward(params,inputs,jnp.ones((2,73),bool),initialization='down')
            both=np.asarray(up.converged)&np.asarray(down.converged)
            gap=np.max(abs(np.asarray(up.protonated)-np.asarray(down.protonated)),axis=(-2,-1))
            diag.append(dict(complex_id=cid,hysteresis_max=float(gap[both].max()) if both.any() else None,
                invalid_up=int((~np.asarray(up.converged)).sum()),invalid_down=int((~np.asarray(down.converged)).sum()),
                production_gap=float(np.max(abs(np.asarray(up.protonated)-h)))))
    pq.write_table(pa.Table.from_pylist(rows),dest/'predictions.parquet')
    atomic_json(dest/'complex_metrics.json',metrics); atomic_json(dest/'branch_diagnostics.json',diag)
    result,groups=aggregate(metrics); atomic_json(dest/'scores.json',{'aggregate':result,'groups':groups,'missing':missing,'total':total})
    atomic_json(dest/'verification.json',{'passed':True,'complexes':len(ids),'test_data_included':False,'predictions_sha256':digest(dest/'predictions.parquet'),'identity':identity})
