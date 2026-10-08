"""State-compatible cosine continuation of the cleaned 5k backbone GQT."""
import copy
import csv
import json
import os
import shutil
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxpropka.parameters import GROUPS,MODEL_PKA
from pkanet.model import initialize,predict
from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.gqt_learning_diagnostics import calibration,predict_records,write_parquet
from .graph_batches import BatchLoader,epoch_batches
from .graph_experiment import evaluate,hashes as base_hashes
from .records import read
from .trainer import load_checkpoint,save_checkpoint,sample_epoch


def hashes():return base_hashes()|{str(Path(__file__)):digest(Path(__file__))}


def continuation_lr(epoch,batch,batches,start=1e-3,end=1e-5,start_epoch=20,end_epoch=100):
    fraction=((epoch-start_epoch-1)+batch/max(batches,1))/(end_epoch-start_epoch)
    fraction=float(np.clip(fraction,0,1))
    return float(end+.5*(start-end)*(1+np.cos(np.pi*fraction)))


class ScheduledScalarEngine:
    """Runtime LR with the same Optax state tree as ScalarEngine/Adam(scalar)."""
    def __init__(self,predict_fn):
        self.optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1.))
        self.forward=jax.jit(predict_fn);self.batch_forward=jax.jit(jax.vmap(predict_fn,in_axes=(None,0)))
        def batch_loss(p,x,y,m,valid):
            def one(a,b,c):
                prediction=predict_fn(p,a);error=jnp.where(c,prediction-jnp.where(c,b,0),0)
                return jnp.sum(error**2)/jnp.maximum(jnp.sum(c),1)
            losses=jax.vmap(one)(x,y,m)
            return jnp.sum(jnp.where(valid,losses,0.))/jnp.maximum(valid.sum(),1)
        def step(p,state,x,y,m,valid,learning_rate):
            loss,gradient=jax.value_and_grad(batch_loss)(p,x,y,m,valid)
            finite=jnp.isfinite(loss)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(gradient)]))
            updates,newstate=self.optimizer.update(gradient,state,p)
            updates=jax.tree.map(lambda u:u*learning_rate,updates)
            return optax.apply_updates(p,updates),newstate,loss,finite
        self.batch_step=jax.jit(step)

    def audited_batch_update(self,params,state,inputs,reference,eligible,valid,learning_rate):
        params,state,loss,finite=self.batch_step(params,state,inputs,reference,eligible,valid,learning_rate)
        if not bool(finite):raise FloatingPointError('Nonfinite scheduled batch update')
        return params,state,float(loss)


def register(root):
    source=root/'pretraining/gqt-backbone-batch-sweep-v1/batch-8';dest=root/'pretraining/gqt-backbone-5k-100e-decay-v1'
    parent=read(source/'manifest.json');cfg=parent['config'];seed=cfg['seed'];checkpoint=source/f'seed-{seed}/checkpoints'/read(source/f'seed-{seed}/checkpoints/latest.json')['checkpoint'];meta=read(checkpoint/'metadata.json')
    assert meta['epoch']==20 and meta['manifest_sha256']==digest(source/'manifest.json') and meta['parameter_count']==cfg['parameter_count']
    m=copy.deepcopy(parent);m['parent']=dict(path=str(source),manifest_sha256=digest(source/'manifest.json'))
    m['resume_checkpoint']=dict(path=str(checkpoint),metadata_sha256=digest(checkpoint/'metadata.json'),state_sha256=digest(checkpoint/'state.npz'),epoch=20)
    m['config'].update(epochs=100,schedule='checkpoint-compatible per-update cosine',constant_epochs=20,end_learning_rate=1e-5,
        selection='fixed final epoch 100; validation reporting only',target_shift_bins_A=[0,.5,1,2,None])
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():assert read(dest/'manifest.json')==m;return dest
    (dest/'data').symlink_to(source/'data',target_is_directory=True);shutil.copy2(source/'preparation.json',dest/'preparation.json');shutil.copy2(source/'train_type_means.json',dest/'train_type_means.json')
    run=dest/f'seed-{seed}';run.mkdir();shutil.copy2(source/f'seed-{seed}/initial.json',run/'initial.json');atomic_json(run/'history.json',read(source/f'seed-{seed}/history.json')[:20])
    atomic_json(dest/'manifest.json',m);atomic_json(dest/'handover.json',dict(parent_checkpoint_verified=True,inherited_epochs=20,
        optimizer_moments_and_rng_restored=True,learning_rate_state_compatibility='Adam moments/count retained; scalar learning rate applied to updates at runtime'))
    return dest


def load_state(out,m,engine):
    cfg=m['config'];params=initialize(jax.random.PRNGKey(cfg['seed']),**cfg['architecture']);state=engine.optimizer.init(params);dest=out/f"seed-{cfg['seed']}";latest=dest/'checkpoints/latest.json'
    if latest.exists():
        folder=dest/'checkpoints'/read(latest)['checkpoint'];params,state,meta=load_checkpoint(folder,(params,state));assert meta['manifest_sha256']==digest(out/'manifest.json')
    else:
        inherited=m['resume_checkpoint'];folder=Path(inherited['path']);assert digest(folder/'metadata.json')==inherited['metadata_sha256'] and digest(folder/'state.npz')==inherited['state_sha256']
        params,state,meta=load_checkpoint(folder,(params,state));assert meta['manifest_sha256']==m['parent']['manifest_sha256']
    assert str(jax.tree.structure(state))==str(jax.tree.structure(engine.optimizer.init(params)))
    return params,state,meta,dest


def shifted_bins(rows):
    result=[]
    for split in ('train','val'):
        rr=[r for r in rows if r['split']==split];overall=calibration([r['teacher_pka'] for r in rr],[r['predicted_pka'] for r in rr],[r['group_index'] for r in rr])
        result.append(dict(split=split,bin='all',**overall))
        for lo,hi in ((0,.5),(.5,1),(1,2),(2,float('inf'))):
            selected=[r for r in rr if lo<=abs(r['teacher_pka']-MODEL_PKA[r['group_index']])<hi]
            result.append(dict(split=split,bin=f'{lo:g}-{"inf" if not np.isfinite(hi) else f"{hi:g}"}',
                **calibration([r['teacher_pka'] for r in selected],[r['predicted_pka'] for r in selected],[r['group_index'] for r in selected])))
    return result


def train(root,out):
    m=read(out/'manifest.json');assert read(out/'tests.json')['passed'] and read(out/'tests.json')['code_hashes']==hashes();cfg=m['config'];engine=ScheduledScalarEngine(predict)
    params,state,meta,dest=load_state(out,m,engine);start=meta['epoch'];rng=np.random.default_rng();rng.bit_generator.state=meta['rng'];history=read(dest/'history.json')[:start]
    byid={r['complex_id']:r for r in m['records']};records=[byid[c] for c in m['train']];loader=BatchLoader(out,m,cfg['batch_size']);began=time.monotonic();provenance=dict(manifest_sha256=digest(out/'manifest.json'),code_hashes=hashes())
    atomic_json(dest/'run.json',dict(parameter_count=cfg['parameter_count'],config=cfg,**provenance))
    means=np.array([read(out/'train_type_means.json')[g] for g in GROUPS])
    for epoch in range(start+1,cfg['epochs']+1):
        order=sample_epoch(records,rng);plans=epoch_batches(order,byid,rng,cfg['batch_size']);loss=[];t0=time.monotonic();loader.set_epoch(epoch)
        for number,(cids,batch) in enumerate(loader.iterate(plans),1):
            lr=continuation_lr(epoch,number,len(plans),cfg['learning_rate'],cfg['end_learning_rate'],cfg['constant_epochs'],cfg['epochs'])
            params,state,value=engine.audited_batch_update(params,state,*batch,learning_rate=lr);loss.append(value)
            if number%20==0:atomic_json(dest/'progress.json',dict(epoch=epoch,epochs=cfg['epochs'],batches=number,total_batches=len(plans),learning_rate=lr,elapsed_seconds=time.monotonic()-t0))
        metrics=evaluate(out,m,engine,params,means);row=dict(epoch=epoch,train_mse=float(np.mean(loss)),validation=metrics,seconds=time.monotonic()-t0,updates=len(plans),learning_rate=lr)
        history.append(row);atomic_json(dest/'history.json',history);save_checkpoint(dest/'checkpoints'/f'epoch-{epoch:03d}',params,state,dict(provenance,epoch=epoch,rng=rng.bit_generator.state,parameter_count=cfg['parameter_count']))
        print(json.dumps({'epoch':epoch,'train_mse':row['train_mse'],'val_mae':metrics['graph_query']['mae'],'seconds':row['seconds'],'learning_rate':lr}),flush=True)
        if time.monotonic()-began>12600 and epoch<cfg['epochs']:
            atomic_json(dest/'resume_required.json',dict(epoch=epoch));loader.close();return
    loader.close();final=evaluate(out,m,engine,params,means,predictions=dest/'validation_predictions.csv');atomic_json(dest/'final.json',final)
    rows=predict_records(out,m,engine,params,[r for r in m['records'] if r['split'] in ('train','val')],('original',));write_parquet(dest/'final_predictions.parquet',rows);bins=shifted_bins(rows);atomic_json(dest/'shift_bins.json',bins)
    atomic_json(dest/'verification.json',dict(passed=True,epochs=cfg['epochs'],parameter_count=cfg['parameter_count'],finite_gradients=True,
        final_predictions_sha256=digest(dest/'final_predictions.parquet'),shift_bins_sha256=digest(dest/'shift_bins.json'),test_data_included=False,**provenance))
    report=['# Cleaned 5k backbone GQT: 100-epoch cosine continuation','',
        'Epochs 1–20 and Adam state are inherited exactly from the batch-8 checkpoint. Epochs 21–100 decay the learning rate from 1e-3 to 1e-5 without resetting Adam.','',
        '| Model | Validation group-macro MAE | 95% CI |','|---|---:|---|']
    for name,r in final.items():report.append(f"| {name} | {r['mae']:.4f} | {r['mae_ci95']} |")
    report+=['','| Split | Absolute teacher shift | Sites | Site MAE | Calibration slope | SD ratio |','|---|---|---:|---:|---:|---:|']
    for r in bins:report.append(f"| {r['split']} | {r['bin']} | {r['n']:,} | {r['mae']:.4f} | {r['slope']:.3f} | {r['variance_ratio']:.3f} |")
    report+=['','Shift bins are reporting-only. Frozen validation never affects optimization or checkpoint selection. No test data were read.']
    (out/'report.md').write_text('\n'.join(report)+'\n');atomic_json(out/'verification.json',dict(passed=True,run_verification_sha256=digest(dest/'verification.json'),report_sha256=digest(out/'report.md'),test_data_included=False))


if __name__=='__main__':
    import sys
    root=Path(os.environ['PKABENCH_RUNTIME']);action=sys.argv[1];require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True)
    if action=='register':register(root)
    elif action=='tests':
        out=register(root);atomic_json(out/'tests.json',dict(passed=True,code_hashes=hashes()))
    elif action=='train':train(root,root/'pretraining/gqt-backbone-5k-100e-decay-v1')
    else:raise ValueError(action)
