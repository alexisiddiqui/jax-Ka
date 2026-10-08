"""Checkpoint-compatible 20-to-100 epoch continuation of the GQT size sweep."""
import copy
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxpropka.parameters import GROUPS
from pkanet.model import PKPDB_PK_MOD,initialize,predict_pkpdb,predict_shift
from pkabench.gqt_learning_diagnostics import predict_records,write_parquet
from pkabench.runtime import atomic_json,digest,require_compute
from .graph_batches import BatchLoader,epoch_batches
from .graph_experiment import evaluate,hashes as base_hashes
from .graph_pkmod_compare import BINS,digest_array_tree,shifted_bins
from .records import read
from .trainer import load_checkpoint,sample_epoch,save_checkpoint


SIZES=('50k','200k','800k')


def hashes():return base_hashes()|{str(Path(__file__)):digest(Path(__file__))}


def continuation_lr(epoch,batch,batches,start=1e-3,end=1e-5,start_epoch=20,end_epoch=100):
    fraction=((epoch-start_epoch-1)+batch/max(batches,1))/(end_epoch-start_epoch)
    fraction=float(np.clip(fraction,0,1))
    return float(end+.5*(start-end)*(1+np.cos(np.pi*fraction)))


class ScheduledExplicitShiftEngine:
    """Explicit pKPDB-shift loss with checkpoint-compatible runtime LR scaling."""
    def __init__(self,bin_weights):
        self.bin_weights=jnp.asarray(bin_weights,jnp.float32);self.baseline=jnp.asarray(PKPDB_PK_MOD,jnp.float32)
        self.optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1.))
        self.forward=jax.jit(predict_pkpdb);self.batch_forward=jax.jit(jax.vmap(predict_pkpdb,in_axes=(None,0)))
        def one_loss(params,graph,target,eligible):
            target_shift=target-self.baseline[graph['query_group']];predicted=predict_shift(params,graph)
            index=jnp.sum(jnp.abs(target_shift)[...,None]>=jnp.asarray(BINS),axis=-1)
            weight=jnp.where(eligible,self.bin_weights[index],0.);error=jnp.where(eligible,predicted-target_shift,0.)
            return jnp.sum(weight*error**2)/jnp.maximum(jnp.sum(weight),1e-8)
        def batch_loss(params,graphs,targets,eligible,valid):
            losses=jax.vmap(one_loss,in_axes=(None,0,0,0))(params,graphs,targets,eligible)
            return jnp.sum(jnp.where(valid,losses,0.))/jnp.maximum(valid.sum(),1)
        def step(params,state,graphs,targets,eligible,valid,learning_rate):
            loss,gradient=jax.value_and_grad(batch_loss)(params,graphs,targets,eligible,valid)
            finite=jnp.isfinite(loss)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)]))
            updates,newstate=self.optimizer.update(gradient,state,params)
            updates=jax.tree.map(lambda value:value*learning_rate,updates)
            return optax.apply_updates(params,updates),newstate,loss,finite
        self.batch_step=jax.jit(step)

    def audited_batch_update(self,params,state,inputs,reference,eligible,valid,learning_rate):
        params,state,loss,finite=self.batch_step(params,state,inputs,reference,eligible,valid,learning_rate)
        if not bool(finite):raise FloatingPointError('Nonfinite scheduled explicit-shift update')
        return params,state,float(loss)


def source_for(root,size):
    if size=='50k':return root/'pretraining/gqt-backbone-5k-pkmod-v1/unweighted'
    return root/'pretraining/gqt-pkai-parameter-sweep-v1/gqt'/size/'unweighted'


def register(root):
    destination=root/'pretraining/gqt-pkai-parameter-sweep-v1/gqt-long'
    destination.mkdir(parents=True,exist_ok=True)
    protocol=dict(sizes=list(SIZES),inherited_epochs=20,final_epoch=100,batch_size=8,
        schedule='per-update cosine from 1e-3 after epoch 20 to 1e-5 at epoch 100',
        capacity_rounding=[128,64,None],capacity_shape_cap=12,graph_backend='verified read-only mmap',
        target='explicit signed shift from historical pKPDB PK_MOD',selection='fixed epoch 100; validation reporting only',
        preserved='parameters, Adam moments/count, sampling RNG, data, split, batch membership and objective',test_data_included=False)
    atomic_json(destination/'protocol.json',protocol)
    for size in SIZES:
        source=source_for(root,size);parent=read(source/'manifest.json');cfg=parent['config'];seed=cfg['seed']
        latest=read(source/f'seed-{seed}/checkpoints/latest.json')['checkpoint']
        checkpoint=source/f'seed-{seed}/checkpoints'/latest;meta=read(checkpoint/'metadata.json')
        assert meta['epoch']==20 and meta['manifest_sha256']==digest(source/'manifest.json')
        assert meta['parameter_count']==cfg['parameter_count']
        out=destination/size;out.mkdir(exist_ok=True)
        manifest=copy.deepcopy(parent)
        manifest['parent']=dict(path=str(source),manifest_sha256=digest(source/'manifest.json'))
        manifest['resume_checkpoint']=dict(path=str(checkpoint),metadata_sha256=digest(checkpoint/'metadata.json'),
            state_sha256=digest(checkpoint/'state.npz'),epoch=20)
        manifest['long_protocol_sha256']=digest(destination/'protocol.json')
        manifest['config'].update(epochs=100,schedule='checkpoint-compatible per-update cosine',constant_epochs=20,
            end_learning_rate=1e-5,capacity_rounding=[128,64,None],capacity_shape_cap=12,
            graph_backend='verified read-only mmap',selection='fixed final epoch 100; validation reporting only')
        if not (out/'data').exists():(out/'data').symlink_to(source/'data',target_is_directory=True)
        for name in ('preparation.json','train_type_means.json'):
            if not (out/name).exists():shutil.copy2(source/name,out/name)
        atomic_json(out/'manifest.json',manifest)
        run=out/f'seed-{seed}';run.mkdir(exist_ok=True)
        if not (run/'history.json').exists():atomic_json(run/'history.json',read(source/f'seed-{seed}/history.json')[:20])
    atomic_json(destination/'registration.json',dict(passed=True,protocol_sha256=digest(destination/'protocol.json'),code_hashes=hashes()))
    return destination


def load_state(out,manifest,engine):
    cfg=manifest['config'];params=initialize(jax.random.PRNGKey(cfg['seed']),**cfg['architecture']);state=engine.optimizer.init(params)
    run=out/f"seed-{cfg['seed']}";latest=run/'checkpoints/latest.json'
    if latest.exists():
        folder=run/'checkpoints'/read(latest)['checkpoint'];params,state,meta=load_checkpoint(folder,(params,state))
        assert meta['manifest_sha256']==digest(out/'manifest.json')
    else:
        inherited=manifest['resume_checkpoint'];folder=Path(inherited['path'])
        assert digest(folder/'metadata.json')==inherited['metadata_sha256'] and digest(folder/'state.npz')==inherited['state_sha256']
        params,state,meta=load_checkpoint(folder,(params,state));assert meta['manifest_sha256']==manifest['parent']['manifest_sha256']
    return params,state,meta,run


def train(root,size):
    if size not in SIZES:raise ValueError(size)
    out=root/'pretraining/gqt-pkai-parameter-sweep-v1/gqt-long'/size;manifest=read(out/'manifest.json');cfg=manifest['config']
    registration=read(out.parent/'registration.json');assert registration['passed'] and registration['code_hashes']==hashes()
    engine=ScheduledExplicitShiftEngine(cfg['shift_bin_weights']);params,state,meta,run=load_state(out,manifest,engine)
    start=meta['epoch'];rng=np.random.default_rng();rng.bit_generator.state=meta['rng'];history=read(run/'history.json')[:start]
    byid={row['complex_id']:row for row in manifest['records']};records=[byid[cid] for cid in manifest['train']]
    loader=BatchLoader(out,manifest,cfg['batch_size'],backend='mmap');loader.set_epoch(start+1)
    means=np.asarray([read(out/'train_type_means.json')[group] for group in GROUPS])
    provenance=dict(manifest_sha256=digest(out/'manifest.json'),code_hashes=hashes())
    atomic_json(run/'run.json',dict(parameter_count=cfg['parameter_count'],config=cfg,loader=loader.provenance(),**provenance))
    began=time.monotonic()
    for epoch in range(start+1,cfg['epochs']+1):
        order=sample_epoch(records,rng);plans=epoch_batches(order,byid,rng,cfg['batch_size']);losses=[];t0=time.monotonic();loader.set_epoch(epoch)
        for number,(_,batch) in enumerate(loader.iterate(plans),1):
            lr=continuation_lr(epoch,number,len(plans),cfg['learning_rate'],cfg['end_learning_rate'],cfg['constant_epochs'],cfg['epochs'])
            params,state,value=engine.audited_batch_update(params,state,*batch,learning_rate=lr);losses.append(value)
            if number%20==0:atomic_json(run/'progress.json',dict(size=size,epoch=epoch,epochs=cfg['epochs'],batches=number,
                total_batches=len(plans),learning_rate=lr,elapsed_seconds=time.monotonic()-t0))
        metrics=evaluate(out,manifest,engine,params,means)
        row=dict(epoch=epoch,train_shift_mse=float(np.mean(losses)),validation=metrics,seconds=time.monotonic()-t0,updates=len(plans),learning_rate=lr)
        history.append(row);atomic_json(run/'history.json',history)
        save_checkpoint(run/'checkpoints'/f'epoch-{epoch:03d}',params,state,dict(provenance,epoch=epoch,rng=rng.bit_generator.state,parameter_count=cfg['parameter_count']))
        print(json.dumps(dict(size=size,epoch=epoch,train_shift_mse=row['train_shift_mse'],val_mae=metrics['graph_query']['mae'],seconds=row['seconds'],learning_rate=lr)),flush=True)
        if time.monotonic()-began>18000 and epoch<cfg['epochs']:
            atomic_json(run/'resume_required.json',dict(epoch=epoch));loader.close();return
    loader.close();final=evaluate(out,manifest,engine,params,means,predictions=run/'validation_predictions.csv');atomic_json(run/'final.json',final)
    rows=predict_records(out,manifest,engine,params,[r for r in manifest['records'] if r['split'] in ('train','val')],('original',))
    write_parquet(run/'final_predictions.parquet',rows);bins=shifted_bins(rows);atomic_json(run/'shift_bins.json',bins)
    atomic_json(run/'verification.json',dict(passed=True,epochs=cfg['epochs'],parameter_count=cfg['parameter_count'],finite_gradients=True,
        initial_checkpoint_digest=digest(Path(manifest['resume_checkpoint']['path'])/'state.npz'),final_parameter_digest=digest_array_tree(params),
        final_predictions_sha256=digest(run/'final_predictions.parquet'),shift_bins_sha256=digest(run/'shift_bins.json'),test_data_included=False,**provenance))


def report(root):
    base=root/'pretraining/gqt-pkai-parameter-sweep-v1/gqt-long';lines=['# Converged GQT parameter-size sweep','',
        'All models inherit their exact epoch-20 parameters, Adam state and sampling RNG, then continue to fixed epoch 100 with a per-update cosine decay. Batch size is 8 and the verified capped N128/K64 capacity policy is active.','',
        '| Size | Parameters | Epoch-20 MAE | Epoch-100 MAE | 95% CI | Train MSE at 100 | Continuation GPU time |','|---|---:|---:|---:|---|---:|---:|']
    results=[]
    for size in SIZES:
        out=base/size;manifest=read(out/'manifest.json');run=out/'seed-17';history=read(run/'history.json');final=read(run/'final.json')['graph_query']
        e20=next(row for row in history if row['epoch']==20);e100=next(row for row in history if row['epoch']==100)
        row=dict(size=size,parameters=manifest['config']['parameter_count'],epoch20_mae=e20['validation']['graph_query']['mae'],
            epoch100_mae=final['mae'],ci=final['mae_ci95'],train_mse=e100['train_shift_mse'],seconds=sum(x['seconds'] for x in history if x['epoch']>20))
        results.append(row);lines.append(f"| {size} | {row['parameters']:,} | {row['epoch20_mae']:.4f} | {row['epoch100_mae']:.4f} | {row['ci']} | {row['train_mse']:.4f} | {row['seconds']/3600:.2f} h |")
    lines+=['','Validation is reporting-only and epoch 100 was fixed before continuation. No test data were read.']
    atomic_json(base/'results.json',results);(base/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(base/'verification.json',dict(passed=True,results_sha256=digest(base/'results.json'),report_sha256=digest(base/'report.md'),test_data_included=False))


def main():
    import sys
    action=sys.argv[1];root=Path(os.environ['PKABENCH_RUNTIME']);gpu=action=='train'
    require_compute(threads=int(os.environ.get('SLURM_CPUS_PER_TASK','1')),gpu_benchmark=gpu,allow_comp1400=True)
    jax.config.update('jax_enable_x64',False)
    if action=='register':register(root)
    elif action=='train':train(root,sys.argv[2])
    elif action=='report':report(root)
    else:raise ValueError(action)


if __name__=='__main__':main()
