"""Small single-state pretraining experiment using the shared optimizer/checkpoints."""
import json
import os
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from pkanet.model import initialize,predict
from jaxpropka.parameters import MODEL_PKA,GROUPS
from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.frozen_score import measures,aggregate,write_csv
from .records import read
from .graph_data import prepare,load,bucket,mask_features
from .trainer import ScalarEngine,save_checkpoint,load_checkpoint,sample_epoch,scalar_schedule


def hashes():
    base=Path(__file__).parent
    paths=list((base.parent/'pkanet').glob('*.py'))+[base/name for name in
        ('graph_data.py','graph_batches.py','graph_mmap.py','context_augmentation.py','compilation_cache.py','graph_experiment.py','trainer.py','losses.py')]
    return {str(p):digest(p) for p in paths}


def inputs(out,r,m):
    graph,y,mask=load(out,r,m['capacities'][bucket(r)])
    return mask_features(graph,m['config']),y,mask


def constants(out,m):
    # Match group-uniform/complex-uniform training weights, fitted on train only.
    from collections import Counter
    group_counts=Counter(r['component_id'] for r in m['records'] if r['split']=='train')
    sums=np.zeros(9);counts=np.zeros(9)
    for r in m['records']:
        if r['split']!='train':continue
        g,y,mask=inputs(out,r,m);weight=1/(group_counts[r['component_id']]*r['q'])
        for group,value in zip(g['query_group'][mask],y[mask]):sums[group]+=value*weight;counts[group]+=weight
    return np.divide(sums,counts,out=MODEL_PKA.copy(),where=counts>0)


def evaluate(out,m,engine,params,means,*,replicates=2000,predictions=None):
    rows=[];scores={name:[] for name in ('graph_query','model_values','train_type_mean')}
    for r in m['records']:
        if r['split']!='val':continue
        graph,y,mask=inputs(out,r,m);pred=np.asarray(engine.forward(params,graph))[mask];y=y[mask]
        assert np.isfinite(pred).all()
        group=graph['query_group'][mask];base=MODEL_PKA[group]
        for name,value in (('graph_query',pred),('model_values',base),('train_type_mean',means[group])):
            scores[name].append(dict(component_id=r['component_id'],complex_id=r['complex_id'],n=len(y),**measures(y-base,value-base)))
        if predictions is not None:
            for key,target,predicted,mean in zip(r['keys'],y,pred,means[group]):
                rows.append(dict(zip(('complex_id','chain','resnum','icode','group'),key),
                    component_id=r['component_id'],teacher_pka=float(target),predicted_pka=float(predicted),train_type_mean=float(mean)))
    result={name:aggregate(values,replicates=replicates)[0] for name,values in scores.items()}
    if predictions is not None:write_csv(predictions,rows)
    return result


def train(out):
    m=read(out/'manifest.json');assert read(out/'preparation.json')['passed']
    gate=read(out/'tests.json');assert gate['passed'] and gate['code_hashes']==hashes(),'tests must match current code'
    cfg=m['config'];seed=cfg['seed']
    from .compilation_cache import configure
    cache=configure(cfg,hashes())
    params=initialize(jax.random.PRNGKey(seed),**cfg.get('architecture',{}))
    if 'matmul_precision' in cfg:
        assert str(jax.config.jax_default_matmul_precision)==cfg['matmul_precision']
    effective_batch=cfg.get('batch_size',cfg['accumulation'])
    updates_per_epoch=(len(m['train'])+effective_batch-1)//effective_batch
    schedule=scalar_schedule(cfg,updates_per_epoch)
    dropout_rate=cfg.get('dropout_rate',0.)
    training_predict=(lambda p,g,k:predict(p,g,key=k,dropout_rate=dropout_rate)) if dropout_rate else None
    engine=ScalarEngine(predict,schedule,training_predict=training_predict)
    state=engine.optimizer.init(params);rng=np.random.default_rng(seed)
    provenance=dict(manifest_sha256=digest(out/'manifest.json'),code_hashes=hashes())
    dest=out/f'seed-{seed}';dest.mkdir(exist_ok=True);latest=dest/'checkpoints/latest.json';start=0
    if latest.exists():
        folder=dest/'checkpoints'/read(latest)['checkpoint'];params,state,meta=load_checkpoint(folder,(params,state))
        assert all(meta[k]==v for k,v in provenance.items());start=meta['epoch'];rng.bit_generator.state=meta['rng']
    elif 'resume_checkpoint' in m:
        inherited=m['resume_checkpoint'];folder=Path(inherited['path'])
        assert digest(folder/'metadata.json')==inherited['metadata_sha256']
        params,state,meta=load_checkpoint(folder,(params,state))
        assert meta['manifest_sha256']==m['parent']['manifest_sha256']
        start=meta['epoch'];rng.bit_generator.state=meta['rng']
    means=np.array([read(out/'train_type_means.json')[g] for g in GROUPS]) if 'resume_checkpoint' in m and (out/'train_type_means.json').exists() else constants(out,m)
    atomic_json(out/'train_type_means.json',dict(zip(GROUPS,means.tolist())))
    parameter_count=sum(x.size for x in jax.tree.leaves(params))
    assert parameter_count==cfg.get('parameter_count',28565) and all(x.dtype==jnp.float32 for x in jax.tree.leaves(params))
    byid={r['complex_id']:r for r in m['records']};records=[byid[c] for c in m['train']]
    batched=cfg.get('batch_size',0)>0
    if batched:
        from .graph_batches import BatchLoader,epoch_batches
        loader=BatchLoader(out,m,cfg['batch_size'])
    atomic_json(dest/'run.json',dict(parameter_count=parameter_count,config=cfg,compilation_cache=cache,
        graph_storage=loader.provenance() if batched else dict(backend='npz'),**provenance))
    if not (dest/'initial.json').exists():
        atomic_json(dest/'initial.json',evaluate(out,m,engine,params,means,replicates=2000))
    began=time.monotonic();history=read(dest/'history.json')[:start] if (dest/'history.json').exists() else []
    for epoch in range(start,cfg['epochs']):
        order=sample_epoch(records,rng);losses=[];gradients=[];updates=0;t0=time.monotonic()
        if batched:
            context_sha256=loader.set_epoch(epoch+1)
            batches=epoch_batches(order,byid,rng,cfg['batch_size']);done=0
            for cids,batch in loader.iterate(batches):
                key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed),epoch+1),updates) if dropout_rate else None
                params,state,loss=engine.audited_batch_update(params,state,*batch,key=key)
                losses.extend([loss]*len(cids));updates+=1;done+=len(cids)
                if updates%10==0:
                    atomic_json(dest/'progress.json',dict(epoch=epoch+1,epochs=cfg['epochs'],complexes=done,total=len(order),
                        elapsed_seconds=time.monotonic()-t0,batch_size=cfg['batch_size'],updates=updates))
        for i,cid in enumerate(order if not batched else []):
            r=byid[cid];graph,y,mask=inputs(out,r,m)
            gradient,loss=engine.audited_gradient(params,graph,y,mask);gradients.append(gradient);losses.append(loss)
            if len(gradients)==cfg['accumulation'] or i==len(order)-1:
                params,state=engine.update(params,state,gradients);gradients=[];updates+=1
            if (i+1)%50==0:
                atomic_json(dest/'progress.json',dict(epoch=epoch+1,epochs=cfg['epochs'],complexes=i+1,total=len(order),elapsed_seconds=time.monotonic()-t0))
        metrics=evaluate(out,m,engine,params,means,predictions=dest/f'validation_epoch_{epoch+1:03d}.csv' if epoch+1==20 else None)
        row=dict(epoch=epoch+1,train_mse=float(np.mean(losses)),validation=metrics,
            seconds=time.monotonic()-t0,updates=updates,
            context_mask_sha256=context_sha256 if batched else None,
            learning_rate=float(schedule((epoch+1)*updates_per_epoch-1)) if callable(schedule) else float(schedule))
        history.append(row);atomic_json(dest/'history.json',history)
        save_checkpoint(dest/'checkpoints'/f'epoch-{epoch+1:03d}',params,state,
            dict(provenance,epoch=epoch+1,rng=rng.bit_generator.state,parameter_count=parameter_count))
        print(json.dumps({'epoch':epoch+1,'train_mse':row['train_mse'],'val_mae':metrics['graph_query']['mae'],'seconds':row['seconds'],'learning_rate':row['learning_rate']}),flush=True)
        if time.monotonic()-began>32400 and epoch+1<cfg['epochs']:
            atomic_json(dest/'resume_required.json',dict(epoch=epoch+1));return
    if batched:loader.close()
    final=evaluate(out,m,engine,params,means,replicates=2000,predictions=dest/'validation_predictions.csv')
    atomic_json(dest/'final.json',final)
    memory=jax.devices()[0].memory_stats() or {}
    atomic_json(dest/'verification.json',dict(passed=True,epochs=cfg['epochs'],parameter_count=parameter_count,
        finite_gradients=True,gpu_bytes_in_use=memory.get('bytes_in_use'),
        gpu_peak_bytes_in_use=memory.get('peak_bytes_in_use'),**provenance))
    atomic_json(dest/'progress.json',dict(complete=True,epoch=cfg['epochs'],epochs=cfg['epochs']))
    report=['# Small graph-query pretraining pilot','',
        f'{parameter_count:,} parameters; full float32; {len(m["train"])} train / {len(m["val"])} validation complexes.',
        m.get('label_source','Current PypKa AB midpoints')+'. '+m.get('scope','Frozen split; no test-set access.'),
        '', '| Model | Validation group-macro MAE | 95% group-bootstrap CI |', '|---|---:|---|']
    for name,s in final.items():report.append(f'| {name} | {s["mae"]:.4f} | {s["mae_ci95"]} |')
    report+=['','Final epoch selected in advance. 2,000 bootstrap replicates by sequence component.',
        'Raw/clean effects are conditional on the selected cohort and teacher-label provenance.',
        'Structural geometry is precomputed for this frozen-structure pilot; coordinate differentiation and LocalTerms heads are not implemented.']
    (out/'report.md').write_text('\n'.join(report)+'\n')


def memory_probe(out):
    """Run one worst-capacity training update and report allocator high-water marks."""
    m=read(out/'manifest.json');cfg=m['config'];seed=cfg['seed']
    params=initialize(jax.random.PRNGKey(seed),**cfg.get('architecture',{}))
    records=[r for r in m['records'] if r['split']=='train']
    largest=max(m['capacities'],key=lambda b:np.prod(m['capacities'][b]))
    members=sorted((r for r in records if bucket(r)==largest),key=lambda r:(r['n'],r['k'],r['q']),reverse=True)
    from .graph_batches import BatchLoader
    batch_size=cfg.get('batch_size',8);loader=BatchLoader(out,m,batch_size);loader.set_epoch(1)
    batch=loader.load([r['complex_id'] for r in members[:batch_size]])
    updates_per_epoch=(len(m['train'])+batch_size-1)//batch_size
    dropout_rate=cfg.get('dropout_rate',0.)
    training_predict=(lambda p,g,k:predict(p,g,key=k,dropout_rate=dropout_rate)) if dropout_rate else None
    engine=ScalarEngine(predict,scalar_schedule(cfg,updates_per_epoch),training_predict=training_predict)
    state=engine.optimizer.init(params)
    params,state,loss=engine.audited_batch_update(params,state,*batch,key=jax.random.PRNGKey(seed) if dropout_rate else None)
    jax.block_until_ready(loss);stats=jax.devices()[0].memory_stats() or {};loader.close()
    print(json.dumps(dict(bucket=largest,capacity=m['capacities'][largest],batch_size=batch_size,
        loss=float(loss),bytes_in_use=stats.get('bytes_in_use'),peak_bytes_in_use=stats.get('peak_bytes_in_use'),
        peak_gib=None if stats.get('peak_bytes_in_use') is None else stats['peak_bytes_in_use']/2**30)),flush=True)


def main():
    import sys
    action=sys.argv[1];out=Path(sys.argv[2]).resolve()
    gpu_actions=('tests','train','memory-probe','register-size','register-control','register-long')
    require_compute(threads=8 if action in gpu_actions else 1,gpu_benchmark=action in gpu_actions,
        allow_comp1400=action=='prepare-sidechains')
    jax.config.update('jax_enable_x64',False)
    if action=='register-long':
        import shutil
        source=Path(sys.argv[3]);seed=int(sys.argv[4]);assert seed in (17,29,43)
        expected=read(source/'manifest.json')
        expected['parent']={'path':str(source),'manifest_sha256':digest(source/'manifest.json')}
        expected['config'].update(seed=seed,epochs=100,strict_backbone=True,
            schedule='constant_then_cosine',constant_epochs=20,end_learning_rate=1e-5,
            input_features='residue identity, backbone geometry and termini; disulfide flag disabled')
        out.mkdir(parents=True,exist_ok=True)
        if (out/'manifest.json').exists():assert read(out/'manifest.json')==expected
        else:
            (out/'data').symlink_to(source/'data',target_is_directory=True)
            shutil.copy2(source/'preparation.json',out/'preparation.json');atomic_json(out/'manifest.json',expected)
    elif action=='prepare-sidechains':
        from .graph_data import prepare_sidechains
        prepare_sidechains(Path(sys.argv[3]),out)
    elif action=='register-control':
        import shutil
        source=Path(sys.argv[3]);expected=read(source/'manifest.json')
        expected['config']['zero_sidechains']=True
        expected['config']['input_features']='matched-capacity backbone control; side-chain columns all zero'
        expected['parent']={'path':str(source),'manifest_sha256':digest(source/'manifest.json')}
        out.mkdir(parents=True,exist_ok=True)
        if (out/'manifest.json').exists():assert read(out/'manifest.json')==expected
        else:
            (out/'data').symlink_to(source/'data',target_is_directory=True)
            shutil.copy2(source/'preparation.json',out/'preparation.json');atomic_json(out/'manifest.json',expected)
    elif action=='register-size':
        import shutil
        source=Path(sys.argv[3]);size=sys.argv[4]
        width,ff,count={'10k':(20,32,10229),'50k':(44,88,49709)}[size]
        expected=read(source/'manifest.json')
        expected['parent']={'path':str(source),'manifest_sha256':digest(source/'manifest.json')}
        expected['config'].update(architecture=dict(width=width,ff=ff),parameter_count=count)
        out.mkdir(parents=True,exist_ok=True)
        if (out/'manifest.json').exists():assert read(out/'manifest.json')==expected
        else:
            (out/'data').symlink_to(source/'data',target_is_directory=True)
            shutil.copy2(source/'preparation.json',out/'preparation.json')
            atomic_json(out/'manifest.json',expected)
    elif action=='prepare':prepare(Path(sys.argv[3]),out)
    elif action=='memory-probe':memory_probe(out)
    elif action=='tests':
        import subprocess
        subprocess.run([sys.executable,'-m','pytest','-q','-x','tests/test_pkanet.py'],check=True)
        atomic_json(out/'tests.json',dict(passed=True,code_hashes=hashes()))
    elif action=='train':train(out)
    else:raise ValueError(action)


if __name__=='__main__':main()
