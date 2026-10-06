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
from .graph_data import prepare,load,bucket
from .trainer import ScalarEngine,save_checkpoint,load_checkpoint,sample_epoch


def hashes():
    base=Path(__file__).parent
    paths=list((base.parent/'pkanet').glob('*.py'))+[base/name for name in
        ('graph_data.py','graph_experiment.py','trainer.py','losses.py')]
    return {str(p):digest(p) for p in paths}


def inputs(out,r,m):return load(out,r,m['capacities'][bucket(r)])


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
    cfg=m['config'];seed=cfg['seed'];params=initialize(jax.random.PRNGKey(seed),**cfg.get('architecture',{}));engine=ScalarEngine(predict,cfg['learning_rate'])
    state=engine.optimizer.init(params);rng=np.random.default_rng(seed)
    provenance=dict(manifest_sha256=digest(out/'manifest.json'),code_hashes=hashes())
    dest=out/f'seed-{seed}';dest.mkdir(exist_ok=True);latest=dest/'checkpoints/latest.json';start=0
    if latest.exists():
        folder=dest/'checkpoints'/read(latest)['checkpoint'];params,state,meta=load_checkpoint(folder,(params,state))
        assert all(meta[k]==v for k,v in provenance.items());start=meta['epoch'];rng.bit_generator.state=meta['rng']
    means=constants(out,m);atomic_json(out/'train_type_means.json',dict(zip(GROUPS,means.tolist())))
    parameter_count=sum(x.size for x in jax.tree.leaves(params))
    assert parameter_count==cfg.get('parameter_count',28565) and all(x.dtype==jnp.float32 for x in jax.tree.leaves(params))
    atomic_json(dest/'run.json',dict(parameter_count=parameter_count,config=cfg,**provenance))
    if not (dest/'initial.json').exists():
        atomic_json(dest/'initial.json',evaluate(out,m,engine,params,means,replicates=2000))
    byid={r['complex_id']:r for r in m['records']};records=[byid[c] for c in m['train']]
    began=time.monotonic();history=read(dest/'history.json') if (dest/'history.json').exists() else []
    for epoch in range(start,cfg['epochs']):
        order=sample_epoch(records,rng);losses=[];gradients=[];updates=0;t0=time.monotonic()
        for i,cid in enumerate(order):
            r=byid[cid];graph,y,mask=inputs(out,r,m)
            gradient,loss=engine.audited_gradient(params,graph,y,mask);gradients.append(gradient);losses.append(loss)
            if len(gradients)==cfg['accumulation'] or i==len(order)-1:
                params,state=engine.update(params,state,gradients);gradients=[];updates+=1
            if (i+1)%50==0:
                atomic_json(dest/'progress.json',dict(epoch=epoch+1,epochs=cfg['epochs'],complexes=i+1,total=len(order),elapsed_seconds=time.monotonic()-t0))
        metrics=evaluate(out,m,engine,params,means)
        row=dict(epoch=epoch+1,train_mse=float(np.mean(losses)),validation=metrics,
            seconds=time.monotonic()-t0,updates=updates)
        history.append(row);atomic_json(dest/'history.json',history)
        save_checkpoint(dest/'checkpoints'/f'epoch-{epoch+1:03d}',params,state,
            dict(provenance,epoch=epoch+1,rng=rng.bit_generator.state,parameter_count=parameter_count))
        print(json.dumps({'epoch':epoch+1,'train_mse':row['train_mse'],'val_mae':metrics['graph_query']['mae'],'seconds':row['seconds']}),flush=True)
        if time.monotonic()-began>32400 and epoch+1<cfg['epochs']:
            atomic_json(dest/'resume_required.json',dict(epoch=epoch+1));return
    final=evaluate(out,m,engine,params,means,replicates=2000,predictions=dest/'validation_predictions.csv')
    atomic_json(dest/'final.json',final)
    atomic_json(dest/'verification.json',dict(passed=True,epochs=cfg['epochs'],parameter_count=parameter_count,
        finite_gradients=True,**provenance))
    report=['# Small graph-query pretraining pilot','',
        f'{parameter_count:,} parameters; full float32; {len(m["train"])} train / {len(m["val"])} validation complexes.',
        'Current PypKa AB midpoints, all eligible sites. Frozen component split; no paired objective, physical solver, or test-set access.',
        '', '| Model | Validation group-macro MAE | 95% group-bootstrap CI |', '|---|---:|---|']
    for name,s in final.items():report.append(f'| {name} | {s["mae"]:.4f} | {s["mae_ci95"]} |')
    report+=['','Final epoch selected in advance. 2,000 bootstrap replicates by sequence component.',
        'This tests the small-model training path, not historical pKPDB pretraining or a causal benefit of cleaning.',
        'Backbone geometry is precomputed for this frozen-structure pilot; coordinate differentiation and LocalTerms heads are not implemented.']
    (out/'report.md').write_text('\n'.join(report)+'\n')


def main():
    import sys
    action=sys.argv[1];out=Path(sys.argv[2]).resolve()
    require_compute(threads=8 if action in ('tests','train','register-size') else 1,gpu_benchmark=action in ('tests','train','register-size'))
    jax.config.update('jax_enable_x64',False)
    if action=='register-size':
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
    elif action=='tests':
        import subprocess
        subprocess.run([sys.executable,'-m','pytest','-q','-x','tests/test_pkanet.py'],check=True)
        atomic_json(out/'tests.json',dict(passed=True,code_hashes=hashes()))
    elif action=='train':train(out)
    else:raise ValueError(action)


if __name__=='__main__':main()
