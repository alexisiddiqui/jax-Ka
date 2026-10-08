"""Minimal diagnostics for compression and context use in backbone-only GQT."""
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jaxpropka.parameters import GROUPS,GROUP_AA,MODEL_PKA
from pkanet.model import initialize,predict
from pkatrain.graph_batches import BatchLoader,epoch_batches
from pkatrain.graph_data import bucket
from pkatrain.trainer import ScalarEngine,load_checkpoint,save_checkpoint
from .runtime import atomic_json,digest,require_compute

COUPLING_BINS=((0.,.25),(.25,.5),(.5,1.),(1.,2.),(2.,float('inf')))
VARIANTS=('original','local_only','remove_titratable','remove_0_6A','remove_6_10A','remove_10_15A','remove_15_20A','remove_identity','remove_geometry')


def read(path):return json.loads(Path(path).read_text())


def calibration(target,predicted,group):
    target=np.asarray(target,float)-MODEL_PKA[np.asarray(group,int)];predicted=np.asarray(predicted,float)-MODEL_PKA[np.asarray(group,int)]
    slope=float(np.cov(target,predicted,ddof=0)[0,1]/np.var(target)) if np.var(target)>0 else None
    return dict(n=len(target),mae=float(np.mean(abs(predicted-target))),rmse=float(np.sqrt(np.mean((predicted-target)**2))),
        slope=slope,intercept=None if slope is None else float(predicted.mean()-slope*target.mean()),
        correlation=float(np.corrcoef(target,predicted)[0,1]) if target.std()>0 and predicted.std()>0 else None,
        teacher_std=float(target.std()),prediction_std=float(predicted.std()),variance_ratio=float(predicted.std()/target.std()) if target.std()>0 else None,
        mean_tanh_derivative=float(np.mean(8*(1-np.clip(predicted/8,-1,1)**2))),
        low_derivative_fraction=float(np.mean(8*(1-np.clip(predicted/8,-1,1)**2)<.8)))


def edge_distance(edge):
    centers=np.linspace(0,20,16,dtype=np.float32)
    weights=np.asarray(edge)[...,:16]
    return (weights*centers).sum(-1)/np.maximum(weights.sum(-1),1e-12)


def perturb_batch(graph,eligible,variant):
    """Create global context ablations; self edges and supervised centres are retained."""
    result={k:np.array(v,copy=True) for k,v in graph.items()};assert result['neighbors'].ndim==3
    b,n,k=result['neighbors'].shape;receivers=np.arange(n)[None,:,None]
    selfedge=result['neighbors']==receivers;context=result['edge_mask']&~selfedge
    if variant=='original':return result
    if variant=='local_only':remove=context
    elif variant=='remove_titratable':
        aa=result['nodes'][...,:20].argmax(-1);tit=np.isin(aa,np.asarray(GROUP_AA))|(result['nodes'][...,20]>.5)|(result['nodes'][...,21]>.5)
        remove=context&tit[np.arange(b)[:,None,None],result['neighbors']]
    elif variant.startswith('remove_') and variant.endswith('A'):
        lo,hi={'remove_0_6A':(0,6),'remove_6_10A':(6,10),'remove_10_15A':(10,15),'remove_15_20A':(15,20)}[variant]
        distance=edge_distance(result['edge']);remove=context&(distance>=lo)&(distance<hi)
    elif variant=='remove_identity':
        protected=np.zeros((b,n),bool)
        for i in range(b):protected[i,result['query_residue'][i,np.asarray(eligible[i],bool)]]=True
        result['nodes'][~protected,:20]=0
        return result
    elif variant=='remove_geometry':
        result['edge'][context,:19]=0
        return result
    else:raise ValueError(variant)
    result['edge_mask'][remove]=False;result['switch'][remove]=0
    return result


def load_frozen(source):
    m=read(source/'manifest.json');cfg=m['config'];params=initialize(jax.random.PRNGKey(cfg['seed']),**cfg['architecture'])
    engine=ScalarEngine(predict,cfg['learning_rate']);state=engine.optimizer.init(params)
    checkpoint=source/f"seed-{cfg['seed']}/checkpoints"/read(source/f"seed-{cfg['seed']}/checkpoints/latest.json")['checkpoint']
    params,_,meta=load_checkpoint(checkpoint,(params,state));assert meta['epoch']==cfg['epochs'] and meta['parameter_count']==cfg['parameter_count']
    model_path=Path(__file__).parent.parent/'pkanet/model.py';assert meta['code_hashes'][str(model_path)]==digest(model_path)
    return m,engine,params,checkpoint,meta


def batches(records,size):
    grouped=defaultdict(list)
    for r in records:grouped[bucket(r)].append(r['complex_id'])
    return [ids[i:i+size] for name in sorted(grouped) for ids in (grouped[name],) for i in range(0,len(ids),size)]


def predict_records(source,m,engine,params,records,variants=('original',)):
    loader=BatchLoader(source,m,m['config']['batch_size']);loader.set_epoch(1);byid={r['complex_id']:r for r in m['records']};rows=[]
    for number,cids in enumerate(batches(records,m['config']['batch_size']),1):
        graph,y,mask,valid=loader.load(cids)
        for variant in variants:
            view=perturb_batch(graph,mask,variant);pred=np.asarray(engine.batch_forward(params,view))
            for i,cid in enumerate(cids):
                r=byid[cid];q=int(mask[i].sum())
                for key,target,value,g in zip(r['keys'][:q],y[i,:q],pred[i,:q],graph['query_group'][i,:q]):
                    rows.append(dict(variant=variant,split=r['split'],component_id=r['component_id'],complex_id=cid,
                        chain=key[1],resnum=key[2],icode=key[3],group=key[4],group_index=int(g),teacher_pka=float(target),predicted_pka=float(value)))
        if number%100==0:print(json.dumps({'prediction_batches':number,'total':len(batches(records,m['config']['batch_size']))}),flush=True)
    loader.close();return rows


def summarize_predictions(rows):
    result={}
    for key in sorted({(r['split'],r['variant']) for r in rows}):
        rr=[r for r in rows if (r['split'],r['variant'])==key]
        result[':'.join(key)]=calibration([r['teacher_pka'] for r in rr],[r['predicted_pka'] for r in rr],[r['group_index'] for r in rr])
    base={(r['complex_id'],r['chain'],r['resnum'],r['icode'],r['group']):r for r in rows if r['split']=='val' and r['variant']=='original'}
    for variant in VARIANTS[1:]:
        rr=[r for r in rows if r['split']=='val' and r['variant']==variant]
        delta=[abs(r['predicted_pka']-base[(r['complex_id'],r['chain'],r['resnum'],r['icode'],r['group'])]['predicted_pka']) for r in rr]
        result['val:'+variant]['mean_absolute_prediction_change']=float(np.mean(delta));result['val:'+variant]['p95_absolute_prediction_change']=float(np.quantile(delta,.95))
    return result


def coupling_map(root):
    import pyarrow.parquet as pq
    rows=pq.read_table(root/'audits/gqt-data-alignment-v1/prediction_audit.parquet').to_pylist();out={}
    for r in rows:
        if r['model']=='gqt_sidechain_dropout':out[(r['complex_id'],r['chain'],r['resnum'],r['icode'],r['group'])]=r['max_teacher_coupling_kbt']
    return out


def tree_add(acc,gradient,weight):
    if acc is None:return jax.tree.map(lambda x:np.asarray(x,dtype=np.float64)*weight,gradient)
    return jax.tree.map(lambda a,x:a+np.asarray(x,dtype=np.float64)*weight,acc,gradient)


def block_norms(gradient):
    norm=lambda tree:float(np.sqrt(sum(np.sum(np.asarray(x,dtype=np.float64)**2) for x in jax.tree.leaves(tree))))
    return dict(total=norm(gradient),head=norm(gradient['head']),query=norm(gradient['query']),embed=norm(gradient['embed']),groups=norm(gradient['groups']),
        encoder_0=norm(gradient['blocks'][0]),encoder_1=norm(gradient['blocks'][1]))


def gradient_audit(root,source,m,engine,params):
    coupling=coupling_map(root);records=[r for r in m['records'] if r['split']=='val'];byid={r['complex_id']:r for r in records};loader=BatchLoader(source,m,m['config']['batch_size']);loader.set_epoch(1)
    gradients={};summaries=[]
    for lo,hi in COUPLING_BINS:
        acc=None;weight=0;sites=0
        for cids in batches(records,m['config']['batch_size']):
            graph,y,eligible,valid=loader.load(cids);selected=np.zeros_like(eligible)
            for i,cid in enumerate(cids):
                values=[coupling.get(tuple(key)) for key in byid[cid]['keys']]
                selected[i,:len(values)]=[v is not None and lo<=v<hi for v in values]
            chosen=selected.any(axis=1)&np.asarray(valid);count=int(chosen.sum())
            if not count:continue
            _,gradient=engine.batch_value_grad(params,graph,y,selected,chosen);acc=tree_add(acc,gradient,count);weight+=count;sites+=int(selected.sum())
        assert weight and sites;mean=jax.tree.map(lambda x:x/weight,acc);name=f'{lo:g}-{"inf" if not np.isfinite(hi) else f"{hi:g}"}'
        gradients[name]=mean;summaries.append(dict(bin=name,lo_kbt=lo,hi_kbt=None if not np.isfinite(hi) else hi,complexes=weight,sites=sites,**block_norms(mean)))
    names=list(gradients);cosines=[]
    flat={name:np.concatenate([np.ravel(x) for x in jax.tree.leaves(g)]) for name,g in gradients.items()}
    for i,a in enumerate(names):
        for b in names[i+1:]:cosines.append(dict(bin_a=a,bin_b=b,cosine=float(np.dot(flat[a],flat[b])/(np.linalg.norm(flat[a])*np.linalg.norm(flat[b])))))
    loader.close();return dict(bins=summaries,cosines=cosines)


def select_memorization(records,seed=17):
    rng=np.random.default_rng(seed);bybucket=defaultdict(list)
    for r in records:bybucket[bucket(r)].append(r)
    quotas={'384':48,'768':12,'100000':4};chosen=[];components=set()
    for name,quota in quotas.items():
        candidates=bybucket[name].copy();rng.shuffle(candidates)
        for r in candidates:
            if r['component_id'] in components:continue
            chosen.append(r);components.add(r['component_id'])
            if sum(bucket(x)==name for x in chosen)==quota:break
    assert len(chosen)==64 and len(components)==64
    return chosen


def subset_metrics(loader,engine,params,records,batch_size):
    byid={r['complex_id']:r for r in records};target=[];prediction=[];groups=[]
    for cids in batches(records,batch_size):
        graph,y,mask,valid=loader.load(cids);pred=np.asarray(engine.batch_forward(params,graph))
        for i,cid in enumerate(cids):
            q=int(mask[i].sum());target.extend(y[i,:q]);prediction.extend(pred[i,:q]);groups.extend(graph['query_group'][i,:q])
    return calibration(target,prediction,groups)


def memorization(source,m,out):
    cfg=m['config'];records=select_memorization([r for r in m['records'] if r['split']=='train']);byid={r['complex_id']:r for r in records};batch_size=8
    params=initialize(jax.random.PRNGKey(cfg['seed']),**cfg['architecture']);engine=ScalarEngine(predict,cfg['learning_rate']);state=engine.optimizer.init(params)
    loader=BatchLoader(source,m,batch_size);loader.set_epoch(1);rng=np.random.default_rng(cfg['seed']);history=[];best=float('inf');last_improvement=0;began=time.monotonic()
    for epoch in range(1,501):
        order=[r['complex_id'] for r in records];rng.shuffle(order);loss=[]
        for cids in epoch_batches(order,byid,rng,batch_size):
            graph,y,mask,valid=loader.load(cids);params,state,value=engine.audited_batch_update(params,state,graph,y,mask,valid);loss.append(value)
        if epoch==1 or epoch%10==0:
            metrics=subset_metrics(loader,engine,params,records,batch_size);row=dict(epoch=epoch,train_batch_mse=float(np.mean(loss)),seconds=time.monotonic()-began,**metrics);history.append(row);atomic_json(out/'memorization-history.json',history)
            if best-metrics['mae']>=.001:best=metrics['mae'];last_improvement=epoch
            print(json.dumps({'memorization_epoch':epoch,'mae':metrics['mae'],'slope':metrics['slope']}),flush=True)
            if metrics['mae']<=.05 or (epoch>=150 and epoch-last_improvement>=100):break
    save_checkpoint(out/'memorization-checkpoint',params,state,dict(epoch=epoch,selected=[r['complex_id'] for r in records],parameter_count=cfg['parameter_count']))
    loader.close();final=history[-1];final.update(stopped='target_mae' if final['mae']<=.05 else 'plateau' if epoch<500 else 'epoch_cap',selected=[r['complex_id'] for r in records])
    return final


def write_parquet(path,rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows),path)


def run(root,out):
    require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True);source=root/'pretraining/gqt-backbone-batch-sweep-v1/batch-8';m,engine,params,checkpoint,meta=load_frozen(source)
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),checkpoint=str(checkpoint),checkpoint_state_sha256=digest(checkpoint/'state.npz'),
        alignment_verification_sha256=digest(root/'audits/gqt-data-alignment-v1/verification.json'),variants=list(VARIANTS),
        coupling_bins=[[lo,None if not np.isfinite(hi) else hi] for lo,hi in COUPLING_BINS],
        scope='Backbone-only frozen checkpoint diagnostics plus 64-complex train-only memorization; validation is reporting-only; test excluded'))
    train=[r for r in m['records'] if r['split']=='train'];val=[r for r in m['records'] if r['split']=='val']
    rows=predict_records(source,m,engine,params,train,('original',))+predict_records(source,m,engine,params,val,VARIANTS);write_parquet(out/'predictions.parquet',rows)
    diagnostics=summarize_predictions(rows);atomic_json(out/'prediction-diagnostics.json',diagnostics);atomic_json(out/'progress.json',dict(stage='predictions',complete=True))
    gradients=gradient_audit(root,source,m,engine,params);atomic_json(out/'gradient-audit.json',gradients);atomic_json(out/'progress.json',dict(stage='gradients',complete=True))
    memory=memorization(source,m,out);atomic_json(out/'memorization.json',memory)
    base_train=diagnostics['train:original'];base_val=diagnostics['val:original'];local=diagnostics['val:local_only'];titr=diagnostics['val:remove_titratable']
    result=dict(passed=True,training=base_train,validation=base_val,perturbations={k:v for k,v in diagnostics.items() if k.startswith('val:')},gradients=gradients,memorization=memory,test_data_included=False)
    atomic_json(out/'report.json',result)
    lines=['# Backbone-only GQT learning diagnostics','',
        f"Frozen checkpoint train/validation shift slopes: **{base_train['slope']:.3f} / {base_val['slope']:.3f}**; variance ratios: **{base_train['variance_ratio']:.3f} / {base_val['variance_ratio']:.3f}**.",'',
        '| Validation view | MAE | Shift slope | Mean absolute prediction change |','|---|---:|---:|---:|']
    for variant in VARIANTS:
        r=diagnostics['val:'+variant];lines.append(f"| {variant} | {r['mae']:.4f} | {r['slope']:.3f} | {r.get('mean_absolute_prediction_change',0):.4f} |")
    lines+=['','| Coupling bin (kBT) | Sites | Head grad | Query grad | Encoder 0 | Encoder 1 |','|---|---:|---:|---:|---:|---:|']
    for r in gradients['bins']:lines.append(f"| {r['bin']} | {r['sites']:,} | {r['head']:.3g} | {r['query']:.3g} | {r['encoder_0']:.3g} | {r['encoder_1']:.3g} |")
    lines+=['',f"The 64-complex memorization run stopped by **{memory['stopped']}** at epoch {memory['epoch']}: MAE {memory['mae']:.4f}, slope {memory['slope']:.3f}.",
        '', 'Perturbations are global graph ablations. `local_only` retains self edges; shell assignments reconstruct distance from the stored 16-channel RBF.',
        'Native coupling is used only to stratify frozen validation diagnostics. It is never an optimization target. No test data were read.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');atomic_json(out/'verification.json',dict(passed=True,report_sha256=digest(out/'report.json'),predictions_sha256=digest(out/'predictions.parquet'),test_data_included=False))


if __name__=='__main__':
    root=Path(os.environ['PKABENCH_RUNTIME']);final=root/'audits/gqt-learning-diagnostics-v1'
    if final.exists():assert read(final/'verification.json')['passed'];print(json.dumps({'status':'already_complete'}))
    else:
        work=final.with_name('.'+final.name+'-'+os.environ['SLURM_JOB_ID']);run(root,work);os.replace(work,final)
