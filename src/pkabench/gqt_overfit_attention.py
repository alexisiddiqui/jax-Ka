"""Best-versus-late checkpoint attention and causal-context audit."""
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from .runtime import atomic_json,digest,require_compute

SIZES=('50k','200k','800k')
VARIANTS=('original','local_only','remove_titratable','remove_0_6A','remove_6_10A',
          'remove_10_15A','remove_15_20A','remove_identity','remove_geometry')


def read(path):return json.loads(Path(path).read_text())


def write_parquet(path,rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=Path(path);pending=path.with_name('.pending-'+path.name)
    pq.write_table(pa.Table.from_pylist(rows),pending);os.replace(pending,path)


def sources(root,size):
    early=(root/'pretraining/gqt-backbone-5k-pkmod-v1/unweighted' if size=='50k' else
           root/'pretraining/gqt-pkai-parameter-sweep-v1/gqt'/size/'unweighted')
    late=root/'pretraining/gqt-pkai-parameter-sweep-v1/gqt-long'/size
    return early,late


def checkpoint_spec(root,size):
    early,late=sources(root,size);history=read(late/'seed-17/history.json')
    best=min(history,key=lambda row:row['validation']['graph_query']['mae'])
    best_source=early if best['epoch']<=20 else late
    best_path=best_source/f"seed-17/checkpoints/epoch-{best['epoch']:03d}"
    latest=read(late/'seed-17/checkpoints/latest.json')['checkpoint'];late_path=late/'seed-17/checkpoints'/latest
    late_meta=read(late_path/'metadata.json')
    return dict(size=size,best_epoch=best['epoch'],best_validation_mae=best['validation']['graph_query']['mae'],
        best_source=str(best_source),best_checkpoint=str(best_path),best_sha256=digest(best_path/'state.npz'),
        late_epoch=late_meta['epoch'],late_checkpoint=str(late_path),late_sha256=digest(late_path/'state.npz'))


def sample_train(records,n=128,seed=1701):
    rng=np.random.default_rng(seed);order=np.arange(len(records));rng.shuffle(order);selected=[];components=set()
    for index in order:
        row=records[int(index)]
        if row['component_id'] in components:continue
        selected.append(row);components.add(row['component_id'])
        if len(selected)==n:break
    if len(selected)!=n:raise ValueError(f'Only {len(selected)} component-unique training records available')
    return selected


def batches(records,size=4):
    from pkatrain.graph_data import bucket
    groups=defaultdict(list)
    for row in records:groups[bucket(row)].append(row['complex_id'])
    return [members[i:i+size] for name in sorted(groups) for members in (groups[name],) for i in range(0,len(members),size)]


def load_params(manifest,path,scheduled):
    import jax
    from pkanet.model import initialize
    from pkatrain.graph_pkmod_compare import ExplicitShiftEngine
    from pkatrain.graph_pkmod_long import ScheduledExplicitShiftEngine
    from pkatrain.trainer import load_checkpoint
    cfg=manifest['config'];params=initialize(jax.random.PRNGKey(cfg['seed']),**cfg['architecture'])
    engine=(ScheduledExplicitShiftEngine(cfg['shift_bin_weights']) if scheduled else
            ExplicitShiftEngine(cfg['shift_bin_weights'],cfg['learning_rate']))
    state=engine.optimizer.init(params);params,_,meta=load_checkpoint(path,(params,state))
    return params,engine,meta


def edge_distance(edge):
    centers=np.linspace(0,20,16,dtype=np.float32);weight=np.asarray(edge)[...,:16]
    return (weight*centers).sum(-1)/np.maximum(weight.sum(-1),1e-12)


def attention_metrics(weights,graph,query_rows,query_count):
    from jaxpropka.parameters import GROUP_AA
    weights=np.asarray(weights)[:query_count];neighbors=np.asarray(graph['neighbors'])[query_rows[:query_count]]
    mask=np.asarray(graph['edge_mask'])[query_rows[:query_count]];edge=np.asarray(graph['edge'])[query_rows[:query_count]]
    nodes=np.asarray(graph['nodes']);receivers=np.asarray(query_rows[:query_count])[:,None]
    distance=edge_distance(edge);aa=nodes[:,:20].argmax(-1)
    tit=np.isin(aa,np.asarray(GROUP_AA))|(nodes[:,20]>.5)|(nodes[:,21]>.5)
    neighbor_tit=tit[neighbors];same_chain=edge[...,19]>.5;self_edge=neighbors==receivers
    safe=np.where(mask[...,None],weights,0.);entropy=-np.sum(np.where(safe>0,safe*np.log(np.maximum(safe,1e-30)),0),axis=1)
    def mass(select):return np.sum(safe*np.asarray(select)[...,None],axis=1)
    return dict(entropy=entropy,effective=np.exp(entropy),maximum=np.max(safe,axis=1),
        mean_distance=np.sum(safe*distance[...,None],axis=1),mass_self=mass(self_edge),
        mass_0_6=mass(mask&~self_edge&(distance<6)),mass_6_10=mass(mask&(distance>=6)&(distance<10)),
        mass_10_15=mass(mask&(distance>=10)&(distance<15)),mass_15_20=mass(mask&(distance>=15)),
        mass_cross_chain=mass(mask&~same_chain),mass_titratable=mass(mask&neighbor_tit))


def run(root,out):
    require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True)
    import jax
    import jax.numpy as jnp
    from pkanet.model import PKPDB_PK_MOD,predict_pkpdb_with_trace
    from pkatrain.graph_batches import BatchLoader
    from pkabench.gqt_learning_diagnostics import perturb_batch
    out.mkdir(parents=True,exist_ok=True)
    specs=[checkpoint_spec(root,size) for size in SIZES]
    base_manifest=read(sources(root,'50k')[1]/'manifest.json')
    train=sample_train([r for r in base_manifest['records'] if r['split']=='train'])
    val=[r for r in base_manifest['records'] if r['split']=='val'];selected=train+val
    selection=dict(train_complexes=[r['complex_id'] for r in train],validation_complexes=[r['complex_id'] for r in val],
        train_components_unique=True,seed=1701)
    atomic_json(out/'manifest.json',dict(checkpoints=specs,selection=selection,variants=list(VARIANTS),
        attention_metrics=['entropy','effective','maximum','mean_distance','mass_self','mass_0_6','mass_6_10','mass_10_15','mass_15_20','mass_cross_chain','mass_titratable'],
        scope='128 component-unique training complexes and every frozen validation complex; no test data',test_data_included=False))
    site_rows=[];attention_rows=[]
    for spec in specs:
        size=spec['size'];early,late=sources(root,size);manifest=read(late/'manifest.json');byid={r['complex_id']:r for r in manifest['records']}
        loader=BatchLoader(late,manifest,4,backend='mmap');loader.set_epoch(1)
        checkpoints=(('best',Path(spec['best_checkpoint']),False),('late',Path(spec['late_checkpoint']),True))
        for checkpoint,path,scheduled in checkpoints:
            params,engine,meta=load_params(manifest,path,scheduled);trace_batch=jax.jit(jax.vmap(predict_pkpdb_with_trace,in_axes=(None,0)))
            for number,cids in enumerate(batches(selected),1):
                graph,y,eligible,valid=loader.load(cids);trace=trace_batch(params,{k:jnp.asarray(v) for k,v in graph.items()})
                predictions={}
                for variant in VARIANTS:
                    view=perturb_batch(graph,eligible,variant);predictions[variant]=np.asarray(engine.batch_forward(params,view))
                for bi,cid in enumerate(cids):
                    record=byid[cid];q=int(eligible[bi].sum());groups=np.asarray(graph['query_group'][bi,:q]);baseline=np.asarray(PKPDB_PK_MOD)[groups]
                    target=np.asarray(y[bi,:q])-baseline;original=predictions['original'][bi,:q]-baseline
                    for qi,(key,t,p) in enumerate(zip(record['keys'][:q],target,original)):
                        row=dict(size=size,checkpoint=checkpoint,epoch=int(meta['epoch']),split=record['split'],component_id=record['component_id'],
                            complex_id=cid,chain=key[1],resnum=key[2],icode=key[3],group=key[4],query_index=qi,
                            teacher_shift=float(t),predicted_shift=float(p),absolute_error=float(abs(p-t)))
                        for variant in VARIANTS[1:]:
                            value=predictions[variant][bi,qi]-baseline[qi]
                            row['predicted_'+variant]=float(value);row['effect_'+variant]=float(value-p)
                        site_rows.append(row)
                    layers=[trace['encoder'][0]['weights'][bi],trace['encoder'][1]['weights'][bi],trace['query']['weights'][bi]]
                    query_rows=np.asarray(graph['query_residue'][bi,:q])
                    for li,weights in enumerate(layers):
                        local_weights=np.asarray(weights)[query_rows] if li<2 else np.asarray(weights)[:q]
                        metrics=attention_metrics(local_weights, {k:np.asarray(v[bi]) for k,v in graph.items()}, query_rows, q)
                        for qi,key in enumerate(record['keys'][:q]):
                            for head in range(weights.shape[-1]):
                                attention_rows.append(dict(size=size,checkpoint=checkpoint,epoch=int(meta['epoch']),split=record['split'],
                                    component_id=record['component_id'],complex_id=cid,chain=key[1],resnum=key[2],icode=key[3],group=key[4],query_index=qi,
                                    layer=('encoder_0','encoder_1','query')[li],head=head,
                                    **{name:float(value[qi,head]) for name,value in metrics.items()}))
                if number%25==0:
                    atomic_json(out/'progress.json',dict(size=size,checkpoint=checkpoint,epoch=int(meta['epoch']),batches=number,total=len(batches(selected))))
            del trace_batch,params,engine
        loader.close()
    write_parquet(out/'sites.parquet',site_rows);write_parquet(out/'attention.parquet',attention_rows)
    atomic_json(out/'verification.json',dict(complete=True,sites=len(site_rows),attention_rows=len(attention_rows),
        manifest_sha256=digest(out/'manifest.json'),sites_sha256=digest(out/'sites.parquet'),attention_sha256=digest(out/'attention.parquet'),test_data_included=False))


def _slope(target,predicted):
    return float(np.cov(target,predicted,ddof=0)[0,1]/np.var(target)) if np.var(target)>0 else None


def report(out):
    require_compute(threads=4,allow_comp1400=True)
    import pyarrow.parquet as pq
    from scipy.stats import spearmanr
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);sites=pq.read_table(out/'sites.parquet').to_pylist();attention=pq.read_table(out/'attention.parquet').to_pylist()
    site_key=lambda r:(r['size'],r['split'],r['component_id'],r['complex_id'],r['chain'],r['resnum'],r['icode'],r['group'],r['query_index'])
    att_key=lambda r:(r['size'],r['split'],r['component_id'],r['complex_id'],r['chain'],r['resnum'],r['icode'],r['group'],r['query_index'],r['layer'],r['head'])
    best={site_key(r):r for r in sites if r['checkpoint']=='best'};late={site_key(r):r for r in sites if r['checkpoint']=='late'};assert best.keys()==late.keys()
    metrics=[];bins=[];groups=[];causal=[]
    edges=(0,.5,1,2,float('inf'))
    for size in SIZES:
        for split in ('train','val'):
            for checkpoint in ('best','late'):
                rr=[r for r in sites if r['size']==size and r['split']==split and r['checkpoint']==checkpoint]
                target=np.asarray([r['teacher_shift'] for r in rr]);pred=np.asarray([r['predicted_shift'] for r in rr])
                metrics.append(dict(size=size,split=split,checkpoint=checkpoint,epoch=rr[0]['epoch'],sites=len(rr),
                    mae=float(np.mean(abs(pred-target))),rmse=float(np.sqrt(np.mean((pred-target)**2))),slope=_slope(target,pred),prediction_sd=float(pred.std())))
                if split=='val':
                    for lo,hi in zip(edges[:-1],edges[1:]):
                        ss=[r for r in rr if lo<=abs(r['teacher_shift'])<hi]
                        bins.append(dict(size=size,checkpoint=checkpoint,bin=f'{lo:g}-{"inf" if not np.isfinite(hi) else f"{hi:g}"}',sites=len(ss),mae=float(np.mean([r['absolute_error'] for r in ss]))))
                    for group in sorted({r['group'] for r in rr}):
                        ss=[r for r in rr if r['group']==group];groups.append(dict(size=size,checkpoint=checkpoint,group=group,sites=len(ss),mae=float(np.mean([r['absolute_error'] for r in ss]))))
                    for variant in VARIANTS[1:]:
                        ablated=np.asarray([r['predicted_'+variant] for r in rr])
                        causal.append(dict(size=size,checkpoint=checkpoint,variant=variant,
                            mean_absolute_effect=float(np.mean(abs(ablated-pred))),ablated_mae=float(np.mean(abs(ablated-target)))))
    attention_best={att_key(r):r for r in attention if r['checkpoint']=='best'};attention_late={att_key(r):r for r in attention if r['checkpoint']=='late'}
    assert attention_best.keys()==attention_late.keys();attention_change=[]
    names=('entropy','effective','maximum','mean_distance','mass_self','mass_0_6','mass_6_10','mass_10_15','mass_15_20','mass_cross_chain','mass_titratable')
    for size in SIZES:
        for split in ('train','val'):
            for layer in ('encoder_0','encoder_1','query'):
                for head in range(4):
                    keys=[k for k in attention_best if k[0]==size and k[1]==split and k[-2:]==(layer,head)]
                    row=dict(size=size,split=split,layer=layer,head=head,n=len(keys))
                    for name in names:
                        delta=np.asarray([attention_late[k][name]-attention_best[k][name] for k in keys]);row['delta_'+name]=float(delta.mean())
                    attention_change.append(row)
    correlations=[]
    for size in SIZES:
        for layer in ('encoder_0','encoder_1','query'):
            for name in names:
                xs=[];ys=[]
                for key,a in attention_best.items():
                    if key[0]!=size or key[1]!='val' or key[-2]!=layer:continue
                    site=key[:-2];xs.append(attention_late[key][name]-a[name]);ys.append(late[site]['absolute_error']-best[site]['absolute_error'])
                rho,p=spearmanr(xs,ys);correlations.append(dict(size=size,layer=layer,metric=name,rho=float(rho),p=float(p),n=len(xs)))
    deterioration=[]
    for size in SIZES:
        for split in ('train','val'):
            keys=[k for k in best if k[0]==size and k[1]==split]
            deterioration.append(dict(size=size,split=split,n=len(keys),mean_error_change=float(np.mean([late[k]['absolute_error']-best[k]['absolute_error'] for k in keys])),
                fraction_worse=float(np.mean([late[k]['absolute_error']>best[k]['absolute_error'] for k in keys]))))
    results=dict(metrics=metrics,shift_bins=bins,groups=groups,causal=causal,attention_change=attention_change,
        attention_error_correlations=correlations,deterioration=deterioration)
    atomic_json(out/'results.json',results)
    plots=out/'plots';plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,size in zip(axes,SIZES):
        rows=[r for r in attention_change if r['size']==size and r['split']=='val']
        matrix=np.asarray([[r['delta_mass_self'],r['delta_mass_0_6'],r['delta_mass_6_10'],r['delta_mass_10_15'],r['delta_mass_15_20']] for r in rows])
        image=ax.imshow(matrix,aspect='auto',cmap='coolwarm',vmin=-max(abs(matrix.min()),abs(matrix.max())),vmax=max(abs(matrix.min()),abs(matrix.max())))
        ax.set_title(size);ax.set_xticks(range(5),['self','0–6','6–10','10–15','15–20'],rotation=35,ha='right');ax.set_yticks(range(len(rows)),[f"{r['layer']}/h{r['head']}" for r in rows])
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Late − best attention mass');fig.suptitle('Validation attention redistribution during overfitting');fig.savefig(plots/'attention_mass_change.png',dpi=180,bbox_inches='tight');plt.close(fig)
    lines=['# What does the GQT overfit?','',
        'Best-validation and last-completed checkpoints are compared on the same 128 component-unique training complexes and every frozen validation complex. Attention changes are descriptive; context ablations provide the causal check. No test data were read.','',
        '| Size | Split | Best epoch MAE | Late epoch MAE | Error change | Fraction of sites worse |','|---|---|---:|---:|---:|---:|']
    md={(r['size'],r['split'],r['checkpoint']):r for r in metrics};dd={(r['size'],r['split']):r for r in deterioration}
    for size in SIZES:
        for split in ('train','val'):
            a,b=md[size,split,'best'],md[size,split,'late'];d=dd[size,split]
            lines.append(f"| {size} | {split} | e{a['epoch']}: {a['mae']:.4f} | e{b['epoch']}: {b['mae']:.4f} | {d['mean_error_change']:+.4f} | {d['fraction_worse']:.3f} |")
    lines+=['','![Attention redistribution](plots/attention_mass_change.png)','',
        'Detailed shift-bin, residue-group, attention-head, attention/error-correlation and causal-ablation tables are recorded in `results.json`.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');verification=read(out/'verification.json');verification.update(report_complete=True,
        results_sha256=digest(out/'results.json'),report_sha256=digest(out/'report.md'));atomic_json(out/'verification.json',verification)


def main():
    import sys
    action=sys.argv[1];root=Path(os.environ['PKABENCH_RUNTIME']);out=root/'audits/gqt-overfit-attention-v1'
    if action=='run':run(root,out)
    elif action=='report':report(out)
    else:raise ValueError(action)


if __name__=='__main__':main()
