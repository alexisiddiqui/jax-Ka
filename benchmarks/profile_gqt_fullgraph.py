"""Full-graph GQT loader, fused-update, validation, and cache profiling."""
import json
import os
from pathlib import Path
import time

import jax
import numpy as np

from jaxpropka.parameters import GROUPS
from pkanet.model import initialize
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.compilation_cache import configure
from pkatrain.graph_batches import BatchLoader,LoaderTelemetry,epoch_batches
from pkatrain.graph_data import bucket,load,mask_features
from pkatrain.graph_experiment import evaluate
from pkatrain.graph_pkmod_compare import ExplicitShiftEngine,hashes
from pkatrain.records import read
from pkatrain.trainer import sample_epoch


MODELS={
    '50k':'pretraining/gqt-backbone-5k-pkmod-v1/unweighted',
    '200k':'pretraining/gqt-pkai-parameter-sweep-v1/gqt/200k/unweighted',
}


def ready(tree):
    return jax.tree.map(lambda value:value.block_until_ready() if hasattr(value,'block_until_ready') else value,tree)


def quantiles(values):
    values=np.asarray(values,float)
    if not len(values):return dict(n=0,total=0.,mean=None,median=None,p95=None,max=None)
    return dict(n=len(values),total=float(values.sum()),mean=float(values.mean()),
        median=float(np.median(values)),p95=float(np.quantile(values,.95)),max=float(values.max()))


def summarize_telemetry(raw):
    structures=raw['structures'];batches=raw['batches'];waits=raw['prefetch_waits']
    return dict(
        structures=len(structures),batches=len(batches),
        verify=quantiles([x['verify_seconds'] for x in structures if x['verified']]),
        read=quantiles([x['read_seconds'] for x in structures]),
        augment=quantiles([x['augment_seconds'] for x in structures]),
        pad_and_mask=quantiles([x['pad_and_mask_seconds'] for x in structures]),
        batch_load=quantiles([x['load_seconds'] for x in batches]),
        batch_stack=quantiles([x['stack_seconds'] for x in batches]),
        startup_wait=quantiles([x['seconds'] for x in waits if x['startup']]),
        steady_wait=quantiles([x['seconds'] for x in waits if not x['startup']]),
        actual_nodes=sum(x['actual_nodes'] for x in batches),padded_nodes=sum(x['padded_nodes'] for x in batches),
        actual_edges=sum(x['actual_edges'] for x in batches),padded_edges=sum(x['padded_edges'] for x in batches),
        actual_queries=sum(x['actual_queries'] for x in batches),padded_queries=sum(x['padded_queries'] for x in batches))


def manifest_for(root,size):
    out=root/MODELS[size];manifest=read(out/'manifest.json')
    assert manifest['config']['parameter_count']==({'50k':49709,'200k':209645}[size])
    return out,manifest


def make_engine(manifest):
    cfg=manifest['config'];params=initialize(jax.random.PRNGKey(cfg['seed']),**cfg['architecture'])
    engine=ExplicitShiftEngine(cfg['shift_bin_weights'],cfg['learning_rate'])
    return engine,params,engine.optimizer.init(params)


def plans_for(manifest,seed=17):
    byid={row['complex_id']:row for row in manifest['records']}
    records=[byid[cid] for cid in manifest['train']]
    rng=np.random.default_rng(seed);order=sample_epoch(records,rng)
    return byid,epoch_batches(order,byid,rng,manifest['config']['batch_size'])


def attributed_profile(out,manifest,engine,params,state):
    byid,plans=plans_for(manifest);bybucket={name:[] for name in manifest['capacities']}
    for members in plans:bybucket[bucket(byid[members[0]])].append(members)
    rows=[]
    for name,memberships in sorted(bybucket.items(),key=lambda item:int(item[0])):
        selected=memberships[:min(30,len(memberships))]
        telemetry=LoaderTelemetry();loader=BatchLoader(out,manifest,manifest['config']['batch_size'],telemetry=telemetry)
        load_times=[];transfer_times=[];times=[];compile_and_step=None
        for number,members in enumerate(selected):
            t=time.perf_counter();batch=loader.load(members);load_seconds=time.perf_counter()-t
            t=time.perf_counter();device_batch=ready(jax.device_put(batch));transfer_seconds=time.perf_counter()-t
            load_times.append(load_seconds);transfer_times.append(transfer_seconds)
            if number==0:
                t=time.perf_counter();ready(engine.batch_step(params,state,*device_batch));compile_and_step=time.perf_counter()-t
            t=time.perf_counter();ready(engine.batch_step(params,state,*device_batch));times.append(time.perf_counter()-t)
        loader.close();raw=telemetry.snapshot()
        rows.append(dict(bucket=name,capacities=manifest['capacities'][name],sampled_batches=len(selected),
            compile_and_first_step_seconds=compile_and_step,
            explicit_load=quantiles(load_times),host_to_device=quantiles(transfer_times),
            fused_step=quantiles(times),loader=summarize_telemetry(raw)))
    return rows


def throughput_epoch(out,manifest,engine,params,state,epoch):
    byid,plans=plans_for(manifest,17+epoch);telemetry=LoaderTelemetry()
    loader=BatchLoader(out,manifest,manifest['config']['batch_size'],telemetry=telemetry);loader.set_epoch(epoch)
    began=time.perf_counter();steps=[]
    for _,batch in loader.iterate(plans):
        t=time.perf_counter();params,state,loss=engine.audited_batch_update(params,state,*batch);steps.append(time.perf_counter()-t)
    elapsed=time.perf_counter()-began;loader.close();raw=telemetry.snapshot();summary=summarize_telemetry(raw)
    return params,state,dict(epoch=epoch,seconds=elapsed,updates=len(plans),fused_call=quantiles(steps),loader=summary,
        steady_prefetch_wait_fraction=summary['steady_wait']['total']/elapsed)


def validation_profile(out,manifest,engine,params):
    means=np.asarray([read(out/'train_type_means.json')[group] for group in GROUPS])
    val=[row for row in manifest['records'] if row['split']=='val']
    # Compile every validation shape outside the timed stage attribution.
    warmed=set();warm_seconds=0.
    for row in val:
        name=bucket(row)
        if name in warmed:continue
        graph,_,_=load(out,row,manifest['capacities'][name]);graph=mask_features(graph,manifest['config'])
        t=time.perf_counter();ready(engine.forward(params,jax.device_put(graph)));warm_seconds+=time.perf_counter()-t;warmed.add(name)
    stages={name:[] for name in ('hash','read','pad_mask','transfer','inference')};scores={name:[] for name in ('graph_query','model_values','train_type_mean')}
    from pkabench.frozen_score import measures,aggregate
    for row in val:
        path=out/'data'/row['complex_id']/'graph.npz'
        t=time.perf_counter();assert digest(path)==row['sha256'];stages['hash'].append(time.perf_counter()-t)
        t=time.perf_counter()
        with np.load(path,allow_pickle=False) as data:
            raw={key:data[key] for key in data.files if key!='labels'};labels=data['labels']
        stages['read'].append(time.perf_counter()-t)
        from pkatrain.graph_data import pad
        t=time.perf_counter();graph,y,mask=pad(raw,labels,manifest['capacities'][bucket(row)]);graph=mask_features(graph,manifest['config']);stages['pad_mask'].append(time.perf_counter()-t)
        t=time.perf_counter();device_graph=ready(jax.device_put(graph));stages['transfer'].append(time.perf_counter()-t)
        t=time.perf_counter();pred=np.asarray(ready(engine.forward(params,device_graph)))[mask];stages['inference'].append(time.perf_counter()-t)
        y=y[mask];group=graph['query_group'][mask]
        from jaxpropka.parameters import MODEL_PKA
        base=MODEL_PKA[group]
        for name,value in (('graph_query',pred),('model_values',base),('train_type_mean',means[group])):
            scores[name].append(dict(component_id=row['component_id'],complex_id=row['complex_id'],n=len(y),**measures(y-base,value-base)))
    t=time.perf_counter();attributed={name:aggregate(values,replicates=2000)[0] for name,values in scores.items()};bootstrap_seconds=time.perf_counter()-t
    t=time.perf_counter();normal=evaluate(out,manifest,engine,params,means,replicates=2000);normal_seconds=time.perf_counter()-t
    for name in normal:
        np.testing.assert_allclose(normal[name]['mae'],attributed[name]['mae'],rtol=0,atol=0)
    return dict(complexes=len(val),warm_compile_seconds=warm_seconds,stages={k:quantiles(v) for k,v in stages.items()},
        bootstrap_seconds=bootstrap_seconds,normal_pipeline_seconds=normal_seconds,metrics=normal)


def gpu_trace(out,manifest,engine,params,state,destination):
    byid,plans=plans_for(manifest);target=sorted(manifest['capacities'],key=int)[len(manifest['capacities'])//2]
    members=next(x for x in plans if bucket(byid[x[0]])==target)
    loader=BatchLoader(out,manifest,manifest['config']['batch_size']);batch=jax.device_put(loader.load(members));loader.close()
    ready(engine.batch_step(params,state,*batch));destination.mkdir(parents=True,exist_ok=True)
    jax.profiler.start_trace(str(destination))
    for _ in range(10):ready(engine.batch_step(params,state,*batch))
    jax.profiler.stop_trace()
    return dict(bucket=target,steps=10,path=str(destination))


def run_profile(root):
    destination=root/'audits/gqt-fullgraph-profile-v1';destination.mkdir(parents=True,exist_ok=True)
    results={}
    for size in ('50k','200k'):
        out,manifest=manifest_for(root,size);engine,params,state=make_engine(manifest)
        attributed=attributed_profile(out,manifest,engine,params,state)
        epochs=[]
        for epoch in (1,2):params,state,row=throughput_epoch(out,manifest,engine,params,state,epoch);epochs.append(row)
        validation=validation_profile(out,manifest,engine,params)
        trace=gpu_trace(out,manifest,engine,params,state,destination/f'trace-{size}') if size=='200k' else None
        results[size]=dict(path=str(out),parameter_count=manifest['config']['parameter_count'],attributed=attributed,
                           throughput_epochs=epochs,validation=validation,trace=trace)
        atomic_json(destination/'profile.json',dict(complete=False,models=results))
    atomic_json(destination/'profile.json',dict(complete=True,models=results))
    write_report(destination,results)


def write_report(destination,results,cache=None):
    lines=['# Full-graph GQT runtime profile','',
        'Full float32 on one A40 with the existing batch membership, padding, four structure workers, one-batch prefetch and per-epoch validation. Compilation is excluded from warm throughput.','',
        '| Model | Parameters | Warm train epoch | Steady loader wait | Validation | Total train + validation |','|---|---:|---:|---:|---:|---:|']
    for name,result in results.items():
        epoch=result['throughput_epochs'][-1];validation=result['validation']['normal_pipeline_seconds']
        lines.append(f"| {name} | {result['parameter_count']:,} | {epoch['seconds']:.2f} s | {100*epoch['steady_prefetch_wait_fraction']:.2f}% | {validation:.2f} s | {epoch['seconds']+validation:.2f} s |")
    lines+=['','| Model | Bucket | Fused step median | H2D median | Actual/padded edges |','|---|---:|---:|---:|---:|']
    for name,result in results.items():
        for row in result['attributed']:
            load=row['loader'];ratio=load['actual_edges']/load['padded_edges']
            lines.append(f"| {name} | {row['bucket']} | {row['fused_step']['median']:.4f} s | {row['host_to_device']['median']:.4f} s | {ratio:.3f} |")
    lines+=['','The mmap backend gate is a steady loader-wait fraction above 5% or a projected end-to-end gain of at least 10%. The compilation cache is reported separately as a startup/restart optimization. Cropping is outside this experiment.']
    if cache is not None:
        lines+=['','## Opt-in persistent compilation cache','',
            '| Process | First fused call | Loss | Parameter checksum |','|---|---:|---:|---:|']
        for row in cache['rows']:
            lines.append(f"| {row['label']} | {row['seconds']:.3f} s | {row['loss']:.8f} | {row['parameter_checksum']:.8f} |")
        lines+=['',f"Concurrent-writer integrity: **{'passed' if cache['passed'] else 'failed'}**. Cache contained {cache['cache_files']} files / {cache['cache_bytes']/2**20:.1f} MiB. The 5 GiB bound applies to this namespace; other namespaces have independent bounds."]
    (destination/'report.md').write_text('\n'.join(lines)+'\n')


def cache_worker(root,label):
    destination=root/'audits/gqt-fullgraph-profile-v1/cache-probe';destination.mkdir(parents=True,exist_ok=True)
    out,manifest=manifest_for(root,'50k');cache=configure(manifest['config'],hashes())
    engine,params,state=make_engine(manifest);byid,plans=plans_for(manifest)
    members=next(x for x in plans if bucket(byid[x[0]])==sorted(manifest['capacities'],key=int)[0])
    loader=BatchLoader(out,manifest,manifest['config']['batch_size']);batch=loader.load(members);loader.close()
    t=time.perf_counter();updated,new_state,loss,finite=ready(engine.batch_step(params,state,*batch));seconds=time.perf_counter()-t
    assert bool(finite)
    checksum=float(sum(np.asarray(x,dtype=np.float64).sum() for x in jax.tree.leaves(updated)))
    atomic_json(destination/f'{label}.json',dict(label=label,seconds=seconds,loss=float(loss),parameter_checksum=checksum,cache=cache))


def cache_report(root):
    destination=root/'audits/gqt-fullgraph-profile-v1/cache-probe';rows=[read(destination/f'{x}.json') for x in ('writer-a','writer-b','reader')]
    assert len({row['loss'] for row in rows})==1
    checksums=np.asarray([row['parameter_checksum'] for row in rows])
    checksum_spread=float(np.ptp(checksums));checksum_relative_spread=checksum_spread/max(float(abs(checksums).max()),1e-12)
    # GPU graph-gradient reductions are not bitwise deterministic across
    # concurrent processes. The scalar output is exact; parameter updates must
    # agree well below the existing 2e-4 gradient-equivalence gate.
    assert checksum_relative_spread<1e-6,(checksum_spread,checksum_relative_spread)
    paths={row['cache']['path'] for row in rows};assert len(paths)==1
    path=Path(next(iter(paths)));files=[p for p in path.iterdir() if p.is_file()]
    result=dict(passed=True,rows=rows,parameter_checksum_spread=checksum_spread,
        parameter_checksum_relative_spread=checksum_relative_spread,
        cache_files=len(files),cache_bytes=sum(p.stat().st_size for p in files),
        reader_faster_than_slowest_writer=rows[2]['seconds']<max(rows[0]['seconds'],rows[1]['seconds']))
    atomic_json(destination/'verification.json',result)


def finalize(root):
    destination=root/'audits/gqt-fullgraph-profile-v1';profile=read(destination/'profile.json');assert profile['complete']
    cache=read(destination/'cache-probe/verification.json');assert cache['passed']
    write_report(destination,profile['models'],cache)
    mmap_recommended=any(model['throughput_epochs'][-1]['steady_prefetch_wait_fraction']>.05 for model in profile['models'].values())
    atomic_json(destination/'verification.json',dict(passed=True,mmap_recommended=mmap_recommended,
        mmap_gate='steady prefetch wait >5%; projected 10% end-to-end gain requires follow-up prototype if this gate passes',
        profile_sha256=digest(destination/'profile.json'),cache_sha256=digest(destination/'cache-probe/verification.json'),
        report_sha256=digest(destination/'report.md'),cropping_implemented=False))


if __name__=='__main__':
    import sys
    action=sys.argv[1];gpu=action in ('profile','cache-worker')
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),gpu_benchmark=gpu,allow_comp1400=True)
    root=Path(os.environ['PKABENCH_RUNTIME'])
    jax.config.update('jax_enable_x64',False)
    if action=='profile':run_profile(root)
    elif action=='cache-worker':cache_worker(root,sys.argv[2])
    elif action=='cache-report':cache_report(root)
    elif action=='finalize':finalize(root)
    else:raise ValueError(action)
