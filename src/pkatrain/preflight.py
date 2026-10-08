"""Measure registered production buckets and release only after numerical checks."""
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from pkabench.runtime import require_compute,atomic_json,digest
from .records import read
from .experiment import make_engine,code_hashes
from .minibatch import pack_records

def run(out):
    require_compute(threads=8,gpu_benchmark=True)
    jax.config.update('jax_enable_x64',False)
    assert jax.default_backend()=='gpu'
    m=read(out/'manifest.json');plan=read(out/'buckets.json')
    for path,h in m['reference_gates'].items():assert digest(path)==h and read(path)['passed']
    engine=make_engine(out);theta=jnp.zeros(3,jnp.float32);results=[]
    for bucket in plan['buckets']:
        ids=bucket['representative_ids'];began=time.monotonic()
        d,r,e,_=pack_records(out,ids,np.float32,bucket['capacities'])
        packing=time.monotonic()-began
        d,r,e=jax.device_put((d,r,e));timings=[]
        started=time.monotonic();gradient,log,curves=engine.audited_batch_gradient(theta,d,r,e,1.)
        cold=time.monotonic()-started
        for _ in range(3):
            started=time.monotonic();gradient,log,curves=engine.audited_batch_gradient(theta,d,r,e,1.)
            timings.append(time.monotonic()-started)
        gg=[];hh=[];vv=[]
        for i in range(len(ids)):
            single=jax.tree.map(lambda x:x[i],d)
            g,_=engine.audited_gradient(theta,single,r[i],e[i],1.)
            c=engine.forward(theta,single,jnp.ones((2,73),bool))
            gg.append(np.asarray(g));hh.append(np.asarray(c.protonated));vv.append(np.asarray(c.converged))
        expected=np.mean(gg,axis=0);gap=float(np.max(abs(np.asarray(curves.protonated)-np.stack(hh))))
        assert gap<=2e-5 and np.array_equal(curves.converged,np.stack(vv))
        assert np.allclose(gradient,expected,rtol=1e-3,atol=1e-6)
        assert log['missing']/max(log['total'],1)<=.01
        row=dict(bucket=bucket['name'],batch_size=len(ids),complex_ids=ids,packing_seconds=packing,cold_seconds=cold,
            warm_seconds=timings,seconds_per_complex=(float(np.median(timings))+packing)/len(ids),
            curve_gap=gap,coverage=log,device_memory=jax.devices()[0].memory_stats())
        results.append(row);print(json.dumps(row),flush=True)
        atomic_json(out/'preflight/progress.json',dict(rows=results,complete=False))
    # Expected counts follow the actual group-uniform sampler, not uniform complexes.
    groups={}
    for cid in m['train']:
        group=read(out/'records'/f'{cid}.json')['component_id'];groups.setdefault(group,[]).append(cid)
    weights={b['name']:0. for b in plan['buckets']}
    for members in groups.values():
        for cid in members:weights[plan['assignment'][cid]]+=1/len(groups)/len(members)
    estimate=len(m['train'])*m['config']['epochs']*sum(weights[r['bucket']]*r['seconds_per_complex'] for r in results)
    report=dict(passed=True,rows=results,group_uniform_bucket_probabilities=weights,
        estimated_training_seconds_per_seed=estimate,
        estimate_excludes='Validation, compilation, checkpoints, partial batches and loss profiles; representatives chosen near bucket upper sizes',
        manifest_sha256=digest(out/'manifest.json'),buckets_sha256=digest(out/'buckets.json'),code_sha256=code_hashes(),
        reference_gates=m['reference_gates'],dtype='float32',x64_enabled=False,seed_steps=1024,lm_steps=32)
    atomic_json(out/'release.json',report)
    lines=['# Registered float32 batching preflight','',
        '| Bucket | Train complexes | Batch size | Warm gradient seconds | Including packing: seconds / complex |','|---|---:|---:|---:|---:|']
    for row,bucket in zip(results,plan['buckets']):
        lines.append(f"| {row['bucket']} | {bucket['train_count']} | {row['batch_size']} | {np.median(row['warm_seconds']):.2f} | {row['seconds_per_complex']:.2f} |")
    lines+=['',f'Group-weighted training-only estimate: {estimate/3600:.2f} hours per seed.',report['estimate_excludes']]
    (out/'preflight/report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]).resolve())
