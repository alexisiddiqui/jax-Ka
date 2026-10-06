"""Production-seed, distinct-complex GPU precision checks and 20-update smoke."""
import argparse
import json
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','float64','mixed','float32','smoke'])
    a=p.parse_args()
    from pkabench.runtime import require_compute,atomic_json,digest
    require_compute(threads=1 if a.action=='prepare' else 8,gpu_benchmark=a.action!='prepare')
    from pkatrain.records import read,prepare
    root=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/training/shared-v3-production')
    dest=root/'gpu-pilot';dest.mkdir(exist_ok=True)
    manifest=read(root/'manifest.json')
    if a.action=='prepare':
        records=[read(root/'records'/f'{cid}.json') for cid in manifest['train']]
        chosen=[];groups=set()
        for role in sorted({r['role'] for r in records}):
            count=0
            for r in sorted((r for r in records if r['role']==role),key=lambda r:(abs(r['n_residues']-200),r['complex_id'])):
                if r['component_id'] in groups:continue
                chosen.append(r);groups.add(r['component_id']);count+=1
                if count==4:break
        assert len(chosen)==8
        atomic_json(dest/'selection.json',{'complexes':[{k:r[k] for k in ('complex_id','component_id','role','n_residues')} for r in chosen],
            'rule':'Four per role, distinct split components, closest residue counts to 200',
            'manifest_sha256':digest(root/'manifest.json')})
        for r in chosen:prepare(root,r['complex_id'])
        atomic_json(dest/'prepared.json',{'passed':True});return
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pkatrain.experiment import make_engine,code_hashes
    from pkatrain.trainer import Engine,save_checkpoint,load_checkpoint
    from pkatrain.minibatch import pack_records
    assert jax.default_backend()=='gpu'
    full_float32=a.action in ('float32','smoke')
    jax.config.update('jax_enable_x64',not full_float32)
    dtype=np.float32 if full_float32 else np.float64
    ids=[r['complex_id'] for r in read(dest/'selection.json')['complexes']]
    inputs,ref,eligible,layout=pack_records(root,ids,dtype)
    inputs,ref,eligible=jax.device_put((inputs,ref,eligible))
    base=make_engine(root)
    engine=Engine(base.terms_fn,base.prior_fn,base.config,base.solver_config,
        seed_steps=None,seed_dtype='float32' if a.action=='mixed' else None,dtype=dtype)
    theta=jnp.zeros(3,dtype);state=engine.optimizer.init(theta)
    def step(t,s):
        g,log,c=engine.audited_batch_gradient(t,inputs,ref,eligible,1.)
        t,s=engine.update(t,s,[g]);jax.block_until_ready((t,s))
        return t,s,g,log,c
    if a.action=='smoke':
        assert read(dest/'float64.json')['passed']
        comparison=read(dest/'float32.json')
        # Keep the stricter float64-parity verdict unchanged in its artifact.
        # Float32 training uses occupancy accuracy at the solver tolerance,
        # matching validity and a separate relative-gradient accuracy gate.
        admission=dict(precision='float32',x64_enabled=False,curve_tolerance=2e-5,
            gradient_relative_tolerance=1e-3,require_identical_validity=True,
            reason='User selected full-float32 training; curve tolerance equals the solver residual tolerance',
            comparison_sha256=digest(dest/'float32.json'))
        atomic_json(dest/'smoke/precision_admission.json',admission)
        assert all(r['curve_gap'] is not None and r['curve_gap']<=2e-5
            and r['gradient_relative_difference']<=1e-3 and r['validity_equal']
            and r['max_residual']<=2e-5 and r['coverage']['missing']/max(r['coverage']['total'],1)<=.01
            for r in comparison['rows']),comparison
        missing=total=0;history=[];start=time.monotonic()
        for i in range(20):
            theta,state,g,log,c=step(theta,state);missing+=log['missing'];total+=log['total']
            assert missing/max(total,1)<=.01,(missing,total)
            history.append(dict(step=i+1,theta=np.asarray(theta).tolist(),**log))
            print(json.dumps(history[-1]),flush=True)
            if i==9:
                save_checkpoint(dest/'smoke/checkpoints/000010',theta,state,{'step':10,'manifest_sha256':digest(root/'manifest.json')})
                rt,rs,_=load_checkpoint(dest/'smoke/checkpoints/000010',(theta,state))
                for x,y in zip(jax.tree.leaves((theta,state)),jax.tree.leaves((rt,rs))):np.testing.assert_array_equal(x,y)
                theta,state=rt,rs
        assert np.any(np.asarray(theta)!=0)
        atomic_json(dest/'smoke/verification.json',dict(passed=True,updates=20,checkpoint_roundtrip=True,precision='float32',x64_enabled=False,
            seconds=time.monotonic()-start,history=history,manifest_sha256=digest(root/'manifest.json')))
        lines=['# Production-seed GPU pilot','',
            'Eight distinct training complexes from eight split groups: four antibody–antigen and four general protein pairs. '
            '1024 seed steps, LM cap 32; timings include validity audit, mean loss/gradient and Optax update, excluding compilation and input transfer.','',
            '| Precision | Batch-8 seconds (two parameter settings) | Peak live GPU GB | Max curve difference | Gradient relative difference | Gates |',
            '|---|---:|---:|---:|---:|---|']
        for precision in ('float64','mixed','float32'):
            if not (dest/f'{precision}.json').exists():continue
            result=read(dest/f'{precision}.json');rr=result['rows']
            times=[float(np.median(row['warm_seconds'])) for row in rr]
            memory=max(row['device_memory']['peak_bytes_in_use'] for row in rr)/1e9
            lines.append(f"| {precision} | {min(times):.3f}–{max(times):.3f} | {memory:.2f} | {max(row['curve_gap'] for row in rr):.3g} | {max(row['gradient_relative_difference'] for row in rr):.3g} | {'pass' if result['passed'] else 'fail'} |")
        lines+=['','Float64 compares batched outputs with independent per-complex solves. Other precisions compare with float64. '
            'The table retains the strict 1e-6 curve-parity verdict. Full-float32 smoke admission is recorded separately: '
            'curve difference and residual <=2e-5, relative gradient difference <=1e-3, and identical validity masks.','',
            f'Full-float32 optimizer smoke (x64 disabled): 20 updates completed; loss {history[0]["loss"]:.6g} to {history[-1]["loss"]:.6g}. '
            f'Missing observations: {missing}/{total}. Checkpoint round trip passed at update 10.','',
            'The smoke repeatedly optimizes this fixed batch; it is not a generalization result or a whole-dataset throughput estimate. '
            'A dataset-weighted ETA still requires measurements of larger batches/buckets.']
        (dest/'report.md').write_text('\n'.join(lines)+'\n')
        return
    records=[];payload={};overall=True
    for point,theta in enumerate((jnp.zeros(3,dtype),jnp.asarray([.1,-.1,.1],dtype))):
        began=time.monotonic();_,_,g,log,c=step(theta,state);cold=time.monotonic()-began
        times=[]
        for _ in range(3):
            began=time.monotonic();_,_,g,log,c=step(theta,state);times.append(time.monotonic()-began)
        h=np.asarray(c.protonated);valid=np.asarray(c.converged);grad=np.asarray(g)
        payload[f'curve{point}']=h;payload[f'valid{point}']=valid;payload[f'gradient{point}']=grad
        if a.action=='float64':
            single_grad=[];single_h=[];single_valid=[]
            for i in range(8):
                d=jax.tree.map(lambda x:x[i],inputs)
                gg,_=engine.audited_gradient(theta,d,ref[i],eligible[i],1.)
                cc=engine.forward(theta,d,jnp.ones((2,73),bool))
                single_grad.append(np.asarray(gg));single_h.append(np.asarray(cc.protonated));single_valid.append(np.asarray(cc.converged))
            rh=np.stack(single_h);rv=np.stack(single_valid);rg=np.mean(single_grad,axis=0)
        else:
            with np.load(dest/'float64.npz') as original:
                rh=original[f'curve{point}'];rv=original[f'valid{point}'];rg=original[f'gradient{point}']
        common=valid&rv;gap=float(np.max(abs(h-rh)[common])) if common.any() else None
        passed=bool(gap is not None and gap<=1e-6 and np.array_equal(valid,rv) and np.allclose(grad,rg,rtol=.01,atol=1e-7)
            and log['missing']/max(log['total'],1)<=.01)
        overall &= passed
        records.append(dict(theta=np.asarray(theta).tolist(),cold_seconds=cold,warm_seconds=times,
            seconds_per_complex=float(np.median(times))/8,curve_gap=gap,
            gradient_relative_difference=float(np.linalg.norm(grad-rg)/max(np.linalg.norm(rg),1e-15)),
            validity_equal=bool(np.array_equal(valid,rv)),passed=passed,coverage=log,
            max_residual=float(jnp.max(c.residual)),device_memory=jax.devices()[0].memory_stats()))
        print(json.dumps(records[-1]),flush=True)
    np.savez(dest/f'{a.action}.npz',**payload)
    atomic_json(dest/f'{a.action}.json',dict(passed=overall,precision=a.action,layout=layout,rows=records,
        seed_steps=1024,lm_steps=32,x64_enabled=bool(jax.config.jax_enable_x64),code_sha256=code_hashes(),
        manifest_sha256=digest(root/'manifest.json'),complex_ids=ids))

if __name__=='__main__':main()
