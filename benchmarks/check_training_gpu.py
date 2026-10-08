"""Matched CPU/GPU timing and numerical checks in a GPU Slurm allocation."""
import argparse
import json
import os
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('cid')
    p.add_argument('backend',choices=['cpu','gpu']);p.add_argument('output',type=Path)
    a=p.parse_args()
    from pkabench.runtime import require_compute,atomic_json,digest
    require_compute(threads=8,gpu_benchmark=True)
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    import numpy as np
    from dataclasses import replace
    from pkatrain.records import load
    from pkatrain.experiment import make_engine,code_hashes
    from pkatrain.trainer import Engine
    assert jax.default_backend()==a.backend,jax.devices()
    hashes=code_hashes();inputs,reference,eligible,_=load(a.root,a.cid)
    base=make_engine(a.root)
    engine=Engine(base.terms_fn,base.prior_fn,base.config,replace(base.solver_config,max_steps=32),seed_steps=64)
    rows=[];payload={}
    for i,theta in enumerate((jnp.zeros(3),jnp.asarray([.1,-.1,.1]))):
        began=time.monotonic()
        curve=jax.block_until_ready(engine.forward(theta,inputs,jnp.ones((2,73),bool)))
        gradient,log=engine.audited_gradient(theta,inputs,reference,eligible,1.)
        cold=time.monotonic()-began;times=[]
        for _ in range(2):
            began=time.monotonic();gradient,log=engine.audited_gradient(theta,inputs,reference,eligible,1.)
            times.append(time.monotonic()-began)
        payload[f'curve{i}']=np.asarray(curve.protonated)
        payload[f'valid{i}']=np.asarray(curve.converged)
        payload[f'gradient{i}']=np.asarray(gradient)
        rows.append(dict(theta=np.asarray(theta).tolist(),cold_seconds=cold,warm_seconds=times,
            gradient=np.asarray(gradient).tolist(),loss=log))
        print(json.dumps(dict(backend=a.backend,complex_id=a.cid,**rows[-1])),flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    dest=a.output/f'{a.cid}-{a.backend}'
    np.savez(dest.with_suffix('.npz'),**payload)
    atomic_json(dest.with_suffix('.json'),dict(rows=rows,backend=a.backend,devices=list(map(str,jax.devices())),
        jax_version=jax.__version__,code_sha256=hashes,manifest_sha256=digest(a.root/'manifest.json'),
        seed_steps=64,lm_steps=32,job=os.environ['SLURM_JOB_ID']))
    other=a.output/f'{a.cid}-cpu.npz'
    gpu=a.output/f'{a.cid}-gpu.npz'
    if other.exists() and gpu.exists():
        checks=[]
        with np.load(other) as cpu,np.load(gpu) as device:
            for i in range(2):
                valid=cpu[f'valid{i}']&device[f'valid{i}']
                gap=float(np.max(abs(cpu[f'curve{i}']-device[f'curve{i}'])[valid])) if valid.any() else None
                checks.append(dict(curve_gap=gap,passed=bool(gap is not None and gap<=1e-6
                    and np.array_equal(cpu[f'valid{i}'],device[f'valid{i}'])
                    and np.allclose(cpu[f'gradient{i}'],device[f'gradient{i}'],rtol=.01,atol=1e-7))))
        atomic_json(a.output/f'{a.cid}-comparison.json',dict(passed=all(c['passed'] for c in checks),checks=checks))

if __name__=='__main__':main()
