"""Dynamic-input batch-1 versus batch-8 GPU training-step benchmark."""
import json
import os
import time
from pathlib import Path

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--precision',choices=['float64','mixed','float32'],default='float64')
    args=parser.parse_args()
    from pkabench.runtime import require_compute,atomic_json,digest
    require_compute(threads=8,gpu_benchmark=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from dataclasses import replace
    from pkatrain.records import load
    from pkatrain.experiment import make_engine,code_hashes
    from pkatrain.trainer import Engine
    from pkatrain.losses import coverage
    jax.config.update('jax_enable_x64',args.precision!='float32')
    assert jax.default_backend()=='gpu'
    root=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/training')
    cid='a779dcd261ce4357';inputs,reference,eligible,receipt=load(root/'shared-v1',cid)
    # Only one prepared record has this exact bucket. Repeated inputs stay
    # dynamic arguments, so XLA cannot constant-fold away the batch axis.
    inputs.pop('teacher_pka')
    dtype=np.float32 if args.precision=='float32' else np.float64
    inputs=jax.tree.map(lambda x:x.astype(dtype) if np.issubdtype(x.dtype,np.floating) else x,inputs)
    reference=reference.astype(dtype)
    base=make_engine(root/'shared-v1')
    engine=Engine(base.terms_fn,base.prior_fn,base.config,replace(base.solver_config,max_steps=32),seed_steps=64,
        seed_dtype='float32' if args.precision=='mixed' else None,dtype=dtype)
    theta=jnp.zeros(3);state=engine.optimizer.init(theta)
    forward=jax.jit(jax.vmap(engine.forward,in_axes=(None,0,0)))
    def loss(t,d,r,e,a):
        values,_=jax.vmap(engine.objective,in_axes=(None,0,0,0,0,None))(t,d,r,e,a,1.)
        return jnp.mean(values)
    vg=jax.jit(jax.value_and_grad(loss))
    @jax.jit
    def update(t,s,g):
        updates,s=engine.optimizer.update(g,s,t)
        return optax.apply_updates(t,updates),s
    rows=[];single=None;hashes=code_hashes()
    for size in (1,8):
        d=jax.device_put(jax.tree.map(lambda x:np.stack([x]*size),inputs))
        r=jax.device_put(np.stack([reference]*size));e=jax.device_put(np.stack([eligible]*size))
        mask=jnp.ones((size,2,73),bool)
        def step():
            curves=forward(theta,d,mask)
            accepted=np.asarray(curves.converged)&np.asarray(jnp.all(jnp.isfinite(curves.protonated),axis=(-2,-1)))
            cov=[coverage(eligible,a) for a in accepted]
            assert max(c['fraction'] for c in cov)<=.05
            value,gradient=jax.block_until_ready(vg(theta,d,r,e,jnp.asarray(accepted)))
            assert np.isfinite(float(value)) and np.isfinite(np.asarray(gradient)).all()
            jax.block_until_ready(update(theta,state,gradient))
            return curves,value,gradient,cov
        began=time.monotonic();curves,value,gradient,cov=step();cold=time.monotonic()-began
        times=[]
        for _ in range(3):
            began=time.monotonic();curves,value,gradient,cov=step();times.append(time.monotonic()-began)
        h=np.asarray(curves.protonated);valid=np.asarray(curves.converged);g=np.asarray(gradient)
        if single is None:single=(h[0],valid[0],g,float(value))
        gap=float(np.max(abs(h-single[0])))
        passed=bool(gap<=1e-6 and np.all(valid==single[1]) and np.allclose(g,single[2],rtol=.01,atol=1e-7)
            and np.isclose(float(value),single[3],rtol=1e-8,atol=1e-10))
        saved=root/'gpu-check-v1'/f'{cid}-gpu.npz'
        with np.load(saved) as fp64:
            reference_gap=float(np.max(abs(h-fp64['curve0'])))
            reference_gradient=fp64['gradient0']
            relative_gradient=float(np.linalg.norm(g-reference_gradient)/max(np.linalg.norm(reference_gradient),1e-15))
            validity_equal=bool(np.all(valid==fp64['valid0']))
        row=dict(batch_size=size,cold_seconds=cold,warm_seconds=times,
            median_seconds=float(np.median(times)),seconds_per_complex=float(np.median(times))/size,
            curve_gap=gap,gradient=g.tolist(),passed=passed,coverage=cov,
            fp64_curve_gap=reference_gap,fp64_gradient_relative_difference=relative_gradient,
            fp64_validity_equal=validity_equal,max_residual=float(jnp.max(curves.residual)),
            allocator_memory_stats=jax.devices()[0].memory_stats())
        rows.append(row);print(json.dumps(row),flush=True)
        destination='gpu-batch-check-v1' if args.precision=='float64' else f'gpu-batch-{args.precision}-v1'
        atomic_json(root/destination/'result.json',dict(complete=size==8,rows=rows,precision=args.precision,
            x64_enabled=bool(jax.config.jax_enable_x64),working_dtype=str(theta.dtype),
            complex_id=cid,layout=receipt['layout'],batch_content='Repeated identical complex; dynamic input arrays',
            resident_inputs=True,timed_operations='forward validity audit + mean loss/gradient + Optax update',
            seed_steps=64,lm_steps=32,code_sha256=hashes,job=os.environ['SLURM_JOB_ID'],
            manifest_sha256=digest(root/'shared-v1/manifest.json'),devices=list(map(str,jax.devices())),
            throughput_speedup=(rows[0]['median_seconds']*8/rows[-1]['median_seconds']) if size==8 else None))

if __name__=='__main__':main()
