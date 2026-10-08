"""Separate seed-budget branch effects from LM-budget convergence failures."""
import os
import time
from pathlib import Path

def main():
    from pkabench.runtime import require_compute,atomic_json
    require_compute(threads=8,gpu_benchmark=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from dataclasses import replace
    from pkatrain.records import load
    from pkatrain.experiment import make_engine
    from pkatrain.adapters.jaxka import local_terms
    from pkatrain.forward import solve_branches
    jax.config.update('jax_enable_x64',True)
    assert jax.default_backend()=='gpu'
    root=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/training')
    cid='d5334196bf4ab5e6';inputs,_,_,_=load(root/'shared-v1',cid)
    engine=make_engine(root/'shared-v1');theta=jnp.zeros(3)
    terms=jax.vmap(lambda d,p:local_terms(theta,d,p,engine.config))(inputs['arrays'],inputs['probabilities'])
    rows=[];reference=None
    for seeds,steps in ((1024,512),(64,512),(64,32)):
        began=time.monotonic()
        result,extra=jax.block_until_ready(solve_branches(inputs,terms,engine.ph,config=engine.config,
            solver_config=replace(engine.solver_config,max_steps=steps),seed_steps=seeds))
        valid=np.asarray(result.converged);h=np.asarray(result.protonated)
        if reference is None:reference=(valid,h)
        common=valid&reference[0]
        failed=[dict(branch=('AB','free')[b],ph=float(engine.ph[k]),residual=float(result.residual[b,k]),
            lm_steps=int(extra['newton_steps'][b,k]),lm_success=bool(extra['optx_success'][b,k])) for b,k in zip(*np.nonzero(~valid))]
        row=dict(seed_steps=seeds,lm_steps=steps,seconds=time.monotonic()-began,failed=failed,
            validity_equal=bool(np.array_equal(valid,reference[0])),
            common_curve_gap=float(np.max(abs(h-reference[1])[common])) if common.any() else None)
        rows.append(row);print(__import__('json').dumps(row),flush=True)
        atomic_json(root/'gpu-check-v1/seed_coverage.json',dict(complex_id=cid,rows=rows,job=os.environ['SLURM_JOB_ID']))

if __name__=='__main__':main()
