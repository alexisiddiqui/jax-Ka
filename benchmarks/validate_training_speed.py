"""Compare seeding/LM budgets and CPU threading without changing a registered run."""
import argparse
import os
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('cid');p.add_argument('threads',type=int)
    a=p.parse_args()
    from pkabench.runtime import require_compute,atomic_json,digest
    require_compute(threads=a.threads)
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    import numpy as np
    from pkatrain.records import load
    from pkatrain.experiment import make_engine,code_hashes
    from pkatrain.trainer import Engine
    from dataclasses import replace
    inputs,reference,eligible,_=load(a.root,a.cid)
    baseline=make_engine(a.root)
    engines=[('1024/512',baseline),('64/512',Engine(baseline.terms_fn,baseline.prior_fn,
        baseline.config,replace(baseline.solver_config,max_steps=512),seed_steps=64)),
        ('64/32',Engine(baseline.terms_fn,baseline.prior_fn,baseline.config,
        replace(baseline.solver_config,max_steps=32),seed_steps=64))]
    rows=[]; all_passed=True
    for theta in (jnp.zeros(3),jnp.asarray([.1,-.1,.1])):
        ref_curve=ref_valid=ref_gradient=None
        for name,engine in engines:
            began=time.monotonic()
            curve=jax.block_until_ready(engine.forward(theta,inputs,jnp.ones((2,73),bool)))
            forward_seconds=time.monotonic()-began
            gradient,log=engine.audited_gradient(theta,inputs,reference,eligible,1.)
            cold_seconds=time.monotonic()-began
            began=time.monotonic()
            gradient,log=engine.audited_gradient(theta,inputs,reference,eligible,1.)
            warm_seconds=time.monotonic()-began
            h=np.asarray(curve.protonated); valid=np.asarray(curve.converged); g=np.asarray(gradient)
            if ref_curve is None:ref_curve,ref_valid,ref_gradient=h,valid,g
            both=ref_valid&valid
            gap=float(np.max(abs(h-ref_curve)[both])) if both.any() else None
            passed=bool(np.array_equal(valid,ref_valid) and gap is not None and gap<=1e-6
                and np.allclose(g,ref_gradient,rtol=.01,atol=1e-7))
            all_passed &= passed
            row=dict(config=name,theta=np.asarray(theta).tolist(),forward_seconds=forward_seconds,
                cold_seconds=cold_seconds,warm_seconds=warm_seconds,curve_gap=gap,
                gradient=g.tolist(),gradient_gap=float(np.max(abs(g-ref_gradient))),
                validity_equal=bool(np.array_equal(valid,ref_valid)),passed=passed,loss=log)
            rows.append(row)
            print(__import__('json').dumps(row),flush=True)
            atomic_json(a.root/'speed_checks'/f'{a.cid}-threads{a.threads}.json',dict(
                complete=False,rows=rows,threads=a.threads,complex_id=a.cid))
    import resource
    atomic_json(a.root/'speed_checks'/f'{a.cid}-threads{a.threads}.json',dict(
        complete=True,passed=all_passed,rows=rows,threads=a.threads,complex_id=a.cid,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        manifest_sha256=digest(a.root/'manifest.json'),code_sha256=code_hashes(),
        job=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME']))


if __name__=='__main__':main()
