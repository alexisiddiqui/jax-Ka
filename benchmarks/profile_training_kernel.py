"""Cost and equivalence of the shared training kernel's seeding path.

``local_terms_curve_kernel`` seeds each optimistix root solve with a detached damped
solve. Run per pH inside the continuation scan that is ``config.steps * len(ph)``
sequential iterations on one small [N,9] state -- ~150k for a two-branch step at 73 pH
and 1024 damped steps. Vmapped over the grid it is ``config.steps`` iterations on
[H,N,9]: the same arithmetic with H times fewer loop trips, which is how the production
readout already does it (``model._over_ph``).

The seed is stop_gradient'd, so it cannot change the converged root. This measures the
saving and checks that claim on a real prepared record: curves and the loss gradient
must match a per-pH reference to roundoff. ``--seed-steps`` then asks the separate,
numerics-changing question of whether 1024 damped steps are needed to seed at all,
reporting the curve gap against the full-seed result so a branch change would show.

The per-pH reference is reconstructed by calling the kernel on one pH at a time, which
is the arithmetic the scan used to do.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--complex', help='prepared complex id; default = first in manifest')
    parser.add_argument('--seed-steps', type=int, nargs='*', default=[256, 64],
                        help='additional seed step counts to compare against the full seed')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--skip-reference', action='store_true',
                        help='skip the per-pH reference timing (it is slow by construction)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', True)   # as pkatrain.run does; the manifest is float64
    import jax.numpy as jnp
    from jaxpropka.optx_solver import local_terms_curve_kernel
    from pkatrain.records import read, load
    from pkatrain.experiment import make_engine
    from pkatrain.forward import solve_branches

    engine = make_engine()
    manifest = read(args.training_root/'manifest.json')
    cid = args.complex or (manifest.get('smoke') or manifest['train'])[0]
    inputs, reference, eligible, receipt = load(args.training_root, cid)
    inputs = jax.tree.map(jnp.asarray, inputs)
    params = jnp.zeros(3, jnp.float64)
    ph = engine.ph
    n_ph = int(ph.size)
    real_residues = receipt.get('layout', {}).get('real_residues')

    def terms_of(theta):
        from pkatrain.adapters.jaxka import local_terms
        return jax.vmap(lambda d, p: local_terms(theta, d, p, engine.config))(
            inputs['arrays'], inputs['probabilities'])

    def timed(fn):
        out = jax.block_until_ready(fn())                      # compile
        best = np.inf
        for _ in range(args.repeats):
            started = time.perf_counter()
            out = jax.block_until_ready(fn())
            best = min(best, time.perf_counter()-started)
        return out, best

    def forward(theta, **kwargs):
        return solve_branches(inputs, terms_of(theta), ph, config=engine.config,
                              solver_config=engine.solver_config, **kwargs)

    def loss(theta, **kwargs):
        curves = forward(theta, **kwargs)[0]
        return jnp.sum(curves.protonated**2)      # shape-only probe of the adjoint cost

    results = []
    # 1. The vmapped seed, at the production step count: the numerics-preserving change.
    (curves, extra), forward_seconds = timed(lambda: forward(params))
    (value, grad), grad_seconds = timed(
        lambda: jax.value_and_grad(loss)(params))
    full = np.asarray(curves.protonated, float)
    results.append(dict(variant='vmapped_seed', seed_steps=engine.config.steps,
                        forward_seconds=forward_seconds, value_and_grad_seconds=grad_seconds,
                        max_residual=float(jnp.max(curves.residual)),
                        all_converged=bool(jnp.all(curves.converged)),
                        newton_steps_max=int(jnp.max(extra['newton_steps'])),
                        newton_steps_mean=float(jnp.mean(extra['newton_steps'])),
                        optx_success_all=bool(jnp.all(extra['optx_success'])),
                        gradient=[float(x) for x in grad],
                        max_curve_gap_vs_full_seed=0.))
    print(json.dumps(results[-1]), flush=True)

    # 2. Per-pH reference: same arithmetic the scan used to do, one pH at a time.
    if not args.skip_reference:
        def per_ph():
            out = []
            for i in range(n_ph):
                one = jax.vmap(lambda d, t, a, v: local_terms_curve_kernel(
                    d, t, ph[i:i+1], a, v, config=engine.config,
                    solver_config=engine.solver_config))(
                    inputs['arrays'], terms_of(params), inputs['active'], inputs['active_valid'])
                out.append(np.asarray(one[0].protonated, float))
            return np.concatenate(out, axis=1)
        started = time.perf_counter()
        ref = jax.block_until_ready(per_ph())
        reference_seconds = time.perf_counter()-started
        results.append(dict(variant='per_ph_reference', seed_steps=engine.config.steps,
                            forward_seconds=reference_seconds,
                            max_curve_gap_vs_full_seed=float(np.max(np.abs(ref-full))),
                            note='includes 73 separate compilations; an upper bound'))
        print(json.dumps(results[-1]), flush=True)

    # 3. Cheaper seeds: changes numerics in principle, so report the curve gap.
    for steps in args.seed_steps:
        (curves_s, extra_s), seconds = timed(lambda s=steps: forward(params, seed_steps=s))
        (_, grad_s), grad_s_seconds = timed(
            lambda s=steps: jax.value_and_grad(loss)(params, seed_steps=s))
        results.append(dict(variant='vmapped_seed', seed_steps=steps,
                            forward_seconds=seconds, value_and_grad_seconds=grad_s_seconds,
                            max_residual=float(jnp.max(curves_s.residual)),
                            all_converged=bool(jnp.all(curves_s.converged)),
                            newton_steps_max=int(jnp.max(extra_s['newton_steps'])),
                            newton_steps_mean=float(jnp.mean(extra_s['newton_steps'])),
                            optx_success_all=bool(jnp.all(extra_s['optx_success'])),
                            gradient=[float(x) for x in grad_s],
                            max_curve_gap_vs_full_seed=float(
                                np.max(np.abs(np.asarray(curves_s.protonated, float)-full))),
                            max_gradient_gap_vs_full_seed=float(
                                np.max(np.abs(np.asarray(grad_s, float)-np.asarray(grad, float))))))
        print(json.dumps(results[-1]), flush=True)

    atomic_json(args.output/'profile.json',
                dict(job=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename,
                     complex_id=cid, n_ph=n_ph, real_residues=real_residues,
                     config_steps=engine.config.steps,
                     solver_max_steps=engine.solver_config.max_steps,
                     cpus=os.environ.get('SLURM_CPUS_PER_TASK'), results=results))


if __name__ == '__main__':
    main()
