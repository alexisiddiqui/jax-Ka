"""Minimal, incremental timing of one shared-training step.

profile_training_kernel.py prints only after compile + 2 forwards + compile + 2
gradients, which tells you nothing while a single evaluation is still running. This
times one operation at a time and flushes after each, so a slow path still yields
numbers. Compile and run are separated, because on this kernel they are easily
confused and the remedies differ.
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
    parser.add_argument('--complex')
    parser.add_argument('--seed-steps', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None, help='override LM max_steps')
    parser.add_argument('--n-ph', type=int, default=None, help='truncate the pH grid')
    parser.add_argument('--gradient', action='store_true', help='also time value_and_grad')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from dataclasses import replace
    from pkatrain.records import read, load
    from pkatrain.experiment import make_engine
    from pkatrain.forward import solve_branches
    from pkatrain.adapters.jaxka import local_terms

    mark = [time.perf_counter()]

    def stamp(label, **extra):
        now = time.perf_counter()
        record = dict(stage=label, seconds=round(now-mark[0], 2), **extra)
        print(json.dumps(record), flush=True)
        mark[0] = time.perf_counter()
        return record

    stages = []
    engine = make_engine()
    solver_config = (engine.solver_config if args.max_steps is None
                     else replace(engine.solver_config, max_steps=args.max_steps))
    manifest = read(args.training_root/'manifest.json')
    cid = args.complex or (manifest.get('smoke') or manifest['train'])[0]
    inputs, reference, eligible, receipt = load(args.training_root, cid)
    inputs = jax.tree.map(jnp.asarray, inputs)
    ph = engine.ph if args.n_ph is None else engine.ph[:args.n_ph]
    params = jnp.zeros(3, jnp.float64)
    layout = receipt.get('layout', {})
    stages.append(stamp('load', complex_id=cid, n_ph=int(ph.size), **{
        k: layout.get(k) for k in ('N', 'M', 'Kc', 'Ke', 'real_residues')}))

    kwargs = dict(config=engine.config, solver_config=solver_config)
    if args.seed_steps is not None:
        kwargs['seed_steps'] = args.seed_steps

    def forward(theta):
        terms = jax.vmap(lambda d, p: local_terms(theta, d, p, engine.config))(
            inputs['arrays'], inputs['probabilities'])
        return solve_branches(inputs, terms, ph, **kwargs)

    lowered = jax.jit(forward).lower(params)
    stages.append(stamp('trace'))
    compiled = lowered.compile()
    stages.append(stamp('compile'))
    curves, extra = jax.block_until_ready(compiled(params))
    stages.append(stamp('forward_run',
                        max_residual=float(jnp.max(curves.residual)),
                        all_converged=bool(jnp.all(curves.converged)),
                        newton_steps_max=int(jnp.max(extra['newton_steps'])),
                        newton_steps_mean=round(float(jnp.mean(extra['newton_steps'])), 2),
                        newton_steps_p90=round(float(np.percentile(
                            np.asarray(extra['newton_steps'], float), 90)), 2),
                        optx_success_all=bool(jnp.all(extra['optx_success']))))
    jax.block_until_ready(compiled(params))
    stages.append(stamp('forward_warm'))

    if args.gradient:
        loss = lambda theta: jnp.sum(forward(theta)[0].protonated**2)
        vg = jax.jit(jax.value_and_grad(loss))
        value, grad = jax.block_until_ready(vg(params))
        stages.append(stamp('value_and_grad_cold', value=float(value),
                            gradient=[float(x) for x in grad]))
        jax.block_until_ready(vg(params))
        stages.append(stamp('value_and_grad_warm'))

    atomic_json(args.output/f'timing-{cid}.json', dict(
        job=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename,
        complex_id=cid, n_ph=int(ph.size), seed_steps=args.seed_steps,
        config_steps=engine.config.steps, lm_max_steps=solver_config.max_steps,
        cpus=os.environ.get('SLURM_CPUS_PER_TASK'), stages=stages))


if __name__ == '__main__':
    main()
