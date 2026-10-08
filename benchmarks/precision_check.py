"""Can the shared training step run in float32?

Production runs float32 (``build_cache`` defaults to it and production_worker_v2 does
not override), while the training manifest specifies float64 -- so the model is trained
at a precision it is never deployed at. Separately, the A40s on the GPU node are GA102,
whose FP64 throughput is roughly 1/64 of FP32, so a float64 benchmark measures the
hardware in its worst mode.

This runs the same step at matched settings in four configurations and compares
everything against the float64 reference:

  float64                     the current training configuration
  float64 solve, float32 seed the seed is detached and discarded, so in principle this
                              costs no accuracy at all
  float32                     throughout
  (any of the above on GPU, by allocation)

Reports wall time, max residual against ``config.residual_tolerance``, Newton steps, and
-- the number that decides it -- the relative difference in the loss gradient. The loss
is a fixed quadratic in the curves, so the comparison needs no labels.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--complex')
    parser.add_argument('--seed-steps', type=int, default=64)
    parser.add_argument('--max-steps', type=int, default=32)
    parser.add_argument('--n-ph', type=int, default=None)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--x64', dest='x64', action='store_true', default=True)
    parser.add_argument('--no-x64', dest='x64', action='store_false',
                        help='run with x64 OFF; required for a clean float32 run, because '
                             'optimistix/lineax build float64 internals under x64 and they '
                             'collide with float32 user arrays inside the LM system')
    parser.add_argument('--only', nargs='*', help='configuration names to run')
    parser.add_argument('--reference-json', type=Path,
                        help='a previous float64 result to compare gradients against')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', args.x64)
    import jax.numpy as jnp
    from dataclasses import replace
    from pkatrain.records import read, load
    from pkatrain.experiment import make_engine
    from pkatrain.forward import solve_branches
    from pkatrain.adapters.jaxka import local_terms

    engine = make_engine()
    solver_config = replace(engine.solver_config, max_steps=args.max_steps)
    manifest = read(args.training_root/'manifest.json')
    cid = args.complex or (manifest.get('smoke') or manifest['train'])[0]
    inputs64, reference, eligible, receipt = load(args.training_root, cid)
    inputs64 = jax.tree.map(jnp.asarray, inputs64)
    ph64 = engine.ph if args.n_ph is None else engine.ph[:args.n_ph]
    tolerance = float(engine.config.residual_tolerance)

    def cast(tree, dtype):
        return jax.tree.map(
            lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, tree)

    def build(dtype, seed_dtype):
        inputs = inputs64 if dtype == jnp.float64 else cast(inputs64, dtype)
        ph = ph64.astype(dtype)
        params = jnp.zeros(3, dtype)
        kwargs = dict(config=engine.config, solver_config=solver_config,
                      seed_steps=args.seed_steps)
        if seed_dtype is not None:
            kwargs['seed_dtype'] = seed_dtype

        def terms_of(theta):
            return jax.vmap(lambda d, p: local_terms(theta, d, p, engine.config))(
                inputs['arrays'], inputs['probabilities'])

        def loss(theta):
            curves = solve_branches(inputs, terms_of(theta), ph, **kwargs)[0]
            return jnp.sum(curves.protonated.astype(jnp.float64)**2), curves

        return params, jax.jit(jax.value_and_grad(loss, has_aux=True)), terms_of, ph, inputs

    configurations = [
        ('float64', jnp.float64, None),
        ('float64_solve_float32_seed', jnp.float64, 'float32'),
        ('float32', jnp.float32, None),
    ]
    if args.only:
        configurations = [c for c in configurations if c[0] in args.only]
    results = []
    baseline = None
    if args.reference_json:
        stored = read(args.reference_json)
        match = next(r for r in stored['results'] if r['configuration'] == 'float64')
        baseline = dict(gradient=np.asarray(match['gradient'], float),
                        value=match['value'], curves=None)
        results.append(dict(configuration='float64_reference', **{
            k: match[k] for k in ('seconds', 'value', 'gradient', 'max_residual')}))
        print(json.dumps(results[-1]), flush=True)
    for name, dtype, seed_dtype in configurations:
        try:
            params, fn, terms_of, ph_used, inputs_used = build(dtype, seed_dtype)
            # A dtype leak shows up as a structure mismatch deep inside lineax; report
            # the dtypes that feed the solve so the source is visible, not guessed.
            probe = jax.eval_shape(terms_of, params)
            seen = dict(theta=str(params.dtype), ph=str(ph_used.dtype),
                        terms={k: str(v.dtype) for k, v in probe._asdict().items()},
                        arrays=sorted({str(v.dtype) for v in inputs_used['arrays'].values()}),
                        probabilities=str(inputs_used['probabilities'].dtype))
            print(json.dumps(dict(configuration=name, dtypes=seen)), flush=True)
            (value, curves), grad = jax.block_until_ready(fn(params))   # compile
            best = np.inf
            for _ in range(args.repeats):
                started = time.perf_counter()
                jax.block_until_ready(fn(params))
                best = min(best, time.perf_counter()-started)
        except Exception as exc:
            detail = traceback.format_exc()
            results.append(dict(configuration=name, error=repr(exc)[:200],
                                traceback=detail[-4000:], dtypes=locals().get('seen')))
            print(detail, flush=True)
            print(json.dumps(results[-1]), flush=True)
            continue
        grad = np.asarray(grad, float)
        residual = float(jnp.max(curves.residual))
        entry = dict(configuration=name, seconds=round(best, 3), value=float(value),
                     gradient=[float(x) for x in grad],
                     max_residual=residual, within_tolerance=bool(residual < tolerance),
                     all_converged=bool(jnp.all(curves.converged)))
        if baseline is None:
            baseline = dict(gradient=grad, value=float(value),
                            curves=np.asarray(curves.protonated, float))
            entry.update(gradient_relative_difference=0., max_curve_difference=0., speedup=1.)
        else:
            denominator = np.maximum(np.abs(baseline['gradient']), 1e-30)
            entry['gradient_relative_difference'] = float(
                np.max(np.abs(grad-baseline['gradient'])/denominator))
            entry['value_relative_difference'] = float(
                abs(float(value)-baseline['value'])/max(abs(baseline['value']), 1e-30))
            if baseline['curves'] is not None:
                entry['max_curve_difference'] = float(np.max(np.abs(
                    np.asarray(curves.protonated, float)-baseline['curves'])))
            if results[0].get('seconds'):
                entry['speedup'] = round(results[0]['seconds']/best, 2)
        results.append(entry)
        print(json.dumps(entry), flush=True)

    atomic_json(args.output/f'precision-{cid}.json', dict(
        job=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename,
        devices=[str(d) for d in jax.devices()], complex_id=cid,
        n_ph=int(ph64.size), seed_steps=args.seed_steps, lm_max_steps=args.max_steps,
        residual_tolerance=tolerance, cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
        x64=args.x64, results=results))


if __name__ == '__main__':
    main()
