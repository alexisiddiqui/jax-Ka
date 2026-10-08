"""Does the seed reach the solver's basin on EVERY prepared record, not one of them?

seed_steps=64 was validated on a single 200-residue complex and generalised; the
1460-residue record then failed to converge at pH 6.5 even with 512 LM steps, because
the damped Picard rate is set by the coupling spectrum and large strongly coupled
systems need more iterations to reach the basin.

The seed now iterates the compact active-set system from ``active_system`` rather than
the full [N,Kc,9,9] field. The two agree exactly on the active channels (leak == 0), so
this is a pure cost change -- roughly 100-700x per iteration -- which makes the full
production step count affordable again.

This sweeps seed step counts across every prepared record and reports, per record:
convergence at every pH, the worst residual and the pH where it occurs, Newton steps,
and wall time. A configuration is only acceptable if every record converges everywhere.
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
    parser.add_argument('--seed-steps', type=int, nargs='*', default=[1024, 256, 64])
    parser.add_argument('--seed-dtype', default=None, choices=[None, 'float32'])
    parser.add_argument('--max-steps', type=int, default=32)
    parser.add_argument('--complexes', nargs='*')
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

    engine = make_engine()
    solver_config = replace(engine.solver_config, max_steps=args.max_steps)
    tolerance = float(engine.config.residual_tolerance)
    ph = engine.ph
    params = jnp.zeros(3, jnp.float64)

    ids = args.complexes
    if not ids:
        prepared = sorted((args.training_root/'prepared').glob('*/receipt.json'))
        ids = [p.parent.name for p in prepared]

    records = []
    for cid in ids:
        inputs, reference, eligible, receipt = load(args.training_root, cid)
        inputs = jax.tree.map(jnp.asarray, inputs)
        layout = receipt.get('layout', {})
        entry = dict(complex_id=cid, **{k: layout.get(k) for k in
                                        ('real_residues', 'N', 'M', 'Kc', 'Ke')}, runs=[])
        for steps in args.seed_steps:
            kwargs = dict(config=engine.config, solver_config=solver_config,
                          seed_steps=steps)
            if args.seed_dtype:
                kwargs['seed_dtype'] = args.seed_dtype

            @jax.jit
            def forward(theta):
                terms = jax.vmap(lambda d, p: local_terms(theta, d, p, engine.config))(
                    inputs['arrays'], inputs['probabilities'])
                return solve_branches(inputs, terms, ph, **kwargs)

            try:
                curves, extra = jax.block_until_ready(forward(params))
                started = time.perf_counter()
                jax.block_until_ready(forward(params))
                seconds = time.perf_counter()-started
            except Exception as exc:
                entry['runs'].append(dict(seed_steps=steps, error=repr(exc)[:200]))
                print(json.dumps(dict(complex_id=cid, **entry['runs'][-1])), flush=True)
                continue
            residual = np.asarray(curves.residual, float)          # [branch, pH]
            converged = np.asarray(curves.converged, bool)
            worst = np.unravel_index(np.argmax(residual), residual.shape)
            entry['runs'].append(dict(
                seed_steps=steps, seconds=round(seconds, 3),
                all_converged=bool(converged.all()),
                failed_ph=[float(ph[i]) for i in
                           sorted({int(j) for _, j in zip(*np.where(~converged))})][:12],
                n_failed=int((~converged).sum()),
                max_residual=float(residual.max()),
                max_residual_ph=float(ph[worst[1]]), max_residual_branch=int(worst[0]),
                within_tolerance=bool(residual.max() < tolerance),
                newton_steps_max=int(np.max(extra['newton_steps'])),
                newton_steps_mean=round(float(np.mean(extra['newton_steps'])), 2),
                active_set_leak=float(np.max(extra['active_set_leak']))))
            print(json.dumps(dict(complex_id=cid, residues=layout.get('real_residues'),
                                  M=layout.get('M'), **entry['runs'][-1])), flush=True)
        records.append(entry)
        atomic_json(args.output/'coverage.json', dict(
            job=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename,
            seed_dtype=args.seed_dtype, lm_max_steps=args.max_steps,
            residual_tolerance=tolerance, n_ph=int(ph.size), records=records))

    for steps in args.seed_steps:
        ok = [r for r in records
              if any(x.get('seed_steps') == steps and x.get('all_converged') for x in r['runs'])]
        print(f'seed_steps={steps}: {len(ok)}/{len(records)} records converged everywhere',
              flush=True)


if __name__ == '__main__':
    main()
