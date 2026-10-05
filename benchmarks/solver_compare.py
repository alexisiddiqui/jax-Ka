"""Optimistix active-set solver vs the production 1024-step damped solver.

Diagnostic only. For one complex (all three states) the published
production-1024-v2 curves are the reference. The damped solver is re-timed on
the same node; optimistix variants are timed and compared on curves, fixed-point
residual, validity and midpoints using the unchanged grid readout. Selection
(by the caller) is size quantiles plus, separately labelled, the complexes with
nonconverged v2 states.
"""
import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

VARIANTS = {
    'lm-1e-5': dict(method='lm', rtol=1e-5, atol=1e-5),
    'newton-1e-5': dict(method='newton', rtol=1e-5, atol=1e-5),
    'lm-1e-6': dict(method='lm', rtol=1e-6, atol=1e-6),
}


def timed(fn):
    t0 = time.perf_counter(); out = fn()
    import jax; jax.block_until_ready(out)
    return out, time.perf_counter()-t0


def run(campaign, cid, label, out):
    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    import numpy as np
    import jax
    import jax.numpy as jnp
    from pkabench.prep import read_cif
    from pkabench.schema import PH
    from jaxpropka import TitrationModel
    from jaxpropka.parameters import ModelConfig
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache, native_identities
    from jaxpropka.model import _grid_pka_result
    from jaxpropka.optx_solver import SolverConfig, active_channels, optx_curve_kernel
    jax.config.update('jax_enable_compilation_cache', False)
    receipt = json.loads((campaign/'jobs/jaxka'/f'{cid}.json').read_text())
    config = ModelConfig(steps=1024); grid = jnp.asarray(PH, jnp.float32)
    states = []
    for state in ('AB', 'A', 'B'):
        ref = np.load(Path(receipt['workdir'])/state/'curves.npz')
        top = load_topology(read_cif(campaign/'structures'/cid/f'{state}.cif'), gap_policy='cap', freeze_disulfides=True)
        cache = build_cache(top, build_candidates(top, missing_sidechain='error'), identities=native_identities(top))
        assert cache.fingerprint() == receipt['extra'][state]['cache_fingerprint']
        model = TitrationModel(cache, config=config, backend='packed'); p = model.native_probabilities
        active = jnp.asarray(active_channels(cache, np.asarray(p)))
        act = np.asarray(ref['active'])
        compiled, compile_s = timed(lambda: model.curves(PH).lower(p).compile())
        picard, picard_s = timed(lambda: compiled(model.arrays, model._edges, p))
        record = dict(state=state, n_residues=int(top.n_residues), active_channels=int(active.size),
                      picard=dict(compile_seconds=compile_s, run_seconds=picard_s,
                                  max_abs_vs_published=float(np.max(np.abs(np.asarray(picard.protonated)-ref['protonated'])))),
                      published=dict(converged=bool(np.all(ref['residual'] < config.residual_tolerance)),
                                     max_residual=float(ref['residual'].max()), valid=int(np.sum(ref['valid']))),
                      variants={})
        for name, kw in VARIANTS.items():
            scfg = SolverConfig(**kw)
            comp, c_s = timed(lambda: optx_curve_kernel.lower(model.arrays, p, grid, active, model._edges,
                                                               config=config, solver_config=scfg).compile())
            (curves, extra), r_s = timed(lambda: comp(model.arrays, p, grid, active, model._edges))
            mid = _grid_pka_result(model.arrays, curves, curves.ph, config)
            h = np.asarray(curves.protonated); valid = np.asarray(mid.valid); value = np.asarray(mid.value)
            both = valid & np.asarray(ref['valid'])
            record['variants'][name] = dict(
                compile_seconds=c_s, run_seconds=r_s,
                steps_mean=float(np.mean(extra['newton_steps'])), steps_max=int(np.max(extra['newton_steps'])),
                optx_success_all=bool(np.all(extra['optx_success'])),
                max_residual=float(np.max(curves.residual)), converged=bool(np.all(curves.converged)),
                curve_max_abs=float(np.max(np.abs(h-ref['protonated'])[:, act])) if act.any() else 0.,
                valid=int(valid.sum()), valid_both=int(both.sum()),
                valid_only_published=int((np.asarray(ref['valid']) & ~valid).sum()),
                valid_only_new=int((valid & ~np.asarray(ref['valid'])).sum()),
                midpoint_max_abs=float(np.max(np.abs(value-ref['midpoint'])[both])) if both.any() else 0.)
        states.append(record)
        print(json.dumps({'complex_id': cid, 'state': state, 'picard_s': picard_s,
                          **{k: (v['run_seconds'], v['curve_max_abs'], v['valid_only_published'], v['valid_only_new'])
                             for k, v in record['variants'].items()}}), flush=True)
    atomic_json(out/f'{cid}.json', dict(complex_id=cid, label=label, node=os.uname().nodename,
                                        job=os.environ.get('SLURM_JOB_ID'), states=states))


def select(campaign, count):
    """Size quantiles (label 'quantile') plus complexes with any nonconverged v2 state ('nonconverged')."""
    from pkabench.schema import read_table
    structures = sorted(read_table(campaign/'structures.parquet'), key=lambda s: (s['n_residues'], s['complex_id']))
    picks = [structures[round(q*(len(structures)-1)/(count-1))]['complex_id'] for q in range(count)]
    chosen = [(c, 'quantile') for c in dict.fromkeys(picks)]
    for s in structures:
        r = json.loads((campaign/'jobs/jaxka'/f"{s['complex_id']}.json").read_text())
        if not all(e['grid_converged'] for e in r['extra'].values()) and s['complex_id'] not in picks:
            chosen.append((s['complex_id'], 'nonconverged'))
    return chosen


def summarize(out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('*.json')) if p.name != 'summary.json']
    summary = {'complexes': len(rows), 'states': sum(len(r['states']) for r in rows)}
    for label in ('quantile', 'nonconverged'):
        sts = [s for r in rows if r['label'] == label for s in r['states']]
        if not sts: continue
        part = {'states': len(sts), 'picard_run_hours': sum(s['picard']['run_seconds'] for s in sts)/3600,
                'published_nonconverged_states': sum(not s['published']['converged'] for s in sts)}
        for name in VARIANTS:
            v = [s['variants'][name] for s in sts]
            part[name] = {'run_hours': sum(x['run_seconds'] for x in v)/3600,
                          'compile_hours': sum(x['compile_seconds'] for x in v)/3600,
                          'nonconverged_states': sum(not x['converged'] for x in v),
                          'optx_failure_states': sum(not x['optx_success_all'] for x in v),
                          'steps_mean': sum(x['steps_mean'] for x in v)/len(v), 'steps_max': max(x['steps_max'] for x in v),
                          'curve_max_abs': max(x['curve_max_abs'] for x in v),
                          'valid_only_published': sum(x['valid_only_published'] for x in v),
                          'valid_only_new': sum(x['valid_only_new'] for x in v),
                          'valid_both': sum(x['valid_both'] for x in v),
                          'midpoint_max_abs': max(x['midpoint_max_abs'] for x in v)}
        summary[label] = part
    (out/'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('stage', choices=['run', 'summarize'])
    a.add_argument('--campaign', type=Path, required=True)
    a.add_argument('--count', type=int, default=30)
    a.add_argument('--index', type=int, default=0)
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'run':
        chosen = select(args.campaign, args.count)
        if args.index < len(chosen):
            run(args.campaign, *chosen[args.index], args.output)
    else:
        summarize(args.output)
