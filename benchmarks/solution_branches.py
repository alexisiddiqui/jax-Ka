"""Multiple mean-field solutions: candidate branches, free energies and selection rules.

Diagnostic only; no production change. Per state and pH, four candidate
solutions on the native active set:
  published  production-1024-v2 damped solver (cold start at every pH)
  cold       optimistix Newton from the same cold start
  up / down  optimistix Newton continuation, ascending / descending pH
  lm_cold    optimistix Levenberg-Marquardt from the cold start (globalized)
Each candidate gets the active-set fixed-point residual and the mean-field free
energy (optx_solver.free_energy). Rules evaluated with the unchanged grid readout:
  (a) min-F   lowest free energy among converged candidates at each pH
  (b) flag    sites whose converged candidates disagree anywhere are invalid
  (c) cold    Newton from the cold start (closest to v2 semantics)
  (d) min-F-optx  as (a) but over optimistix candidates only (no damped solver):
                  the candidate fast v3 rule. Its solve is timed; the damped
                  1024-step solver is timed on the same node for comparison.
Selection of complexes is shared with solver_compare.py (size quantiles plus,
separately labelled, complexes with nonconverged v2 states).
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

TOL_SAME = 1e-3      # candidates closer than this (max |dh| on active) are one solution


def run(campaign, cid, label, out):
    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optimistix as optx
    from pkabench.prep import read_cif
    from pkabench.schema import PH
    from jaxpropka import TitrationModel
    from jaxpropka.parameters import ModelConfig
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache, native_identities
    from jaxpropka.model import _grid_pka_result, _local_terms, _packed_field
    from jaxpropka.optx_solver import SolverConfig, _solver, active_channels, active_system, free_energy
    receipt = json.loads((campaign/'jobs/jaxka'/f'{cid}.json').read_text())
    config = ModelConfig(steps=1024); tol = config.residual_tolerance
    scfg = SolverConfig(method='newton', rtol=1e-5, atol=1e-5)
    solver = _solver(scfg); lm = _solver(SolverConfig(method='lm', rtol=1e-5, atol=1e-5)); log10 = jnp.log(jnp.float32(10)); grid = jnp.asarray(PH, jnp.float32)
    states = []
    for state in ('AB', 'A', 'B'):
        ref = np.load(Path(receipt['workdir'])/state/'curves.npz')
        top = load_topology(read_cif(campaign/'structures'/cid/f'{state}.cif'), gap_policy='cap', freeze_disulfides=True)
        cache = build_cache(top, build_candidates(top, missing_sidechain='error'), identities=native_identities(top))
        model = TitrationModel(cache, config=config, backend='packed'); d = model.arrays; p = model.native_probabilities
        act = jnp.asarray(active_channels(cache, np.asarray(p)))
        gm = d['group_mask']; edges = model._edges

        @jax.jit
        def solve_all(d, edges, act, p):
            gm = d['group_mask']
            terms = _local_terms(d, p, config)
            ia, f0, k = active_system(d, terms, act); w = terms.weights.reshape(-1)[act]
            def residual(u, args):
                x = args
                return u - log10*(ia - x - f0 - k@jax.nn.sigmoid(u))
            def root(u0, x, method=solver):
                sol = optx.root_find(residual, method, u0, args=x, max_steps=scfg.max_steps, throw=False)
                return sol.value
            cold0 = lambda x: log10*(ia - x - f0)
            cold = jax.vmap(lambda x: root(cold0(x), x))(grid)
            lm_cold = jax.vmap(lambda x: root(cold0(x), x, lm))(grid)
            def sweep(xs):
                def step(u, x):
                    u = root(u, x); return u, u
                return jax.lax.scan(step, cold0(xs[0]), xs)[1]
            up = sweep(grid); down = sweep(grid[::-1])[::-1]
            us = jnp.stack([cold, up, down, lm_cold])                    # [4,H,M]
            hs = jax.nn.sigmoid(us)
            def full_target(h_act, x):
                full = jnp.zeros(gm.size, h_act.dtype).at[act].set(h_act).reshape(gm.shape)
                return jnp.where(gm, jax.nn.sigmoid(log10*(terms.intrinsic-x-_packed_field(terms, full, edges))), 0)
            # Full-channel curves (inactive channels: closed-form response), as in optx_solver.
            hfull = jax.vmap(lambda hh: jax.vmap(full_target)(hh, grid))(hs)     # [3,H,N,9]
            return hs, hfull, ia, f0, k, w, terms.weights
        compiled = solve_all.lower(d, edges, act, p).compile()
        t0 = time.perf_counter(); jax.block_until_ready(compiled(d, edges, act, p)); optx_seconds = time.perf_counter()-t0
        hs, hfull, ia, f0, k, w, weights_full = compiled(d, edges, act, p)
        picard = model.curves(PH).lower(p).compile()
        t0 = time.perf_counter(); jax.block_until_ready(picard(d, edges, p)); picard_seconds = time.perf_counter()-t0
        published_full = jnp.asarray(ref['protonated'])
        published_act = published_full.reshape(len(PH), -1)[:, act]
        cand_act = jnp.concatenate([published_act[None], hs])            # [5,H,M]
        cand_full = jnp.concatenate([published_full[None], hfull])       # [5,H,N,9]
        names = ['published', 'cold', 'up', 'down', 'lm_cold']
        target = jax.nn.sigmoid(log10*(ia[None, None] - grid[None, :, None] - f0[None, None]
                                       - jnp.einsum('mn,chn->chm', k, cand_act)))
        res_act = jnp.max(jnp.abs(target-cand_act), axis=-1)            # [4,H]
        energy = jax.vmap(lambda hh: jax.vmap(lambda h, x: free_energy(h, x, ia, f0, k, w))(hh, grid))(cand_act)
        res_act, energy, cand_act = map(np.asarray, (res_act, energy, cand_act))
        converged = res_act < tol
        # Distinct converged solutions per pH and the min-F choice.
        big = np.where(converged, energy, np.inf)
        choice = np.argmin(big, axis=0)                                  # [H]
        any_conv = converged.any(0)
        diff = np.abs(cand_act[:, None]-cand_act[None, :]).max(-1)       # [4,4,H]
        pair_conv = converged[:, None] & converged[None, :]
        multi_ph = np.any(pair_conv & (diff > TOL_SAME), axis=(0, 1))    # [H]
        site_split = np.any((pair_conv[..., None] & (np.abs(cand_act[:, None]-cand_act[None, :]) > TOL_SAME)),
                            axis=(0, 1, 2))                              # [M]
        def readout(curve_full, residual):
            out = SimpleNamespace(ph=grid, protonated=jnp.asarray(curve_full), residual=jnp.asarray(residual),
                                  converged=jnp.asarray(residual) < tol, probability=weights_full)
            mid = _grid_pka_result(d, out, grid, config)
            return {k_: np.asarray(v) for k_, v in mid._asdict().items()}
        cf = np.asarray(cand_full)
        minf_full = cf[choice, np.arange(len(PH))]
        minf_res = np.where(any_conv, res_act[choice, np.arange(len(PH))], 1.)
        # (d) min-F over optimistix candidates only.
        big_o = np.where(converged[1:], energy[1:], np.inf); choice_o = 1+np.argmin(big_o, axis=0)
        any_o = converged[1:].any(0); h_ = np.arange(len(PH))
        mino_full = cf[choice_o, h_]; mino_res = np.where(any_o, res_act[choice_o, h_], 1.)
        rules = {'published': readout(cf[0], res_act[0]), 'cold': readout(cf[1], res_act[1]),
                 'up': readout(cf[2], res_act[2]), 'lm_cold': readout(cf[4], res_act[4]),
                 'min_F': readout(minf_full, minf_res), 'min_F_optx': readout(mino_full, mino_res)}
        both_conv = converged[0] & any_o
        excess = np.where(both_conv, energy[choice_o, h_]-energy[0], 0.)          # >0: optx missed v2's lower solution
        curve_gap = np.abs(cand_act[choice_o, h_]-cand_act[0]).max(-1)             # [H] on active
        activemask = np.zeros(gm.size, bool); activemask[np.asarray(act)] = True; activemask = activemask.reshape(gm.shape)
        flagged = np.zeros(gm.size, bool); flagged[np.asarray(act)[site_split]] = True; flagged = flagged.reshape(gm.shape)
        def summary(r):
            v = r['valid'] & activemask
            return dict(valid=int(v.sum()), nonmonotone=int((activemask & r['bracketed'] & ~r['sampled_monotone']).sum()))
        record = dict(state=state, n_residues=int(top.n_residues), active=int(act.shape[0]),
                      ph_with_multiple_solutions=int(multi_ph.sum()), sites_with_multiple_solutions=int(site_split.sum()),
                      converged_counts={n: int(converged[i].sum()) for i, n in enumerate(names)},
                      min_F_choice_counts={n: int(((choice == i) & any_conv).sum()) for i, n in enumerate(names)},
                      published_is_min_F_ph=int((converged[0] & (energy[0] <= big.min(0)+1e-4)).sum()),
                      published_excess_F_max=float(np.max(np.where(converged[0] & any_conv, energy[0]-big.min(0), 0.))),
                      max_energy_gap=float(max((energy[converged[:, h], h].max()-energy[converged[:, h], h].min())
                                               for h in np.flatnonzero(multi_ph))) if multi_ph.any() else 0.,
                      published_valid_npz=int((np.asarray(ref['valid']) & activemask).sum()),
                      rules={n: summary(r) for n, r in rules.items()},
                      flag_rule_valid=int((rules['min_F']['valid'] & activemask & ~flagged).sum()),
                      optx_seconds=optx_seconds, picard_seconds=picard_seconds,
                      optx_only=dict(published_converged_but_optx_not=int((converged[0] & ~any_o).sum()),
                                     optx_converged_but_published_not=int((~converged[0] & any_o).sum()),
                                     missed_lower_solution_ph=int((excess > 1e-3).sum()),
                                     found_lower_solution_ph=int((excess < -1e-3).sum()),
                                     excess_F_max=float(excess.max()), curve_gap_max_where_both_converged=float(
                                         curve_gap[both_conv].max()) if both_conv.any() else 0.,
                                     same_solution_ph=int((both_conv & (curve_gap <= TOL_SAME)).sum()),
                                     both_converged_ph=int(both_conv.sum())))
        # Midpoint changes vs published for sites valid under both.
        for n in ('cold', 'up', 'lm_cold', 'min_F', 'min_F_optx'):
            both = rules[n]['valid'] & rules['published']['valid'] & activemask
            record['rules'][n]['midpoint_max_abs_vs_published'] = float(np.max(np.abs(rules[n]['value']-rules['published']['value'])[both])) if both.any() else 0.
        states.append(record); print(json.dumps({'complex_id': cid, **{k_: record[k_] for k_ in ('state', 'ph_with_multiple_solutions', 'sites_with_multiple_solutions')}}), flush=True)
    atomic_json(out/f'{cid}.json', dict(complex_id=cid, label=label, node=os.uname().nodename,
                                        job=os.environ.get('SLURM_JOB_ID'), states=states))


def summarize(out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('*.json')) if p.name != 'summary.json']
    result = {}
    for label in ('quantile', 'nonconverged'):
        sts = [s for r in rows if r['label'] == label for s in r['states']]
        if not sts: continue
        part = dict(states=len(sts), states_with_multiple_solutions=sum(s['ph_with_multiple_solutions'] > 0 for s in sts),
                    ph_points_with_multiple_solutions=sum(s['ph_with_multiple_solutions'] for s in sts),
                    sites_with_multiple_solutions=sum(s['sites_with_multiple_solutions'] for s in sts),
                    active_sites=sum(s['active'] for s in sts),
                    published_converged_ph=sum(s['converged_counts']['published'] for s in sts),
                    published_is_min_F_ph=sum(s['published_is_min_F_ph'] for s in sts),
                    published_excess_F_max=max(s['published_excess_F_max'] for s in sts),
                    max_energy_gap=max(s['max_energy_gap'] for s in sts),
                    min_F_choice_counts={n: sum(s['min_F_choice_counts'][n] for s in sts) for n in ('published', 'cold', 'up', 'down', 'lm_cold')},
                    optx_hours=sum(s['optx_seconds'] for s in sts)/3600, picard_hours=sum(s['picard_seconds'] for s in sts)/3600,
                    optx_only={k: (max if 'max' in k else sum)(s['optx_only'][k] for s in sts) for k in sts[0]['optx_only']},
                    flag_rule_valid=sum(s['flag_rule_valid'] for s in sts))
        for n in ('published', 'cold', 'up', 'lm_cold', 'min_F', 'min_F_optx'):
            part[n] = dict(valid=sum(s['rules'][n]['valid'] for s in sts), nonmonotone=sum(s['rules'][n]['nonmonotone'] for s in sts))
            if n != 'published':
                part[n]['midpoint_max_abs_vs_published'] = max(s['rules'][n]['midpoint_max_abs_vs_published'] for s in sts)
        result[label] = part
    (out/'summary.json').write_text(json.dumps(result, indent=1)); print(json.dumps(result, indent=1))


if __name__ == '__main__':
    from solver_compare import select
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
