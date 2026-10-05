"""Is the optx implicit adjoint wrong, or is the linkage genuinely kinked?

``gradient_sanity.py`` compares the optimistix implicit adjoint against a one-sided
finite difference OF THE SAME SOLVER. On 1fcc the two disagree by a bounded, step-size
independent ~2.5e-3 kcal/mol. That comparison cannot separate two explanations:

  (a) the adjoint is wrong, or
  (b) L is genuinely non-smooth at the base point and the one-sided difference measures
      the slope on the far side of a kink.

A second, INDEPENDENT analytic gradient discriminates. Reverse mode through the damped
1024-step fixed-point loop is a different code path with no implicit-function step. The
objective weights only the 9 grid points in [5.5, 7.5], and the damped solver treats each
pH independently (no continuation), so restricting the grid to those points leaves L
unchanged and makes the unrolled adjoint affordable (1024 * [9, N, 9] float64).

Reports a 2x2 of {optx, damped} x {analytic, finite difference} on the same directions.
  damped-analytic == damped-FD, both != optx-analytic  ->  the optx adjoint is at fault.
  damped-analytic == optx-analytic, both != FD         ->  L is kinked; the FD misleads.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from gradient_sanity import (SYSTEMS, DESIGN_AA, PH_LO, PH_HI, build_state,
                             interface_distance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cif-dir', type=Path, required=True)
    parser.add_argument('--systems', nargs='+', default=['1fcc', '1brs'])
    parser.add_argument('--cutoff', type=float, default=5.0)
    parser.add_argument('--max-positions', type=int, default=48)
    parser.add_argument('--probes', type=int, default=4, help='closest design positions probed')
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jaxpropka.parameters import ModelConfig, ALPHABET
    from jaxpropka.optx_solver import SolverConfig, active_channels, optx_curve_kernel
    from jaxpropka.model import packed_curve_kernel
    from jaxpropka.topology import load_topology
    from pkabench.prep import read_cif
    from pkabench.schema import PH

    config = ModelConfig(steps=args.steps)
    scfg = SolverConfig()
    ph_full = np.asarray(PH, dtype=float)
    window = (ph_full >= PH_LO) & (ph_full <= PH_HI)
    inside = ph_full[window]
    quad = np.zeros_like(inside)
    quad[:-1] += np.diff(inside)/2
    quad[1:] += np.diff(inside)/2
    w_window = jnp.asarray(1.364*quad)                 # weights on the 9 window points
    window_index = jnp.asarray(np.flatnonzero(window))

    records = []
    for name in args.systems:
        spec = SYSTEMS[name]
        started = time.monotonic()
        atoms = read_cif(args.cif_dir/f'{name}.cif')
        full = load_topology(atoms, gap_policy='cap', freeze_disulfides=True,
                             chains=spec['a']+spec['b'], ignore_nonprotein=True)
        distance, _ = interface_distance(full, spec['a'])
        native = np.asarray(full.native_index)
        free = ~np.asarray(full.disulfide)
        eligible = np.flatnonzero((distance <= args.cutoff) & free
                                  & ~np.isin(native, [ALPHABET.index(a) for a in 'GP']))
        positions = np.sort(eligible[np.argsort(distance[eligible])][:args.max_positions])
        design_cols = np.array([ALPHABET.index(a) for a in DESIGN_AA])
        n = full.n_residues
        mask = np.zeros((n, 20), bool)
        mask[np.arange(n), native] = True
        mask[np.ix_(positions, design_cols)] = True
        keys_full = [(k.chain, k.number, k.insertion) for k in full.keys]
        mask_by_key = dict(zip(keys_full, mask))

        states = {label: build_state(atoms, chains, mask_by_key, config)
                  for label, chains in (('AB', spec['a']+spec['b']), ('A', spec['a']),
                                        ('B', spec['b']))}
        keys_ab = states['AB'][3]
        index = {k: i for i, k in enumerate(keys_ab)}
        rows = {l: jnp.asarray(np.array([index[k] for k in states[l][3]])) for l in ('A', 'B')}
        rows['AB'] = jnp.arange(len(keys_ab))
        active = {l: jnp.asarray(active_channels(states[l][1], mask[np.asarray(rows[l])].astype(float)))
                  for l in ('AB', 'A', 'B')}
        models = {l: states[l][2] for l in ('AB', 'A', 'B')}

        def L_optx(p):
            # Full 73-point grid: the continuation sweep starts at PH[0], as in production.
            total = 0.
            for sign, l in ((1., 'AB'), (-1., 'A'), (-1., 'B')):
                q = p[rows[l]]
                out, _ = optx_curve_kernel(models[l].arrays, q, jnp.asarray(ph_full, q.dtype),
                                           active[l], config=config, solver_config=scfg)
                total = total + sign*out.total_charge
            return jnp.sum(total[window_index]*w_window)

        def L_damped(p):
            # Only the 9 weighted points: each pH is solved independently, so this is the
            # same L, and reverse mode through 1024 steps stays affordable.
            total = 0.
            for sign, l in ((1., 'AB'), (-1., 'A'), (-1., 'B')):
                q = p[rows[l]]
                out = packed_curve_kernel(models[l].arrays, q, jnp.asarray(inside, q.dtype),
                                          models[l]._edges, config=config)
                total = total + sign*out.total_charge
            return jnp.sum(total*w_window)

        p0 = jnp.asarray(np.eye(20, dtype=np.float64)[native])
        base_optx = float(L_optx(p0)); base_damped = float(L_damped(p0))
        g_optx = np.asarray(jax.grad(L_optx)(p0), float)
        g_damped = np.asarray(jax.grad(L_damped)(p0), float)

        probes = []
        for i in positions[:args.probes]:
            col = int(design_cols[np.argmax(design_cols != native[i])])
            if col == native[i]:
                continue
            d = np.zeros((n, 20), np.float64)
            d[i, col] = 1.; d[i, native[i]] = -1.
            dj = jnp.asarray(d)
            row = dict(index=int(i), native=ALPHABET[native[i]], mutant=ALPHABET[col],
                       analytic_optx=float(g_optx[i, col]-g_optx[i, native[i]]),
                       analytic_damped=float(g_damped[i, col]-g_damped[i, native[i]]))
            for eps in (1e-2, 1e-3):
                row[f'fd_optx_{eps}'] = (float(L_optx(p0+eps*dj))-base_optx)/eps
                row[f'fd_damped_{eps}'] = (float(L_damped(p0+eps*dj))-base_damped)/eps
            row['optx_adjoint_minus_damped_adjoint'] = row['analytic_optx']-row['analytic_damped']
            row['damped_adjoint_minus_damped_fd'] = row['analytic_damped']-row['fd_damped_0.001']
            probes.append(row)
            print(name, row['index'], row['native']+'->'+row['mutant'],
                  'optx_adj %.7f' % row['analytic_optx'],
                  'damped_adj %.7f' % row['analytic_damped'],
                  'damped_fd %.7f' % row['fd_damped_0.001'], flush=True)

        record = dict(system=name, base_L_optx=base_optx, base_L_damped=base_damped,
                      base_gap=base_optx-base_damped, steps=args.steps,
                      n_window_points=int(window.sum()), probes=probes,
                      wall_seconds=time.monotonic()-started)
        atomic_json(args.output/f'{name}.json', record)
        print(json.dumps({k: record[k] for k in
                          ('system', 'base_L_optx', 'base_L_damped', 'base_gap')}, indent=1),
              flush=True)
        records.append(record)
    atomic_json(args.output/'summary.json', dict(records=records))


if __name__ == '__main__':
    main()
