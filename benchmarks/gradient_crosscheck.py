"""Is the optx implicit adjoint wrong, or is the linkage genuinely kinked?

``gradient_sanity.py`` compares the optimistix implicit adjoint against a one-sided
finite difference OF THE SAME SOLVER. On 1fcc the two disagree by a bounded, step-size
independent ~2.5e-3 kcal/mol. That comparison cannot separate two explanations:

  (a) the adjoint is wrong, or
  (b) L is genuinely non-smooth at the base point and the one-sided difference measures
      the slope on the far side of a kink.

A second, INDEPENDENT analytic gradient discriminates. Reverse mode through the damped
fixed-point loop is a different code path with no implicit-function step.

Two things make it affordable. (1) A SINGLE pH point: comparing derivatives of dQ(pH)
answers the question just as well as the pH-integrated L, at 1/9 the cost. (2)
``jax.checkpoint`` on the loop body, so the tape holds only the [N,9] carry per step
rather than the per-edge products -- reverse mode through the PACKED field stores
n_edges * steps values and OOMs at terabyte scale, which is why the production design
uses the implicit adjoint in the first place.

The damped solve here is re-derived from ``_local_terms`` and the dense ``_field`` rather
than called through ``packed_curve_kernel``. Being an independent implementation is the
point; its forward value is asserted against the production kernel to ~1e-10 so a
transcription bug cannot masquerade as a gradient discrepancy.

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
    from jaxpropka.model import packed_curve_kernel, _local_terms
    from jaxpropka.parameters import Q_DEPROT
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

        # One pH point, mid-window. Derivatives of dQ(ph_probe) answer the same question.
        ph_probe = float(inside[len(inside)//2])

        def dq_optx(p):
            total = 0.
            for sign, l in ((1., 'AB'), (-1., 'A'), (-1., 'B')):
                q = p[rows[l]]
                out, _ = optx_curve_kernel(models[l].arrays, q, jnp.asarray(ph_full, q.dtype),
                                           active[l], config=config, solver_config=scfg)
                total = total + sign*out.total_charge[window_index][len(inside)//2]
            return total

        def charge_damped(arrays, p, ph):
            """Damped fixed point, re-derived; mirrors _solve_with_field + total_charge."""
            terms = _local_terms(arrays, p, config)
            gm = arrays['group_mask']
            log10 = jnp.log(jnp.asarray(10, terms.intrinsic.dtype))

            def target(h):
                field = terms.field0 + jnp.einsum('nkgt,nkt->ng', terms.coupling,
                                                  h[arrays['neighbors']])
                return jnp.where(gm, jax.nn.sigmoid(log10*(terms.intrinsic-ph-field)), 0)

            h = jnp.where(gm, jax.nn.sigmoid(log10*(terms.intrinsic-ph-terms.field0)), 0)

            @jax.checkpoint          # rematerialize the body; keep only the [N,9] carry
            def step(_, old):
                return old + config.damping*(target(old)-old)

            h = jax.lax.fori_loop(0, config.steps, step, h)
            charge = terms.weights*(jnp.asarray(Q_DEPROT, h.dtype)+h)
            return jnp.sum(charge), jnp.max(jnp.abs(target(h)-h))

        def dq_damped(p):
            total = 0.
            for sign, l in ((1., 'AB'), (-1., 'A'), (-1., 'B')):
                total = total + sign*charge_damped(models[l].arrays, p[rows[l]], ph_probe)[0]
            return total

        p0 = jnp.asarray(np.eye(20, dtype=np.float64)[native])
        base_optx = float(dq_optx(p0)); base_damped = float(dq_damped(p0))
        # A transcription bug in charge_damped must not look like a gradient discrepancy.
        reference = 0.
        for sign, l in ((1., 'AB'), (-1., 'A'), (-1., 'B')):
            q = p0[rows[l]]
            out = packed_curve_kernel(models[l].arrays, q, jnp.asarray([ph_probe], q.dtype),
                                      models[l]._edges, config=config)
            reference = reference + sign*float(out.total_charge[0])
        forward_mismatch = abs(base_damped-reference)
        if forward_mismatch > 1e-8:
            raise ValueError(f'{name}: re-derived damped solve disagrees with the production '
                             f'kernel by {forward_mismatch:.3g}')
        residuals = {l: float(charge_damped(models[l].arrays, p0[rows[l]], ph_probe)[1])
                     for l in ('AB', 'A', 'B')}
        g_optx = np.asarray(jax.grad(dq_optx)(p0), float)
        g_damped = np.asarray(jax.grad(dq_damped)(p0), float)

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
                row[f'fd_optx_{eps}'] = (float(dq_optx(p0+eps*dj))-base_optx)/eps
                row[f'fd_damped_{eps}'] = (float(dq_damped(p0+eps*dj))-base_damped)/eps
            row['optx_adjoint_minus_damped_adjoint'] = row['analytic_optx']-row['analytic_damped']
            row['damped_adjoint_minus_damped_fd'] = row['analytic_damped']-row['fd_damped_0.001']
            probes.append(row)
            print(name, row['index'], row['native']+'->'+row['mutant'],
                  'optx_adj %.7f' % row['analytic_optx'],
                  'damped_adj %.7f' % row['analytic_damped'],
                  'damped_fd %.7f' % row['fd_damped_0.001'], flush=True)

        record = dict(system=name, ph_probe=ph_probe, base_dq_optx=base_optx,
                      base_dq_damped=base_damped, base_gap=base_optx-base_damped,
                      forward_mismatch_vs_production=forward_mismatch,
                      damped_residuals=residuals, steps=args.steps, probes=probes,
                      wall_seconds=time.monotonic()-started)
        atomic_json(args.output/f'{name}.json', record)
        print(json.dumps({k: record[k] for k in
                          ('system', 'ph_probe', 'base_dq_optx', 'base_dq_damped', 'base_gap',
                           'forward_mismatch_vs_production', 'damped_residuals')}, indent=1),
              flush=True)
        records.append(record)
    atomic_json(args.output/'summary.json', dict(records=records))


if __name__ == '__main__':
    main()
