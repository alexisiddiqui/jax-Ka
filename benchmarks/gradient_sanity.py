"""Tier 3 gradient sanity: does d(linkage)/dP point where the discrete model agrees?

No experimental data and no training. On a set-2b complex the soft sequence P[N,20]
enters AB, A and B matched (the same rows are sliced into each free state), so the
fixed deprotonated charges cancel and

    dQ(pH)  = Q_AB(pH) - Q_A(pH) - Q_B(pH)          protons taken up on binding
    ddG     = 1.364 * integral_{pH_lo}^{pH_hi} dQ dpH

is the pH-dependent part of the binding free energy. Wyman linkage for association is
d(ln Ka)/d(ln[H+]) = dQ, so d(dG_bind)/dpH = +1.364 dQ and

    ddG = dG_bind(pH_hi) - dG_bind(pH_lo),

in kcal/mol: POSITIVE ddG means binding is tighter at the LOW pH. This matches
pkabench.linkage, whose delta_g(pH) = 1.364 * integral_7^pH dQ is dG_bind(pH) relative
to pH 7. FcRn/Fc (tight at pH 5.5, released at 7.4) must therefore come out positive,
and does. That scalar is the design target; its gradient is taken at the native sequence.

The first-order score of substituting identity a at position i is the directional
derivative along the simplex edge native -> a,

    s[i,a] = dL/dP[i,a] - dL/dP[i,native_i],

which is exactly the linear prediction of L(mutant) - L(native). Each top candidate is
then discretised (hard one-hot P) and rescored, and the rank agreement between s and the
exact rescore is the actual test.

Gradients use the optimistix implicit adjoint (optx_solver), not reverse mode through
the 1024-step damped loop: unrolling 1024 steps across 73 pH points stores several GB of
activations. The damped production solver is used for an independent rescore, so the
discretised ranking never depends on the solver the gradient came from.

Caveat, stated once: mutant side chains use ONE frozen CCD conformation in the
backbone-local frame. No repacking, no rotamer search. The prediction is for that
placement, not a relaxed mutant; for buried positions this is a real limitation.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# Design alphabet offered at interface positions, on top of the native identity.
# Titratable (DEHKRY) and non-titratable (ANQS) controls: the sanity claim is that
# the titratable ones rank top, which is only meaningful if the others are offered.
DESIGN_AA = "ADEHKNQRSY"
TITRATABLE = set("DEHKRY")


def category(native, mutant):
    """How a substitution changes the titratable content at a position."""
    before, after = native in TITRATABLE, mutant in TITRATABLE
    return ('swap' if before and after else 'remove' if before
            else 'introduce' if after else 'neutral')
# Grid-aligned endpoints of the linkage window (PH = linspace(-2, 16, 73), step 0.25).
PH_LO, PH_HI = 5.5, 7.5

SYSTEMS = {
    '1brs': dict(a=['A'], b=['D'], note='barnase (A) / barstar (D)'),
    '1fcc': dict(a=['A'], b=['C'], note='IgG Fc (A) / protein G C2 (C)'),
    '1frt': dict(a=['A', 'B'], b=['C'], note='rat FcRn heavy+b2m (A,B) / Fc (C)'),
}


def interface_distance(topology, group_a):
    """Per-residue minimum heavy-atom distance to the other partner, and the side mask."""
    chains = np.asarray([topology.chain_ids[c] for c in topology.chain_index])
    side_a = np.isin(chains, group_a)
    xyz = topology.atoms.coord.astype(np.float64)
    atom_residue = np.repeat(np.arange(topology.n_residues), np.diff(topology.starts))
    atom_a = side_a[atom_residue]
    best = np.full(topology.n_residues, np.inf, np.float64)
    for mine, theirs in ((atom_a, ~atom_a), (~atom_a, atom_a)):
        other = xyz[theirs]
        for start in range(0, int(mine.sum()), 2048):
            block = np.flatnonzero(mine)[start:start+2048]
            d = np.sqrt(((xyz[block][:, None, :]-other[None])**2).sum(-1)).min(1)
            np.minimum.at(best, atom_residue[block], d)
    return best, side_a


def build_state(topology_atoms, chains, identity_mask_by_key, config, gap_policy='cap'):
    """Topology + identity-restricted cache + model for one state (AB, A or B)."""
    from jaxpropka import TitrationModel
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache
    topology = load_topology(topology_atoms, gap_policy=gap_policy, freeze_disulfides=True,
                             chains=chains, ignore_nonprotein=True)
    keys = [(k.chain, k.number, k.insertion) for k in topology.keys]
    mask = np.stack([identity_mask_by_key[k] for k in keys])
    # Deposited entries (not curated campaign structures) carry truncated side
    # chains; rebuild them from the CCD template and report how many.
    candidates = build_candidates(topology, missing_sidechain='template')
    cache = build_cache(topology, candidates, identities=mask, dtype=np.float64)
    model = TitrationModel(cache, config=config, backend='packed')
    rebuilt = getattr(candidates, 'metadata', {}).get('rebuilt_native', [])
    return topology, cache, model, keys, rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cif-dir', type=Path, required=True)
    parser.add_argument('--systems', nargs='+', default=sorted(SYSTEMS))
    parser.add_argument('--cutoff', type=float, default=5.0, help='interface heavy-atom cutoff (A)')
    parser.add_argument('--max-positions', type=int, default=48,
                        help='cap on design positions, nearest-contact first')
    parser.add_argument('--top', type=int, default=20, help='candidates rescored per system')
    parser.add_argument('--controls', type=int, default=10,
                        help='mid/low-ranked candidates also rescored, for a spread')
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--fd-top', type=int, default=5,
                        help='top-scoring directions also finite-difference checked')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', True)   # exact small deltas and a usable FD check
    import jax.numpy as jnp
    from jaxpropka.parameters import ModelConfig, ALPHABET
    from jaxpropka.optx_solver import SolverConfig, active_channels, optx_curve_kernel
    from jaxpropka.model import packed_curve_kernel
    from pkabench.prep import read_cif
    from pkabench.schema import PH

    config = ModelConfig(steps=args.steps)
    scfg = SolverConfig()
    ph = np.asarray(PH, dtype=float)
    window = (ph >= PH_LO) & (ph <= PH_HI)
    # trapezoid weights on the window, times 1.364 kcal/mol per pH unit of ln10 RT
    inside = ph[window]
    if inside.size < 2:
        raise ValueError('linkage window must contain at least two grid points')
    quad = np.zeros_like(inside)
    quad[:-1] += np.diff(inside)/2
    quad[1:] += np.diff(inside)/2
    w = np.zeros_like(ph); w[window] = 1.364*quad
    weights = jnp.asarray(w)

    summary = []
    for name in args.systems:
        spec = SYSTEMS[name]
        started = time.monotonic()
        record = dict(system=name, note=spec['note'], chains_a=spec['a'], chains_b=spec['b'],
                      ph_window=[PH_LO, PH_HI], steps=args.steps, cutoff=args.cutoff)
        atoms = read_cif(args.cif_dir/f'{name}.cif')
        from jaxpropka.topology import load_topology
        full = load_topology(atoms, gap_policy='cap', freeze_disulfides=True,
                             chains=spec['a']+spec['b'], ignore_nonprotein=True)
        distance, side_a = interface_distance(full, spec['a'])
        frozen_native = np.asarray(full.native_index)
        # Design only where the identity is not clamped (disulfides) and the native
        # residue carries a side chain the CCD template can replace.
        free = ~np.asarray(full.disulfide)
        eligible = np.flatnonzero((distance <= args.cutoff) & free
                                  & ~np.isin(frozen_native, [ALPHABET.index(a) for a in 'GP']))
        # Closest contacts first, then back to index order for a stable report.
        positions = np.sort(eligible[np.argsort(distance[eligible])][:args.max_positions])
        design_cols = np.array([ALPHABET.index(a) for a in DESIGN_AA])

        n = full.n_residues
        mask = np.zeros((n, 20), bool)
        mask[np.arange(n), frozen_native] = True
        mask[np.ix_(positions, design_cols)] = True
        keys_full = [(k.chain, k.number, k.insertion) for k in full.keys]
        mask_by_key = dict(zip(keys_full, mask))

        states = {}
        for label, chains in (('AB', spec['a']+spec['b']), ('A', spec['a']), ('B', spec['b'])):
            states[label] = build_state(atoms, chains, mask_by_key, config)
        keys_ab = states['AB'][3]
        if keys_ab != keys_full:
            raise ValueError(f'{name}: AB state does not reproduce the reference residue order')
        record['rebuilt_native'] = {label: len(states[label][4]) for label in states}
        index = {k: i for i, k in enumerate(keys_ab)}
        rows = {label: np.array([index[k] for k in states[label][3]]) for label in ('A', 'B')}
        rows['AB'] = np.arange(len(keys_ab))
        if not np.array_equal(np.sort(np.concatenate([rows['A'], rows['B']])),
                              np.arange(len(keys_ab))):
            raise ValueError(f'{name}: A and B do not partition AB')
        record.update(n_residues=int(n), n_design_positions=int(positions.size),
                      design_alphabet=DESIGN_AA,
                      design_positions=[dict(index=int(i), chain=keys_ab[i][0],
                                             resnum=int(keys_ab[i][1]), icode=keys_ab[i][2],
                                             native=ALPHABET[frozen_native[i]],
                                             interface_distance=float(distance[i]),
                                             side=('A' if side_a[i] else 'B')) for i in positions])

        native_p = np.eye(20, dtype=np.float64)[frozen_native]
        pieces = {}
        for label in ('AB', 'A', 'B'):
            _, cache, model = states[label][:3]
            sub = mask[rows[label]]
            active = jnp.asarray(active_channels(cache, sub.astype(float)))
            pieces[label] = (model, active, jnp.asarray(rows[label]))

        def total_charge(label, p, solver):
            model, active, sel = pieces[label]
            q = p[sel]
            if solver == 'optx':
                out, extra = optx_curve_kernel(model.arrays, q, jnp.asarray(ph, q.dtype),
                                               active, config=config, solver_config=scfg)
                return out.total_charge, out.residual, extra['active_set_leak']
            out = packed_curve_kernel(model.arrays, q, jnp.asarray(ph, q.dtype), model._edges,
                                      config=config)
            return out.total_charge, out.residual, jnp.zeros(())

        def per_state_residuals(p, solver):
            return {label: np.asarray(total_charge(label, p, solver)[1], float).tolist()
                    for label in ('AB', 'A', 'B')}

        def objective(p, solver):
            parts = {label: total_charge(label, p, solver) for label in ('AB', 'A', 'B')}
            dq = parts['AB'][0] - parts['A'][0] - parts['B'][0]
            diagnostics = dict(
                max_residual=jnp.max(jnp.stack([parts[l][1].max() for l in parts])),
                leak=jnp.max(jnp.stack([parts[l][2] for l in parts])))
            return jnp.sum(weights.astype(dq.dtype)*dq), dq, diagnostics

        p0 = jnp.asarray(native_p)
        value_and_grad = jax.value_and_grad(lambda p: objective(p, 'optx')[0])
        base_optx, grad = jax.block_until_ready(value_and_grad(p0))
        base_damped, dq_damped, diag_damped = jax.block_until_ready(objective(p0, 'damped'))
        _, dq_optx, diag_optx = jax.block_until_ready(objective(p0, 'optx'))
        grad = np.asarray(grad, dtype=float)
        record.update(base_ddG_optx=float(base_optx), base_ddG_damped=float(base_damped),
                      base_solver_gap=float(abs(base_optx-base_damped)),
                      base_delta_q_damped=np.asarray(dq_damped, float).tolist(),
                      base_max_residual_optx=float(diag_optx['max_residual']),
                      base_max_residual_damped=float(diag_damped['max_residual']),
                      base_active_set_leak=float(diag_optx['leak']),
                      # Per-pH, per-state: locates a pH where the continuation sweep
                      # failed, which an aggregate max only tells you happened.
                      base_residual_by_ph=dict(ph=ph.tolist(),
                                               optx=per_state_residuals(p0, 'optx'),
                                               damped=per_state_residuals(p0, 'damped')),
                      residual_tolerance=float(config.residual_tolerance),
                      gradient_seconds=time.monotonic()-started)

        scores = []
        for i in positions:
            native_a = frozen_native[i]
            for col in design_cols:
                if col == native_a:
                    continue
                scores.append(dict(index=int(i), chain=keys_ab[i][0], resnum=int(keys_ab[i][1]),
                                   icode=keys_ab[i][2], native=ALPHABET[native_a],
                                   mutant=ALPHABET[col], identity=int(col),
                                   side=('A' if side_a[i] else 'B'),
                                   interface_distance=float(distance[i]),
                                   score=float(grad[i, col]-grad[i, native_a]),
                                   category=category(ALPHABET[native_a], ALPHABET[col])))
        order = np.argsort([-s['score'] for s in scores])
        for rank, j in enumerate(order):
            scores[j]['rank'] = int(rank)
        ranked = [scores[j] for j in order]
        record['n_candidates'] = len(ranked)

        # The implicit adjoint is only as good as the root it is taken at, and the
        # model's clipped burial terms are only piecewise smooth. Check directional
        # derivatives against a forward difference at two step sizes, over BOTH the
        # near-zero directions (first positions, alanine) and the top-scoring ones:
        # an error that is constant in absolute terms is additive and harmless to the
        # ranking, one that tracks the derivative is multiplicative and is not.
        probes = []
        for i in positions[:3]:
            col = int(design_cols[np.argmax(design_cols != frozen_native[i])])
            if col != frozen_native[i]:
                probes.append((int(i), col, 'closest_contact_alanine'))
        for c in ranked[:args.fd_top]:
            probes.append((c['index'], c['identity'], f"rank_{c['rank']}"))
        checks = []
        for i, col, label in probes:
            direction = np.zeros((n, 20), np.float64)
            direction[i, col] = 1.; direction[i, frozen_native[i]] = -1.
            # One-sided along the edge: the opposite direction leaves the simplex.
            forward = {}
            for eps in (1e-2, 1e-3):
                shifted = float(objective(p0+eps*jnp.asarray(direction), 'optx')[0])
                forward[eps] = (shifted-float(base_optx))/eps
            analytic = float(grad[i, col]-grad[i, frozen_native[i]])
            checks.append(dict(index=int(i), probe=label, mutant=ALPHABET[col],
                               native=ALPHABET[frozen_native[i]], analytic=analytic,
                               forward_difference={str(k): v for k, v in forward.items()},
                               absolute_gap=analytic-forward[1e-3],
                               relative_gap=(analytic-forward[1e-3])/max(abs(analytic), 1e-12)))
        record['gradient_finite_difference'] = checks
        gaps = np.array([c['absolute_gap'] for c in checks])
        mags = np.array([abs(c['analytic']) for c in checks])
        record['gradient_check'] = dict(
            max_absolute_gap=float(np.max(np.abs(gaps))),
            max_relative_gap=float(np.max(np.abs(gaps)/np.maximum(mags, 1e-12))),
            # Near 0 => the gap does not grow with the derivative (additive, benign).
            gap_vs_magnitude_correlation=(float(np.corrcoef(mags, np.abs(gaps))[0, 1])
                                          if len(checks) > 2 else None))

        # Does the piecewise-smooth part of the model bite here? Count channels sitting
        # at the burial clip, where the analytic derivative takes the saturated branch.
        saturation = {}
        for label in ('AB', 'A', 'B'):
            model = pieces[label][0]
            terms = model.local_terms()(p0[pieces[label][2]])
            burial = np.asarray(terms.burial); gm = np.asarray(model.cache.group_mask)
            saturation[label] = dict(
                channels=int(gm.sum()),
                at_lower_clip=int(((burial <= 0) & gm).sum()),
                at_upper_clip=int(((burial >= 1) & gm).sum()))
        record['burial_clip_saturation'] = saturation
        # Rescore the extremes of the gradient ranking plus a spread of controls; a
        # correlation measured only on the top is a restricted-range correlation.
        picks = list(range(min(args.top, len(ranked))))
        picks += list(range(max(0, len(ranked)-args.top), len(ranked)))
        if args.controls and len(ranked) > 2*args.top:
            picks += np.linspace(args.top, len(ranked)-args.top-1, args.controls).astype(int).tolist()
        picks = sorted(set(picks))

        rescored = []
        for j in picks:
            candidate = dict(ranked[j])
            p = np.array(native_p)
            p[candidate['index']] = 0.
            p[candidate['index'], candidate['identity']] = 1.
            pj = jnp.asarray(p)
            exact_optx, _, d_optx = jax.block_until_ready(objective(pj, 'optx'))
            exact_damped, _, d_damped = jax.block_until_ready(objective(pj, 'damped'))
            candidate.update(exact_delta_optx=float(exact_optx)-float(base_optx),
                             exact_delta_damped=float(exact_damped)-float(base_damped),
                             max_residual_optx=float(d_optx['max_residual']),
                             max_residual_damped=float(d_damped['max_residual']),
                             active_set_leak=float(d_optx['leak']))
            rescored.append(candidate)
            print(name, candidate['rank'], f"{candidate['chain']}{candidate['resnum']}",
                  f"{candidate['native']}->{candidate['mutant']}",
                  f"score={candidate['score']:+.4f}",
                  f"exact={candidate['exact_delta_damped']:+.4f}", flush=True)

        from scipy.stats import spearmanr, pearsonr
        s = np.array([c['score'] for c in rescored])
        e_damped = np.array([c['exact_delta_damped'] for c in rescored])
        e_optx = np.array([c['exact_delta_optx'] for c in rescored])
        top = [c for c in rescored if c['rank'] < args.top]
        top_s = np.array([c['score'] for c in top]); top_e = np.array([c['exact_delta_damped'] for c in top])
        # A substitution can only move the linkage by changing which groups titrate,
        # so a top candidate that touches no titratable group is the failure signal.
        n_top_titratable = sum(c['category'] != 'neutral' for c in top)
        record['agreement'] = dict(
            n_rescored=len(rescored),
            spearman_all=float(spearmanr(s, e_damped).statistic),
            pearson_all=float(pearsonr(s, e_damped).statistic),
            spearman_within_top=float(spearmanr(top_s, top_e).statistic) if len(top) > 2 else None,
            sign_agreement=float(np.mean(np.sign(s) == np.sign(e_damped))),
            spearman_optx_vs_damped=float(spearmanr(e_optx, e_damped).statistic),
            max_optx_damped_gap=float(np.max(np.abs(e_optx-e_damped))),
            top_touches_titratable_fraction=n_top_titratable/max(len(top), 1),
            top_categories={k: sum(c['category'] == k for c in top)
                            for k in ('introduce', 'remove', 'swap', 'neutral')},
            top_mean_interface_distance=float(np.mean([c['interface_distance'] for c in top])),
            all_mean_interface_distance=float(np.mean([c['interface_distance'] for c in ranked])),
            max_relative_error=float(np.max(np.abs(s-e_damped)/np.maximum(np.abs(e_damped), 1e-9))),
            best_exact_in_top=bool(np.argmax(e_damped) < len(top)))
        record['ranked'] = ranked
        record['rescored'] = rescored
        record['wall_seconds'] = time.monotonic()-started
        atomic_json(args.output/f'{name}.json', record)
        summary.append({k: record[k] for k in
                        ('system', 'n_residues', 'n_design_positions', 'n_candidates',
                         'base_ddG_optx', 'base_ddG_damped', 'base_solver_gap',
                         'base_max_residual_optx', 'base_max_residual_damped',
                         'agreement', 'wall_seconds')})
        print(json.dumps(summary[-1], indent=1), flush=True)

    atomic_json(args.output/'summary.json',
                dict(job=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename,
                     design_alphabet=DESIGN_AA, ph_window=[PH_LO, PH_HI],
                     steps=args.steps, cutoff=args.cutoff, records=summary))


if __name__ == '__main__':
    main()
