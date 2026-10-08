"""How close is each channel to the burial clip boundary?

Follow-up to the tier 3 gradient check (``gradient_sanity.py``). On 1fcc the analytic
adjoint and a one-sided finite difference disagree by a bounded, step-size independent
~2.5e-3 kcal/mol; 1brs and 1frt agree to ~1e-4. ``burial = clip((mass-nmin)/(nmax-nmin),
0, 1)`` has zero analytic gradient where it is saturated, while a +eps step that pushes
mass across the boundary picks up the unclipped slope -- an eps-independent gap of
exactly that shape.

The count of saturated channels does NOT discriminate (1frt is the most saturated and the
most accurate). The margin does: only a channel sitting within a perturbation's reach of
the boundary can be crossed. This reports the margin distribution, overall and restricted
to the design positions whose perturbations produced the gap.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gradient_sanity import SYSTEMS, DESIGN_AA, build_state, interface_distance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cif-dir', type=Path, required=True)
    parser.add_argument('--systems', nargs='+', default=sorted(SYSTEMS))
    parser.add_argument('--cutoff', type=float, default=5.0)
    parser.add_argument('--max-positions', type=int, default=48)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    jax.config.update('jax_enable_x64', True)
    from jaxpropka.parameters import ModelConfig, ALPHABET
    from jaxpropka.topology import load_topology
    from pkabench.prep import read_cif

    config = ModelConfig()
    records = []
    for name in args.systems:
        spec = SYSTEMS[name]
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

        record = dict(system=name, nmin=config.nmin, nmax=config.nmax, states={})
        for label, chains in (('AB', spec['a']+spec['b']), ('A', spec['a']), ('B', spec['b'])):
            _, cache, model, keys, _ = build_state(atoms, chains, mask_by_key, config)
            p = np.eye(20, dtype=np.float64)[np.asarray(cache.native_index)]
            terms = model.local_terms()(p)
            burial = np.asarray(terms.burial)
            gm = np.asarray(cache.group_mask)
            # burial saturates, so it cannot be inverted for mass. Recompute the raw
            # quantity the clip acts on exactly as _local_terms does.
            pn = p[cache.env_neighbors]*cache.env_mask[:, :, None]
            if cache.identity_columns is not None:
                pn = np.take_along_axis(pn, cache.identity_columns[cache.env_neighbors], axis=-1)
            mass = cache.bb_mass + np.einsum('nkga,nka->ng', cache.mass, pn)
            # How far below nmin a saturated channel sits: 0 means it is ON the kink,
            # where a +eps step crosses and the analytic gradient does not follow.
            margin_low = np.where(burial <= 0, config.nmin-mass, np.inf)
            # The SECOND clip: pair_burial gates the dielectric that scales every
            # Coulomb coupling. pair_mass = mass_i + mass_j against 2*nmin.
            nb = np.asarray(cache.neighbors)
            pair_mass = mass[:, None, :, None]+mass[nb][:, :, None, :]
            pm = np.asarray(cache.pair_mask)
            pair_margin = np.where(pm, np.abs(pair_mass-2*config.nmin), np.inf)
            index = {k: i for i, k in enumerate(keys_full)}
            rows = np.array([index[k] for k in keys])
            is_design = np.zeros(len(keys), bool)
            is_design[np.isin(rows, positions)] = True
            out = {}
            for scope, sel in (('all', gm), ('design_positions', gm & is_design[:, None])):
                m = margin_low[sel & np.isfinite(margin_low)]
                out[scope] = dict(
                    channels=int(sel.sum()), saturated_low=int(m.size),
                    within_1=int((m <= 1).sum()), within_5=int((m <= 5).sum()),
                    within_20=int((m <= 20).sum()), within_50=int((m <= 50).sum()),
                    min_margin=(float(m.min()) if m.size else None),
                    at_upper_clip=int(((burial >= 1) & sel).sum()))
            pmv = pair_margin[np.isfinite(pair_margin)]
            out['pair_clip'] = dict(
                entries=int(pm.sum()),
                below_boundary=int((pm & (pair_mass <= 2*config.nmin)).sum()),
                within_1=int((pmv <= 1).sum()), within_5=int((pmv <= 5).sum()),
                within_20=int((pmv <= 20).sum()),
                min_margin=(float(pmv.min()) if pmv.size else None))
            record['states'][label] = out
        records.append(record)
        print(json.dumps(record, indent=1), flush=True)
    atomic_json(args.output/'clip_margin.json', dict(records=records))


if __name__ == '__main__':
    main()
