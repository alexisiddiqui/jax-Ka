"""Mask-radius figures: label error against distance, one panel per removed component or per tested gap length.

Usage: python -m pkabench.cutoff_plots OUT [components|gaps]
"""
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime import atomic_json, digest, require_compute

# Panel, source table under the runtime root, row filter, (training, evaluation) radii in Angstrom.
PANELS = [
    ('Ligands', 'audits/propka-components-v4-ligands/out/site_changes.csv', lambda r: r['kind'] == 'ligand', ((15, 'train'), (25, 'eval'))),
    ('Buffers', 'audits/buffer-propka-v2/site_changes.csv', lambda r: r['component_index'] == 'all', ((15, 'train'), (20, 'eval'))),
    ('Glycans', 'audits/glycan-propka-ccd-v1/site_changes.csv',
     lambda r: r['all_glycans_removed'] == 'True' and r['native_eligible'] == 'True', ((20, 'train'), (25, 'eval'))),
    ('Metals', 'audits/propka-components-v1/site_changes.csv', lambda r: r['kind'] == 'metal', ((25, 'exposed ions'), (30, 'buried metals'))),
]
# Gap panels: kind, deleted length, teacher, radius applied at that panel's distance (anchor_tiers for PypKa, long_gap_policy for pKAI).
GAP_PANELS = [('terminal', n, 'PypKa', r) for n, r in ((1, 10), (3, 15), (5, 15), (10, 20))]
GAP_PANELS += [('terminal', n, 'pKAI', r) for n, r in ((10, 30), (20, 35), (30, 40), (50, 45))]
GAP_PANELS += [('internal', n, 'PypKa', r) for n, r in ((1, 15), (3, 15))] + [('internal', n, 'pKAI', r) for n, r in ((5, 20), (10, 20), (20, 25))]
GAP_SHORT = {'terminal': 'audits/radial-error-v2-nearcomplete/anchor-analysis-v2/observations.csv',
             'internal': 'audits/radial-error-v2-nearcomplete/internal-anchor-analysis-v1/observations.csv'}
GAP_LONG = 'audits/pinder-longgap-v1/results'
FLOOR = 1e-5  # values below the floor are drawn at it; exact zeros get their own row below it
XMAX = 35
GAP_FLOOR = 1e-3  # the gap criterion is 0.1 and PypKa noise sits near 0.01, so lower decades carry nothing
GAP_XMAX = 60
THRESHOLD = .1  # gap criterion: bootstrap upper 95% bound on the band 95th percentile
# Two hues, lighter median and darker 95th percentile; validated for CVD separation and contrast.
BLUE = ('#4A90C8', '#1B5A96')
ORANGE = ('#C9741F', '#963B10')


def launch(out, which='components'):
    require_compute(allow_comp1400=True)
    runtime = Path(os.environ['PKABENCH_RUNTIME'])
    subprocess.run([str(runtime/'envs/radial-plots/bin/python'), '-m', 'pkabench.cutoff_plots', str(out), which], check=True)


def pinned(y, floor=FLOOR):
    import numpy as np
    y = np.abs(y); return np.where(y == 0, .6*floor, np.maximum(y, floor))


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 190})
    return plt


def ybins(floor=FLOOR):
    """Zero row, then eight log bins per decade up to 10."""
    import numpy as np
    decades = round(np.log10(10/floor))
    return np.concatenate([[.45*floor, .8*floor], np.geomspace(floor, 10, 8*decades+1)])


def zero_axis(ax, floor=FLOOR):
    """Exact zeros sit in their own row below the separator."""
    import numpy as np
    ax.axhline(.85*floor, color='0.6', lw=.6)
    ax.set_yscale('log'); ax.set_ylim(.45*floor, 10)
    powers = range(round(np.log10(floor)), 1)
    ax.set_yticks([.6*floor] + [10.**k for k in powers], ['0'] + [f'10{str(k).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))}' if k else '1' for k in powers])
    ax.minorticks_off()


def render(out):
    require_compute(allow_comp1400=True)
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    plt = setup()
    runtime = Path(os.environ['PKABENCH_RUNTIME']); dest = Path(out)/'plots'; dest.mkdir(parents=True, exist_ok=True)
    lines = {'delta_pka_change': ('Binding ΔpKa', *BLUE), 'ab_pka_change': ('Bound-state pKa', *ORANGE)}
    xbins = np.arange(0, XMAX+1, 1); pbins = np.arange(0, XMAX+5, 5)
    data = []
    for title, source, keep, radii in PANELS:
        with (runtime/source).open() as f: rows = [r for r in csv.DictReader(f) if keep(r)]
        data.append((title, source, radii, rows))
    # One colour scale across panels, set by non-zero bins so the zero row saturates instead of washing out the rest.
    counts = [np.histogram2d([float(r['distance_A']) for r in rows], pinned([float(r['delta_pka_change']) for r in rows]),
                             bins=[xbins, ybins()])[0] for *_, rows in data]
    norm = LogNorm(1, max(c[:, 1:].max() for c in counts))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, sharey=True, layout='constrained')
    summaries = []; provenance = {}
    for ax, (title, source, radii, rows) in zip(axes.flat, data):
        x = np.array([float(r['distance_A']) for r in rows])
        mesh = ax.hist2d(x, pinned([float(r['delta_pka_change']) for r in rows]),
                         bins=[xbins, ybins()], norm=norm, cmap='Greys', rasterized=True)
        for metric, (name, light, dark) in lines.items():
            y = np.abs([float(r[metric]) for r in rows]); centres = []; med = []; p95 = []
            for lo, hi in zip(pbins[:-1], pbins[1:]):
                v = y[(x >= lo) & (x < hi)]
                if not len(v): continue
                centres.append((lo+hi)/2); med.append(np.median(v)); p95.append(np.quantile(v, .95))
                summaries.append({'panel': title, 'metric': metric, 'lo_A': int(lo), 'hi_A': int(hi), 'n': int(len(v)),
                                  'median': float(med[-1]), 'p95': float(p95[-1]), 'max': float(v.max())})
            ax.plot(centres, pinned(med), color=light, lw=1.6)
            ax.plot(centres, pinned(p95), color=dark, lw=2.2)
        for target in (.01, .001):
            ax.axhline(target, color='0.35', ls=':', lw=1)
        for radius, label in radii:
            ax.axvline(radius, color='0.2', ls='--', lw=.9)
            ax.text(radius+.4, 6, f'{radius} Å {label}', rotation=90, va='top', ha='left', fontsize=8, color='0.2')
        ax.set_title(f'{title}\n{len({r["complex_id"] for r in rows})} complexes, {len(rows):,} site observations', fontsize=10)
        zero_axis(ax); ax.set_xlim(0, XMAX)
        provenance[title] = {'source': source, 'source_sha256': digest(runtime/source), 'observations': len(rows)}
    for ax in axes[:, 0]:
        ax.set_ylabel('Absolute change on removal (pKa units)')
    for ax in axes[1]:
        ax.set_xlabel('Distance to nearest removed heavy atom (Å)')
    fig.colorbar(mesh[3], ax=axes, shrink=.6, label='Binding ΔpKa observations per bin')
    handles = [Line2D([], [], color=light, lw=1.6, label=f'{name} median') for name, light, _ in lines.values()]
    handles += [Line2D([], [], color=dark, lw=2.2, label=f'{name} 95th percentile') for name, _, dark in lines.values()]
    handles += [Line2D([], [], color='0.35', ls=':', lw=1, label='0.01 and 0.001 targets'),
                Line2D([], [], color='0.2', ls='--', lw=.9, label='Mask radius')]
    fig.legend(handles=[handles[i] for i in (0, 2, 1, 3, 4, 5)], loc='outside upper center', ncol=3, frameon=False, fontsize=9)
    fig.text(.5, -.01, 'PROPKA 3.5.1, fixed coordinates • density shows binding ΔpKa • exact zeros in the bottom row, values below 10⁻⁵ at 10⁻⁵ • no change beyond 27 Å • 5 Å percentile bins • repeated sites correlated',
             ha='center', va='top', fontsize=8.5)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(dest/f'component_cutoffs.{suffix}', bbox_inches='tight')
    plt.close(fig)
    atomic_json(dest/'component_cutoffs.json', {'summaries': summaries, 'provenance': provenance, 'script_sha256': digest(Path(__file__)),
                'distance': 'Nearest removed component heavy atom to the titratable site, Angstrom.',
                'limits': 'PROPKA sensitivity with finite cutoffs, not physical label error. Buffers and glycans: all removed at once; ligands and metals: one component instance removed at a time, others retained.'})
    print(json.dumps(provenance), flush=True)


def gap_observations(runtime):
    """{(kind, length, teacher): (group ids, anchor distance, |delta error|, max |bound|/|free| error)}."""
    import numpy as np
    obs = {key[:3]: ([], [], [], []) for key in GAP_PANELS}
    for kind, source in GAP_SHORT.items():
        with (runtime/source).open() as f:
            for r in csv.DictReader(f):
                key = (kind, int(r['deleted_residues']), 'PypKa')
                if r['status'] != 'ok' or (r['kind'] == 'terminal') != (kind == 'terminal') or key not in obs: continue
                for column, value in zip(obs[key], (r['complex_id'], float(r['anchor_distance_A']), abs(float(r['delta_pka_error'])),
                                                    max(abs(float(r['ab_pka_error'])), abs(float(r['free_pka_error']))))):
                    column.append(value)
    # Site records from pinder-longgap-v1/calib.py; filters and pKAI columns as in its analyze.py.
    for path in sorted(glob.glob(str(runtime/GAP_LONG/'ref_*.jsonl'))):
        for line in open(path):
            o = json.loads(line)
            if o['status'] != 'accepted': continue
            for v in o['variants']:
                key = (v.get('kind'), v.get('length'), 'pKAI')
                if v['status'] != 'ok' or key not in obs: continue
                for s in v['sites']:
                    b, fr, d = s[8:11]
                    if not s[3] or s[4] or s[5] or b is None or fr is None: continue
                    for column, value in zip(obs[key], (o['id'], s[6], abs(d), max(abs(b), abs(fr)))):
                        column.append(value)
    return {key: tuple(np.array(c) for c in cols) for key, cols in obs.items()}


def band_bounds(groups, x, y, rng):
    """Per 5 A band: (lo, n, groups, p95, bootstrap upper 95% bound or None when < 10 groups or < 20 observations)."""
    import numpy as np
    out = []
    for lo in range(0, GAP_XMAX, 5):
        m = (x >= lo) & (x < lo+5)
        if not m.sum(): continue
        u, inv = np.unique(groups[m], return_inverse=True); p95 = float(np.quantile(y[m], .95))
        if len(u) < 10 or m.sum() < 20: out.append((lo, int(m.sum()), len(u), p95, None)); continue
        parts = np.split(y[m][np.argsort(inv, kind='stable')], np.cumsum(np.bincount(inv))[:-1])
        boots = [np.quantile(np.concatenate([parts[i] for i in rng.integers(0, len(u), len(u))]), .95) for _ in range(1000)]
        out.append((lo, int(m.sum()), len(u), p95, float(np.quantile(boots, .95))))
    return out


def band_radius(bands):
    """Upper edge of the last supported failing band, as in pinder-longgap-v1/analyze.py."""
    supported = [b for b in bands if b[4] is not None]
    failing = [b for b in supported if b[4] >= THRESHOLD]
    return float(failing[-1][0]+5) if failing else (0. if supported and supported[0][0] == 0 else None)


def render_gaps(out):
    require_compute(allow_comp1400=True)
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    plt = setup()
    runtime = Path(os.environ['PKABENCH_RUNTIME']); dest = Path(out)/'plots'; dest.mkdir(parents=True, exist_ok=True)
    lines = {'delta': ('Binding ΔpKa', *BLUE), 'absolute': ('Absolute pKa, worse of bound and free', *ORANGE)}
    obs = gap_observations(runtime); rng = np.random.default_rng(20261008)
    xbins = np.arange(0, GAP_XMAX+2, 2)
    # One colour scale across panels, set by non-zero bins (see render).
    counts = [np.histogram2d(obs[key[:3]][1], pinned(obs[key[:3]][2], GAP_FLOOR), bins=[xbins, ybins(GAP_FLOOR)])[0] for key in GAP_PANELS]
    norm = LogNorm(1, max(c[:, 1:].max() for c in counts))
    fig, axes = plt.subplots(3, 5, figsize=(14, 8.2), sharex=True, sharey=True, layout='constrained')
    slots = [axes[0, i] for i in range(4)] + [axes[1, i] for i in range(4)] + list(axes[2])
    panels = []
    for ax, (kind, length, teacher, radius) in zip(slots, GAP_PANELS):
        groups, x, delta, absolute = obs[kind, length, teacher]
        mesh = ax.hist2d(x, pinned(delta, GAP_FLOOR), bins=[xbins, ybins(GAP_FLOOR)], norm=norm, cmap='Greys', rasterized=True)
        summary = {'kind': kind, 'length': length, 'teacher': teacher, 'radius_A': radius,
                   'complexes': len(set(groups.tolist())), 'observations': len(x)}
        for metric, y in (('delta', delta), ('absolute', absolute)):
            name, light, dark = lines[metric]; bands = band_bounds(groups, x, y, rng)
            centres = np.array([b[0]+2.5 for b in bands])
            med = [np.median(y[(x >= b[0]) & (x < b[0]+5)]) for b in bands]
            ax.plot(centres, pinned(med, GAP_FLOOR), color=light, lw=1.4)
            ax.plot(centres, pinned([b[3] for b in bands], GAP_FLOOR), color=dark, lw=2)
            supported = [b for b in bands if b[4] is not None]
            ax.plot([b[0]+2.5 for b in supported], pinned([b[4] for b in supported], GAP_FLOOR), ls='none', marker='_', ms=7, mew=1.6, color=dark)
            summary[metric] = {'bands': [dict(lo_A=b[0], n=b[1], complexes=b[2], median=float(m), p95=b[3], upper95=b[4]) for b, m in zip(bands, med)],
                               'band_radius_A': band_radius(bands)}
        ax.axhline(THRESHOLD, color='0.35', ls=':', lw=1)
        ax.axvline(radius, color='0.2', ls='--', lw=.9)
        ax.text(radius+.7, 6, f'{radius} Å', rotation=90, va='top', ha='left', fontsize=8, color='0.2')
        residues = 'residue' if length == 1 else 'residues'
        ax.set_title(f'{kind.capitalize()}, {length} {residues} ({teacher})\n{summary["complexes"]} complexes, {len(x):,} obs.', fontsize=9.5)
        zero_axis(ax, GAP_FLOOR); ax.set_xlim(0, GAP_XMAX); ax.set_xticks(range(0, GAP_XMAX+1, 20))
        panels.append(summary)
    fig.supylabel('Absolute error from deletion (pKa units)', fontsize=10)
    fig.supxlabel('Distance to gap anchor (Å)', fontsize=10)
    for ax in (axes[0, 4], axes[1, 4]):
        ax.axis('off')
    handles = [Line2D([], [], color=light, lw=1.4, label=f'{name}, median') for name, light, _ in lines.values()]
    handles += [Line2D([], [], color=dark, lw=2, label=f'{name}, 95th percentile') for name, _, dark in lines.values()]
    handles += [Line2D([], [], color='0.25', ls='none', marker='_', ms=7, mew=1.6, label='Upper 95% bootstrap bound\non the 95th percentile'),
                Line2D([], [], color='0.35', ls=':', lw=1, label='0.1 target'),
                Line2D([], [], color='0.2', ls='--', lw=.9, label='Mask radius')]
    axes[0, 4].legend(handles=[handles[i] for i in (0, 2, 1, 3, 4, 5, 6)], loc='center', frameon=False, fontsize=8.5)
    cax = axes[1, 4].inset_axes([.1, .2, .09, .75])
    fig.colorbar(mesh[3], cax=cax, label='Binding ΔpKa observations per bin')
    fig.text(.5, -.01, 'Rows: terminal gaps (PypKa); terminal gaps (pKAI); internal gaps (PypKa, then pKAI) • distance to the observed flank: backbone N/CA/C/O for PypKa, Cα for pKAI\n'
             'density shows binding ΔpKa • exact zeros in the bottom row, values below 10⁻³ at 10⁻³ • 5 Å bands; bounds from 1,000 whole-complex resamples where ≥ 10 complexes and ≥ 20 observations • pKAI reports pKa to 0.01 • repeated sites correlated',
             ha='center', va='top', fontsize=8.5)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(dest/f'gap_cutoffs.{suffix}', bbox_inches='tight')
    plt.close(fig)
    long = hashlib.sha256(''.join(digest(p) for p in sorted(glob.glob(str(runtime/GAP_LONG/'ref_*.jsonl')))).encode()).hexdigest()
    provenance = {**{source: digest(runtime/source) for source in GAP_SHORT.values()}, f'{GAP_LONG}/ref_*.jsonl (sha256 of file digests)': long}
    atomic_json(dest/'gap_cutoffs.json', {'panels': panels, 'provenance': provenance, 'script_sha256': digest(Path(__file__)), 'seed': 20261008,
                'distance': 'Target functional-group atoms to the observed flank: backbone N/CA/C/O (PypKa anchor studies) or CA (pKAI long-gap calibration), Angstrom.',
                'criterion': 'A band passes if the bootstrap upper 95% bound on its 95th-percentile error is below 0.1 pKa; band_radius_A is the upper edge of the last supported failing band.',
                'limits': 'Ordered segments deleted from fully modelled references; conservativeness for disordered tails untested. PypKa panels: radial-error-v2-nearcomplete (23 complexes); pKAI panels: pinder-longgap-v1.'})
    print(json.dumps([{k: p[k] for k in ('kind', 'length', 'teacher', 'radius_A', 'observations')} | {m: p[m]['band_radius_A'] for m in lines} for p in panels]), flush=True)


if __name__ == '__main__':
    {'components': render, 'gaps': render_gaps}[sys.argv[2] if len(sys.argv) > 2 else 'components'](sys.argv[1])
