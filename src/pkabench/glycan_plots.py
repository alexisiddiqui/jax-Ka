"""Glycan sensitivity and retention figures with explicit support counts."""
import csv
import json
import sys
from pathlib import Path
from .runtime import require_compute, atomic_json


def render(out):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    out = Path(out); dest = out/'plots'; dest.mkdir(exist_ok=True)
    with (out/'site_changes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k in ('distance_A', 'sasa_A2', 'exposed_fraction', 'ab_pka_change', 'delta_pka_change'):
            r[k] = float(r[k])
        for k in ('all_glycans_removed', 'native_eligible', 'interface', 'existing_mask_retained'):
            r[k] = r[k] == 'True'
    rows = [r for r in rows if r['native_eligible']]
    allrows = [r for r in rows if r['all_glycans_removed']]
    individual = [r for r in rows if r['variant'].startswith('tree')]
    report = json.loads((out/'report.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    metrics = [('ab_pka_change', 'Absolute change in bound-state pKa'),
               ('delta_pka_change', 'Absolute change in binding ΔpKa')]
    rng = np.random.default_rng(20261004)
    bootstrap_rows = []

    def quantiles(data, metric):
        values = np.abs([r[metric] for r in data])
        ids = sorted({r['complex_id'] for r in data})
        if not len(values):
            return None
        groups = [np.abs([r[metric] for r in data if r['complex_id'] == cid]) for cid in ids]
        ci = None
        if len(ids) >= 5:
            samples = [np.quantile(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), .95) for _ in range(400)]
            ci = np.quantile(samples, [.025, .975]).tolist()
        return {'n': len(values), 'complexes': len(ids), 'median': float(np.median(values)),
                'p95': float(np.quantile(values, .95)), 'p95_bootstrap_95ci': ci,
                'fraction_exact_zero': float(np.mean(values == 0)), 'max': float(values.max())}

    def save(fig, name, note):
        fig.text(.5, .012, note, ha='center', va='bottom', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .93))
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(dest/f'{name}.{ext}', dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (metric, label) in zip(axes, metrics):
        x = np.array([r['distance_A'] for r in allrows]); y = np.abs([r[metric] for r in allrows])
        ax.scatter(x, y, s=8, alpha=.2, color='#287caa', rasterized=True)
        centers = []; med = []; p95 = []; low = []; high = []
        for left in range(0, 60, 5):
            stats = quantiles([r for r in allrows if left <= r['distance_A'] < left+5], metric)
            if stats is None:
                continue
            bootstrap_rows.append({'plot': 'distance', 'metric': metric, 'bin_start': left, **stats})
            centers.append(left+2.5); med.append(stats['median']); p95.append(stats['p95'])
            ci = stats['p95_bootstrap_95ci']; low.append(ci[0] if ci else np.nan); high.append(ci[1] if ci else np.nan)
        ax.plot(centers, med, color='#e1812c', label='Median, 5 Å bins')
        ax.plot(centers, p95, color='#b91c3b', label='95th percentile')
        ax.fill_between(centers, low, high, color='#b91c3b', alpha=.13, label='Complex-bootstrap 95% CI')
        ax.axhline(.1, color='black', ls=':', lw=1)
        for radius in (15, 20, 25):
            ax.axvline(radius, color='grey', alpha=.35, ls='--', lw=.8)
        ax.set(xlabel='Distance to nearest removed glycan atom (Å)', ylabel=label,
               yscale='symlog', ylim=(0, max(.2, float(max(y, default=0))*1.2)), xlim=(0, 60))
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_title(f'{len(allrows):,} unique sites; {sum(x>60)} beyond 60 Å')
        ax.legend(fontsize=8, loc='upper right')
    method = 'PROPKA + CCD sugar typing' if report.get('typing_mode') == 'ccd' else 'Native PROPKA typing'
    fig.suptitle(f'Removing all resolved glycans: {method}', fontsize=17)
    save(fig, 'error_vs_distance', f"{report['completed_complexes']} complexes • Fixed protein coordinates • Native PROPKA cutoffs • Exact zeros shown at zero; linear below 10⁻⁵")

    # Single-tree deletions isolate exposure; other glycans remain in context.
    for xfield, xlabel, filename in [('exposed_fraction', 'Resolved glycan exposed fraction (bound / isolated SASA)', 'error_vs_exposure'),
                                      ('sasa_A2', 'Resolved glycan SASA in complex (Å²)', 'error_vs_sasa')]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for j, radius in enumerate((15, 20, 25)):
            selected = [r for r in individual if r['distance_A'] >= radius]
            for i, (metric, label) in enumerate(metrics):
                ax = axes[i, j]
                x = np.array([r[xfield] for r in selected]); y = np.abs([r[metric] for r in selected])
                ax.scatter(x, y, c=[r['exposed_fraction'] for r in selected], cmap='viridis', norm=Normalize(0, 1), s=9, alpha=.25, rasterized=True)
                bins = [0, .3, .7, 1.001] if xfield == 'exposed_fraction' else np.linspace(0, max(1., max(x, default=1.)), 5)
                for lo, hi in zip(bins[:-1], bins[1:]):
                    data = [r for r in selected if lo <= r[xfield] < hi]
                    stats = quantiles(data, metric)
                    if stats is None:
                        continue
                    bootstrap_rows.append({'plot': xfield, 'metric': metric, 'radius_A': radius, 'bin_start': float(lo), 'bin_end': float(hi), **stats})
                    center = (lo+hi)/2; ci = stats['p95_bootstrap_95ci']
                    ax.plot(center, stats['p95'], 'r_', ms=22)
                    if ci:
                        ax.vlines(center, ci[0], ci[1], color='red', alpha=.6)
                    ax.text(center, .98, f"{stats['complexes']}c", transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
                ax.axhline(.1, color='black', ls=':', lw=1)
                ax.set_yscale('symlog', linthresh=1e-5)
                ax.set(xlabel=xlabel, ylabel=label, ylim=(0, max(.2, max(y, default=0)*1.2)),
                       title=f'≥{radius} Å: {len(selected):,} site–tree observations')
                if xfield == 'exposed_fraction':
                    ax.set_xlim(-.02, 1.02)
        fig.suptitle(f'Single-glycan removal: {method}', fontsize=17)
        save(fig, filename, 'Red: pooled p95 and complex-bootstrap 95% CI (≥5 complexes); c = complex count • Repeated sites are correlated • Exact zeros shown')

    with (out/'retention.csv').open() as stream:
        retention = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, field, title in zip(axes, ('sites', 'interface_sites', 'pairs_with_interface_sites'),
                                ('All titratable sites', 'Interface titratable sites', 'Pairs with ≥1 interface site')):
        for mask, name, color in [('glycan_only', 'Glycan mask only', '#277da8'),
                                  ('combined_existing_gaps', '+ existing missing-region masks', '#cf6b21')]:
            rr = [r for r in retention if r['mask'] == mask]
            ax.plot([int(r['radius_A']) for r in rr], [int(r[field]) for r in rr], lw=2, label=name, color=color)
            for r in rr:
                if int(r['radius_A']) in (15, 20, 25):
                    ax.plot(int(r['radius_A']), int(r[field]), 'o', color=color)
        for radius in (15, 20, 25):
            ax.axvline(radius, color='grey', ls=':', lw=.8)
        ax.set(xlabel='Exclusion radius from all removed glycan atoms (Å)', ylabel='Retained count', title=title, ylim=(0, None))
        ax.legend(fontsize=8)
    fig.suptitle('Retained supervision after stripping all resolved glycans', fontsize=17)
    save(fig, 'retained_sites_vs_radius', f"{report['completed_complexes']} pilot complexes • Unique sites, matched PROPKA coverage • Structural eligibility only; production split unchanged")
    atomic_json(dest/'bootstrap_summary.json', {'replicates': 400, 'resampling_unit': 'complex', 'seed': 20261004,
                'minimum_complexes_for_ci': 5, 'statistics': bootstrap_rows,
                'limits': 'Descriptive pooled site p95; complexes are resampled intact. Sequence-related complexes may remain correlated. Small support and convenience sampling limit generalization.'})
    atomic_json(dest/'manifest.json', {'plots': ['error_vs_distance', 'error_vs_sasa', 'error_vs_exposure', 'retained_sites_vs_radius'],
                                     'formats': ['png', 'pdf', 'svg'], 'limits': report['limits']})
    print(json.dumps({'plots': str(dest), 'complexes': report['completed_complexes'], 'sites': len(allrows)}), flush=True)


if __name__ == '__main__':
    render(sys.argv[1])
