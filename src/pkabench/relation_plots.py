"""Relational edges against width: 50% GQT validation error against parameter count, per seed and mean ± SD.

Usage: python relation_plots.py export RUNS OUT.csv   (on the training machine; standard library only)
       python -m pkabench.relation_plots render RUNS.csv OUT
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

# Label, run prefix under gqt-production-v2/runs (seed suffix -s17/-s29/-s43), group.
# a/b = relation maps in GQT layers 1 and 2: 6 = -2/-1/+1/+2/same chain/other chain, 4 = ±1/±2/same/other, 1 = one shared map.
ARMS = [
    ('w64', 'control-v2-50', 'width'),
    ('w76', 'control-v2-50-w76', 'width'),
    ('1/1', 'control-v2-50-shared-projection', 'relations'),
    ('4/1', 'control-v2-50-relations-4-1', 'relations'),
    ('6/1', 'control-v2-50-relations-6-1', 'relations'),
    ('4/4', 'control-v2-50-relations-4-4', 'relations'),
    ('6/4', 'control-v2-50-relations-6-4', 'relations'),
    ('6/6', 'control-v2-50-relations', 'relations'),
    ('4/1 full', 'control-v2-50-relations-4-1-full', 'full'),
]
SEEDS = (17, 29, 43)
# Panel title, CSV column. Selection is the sum of the other two.
METRICS = [('Selection error', 'selection'), ('PINDER state MAE', 'state_mae'), ('Interface paired MAE', 'interface_paired_mae')]
COLUMNS = ['arm', 'group', 'run', 'seed', 'parameter_count', 'selected_epoch', 'selection', 'state_mae', 'interface_paired_mae',
           'pkpdb_site_mae', 'benchmark_mae', 'selection_json_sha256']
# Label offsets in points, (dx, dy), where the default right-hand label would collide; keyed by arm, or by (metric, arm) for one panel.
OFFSETS = {'4/1 full': (-7, 0), 'w64': (-7, 0), ('interface_paired_mae', '6/1'): (-7, 0)}


def export(runs, out):
    runs = Path(runs); rows = []
    for arm, prefix, group in ARMS:
        for seed in SEEDS:
            path = runs/f'{prefix}-s{seed}'/'selection.json'
            if not path.exists(): continue
            d = json.loads(path.read_text()); v = d['validation']
            rows.append({'arm': arm, 'group': group, 'run': path.parent.name, 'seed': seed, 'parameter_count': d['parameter_count'],
                         'selected_epoch': d['selected_epoch'], 'selection': v['selection'], 'state_mae': v['pinder']['state_mae'],
                         'interface_paired_mae': v['pinder']['interface_paired_mae'], 'pkpdb_site_mae': v['pkpdb']['site_mae'],
                         'benchmark_mae': v['benchmark']['overall']['mae'],
                         'selection_json_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS); w.writeheader(); w.writerows(rows)
    print(json.dumps({arm: sum(r['arm'] == arm for r in rows) for arm, *_ in ARMS}), flush=True)


def render(source, out):
    from matplotlib.lines import Line2D
    from .cutoff_plots import BLUE, ORANGE, setup
    from .runtime import atomic_json, digest
    plt = setup()
    dest = Path(out)/'plots'; dest.mkdir(parents=True, exist_ok=True)
    with open(source) as f: rows = list(csv.DictReader(f))
    colours = {'width': ('0.62', '0.3'), 'relations': BLUE, 'full': ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, layout='constrained')
    summary = []
    for ax, (title, metric) in zip(axes, METRICS):
        means = {}
        for arm, _, group in ARMS:
            sel = [r for r in rows if r['arm'] == arm]
            x = int(sel[0]['parameter_count'])/1000; y = [float(r[metric]) for r in sel]
            light, dark = colours[group]; mean = statistics.mean(y); means[arm] = (x, mean)
            ax.scatter([x]*len(y), y, s=14, color=light, alpha=.75, lw=0, zorder=2)
            if len(y) > 1:
                ax.errorbar(x, mean, yerr=statistics.stdev(y), fmt='o', ms=6.5, color=dark, mec='white', mew=.8, elinewidth=1.4, capsize=3, zorder=3)
            else:
                ax.plot(x, mean, ls='none', marker='D' if group == 'full' else 'o', ms=6.5, mfc='white', mec=dark, mew=1.6, zorder=3)
            dx, dy = OFFSETS.get((metric, arm), OFFSETS.get(arm, (7, 0)))
            ax.annotate(arm, (x, mean), xytext=(dx, dy), textcoords='offset points', va='center', ha='left' if dx > 0 else 'right', fontsize=8.5, color='0.15')
            summary.append({'metric': metric, 'arm': arm, 'group': group, 'parameter_count': int(sel[0]['parameter_count']), 'n': len(y),
                            'mean': mean, 'sd': statistics.stdev(y) if len(y) > 1 else None, 'seeds': {r['seed']: float(r[metric]) for r in sel}})
        (x0, y0), (x1, y1) = means['w64'], means['w76']
        ax.plot([x0, x1], [y0, y1], color='0.55', lw=1.2, ls='--', zorder=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Parameters (thousands)')
        ax.grid(axis='y', color='0.92', lw=.6); ax.set_axisbelow(True)
    axes[0].set_ylabel('Validation MAE (pKa units)')
    axes[0].set_xlim(130, 205)
    handles = [Line2D([], [], color='0.3', marker='o', ls='--', lw=1.2, ms=6, label='Width only, no explicit sequence edges'),
               Line2D([], [], color=BLUE[1], marker='o', ls='none', ms=6, label='Relation maps, layer 1 / layer 2'),
               Line2D([], [], color=ORANGE[1], marker='D', mfc='white', mew=1.6, ls='none', ms=6, label='4/1, layer 2 attends to all residues'),
               Line2D([], [], color='0.3', marker='o', ls='none', ms=6, label='Mean ± SD over seeds 17, 29, 43'),
               Line2D([], [], color='0.3', marker='o', mfc='white', mew=1.6, ls='none', ms=6, label='Seed 17 only'),
               Line2D([], [], color='0.6', marker='o', ls='none', ms=4, alpha=.75, label='Individual seeds')]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False, fontsize=9)
    full = next(r for r in rows if r['arm'] == '4/1 full'); graph = next(r for r in rows if r['arm'] == '4/1' and r['seed'] == full['seed'])
    fig.text(.5, -.01, '50% training pool • validation split • selection error = PINDER state MAE + interface paired MAE\n'
             'relation maps per layer: 6 = −2, −1, +1, +2, same chain, other chain; 4 = ±1, ±2, same, other; 1 = one shared map • '
             f'4/1 graph attention on seed {full["seed"]}: selection {float(graph["selection"]):.4f}, full attention {float(full["selection"]):.4f}',
             ha='center', va='top', fontsize=8.5)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(dest/f'relation_width.{suffix}', bbox_inches='tight')
    plt.close(fig)
    atomic_json(dest/'relation_width.json', {'summary': summary, 'source': str(source), 'source_sha256': digest(source),
                'script_sha256': digest(Path(__file__)),
                'limits': 'Validation split, not test. Three seeds where shown; 4/4, 6/4 and full attention ran on seed 17 only.'})
    print(json.dumps({m: {s['arm']: round(s['mean'], 4) for s in summary if s['metric'] == m} for _, m in METRICS}), flush=True)


if __name__ == '__main__':
    {'export': export, 'render': render}[sys.argv[1]](*sys.argv[2:])
