"""Memory, step time and attended edges of the relation-map GQT configurations (relation_plots.ARMS), 50% pool runs.

  python -m pkabench.relation_memory slurm OUT/slurm.csv              (Isambard: recorded peak host memory per training job)
  python -m pkabench.relation_memory sample RUNTIME OUT               (Isambard: training graphs through the training loaders)
  python -m pkabench.relation_memory measure SAMPLE OUT [backprop]    (Coulson comp1400, one GPU: edges.csv, memory.csv)
  python -m pkabench.relation_memory measure SAMPLE OUT coord         (Coulson comp1400, one GPU: coord.csv)
  python -m pkabench.relation_memory render DIR                       (radial-plots: DIR/plots/relation_memory.*, training path)
  python -m pkabench.relation_memory render-coord DIR                 (radial-plots: DIR/plots/coordinate_gradients.*)

Slurm recorded only host memory for the training jobs (MaxRSS of the batch step: loader workers, prefetch and the
Python process). GPU memory was not recorded: gres/gpumem is listed in AccountingStorageTRES but no GPU gather plugin
runs, so the field is empty. GPU memory and step time are measured here instead.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from .relation_plots import ARMS

# Label -> (width, production_variant), as the runs were registered (protocol.json).
CONFIGS = {'w64': (64, 'none'), 'w76': (76, 'none'), '1/1': (64, 'shared-projection'), '4/1': (64, 'relations-4-1'),
           '6/1': (64, 'relations-6-1'), '4/4': (64, 'relations-4-4'), '6/4': (64, 'relations-6-4'), '6/6': (64, 'relations'),
           '4/1 full': (64, 'relations-4-1-full')}
# Training jobs (Isambard), matched by job name or by the final selection line of the job log.
JOBS = {'control-v2-50-s17': 7364983, 'control-v2-50-s29': 7364984, 'control-v2-50-s43': 7364985,
        'control-v2-50-w76-s17': 7375491, 'control-v2-50-w76-s29': 7375492, 'control-v2-50-w76-s43': 7375493,
        'control-v2-50-shared-projection-s17': 7376944, 'control-v2-50-shared-projection-s29': 7376946, 'control-v2-50-shared-projection-s43': 7376947,
        'control-v2-50-relations-4-1-s17': 7498707, 'control-v2-50-relations-4-1-s29': 7499751, 'control-v2-50-relations-4-1-s43': 7499754,
        'control-v2-50-relations-6-1-s17': 7498705, 'control-v2-50-relations-6-1-s29': 7499752, 'control-v2-50-relations-6-1-s43': 7499755,
        'control-v2-50-relations-4-4-s17': 7498706, 'control-v2-50-relations-6-4-s17': 7498704,
        'control-v2-50-relations-s17': 7366919, 'control-v2-50-relations-s29': 7366920, 'control-v2-50-relations-s43': 7366921,
        'control-v2-50-relations-4-1-full-s17': 7500698}
FRACTION = 0.5
BATCH = 16
DROPOUT = 0.1
SEED = 20261102
PER_DATASET = 150  # PINDER complexes (two branches each) and pKPDB structures: a training step's 2:1 graph ratio
REPEATS = 10  # timed calls after one warm-up call; the median is reported


def slurm(out):
    rows = []
    for arm, prefix, _ in ARMS:
        for run, job in JOBS.items():
            if run.rsplit('-s', 1)[0] != prefix: continue
            text = subprocess.run(['sacct', '-j', f'{job}.batch', '--noheader', '-P', '--format=MaxRSS,NodeList,Elapsed'], check=True,
                                  capture_output=True, text=True).stdout.split('\n')[0]
            rss, node, elapsed = text.split('|')
            scale = {'K': 1, 'M': 1024, 'G': 1024**2}[rss[-1]]
            rows.append({'arm': arm, 'run': run, 'seed': int(run.rsplit('-s', 1)[1]), 'job': job, 'max_rss_kib': round(float(rss[:-1])*scale),
                         'node': node, 'elapsed': elapsed})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({r['run']: r['max_rss_kib'] for r in rows}), flush=True)


def sample(root, out):
    """edges/<dataset>-<i>.npz: backbone table, masks and neighbours of PER_DATASET training structures per dataset;
    memory/bucket-<bound>.npz: the largest training PINDER complex of each size bucket, one structure as the loader pads it."""
    import numpy as np
    from pkatrain.production_graphs import output, read
    from pkatrain.production_inputs import BACKBONE_WIDTH, BackboneTable
    from pkatrain.production_loading import MANIFEST, PinderSource, PkpdbSource, select
    from pkatrain.production_train import _with_policy
    root = Path(root); out = Path(out); rng = np.random.default_rng(SEED)
    manifests = {d: read(output(root, d)/MANIFEST) for d in ('pinder', 'pkpdb')}
    sources = {'pinder': PinderSource(_with_policy(manifests['pinder'], BATCH), sidechains=BackboneTable(root, 'pinder')),
               'pkpdb': PkpdbSource(_with_policy(manifests['pkpdb'], BATCH), sidechains=BackboneTable(root, 'pkpdb'))}
    (out/'edges').mkdir(parents=True, exist_ok=True); (out/'memory').mkdir(parents=True, exist_ok=True); index = []
    for dataset, source in sources.items():
        rows = select(manifests[dataset], 'train', FRACTION)
        for i in rng.choice(len(rows), PER_DATASET, replace=False):
            row = rows[i]; g = source.load([row['id']])[0]; g = {k: v[0] for k, v in g.items()}
            np.savez_compressed(out/'edges'/f'{dataset}-{len(index):04d}.npz', table=g['nodes'][..., -BACKBONE_WIDTH:], node_mask=g['node_mask'],
                                neighbors=g['neighbors'], edge_mask=g['edge_mask'])
            index.append({'file': f'edges/{dataset}-{len(index):04d}.npz', 'dataset': dataset, 'id': row['id'], 'n': row['n']})
    rows = select(manifests['pinder'], 'train', FRACTION); policy = sources['pinder'].policy
    for bound in policy.bounds:
        row = max((r for r in rows if int(policy.bucket(r['n'])) == bound), key=lambda r: r['n']); graphs, targets, mask, *_ = sources['pinder'].load([row['id']])
        np.savez_compressed(out/'memory'/f'bucket-{bound:04d}.npz', targets=targets[0], mask=mask[0], **{f'graph/{k}': v[0] for k, v in graphs.items()})
        index.append({'file': f'memory/bucket-{bound:04d}.npz', 'dataset': 'pinder', 'id': row['id'], 'n': row['n'], 'bucket': bound})
    for s in sources.values(): s.close()
    (out/'index.json').write_text(json.dumps({'fraction': FRACTION, 'seed': SEED, 'per_dataset': PER_DATASET, 'files': index}, indent=1))
    print(json.dumps({'edges': sum('bucket' not in r for r in index), 'buckets': [r['bucket'] for r in index if 'bucket' in r]}), flush=True)


def count_edges(sample_dir, out):
    """Residue-layer attended pairs per graph, with the model's own annotate / pair_mask (production_position)."""
    import numpy as np
    import jax.numpy as jnp
    from pkatrain.production_position import annotate, pair_mask
    index = json.loads((Path(sample_dir)/'index.json').read_text()); rows = []
    for item in index['files']:
        if 'bucket' in item: continue
        z = np.load(Path(sample_dir)/item['file'])
        branches = ('AB', 'free') if z['neighbors'].ndim == 3 else ('single',)
        for b, name in enumerate(branches):
            pick = (lambda x: x[b]) if len(branches) == 2 else (lambda x: x)
            nb, em, nm = pick(z['neighbors']), pick(z['edge_mask']).astype(bool), pick(z['node_mask'])
            g = {'nodes': jnp.asarray(pick(z['table'])), 'node_mask': jnp.asarray(nm), 'neighbors': jnp.asarray(nb), 'edge_mask': jnp.asarray(em),
                 'edge': jnp.zeros(nb.shape + (20,)), 'switch': jnp.zeros(nb.shape)}
            a = annotate(g, 'relations')
            graph_edges = int((em & nm.astype(bool)[:, None]).sum()); sequence_edges = int(np.asarray(a['edge_mask'])[:, nb.shape[1]:].sum())
            rows.append({'dataset': item['dataset'], 'id': item['id'], 'branch': name, 'residues': int(nm.sum()), 'graph_edges': graph_edges,
                         'sequence_edges': sequence_edges, 'full_pairs': int(np.asarray(pair_mask(a)).sum())})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def attended(row, arm):
    """Attended residue pairs over both residue layers for one graph under `arm`."""
    e, s, p = (int(row[k]) for k in ('graph_edges', 'sequence_edges', 'full_pairs'))
    variant = CONFIGS[arm][1]
    return 2*e if variant == 'none' else (e+s)+p if variant.endswith('-full') else 2*(e+s)


def _run(lower, args, row):
    """Compile (timed), one call for the peak memory, then REPEATS timed calls; fills `row`. Out of memory is a result."""
    import time
    import numpy as np
    import jax
    try:
        t = time.perf_counter(); compiled = lower().compile(); row['compile_seconds'] = time.perf_counter() - t
        m = compiled.memory_analysis()
        row.update(temp_bytes=int(m.temp_size_in_bytes), argument_bytes=int(m.argument_size_in_bytes), output_bytes=int(m.output_size_in_bytes))
        jax.block_until_ready(compiled(*args))
        row['peak_bytes'] = int((jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use', -1))
        times = []
        for _ in range(REPEATS):
            t = time.perf_counter(); jax.block_until_ready(compiled(*args)); times.append(time.perf_counter() - t)
        row.update(step_seconds=float(np.median(times)), step_seconds_min=float(min(times)), status='ok')
    except Exception as error:
        row['status'] = f'{type(error).__name__}: {str(error)[:200]}'


def one(arm, sample_dir, out):
    """Peak device memory of DropoutEngine's PINDER objective (forward + backward, the PINDER half of a training step) per
    size bucket, BATCH complexes x two branches; one process per configuration, buckets ascending."""
    import numpy as np
    import jax
    from pkatrain.production_train import DropoutEngine, make_model, model_inputs, parameter_count
    width, variant = CONFIGS[arm]
    params, _, _ = make_model(17, width=width, orientation=True, table=True, dihedrals=True, variant=variant)
    predict = model_inputs(1, True, 'all', 0.0, True, True, False, variant)[1]
    engine = DropoutEngine(params, DROPOUT, predict); index = json.loads((Path(sample_dir)/'index.json').read_text()); rows = []
    for item in sorted((i for i in index['files'] if 'bucket' in i), key=lambda i: i['bucket']):
        z = np.load(Path(sample_dir)/item['file'])
        graphs = {k[6:]: np.broadcast_to(z[k], (BATCH,) + z[k].shape).copy() for k in z.files if k.startswith('graph/')}
        targets = np.broadcast_to(z['targets'], (BATCH,) + z['targets'].shape).copy(); mask = np.broadcast_to(z['mask'], (BATCH,) + z['mask'].shape).copy()
        zeros = np.zeros_like(mask, np.float32); valid = np.ones(BATCH, bool)
        arrays = jax.device_put((graphs, targets, mask, zeros, zeros, valid)); key = jax.random.PRNGKey(0)
        row = {'arm': arm, 'variant': variant, 'parameters': parameter_count(params), 'bucket': item['bucket'], 'residues': item['n'], 'id': item['id'],
               'padded_neighbors': int(graphs['neighbors'].shape[-1])}
        _run(lambda: engine.paired_value_grad.lower(params, *arrays, key), (params, *arrays, key), row)
        rows.append(row); print(json.dumps(row), flush=True); del arrays
    with open(out, 'w') as f: json.dump(rows, f)


def coord_one(arm, sample_dir, out):
    """Peak device memory of the coordinate-only gradient per size bucket: parameters fixed, d(sum of squared shifts)/d(backbone
    coordinates) for one complex in both states, through the structural endpoint's feature rebuild (pkanet.differentiable
    StructureFeatures, dense candidate pairs; pkanet.relational RelationalStructureFeatures for the backbone table and the
    free state's chain isolation) and the configuration's production forward. No dropout."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from pkanet.differentiable import StructureFeatures
    from pkanet.graph import SIDECHAIN_ATOMS
    from pkanet.relational import RelationalStructureFeatures
    from pkatrain.production_inputs import BACKBONE_WIDTH
    from pkatrain.production_train import make_model, model_inputs, parameter_count
    width, variant = CONFIGS[arm]
    params, _, _ = make_model(17, width=width, orientation=True, table=True, dihedrals=True, variant=variant)
    predict = model_inputs(1, True, 'all', 0.0, True, True, False, variant)[1]
    index = json.loads((Path(sample_dir)/'index.json').read_text()); rows = []
    for item in sorted((i for i in index['files'] if 'bucket' in i), key=lambda i: i['bucket']):
        z = np.load(Path(sample_dir)/item['file']); g = {k[6:]: z[k][0] for k in z.files if k.startswith('graph/')}  # AB branch
        n = int(g['node_mask'].sum()); table = g['nodes'][:n, -BACKBONE_WIDTH:]
        template = {'nodes': g['nodes'][:n, :24], 'node_mask': g['node_mask'][:n],
                    **{k: g[k] for k in ('query_residue', 'query_group', 'site_residue', 'site_type', 'site_mask', 'query_site')}}
        base = StructureFeatures(template, table[:, 12].astype(np.int32), table[:, 13].astype(np.int32), np.zeros((n, len(SIDECHAIN_ATOMS)), bool))
        states = [RelationalStructureFeatures(base, s) for s in ('bound', 'free')]
        sequence = jnp.asarray(g['nodes'][:n, :20]); backbone = jnp.asarray(table[:, :12].reshape(n, 4, 3)); side = jnp.zeros((n, len(SIDECHAIN_ATOMS), 3))
        loss = lambda bb: sum(jnp.sum(predict(params, f(sequence, bb, side))**2) for f in states)
        row = {'arm': arm, 'variant': variant, 'parameters': parameter_count(params), 'bucket': item['bucket'], 'residues': n, 'id': item['id']}
        step = jax.jit(jax.grad(loss)); _run(lambda: step.lower(backbone), (backbone,), row)
        rows.append(row); print(json.dumps(row), flush=True)
    with open(out, 'w') as f: json.dump(rows, f)


def _per_config(action, sample_dir, out, name, env):
    (out/f'parts-{name}').mkdir(parents=True, exist_ok=True)
    for k, arm in enumerate(CONFIGS):
        subprocess.run([sys.executable, '-m', 'pkabench.relation_memory', action, arm, str(sample_dir), str(out/f'parts-{name}'/f'{k}.json')], check=True, env=env)
    rows = [r for k in range(len(CONFIGS)) for r in json.loads((out/f'parts-{name}'/f'{k}.json').read_text())]
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out/f'{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def measure(sample_dir, out, which='backprop'):
    """which: 'backprop' (edge counts, then full backprop per configuration -> edges.csv, memory.csv) or 'coord' (coord.csv)."""
    from .runtime import require_compute
    require_compute(threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), gpu_benchmark=True, allow_comp1400=True)
    out = Path(out); env = {**os.environ, 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
    if which == 'coord': return _per_config('coord-one', sample_dir, out, 'coord', env)
    # Edge counts on the CPU: a GPU client in this process would hold device memory through the per-configuration runs.
    subprocess.run([sys.executable, '-c', f'from pkabench.relation_memory import count_edges; count_edges({str(sample_dir)!r}, {str(out/"edges.csv")!r})'],
                   check=True, env={**env, 'JAX_PLATFORMS': 'cpu'})
    _per_config('one', sample_dir, out, 'memory', env)


def _style():
    from .cutoff_plots import ORANGE
    # Width-only arms grey, relation maps a blue ramp by parameter count, full attention orange.
    ramp = ['#9CC3E4', '#74A9D6', '#4A90C8', '#2F74B0', '#1F5E99', '#123F6E']
    return {'w64': '0.7', 'w76': '0.35', **dict(zip(['1/1', '4/1', '6/1', '4/4', '6/4', '6/6'], ramp)), '4/1 full': ORANGE[1]}


def _lines(ax, rows, xkey, value, colour):
    """One line per configuration; x marks for sizes that ran out of memory. Returns {arm: {x: y}}."""
    out = {}
    for arm in CONFIGS:
        pts = sorted((int(r[xkey]), value(r)) for r in rows if r['arm'] == arm and r['status'] == 'ok')
        if not pts: continue
        ax.plot(*zip(*pts), color=colour[arm], lw=2.2 if arm == '4/1 full' else 1.6, marker='o', ms=3.5, ls='--' if arm == 'w76' else '-')
        out[arm] = dict(pts)
    failed = [(r['arm'], int(r[xkey])) for r in rows if r['status'] != 'ok']
    for arm, at in failed: ax.plot(at, ax.get_ylim()[1], ls='none', marker='x', color=colour[arm], clip_on=False)
    if failed: out['failed'] = failed
    ax.grid(color='0.92', lw=.6); ax.set_axisbelow(True)
    return out


# Compiled footprint (temporaries + arguments + outputs) of each size's executable. peak_bytes_in_use is a running
# maximum over the process, so at small sizes it reports start-up overhead; from 640 residues the two agree within 2%.
FOOTPRINT = lambda r: sum(int(r[k]) for k in ('temp_bytes', 'argument_bytes', 'output_bytes'))/1024**3
STEP_MS = lambda r: 1000*float(r['step_seconds'])


def _legend(ax, colour):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colour[a], lw=2.2 if a == '4/1 full' else 1.6, ls='--' if a == 'w76' else '-', marker='o', ms=3.5, label=a) for a in CONFIGS]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=8.5, ncol=2, title='Configuration', title_fontsize=8.5)


def render(directory):
    """Training path: attended edges per training graph, full-backprop GPU memory and step time against size."""
    import numpy as np
    from .cutoff_plots import setup
    from .runtime import atomic_json, digest
    plt = setup(); d = Path(directory); dest = d/'plots'; dest.mkdir(parents=True, exist_ok=True)
    read = lambda name: list(csv.DictReader(open(d/name)))
    edges, memory = read('edges.csv'), read('memory.csv'); arms = list(CONFIGS); colour = _style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), layout='constrained')
    summary = {'attended_pairs': {}}
    ax = axes[0]; x = np.arange(len(arms))
    data = [[attended(r, arm) for r in edges] for arm in arms]
    parts = ax.boxplot(data, positions=x, widths=.6, patch_artist=True, showfliers=True, medianprops={'color': 'white', 'lw': 1.4},
                       flierprops={'marker': '.', 'ms': 3, 'mec': 'none', 'mfc': '0.55'}, whiskerprops={'color': '0.4'}, capprops={'color': '0.4'})
    for box, arm in zip(parts['boxes'], arms): box.set(facecolor=colour[arm], edgecolor=colour[arm])
    for arm, v in zip(arms, data):
        q = np.quantile(v, (.25, .5, .75)); summary['attended_pairs'][arm] = {'q25': float(q[0]), 'median': float(q[1]), 'q75': float(q[2]), 'max': int(max(v))}
    ax.set_yscale('log'); ax.set_ylabel('Attended residue pairs per graph, both residue layers')
    ax.set_title(f'Edges in training graphs ({len(edges)} graphs)', fontsize=10)
    ax.set_xticks(x, arms, rotation=45, ha='right'); ax.grid(axis='y', color='0.92', lw=.6); ax.set_axisbelow(True)
    summary['backprop_gib'] = _lines(axes[1], memory, 'bucket', FOOTPRINT, colour)
    axes[1].set_ylabel('GPU memory of the step (GiB)'); axes[1].set_title('Memory', fontsize=10)
    summary['backprop_ms'] = _lines(axes[2], memory, 'bucket', STEP_MS, colour)
    axes[2].set_ylabel('Step time (ms, median)'); axes[2].set_title('Time', fontsize=10); axes[2].set_yscale('log')
    for ax in axes[1:]: ax.set_xlabel('Padded residues per graph (size bucket)')
    _legend(axes[1], colour)
    fig.text(.5, -.005, f'Left: {sum(r["dataset"] == "pinder" for r in edges)} PINDER graphs (AB and free) and {sum(r["dataset"] == "pkpdb" for r in edges)} pKPDB graphs from the 50% training pool; '
             'graph configurations attend the 20 Å residue graph (the ±1, ±2 sequence edges of the relation maps are already in it); full attention replaces layer 2 with every allowed pair\n'
             f'Middle, right: full backprop (parameter gradients) of the DropoutEngine PINDER objective (dropout {DROPOUT}), the PINDER half of a training step: the largest training PINDER complex of '
             f'each size bucket, padded as in training, {BATCH} copies × 2 states\n'
             f'One A40 (46 GB), one process per configuration, fresh initialisations • memory: compiled footprint (temporaries + arguments + outputs), within 2% of the measured peak from 640 residues • '
             f'time: median of {REPEATS} calls after a warm-up call, log scale • burial auxiliary head and the pKPDB half of the step omitted\n'
             'w76 is slow at every size: 4 heads of 19 channels (not a multiple of 8); its Isambard training jobs also took 35 min against 21 min for w64',
             ha='center', va='top', fontsize=8.3)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(dest/f'relation_memory.{suffix}', bbox_inches='tight')
    plt.close(fig)
    top = max(int(r['bucket']) for r in memory)
    summary['training_peak_gib'] = {r['arm']: int(r['peak_bytes'])/1024**3 for r in memory if int(r['bucket']) == top and r['status'] == 'ok'}
    atomic_json(dest/'relation_memory.json', {'summary': summary, 'sources': {n: digest(d/n) for n in ('edges.csv', 'memory.csv')},
                'script_sha256': digest(Path(__file__)), 'training_peak': f'measured peak_bytes_in_use at the largest bucket ({top} residues)',
                'limits': 'Fresh initialisations, not the trained parameters; the burial auxiliary head and the pKPDB half of the training step are not '
                          'included. GPU memory was not recorded during the Isambard training runs (slurm.csv holds the recorded host MaxRSS, loader-dominated).'})
    print(json.dumps(summary), flush=True)


def render_coord(directory):
    """Coordinate-only gradient through the structural endpoint: GPU memory and time against complex size."""
    from .cutoff_plots import setup
    from .runtime import atomic_json, digest
    plt = setup(); d = Path(directory); dest = d/'plots'; dest.mkdir(parents=True, exist_ok=True)
    coord = list(csv.DictReader(open(d/'coord.csv'))); colour = _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    summary = {'coordinate_gib': _lines(axes[0], coord, 'residues', FOOTPRINT, colour), 'coordinate_ms': _lines(axes[1], coord, 'residues', STEP_MS, colour)}
    axes[0].set_ylabel('GPU memory of the step (GiB)'); axes[0].set_title('Memory', fontsize=10)
    axes[1].set_ylabel('Gradient time (ms, median)'); axes[1].set_title('Time', fontsize=10); axes[1].set_yscale('log')
    for ax in axes: ax.set_xlabel('Residues in the complex')
    _legend(axes[0], colour)
    fig.text(.5, -.01, 'Parameters fixed; gradient of the summed squared shifts with respect to backbone coordinates, one complex in the bound and free states, no dropout\n'
             'Features rebuilt from coordinates by the structural endpoint (pkanet.differentiable StructureFeatures: dense N × N candidate pairs, so every configuration is quadratic)\n'
             f'The largest training PINDER complex of each size bucket, unpadded • one A40, one process per configuration, fresh initialisations • memory: compiled footprint • '
             f'time: median of {REPEATS} calls after a warm-up call, log scale\n'
             'w76 is slow at every size: 4 heads of 19 channels (not a multiple of 8); its Isambard training jobs also took 35 min against 21 min for w64',
             ha='center', va='top', fontsize=8.3)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(dest/f'coordinate_gradients.{suffix}', bbox_inches='tight')
    plt.close(fig)
    atomic_json(dest/'coordinate_gradients.json', {'summary': summary, 'sources': {'coord.csv': digest(d/'coord.csv')}, 'script_sha256': digest(Path(__file__)),
                'limits': 'Current dense-pair feature rebuild, not an optimised online implementation; fresh initialisations.'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    action, *args = sys.argv[1:]
    {'slurm': slurm, 'sample': sample, 'measure': measure, 'one': one, 'coord-one': coord_one, 'render': render, 'render-coord': render_coord}[action](*args)
