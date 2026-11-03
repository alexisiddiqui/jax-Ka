"""Uncapped geodesic scale audit on immutable, already repaired MaSIF surfaces."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.augmentation_store import AugmentationStore
from augmentation_masif_report import view, quantile, weights, write_csv
from augmentation_surface_audit import fit, rms

METRICS = ('global_ca_rmsd_A', 'patch_ca_global_rmsd_A', 'local_ca_rmsd_A',
           'local_heavy_rmsd_A', 'surface_global_distance_A', 'surface_local_distance_A', 'surface_area_relative_change',
           'residue_exposure_TV', 'shape_index_rmse', 'hbond_rmse', 'electrostatics_rmse', 'hydropathy_rmse')


def graph(vertices, faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return csr_matrix((np.r_[lengths, lengths],
                       (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(vertices), len(vertices)))


def supports(v, radii):
    """One bounded shortest-path solve per anchor; nested supports with no cap."""
    g = graph(v['vertices'], v['faces'])
    out = {}
    for i in np.flatnonzero(v['anchor_valid']):
        distances = dijkstra(g, directed=False, indices=int(v['anchor_vertex_ids'][i]), limit=max(radii))
        out[int(i)] = {r: (np.flatnonzero(distances <= r), distances[distances <= r]) for r in radii}
    return out


def surface_metrics(ref, var, rid, vid, rotation, translation, n_residues):
    rp = ref['vertices'][rid]; vp = var['vertices'][vid] @ rotation + translation
    rd, ri = cKDTree(vp).query(rp); vd, vi = cKDTree(rp).query(vp)
    rw, vw = ref['vertex_area'][rid], var['vertex_area'][vid]
    rf, vf = ref['raw_features'][rid], var['raw_features'][vid]
    rmse = np.sqrt(.5 * (np.average((rf-vf[ri])**2, axis=0, weights=rw) +
                        np.average((vf-rf[vi])**2, axis=0, weights=vw)))
    ra = np.bincount(ref['owner'][rid], weights=rw, minlength=n_residues)
    va = np.bincount(var['owner'][vid], weights=vw, minlength=n_residues)
    return dict(distance=float(.5*(np.average(rd, weights=rw)+np.average(vd, weights=vw))),
                rmse=rmse, area_relative_change=float(vw.sum()/rw.sum()-1),
                exposure_tv=float(.5*np.abs(ra/ra.sum()-va/va.sum()).sum()))


def process(task):
    path, identifier, radii, control = task
    with AugmentationStore(path) as store:
        record = store.raw(identifier)
    m, a = record['metadata'], record['arrays']
    if m['status'] != 'complete':
        raise ValueError('Incomplete input parent: '+m['parent'])
    comparisons = [('repair_only', 'native', 'zero'), ('historical_zero_drift', 'zero', 'historical_zero')]
    comparisons += [('selected_vs_zero', 'zero', r) for r in m['roles'] if r.startswith('selected_')]
    comparisons += [('selected_vs_native', 'native', r) for r in m['roles'] if r.startswith('selected_')]
    comparisons += [(r+'_vs_zero', 'historical_zero', r) for r in m['roles'] if r.startswith('amplitude_')]
    comparisons += [(r+'_control', 'zero', r) for r in ('rigid_zero', 'repeat_zero') if r in m['roles']]
    frames = {r: view(record, r) for r in m['roles']}
    # Role aliases share a saved mesh. Compute its geodesics only once.
    cache = {}; patches = {}
    for role in frames:
        fid = m['roles'][role]
        if fid not in cache: cache[fid] = supports(frames[role], radii)
        patches[role] = cache[fid]
    rows = []; excluded = []; support_rows = []
    bb = a['backbone_atoms']; owner = a['atom_residue']
    for role, byanchor in patches.items():
        for i, byr in byanchor.items():
            for radius, (ids, ds) in byr.items():
                support_rows.append(dict(parent=m['parent'], dataset=m['dataset'], role=role,
                    anchor=i, radius_A=radius, vertices=len(ids), effective_radius_A=float(ds.max()),
                    area_A2=float(frames[role]['vertex_area'][ids].sum()),
                    residue_count=len(np.unique(frames[role]['owner'][ids]))))
    for group, refrole, role in comparisons:
        ref, var = frames[refrole], frames[role]
        ca, moving = ref['coordinates'][bb[:, 1]], var['coordinates'][bb[:, 1]]
        gr, gt = fit(moving, ca)
        for i in range(len(ref['anchor_valid'])):
            # Use a common cohort across radii: all four scales must admit a fit.
            reason = None
            if i not in patches[refrole] or i not in patches[role]: reason = 'unmapped_anchor'
            members_by_radius = {}
            if reason is None:
                for radius in radii:
                    rid = patches[refrole][i][radius][0]
                    members = np.unique(ref['owner'][rid]); members_by_radius[radius] = members
                    if len(members)<4 or np.linalg.matrix_rank(ca[members]-ca[members].mean(0))<2:
                        reason = 'insufficient_noncollinear_reference_residues'; break
            if reason:
                excluded.append(dict(parent=m['parent'], group=group, role=role, anchor=i, reason=reason))
                continue
            for radius in radii:
                rid, ds = patches[refrole][i][radius]; vid, vds = patches[role][i][radius]
                members = members_by_radius[radius]
                lr, lt = fit(moving[members], ca[members])
                sm = surface_metrics(ref, var, rid, vid, lr, lt, len(ca))
                gm = surface_metrics(ref, var, rid, vid, gr, gt, len(ca))
                row = dict(parent=m['parent'], dataset=m['dataset'], sample_kind=m['sample_kind'],
                    source='smoke_control' if control else 'full', group=group, reference=refrole, role=role,
                    anchor=int(a['anchor_residue'][i]), anchor_key=json.dumps(m['residue_keys'][int(a['anchor_residue'][i])]),
                    radius_A=radius, effective_radius_A=float(ds.max()), variant_effective_radius_A=float(vds.max()),
                    patch_vertices=len(rid), variant_patch_vertices=len(vid), patch_area_A2=float(ref['vertex_area'][rid].sum()),
                    patch_residues=len(members), global_ca_rmsd_A=rms(moving@gr+gt, ca),
                    patch_ca_global_rmsd_A=rms(moving[members]@gr+gt, ca[members]),
                    local_ca_rmsd_A=rms(moving[members]@lr+lt, ca[members]),
                    local_heavy_rmsd_A=rms(var['coordinates'][np.isin(owner,members)]@lr+lt, ref['coordinates'][np.isin(owner,members)]),
                    surface_global_distance_A=gm['distance'], surface_local_distance_A=sm['distance'],
                    surface_area_relative_change=sm['area_relative_change'], residue_exposure_TV=sm['exposure_tv'])
                row.update({k+'_rmse':float(v) for k,v in zip(('shape_index','hbond','electrostatics','hydropathy'), sm['rmse'])})
                rows.append(row)
    if control:
        rows = [r for r in rows if r['group'] in {'rigid_zero_control', 'repeat_zero_control'}]
        excluded = [r for r in excluded if r['group'] in {'rigid_zero_control', 'repeat_zero_control'}]
    return rows, excluded, support_rows


def run(args):
    require_compute(threads=args.workers)
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []; hashes = {}
    for name, path in [('full', args.input/'full/coordinates'), ('smoke', args.input/'smoke/coordinates')]:
        if name == 'smoke' and not path.exists(): continue
        hashes[name] = digest(path/'manifest.json')
        with AugmentationStore(path) as store:
            tasks += [(str(path), r['id'], args.radii, name == 'smoke') for r in store.rows]
    rows = []; exclusions = []; support_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (rr, ee, ss) in enumerate(pool.map(process, tasks), 1):
            rows += rr; exclusions += ee; support_rows += ss
            print(json.dumps(dict(completed=n, total=len(tasks), comparisons=len(rows))), flush=True)
    write_csv(args.output/'patches.csv', rows); write_csv(args.output/'exclusions.csv', exclusions)
    write_csv(args.output/'supports.csv', support_rows)
    groups = {}
    for group in sorted({r['group'] for r in rows}):
        groups[group] = {}
        for radius in args.radii:
            rr = [r for r in rows if r['group']==group and r['radius_A']==radius]
            groups[group][str(radius)] = dict(parents=len({r['parent'] for r in rr}), comparisons=len(rr),
                metrics={k:quantile(rr, k) for k in METRICS+('patch_vertices','patch_area_A2','effective_radius_A','patch_residues')})
    summary = dict(radii_A=args.radii, vertex_cap=None, parents=len({r['parent'] for r in rows if r['source']=='full'}),
        comparisons=len(rows), excluded_anchor_comparisons=len(exclusions), groups=groups,
        aggregation='equal parent, equal view, equal anchor; matched anchor cohort across radii within each comparison',
        limitations=['30-parent enriched pilot, not a population estimate',
            'Radius is geodesic on the SES mesh, not Euclidean',
            'Exposure TV is area fraction redistributed between original residue identities, not SASA',
            'Nearest-surface chemistry RMSE includes mesh/projection noise',
            'Surface area assigns one third of each incident triangle to its vertex; boundary discretization remains',
            'No radius-altered frozen fingerprints; existing 12 A fingerprint results remain separate',
            'Three-parent smoke controls only; independent MDS seeds leave raw vertex features unchanged'])
    summary['by_dataset'] = {group:{dataset:{str(radius):{k:quantile([r for r in rows if r['group']==group and r['dataset']==dataset and r['radius_A']==radius], k) for k in METRICS} for radius in args.radii} for dataset in sorted({r['dataset'] for r in rows if r['group']==group})} for group in groups}
    atomic_json(args.output/'summary.json', summary)
    atomic_json(args.output/'provenance.json', dict(code_sha256=digest(__file__), dependency_sha256={name:digest(Path(__file__).with_name(name)) for name in ('augmentation_masif_report.py','augmentation_surface_audit.py')}, input_manifests=hashes,
        radii_A=args.radii, patch_csv_sha256=digest(args.output/'patches.csv')))
    plot(rows, args.output, args.radii)
    text = ['Geodesic patch scale audit', '==========================', '',
            f"Completed {summary['parents']} full-pilot parents and {len(rows)} patch comparisons.",
            'No vertex cap. Same mapped, fit-valid anchors across all radii within each comparison.', '']
    for group, scales in groups.items():
        text += [group, '-'*len(group), '']
        for radius, data in scales.items():
            text.append(f"* {radius} A: {data['parents']} parents, {data['comparisons']} comparisons.")
            for key in ('local_ca_rmsd_A','surface_global_distance_A','surface_local_distance_A','electrostatics_rmse','residue_exposure_TV'):
                q = data['metrics'][key]
                text.append(f"  {key}: median {q['median']:.4f}, p95 {q['p95']:.4f}.")
        text.append('')
    text += ['Limitations', '-----------', ''] + ['* '+v for v in summary['limitations']]
    (args.output/'report.rst').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(output=str(args.output), parents=summary['parents'], comparisons=len(rows))), flush=True)


def plot(rows, output, radii):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    groups = ('repair_only','amplitude_0.5_vs_zero','amplitude_1_vs_zero','amplitude_2_vs_zero','selected_vs_zero','rigid_zero_control')
    keys = ('patch_ca_global_rmsd_A','local_ca_rmsd_A','surface_global_distance_A','surface_local_distance_A','electrostatics_rmse','residue_exposure_TV')
    labels = ('Patch CA RMSD, global alignment (Å)','Patch CA RMSD, local alignment (Å)',
              'Surface distance, global alignment (Å)','Surface distance, local alignment (Å)',
              'Normalised electrostatic RMSE','Residue exposed-area fraction TV')
    for ax, key, label in zip(axes.flat, keys, labels):
        for group in groups:
            qq = [quantile([r for r in rows if r['group']==group and r['radius_A']==radius], key) for radius in radii]
            if any(q is None for q in qq): continue
            line, = ax.plot(radii, [q['median'] for q in qq], marker='o', label=group,
                            linestyle='--' if group=='rigid_zero_control' else '-')
            ax.fill_between(radii, [q['p05'] for q in qq], [q['p95'] for q in qq], color=line.get_color(), alpha=.06)
        ax.set(xlabel='Geodesic radius (Å)', ylabel=label, xticks=radii); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Uncapped surface patches: matched anchors across scales; median and 5–95% interval')
    fig.savefig(output/'patch-scale.png', dpi=180); fig.savefig(output/'patch-scale.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for radius in radii:
        rr = [r for r in rows if r['group']=='selected_vs_zero' and r['radius_A']==radius]
        for ax, key in zip(axes, ('local_ca_rmsd_A','surface_local_distance_A')):
            values = np.array([r[key] for r in rr]); w = weights(rr); ix = np.argsort(values)
            ax.plot(values[ix], np.cumsum(w[ix]), label=f'{radius:g} Å')
    axes[0].set(xlabel='Local patch CA RMSD (Å)', ylabel='Fraction, equal parent weight')
    axes[1].set(xlabel='Locally aligned surface distance (Å)', ylabel='Fraction, equal parent weight')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output/'patch-scale-distributions.png', dpi=180); fig.savefig(output/'patch-scale-distributions.svg'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--radii', type=float, nargs='+', default=[6., 9., 12., 15.]); p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if any(not np.isfinite(r) or r<=0 for r in args.radii) or args.radii!=sorted(set(args.radii)):
        p.error('radii must be finite, positive, unique and increasing')
    run(args)
