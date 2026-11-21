"""Turn dense-annotation pilot receipts into native-only runtime/storage estimates."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);args=p.parse_args()
report=json.loads((args.root/'report.json').read_text());sample=json.loads((args.root/'sample.json').read_text())
if not sample.get('all_molecular_components',False):
    raise ValueError('Whole-unit ETA requires all molecular components; legacy MSMS can silently truncate a structure')
parents=report['parents'];groups=defaultdict(list)
for row in parents:
    if not row['stress']:groups[row['stratum']].append(row)
families={};rng=np.random.default_rng(20261115);draws=2000
bootstrap={family:np.zeros(draws) for family in report['families']}
for family in report['families']:
    rows=[r for r in parents if r['family']==family and not r['stress']]
    successes=[r for r in rows if r['status']=='complete']
    cells=[key for key in groups if key.startswith(family+'|')]
    complete_cells=all(all(r['status']=='complete' for r in groups[key]) for key in cells)
    projected_s=0.;projected_bytes=0.;stage_s=defaultdict(float)
    for row in successes:
        steady=row['annotation_s']-row['timings'].get('model_restore_s',0)+row['source_export_s']+row.get('pack_s',0)
        projected_s+=row['weight']*steady;projected_bytes+=row['weight']*row['compressed_bytes']
        for name,value in row['timings'].items():
            if name not in ('model_restore_s','validation_s'):stage_s[name]+=row['weight']*value
    for key in cells:
        cell_rows=groups[key]
        if not all(r['status']=='complete' for r in cell_rows):continue
        values=np.array([r['annotation_s']-r['timings'].get('model_restore_s',0)+r['source_export_s']+r.get('pack_s',0) for r in cell_rows])
        population=cell_rows[0]['population']
        bootstrap[family]+=rng.choice(values,size=(draws,len(values)),replace=True).mean(axis=1)*population/3600
    workers=report['memory_aware_48cpu_projection']['workers']
    families[family]=dict(population=sample['population'][family],sampled=len(rows),successes=len(successes),
        all_probability_cases_succeeded=complete_cells,projected_worker_hours=projected_s/3600,
        conditional_worker_hours_p05_p95=np.quantile(bootstrap[family],[.05,.95]).tolist(),
        hours_8workers=projected_s/3600/8,hours_48cpu_memory_projection=projected_s/3600/max(workers,1),
        projected_compressed_GB=projected_bytes/1e9,stage_worker_hours={k:v/3600 for k,v in stage_s.items()},
        failures=[dict(id=r['id'],stage=r['stage'],error=r['error']) for r in rows if r['status']!='complete'])
totals={}
for name,selected in [('pkpdb_plus_separated_pinder',('pkpdb_assembly','pinder_partner')),
                      ('pkpdb_plus_bound_and_separated_pinder',('pkpdb_assembly','pinder_partner','pinder_bound'))]:
    totals[name]=dict(surfaces=sum(families[k]['population'] for k in selected),
        projected_worker_hours=sum(families[k]['projected_worker_hours'] for k in selected),
        hours_8workers=sum(families[k]['hours_8workers'] for k in selected),
        hours_48cpu_memory_projection=sum(families[k]['hours_48cpu_memory_projection'] for k in selected),
        projected_compressed_GB=sum(families[k]['projected_compressed_GB'] for k in selected),
        all_probability_cases_succeeded=all(families[k]['all_probability_cases_succeeded'] for k in selected),
        worker_hours_p05_p95=np.quantile(sum((bootstrap[k] for k in selected),start=np.zeros(draws)),[.05,.95]).tolist())
result=dict(families=families,totals=totals,memory_projection=report['memory_aware_48cpu_projection'],
    peak_worker_tree_GiB=report['peak_tree_rss_bytes']/1024**3,
    pilot_wall_minutes=report['wall_s']/60,
    caveats=['Bootstrap conditional on this small sample; stress cases excluded from population averages',
        'If any probability case failed, its cell is incomplete: estimates cover successful annotations, not all requested surfaces',
        'Production reuses one model per persistent worker; model construction/restore is paid once per worker, excluded from per-surface projections',
        '48-CPU projection scales measured single-threaded stages with a memory-aware worker reservation; high-concurrency I/O is unmeasured',
        'Runtime excludes queueing and transfer; no ANM views, minimisation or training',
        'Storage includes dense float32 80D descriptors, surface vertices/faces/normals, chemistry and canonical residue mapping; no per-patch feature tensors',
        'CPU author-container implementation; GPU inference not benchmarked'])
(args.root/'estimate.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
fields=['id','family','split','n','actual_residues','stress','status','vertices','annotation_s','source_export_s','compressed_bytes']
with (args.root/'measurements.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(parents)
lines=['Native dense MaSIF annotation pilot','==================================','',
    'Native pKPDB assemblies and native PINDER bound/separated partners, including validation. '
    'Frozen MaSIF-search all-features checkpoint, 80D straight descriptors, 12 A geodesic cutoff, '
    '200 vertices per patch, one descriptor per mesh vertex. All processing is on CPUs.','',
    f"Sample outcomes: {report['counts']}. Pilot wall time: {report['wall_s']/60:.1f} minutes. "
    f"Peak monitored complete worker process tree: {result['peak_worker_tree_GiB']:.2f} GiB.",'',
    'Projection by family','--------------------','']
for family,a in families.items():
    lines.append(f"* {family}: {a['population']:,} surfaces; {a['successes']}/{a['sampled']} probability cases succeeded; "
        f"{a['projected_worker_hours']:.1f} worker-hours; {a['hours_8workers']/24:.2f} days at 8 workers; "
        f"{a['projected_compressed_GB']:.1f} GB compressed.")
lines+=['','Totals','------','']
for name,a in totals.items():
    lines.append(f"* {name}: {a['surfaces']:,} surfaces; {a['projected_worker_hours']:.1f} worker-hours "
        f"(conditional bootstrap p05--p95 {a['worker_hours_p05_p95'][0]:.1f}--{a['worker_hours_p05_p95'][1]:.1f}); "
        f"{a['hours_8workers']/24:.2f} days at 8 workers; "
        f"{a['hours_48cpu_memory_projection']/24:.2f} days at {report['memory_aware_48cpu_projection']['workers']} "
        f"memory-budgeted workers within 48 CPUs; {a['projected_compressed_GB']:.1f} GB compressed.")
lines+=['','Validation and limits','---------------------','',
    'Original heavy atom positions remain fixed. Reduce assigns hydrogens; PDB2PQR may add missing atoms '
    'for charge assignment, with additions and coordinate-preserving aliases recorded in each receipt. '
    'Dense sparse-distance fingerprints are checked at three random real-mesh vertices against the '
    'existing independently upstream-validated MaSIF adapter. Model and binary hashes are recorded separately.','']
lines+=['* '+c for c in result['caveats']]
(args.root/'report.rst').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(totals=totals,memory_projection=result['memory_projection'],peak_worker_tree_GiB=result['peak_worker_tree_GiB']),indent=2))
