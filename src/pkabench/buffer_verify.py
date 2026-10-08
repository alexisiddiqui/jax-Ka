"""Verify completed buffer pilot and report conservative descriptive summaries."""
import csv
import json
from pathlib import Path
from .runtime import require_compute, atomic_json, digest

def verify(out):
    require_compute()
    import numpy as np
    out=Path(out); manifest=json.loads((out/'manifest.json').read_text()); report=json.loads((out/'report.json').read_text())
    with (out/'site_changes.csv').open() as f: rows=list(csv.DictReader(f))
    successful=set(report['successful_complexes']); assert len(rows)==report['observations']
    keys=[tuple(r[k] for k in ('complex_id','component_index','chain','resnum','icode','group')) for r in rows]; assert len(keys)==len(set(keys))
    assert all(r['complex_id'] in successful for r in rows)
    components=[]; checks=0; lost=0
    for task in manifest['tasks']:
        root=out/task['complex_id']; result=json.loads((root/'result.json').read_text())
        if result['status']!='complete': continue
        lost+=task['lost_training_pair']
        assert digest(Path(task['protein_root'])/'original-resolved.cif')==result['source_sha256']
        for state in ('AB','A','B'):
            reference=[l for l in (root/f'{state}.pdb').read_text().splitlines() if l.startswith('ATOM')]
            for path in root.glob(f'{state}-*.pdb'):
                assert [l for l in path.read_text().splitlines() if l.startswith('ATOM')]==reference
                checks+=1
        components.extend(c for c in result['components'] if c['component_index']!='all')
    all_rows=[r for r in rows if r['component_index']=='all']
    assert len(all_rows)==report['retention'][0]['sites']
    assert all(report['retention'][i]['sites']>=report['retention'][i+1]['sites'] for i in range(len(report['retention'])-1))
    selected=[s for s in report['summaries'] if s['exposure']=='all' and s['radius_A'] in (0,10,15,20,25)]
    # Resample whole complexes, retaining all repeated sites within a complex.
    rng=np.random.default_rng(20261004); ci=[]
    for variant in ('individual','all'):
        for radius in (10,15,20,25):
            rr=[r for r in rows if (r['component_index']=='all')==(variant=='all') and float(r['distance_A'])>=radius]
            groups={cid:np.abs([float(r['ab_pka_change']) for r in rr if r['complex_id']==cid]) for cid in sorted({r['complex_id'] for r in rr})}
            values=list(groups.values()); boot=[]
            if len(values)>=5:
                for _ in range(400): boot.append(float(np.quantile(np.concatenate([values[i] for i in rng.integers(0,len(values),len(values))]),.95)))
            ci.append({'variant':variant,'radius_A':radius,'complexes':len(values),'p95_bootstrap_95CI':np.quantile(boot,[.025,.975]).tolist() if boot else None})
    exposure=[]
    for lo,hi in ((0,.3),(.3,.7),(.7,1.01)):
        rr=[r for r in rows if r['component_index']!='all' and lo<=float(r['exposed_fraction'])<hi]
        exposure.append({'exposure_interval':[lo,hi],'complexes':len({r['complex_id'] for r in rr}),'components':len({(r['complex_id'],r['component_index']) for r in rr})})
    all_by_site={tuple(r[k] for k in ('complex_id','chain','resnum','icode','group')):r for r in all_rows}
    outliers=[]
    for r in rows:
        if r['component_index']=='all' or float(r['distance_A'])<15 or abs(float(r['ab_pka_change']))<=.1: continue
        a=all_by_site[tuple(r[k] for k in ('complex_id','chain','resnum','icode','group'))]
        outliers.append(dict(r,nearest_any_buffer_A=float(a['distance_A']),all_buffers_ab_change=float(a['ab_pka_change'])))
    atomic_json(out/'individual_outliers_beyond15.json',outliers)
    for name in ('error_vs_distance','error_vs_sasa','error_vs_exposure','retained_sites_vs_radius'):
        assert (out/'plots'/f'{name}.png').stat().st_size>1000
    data={'verified':True,'individual_outliers_beyond15':outliers,'protein_geometry_files_checked':checks,'successful_complexes':len(successful),'successful_lost_training_pairs':lost,'buffer_instances':len(components),'identities':dict(__import__('collections').Counter(c['name'] for c in components)),
        'radius_summaries':selected,'complex_bootstrap':ci,'exposure_support':exposure,'exposure_summaries':[s for s in report['summaries'] if s['variant']=='individual' and s['exposure']!='all' and s['radius_A'] in (10,20) and s['metric']=='ab_pka_change'],'bootstrap_limit':'Complex resampling accounts for repeated sites, not sequence-family correlations or chemistry/sample-selection bias.',
        'proposal_sha256':digest(Path(__import__('os').environ['PKABENCH_RUNTIME'])/'universe/combined-split-v1/usable-proposal-v2/proposal.parquet')}
    assert data['proposal_sha256']=='4a51bc4e8d8b21ef62bed308d99b8ae9761ce7102745ed2d81e79bb16472e212'
    atomic_json(out/'verification.json',data); print(json.dumps(data,indent=2),flush=True)
