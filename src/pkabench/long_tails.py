"""Bounded 20/30-residue terminal-deletion pilot on existing references."""
import json
from pathlib import Path
from .runtime import require_compute, atomic_json, digest


def deletion_plan(atoms):
    import numpy as np
    import biotite.structure as struc
    from .radial import residue_key
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True); chains={}
    sasa=np.asarray(struc.sasa(atoms,probe_radius=1.4,point_number=1000),float)
    for s,e in zip(starts[:-1],starts[1:]):
        r=atoms[s:e]; isolated=float(np.nansum(struc.sasa(r,probe_radius=1.4,point_number=1000)))
        chains.setdefault(str(r.chain_id[0]),[]).append({'key':list(residue_key(r)),'name':str(r.res_name[0]),'coordinates':r.coord.astype(float).tolist(),'exposure_fraction':float(np.nansum(sasa[s:e]))/isolated if isolated else 0.})
    chain=next((c for c in sorted(chains) if len(chains[c])>=80),None)
    if chain is None: return {}
    rr=chains[chain]; variants={}
    for end in ('N','C'):
        for n in (20,30):
            removed=rr[:n] if end=='N' else rr[-n:]; fractions=[r['exposure_fraction'] for r in removed]
            variants[f'{end}{n}']={'kind':'terminal','end':end,'length':n,'residues':removed,
                'removed_coordinates':[xyz for r in removed for xyz in r['coordinates']],
                'deleted_ionisable_residues':sum(r['name'] in ('ASP','GLU','HIS','LYS','ARG','CYS','TYR') for r in removed),
                'reference_exposure_mean':float(np.mean(fractions)),
                'reference_exposure_class':'exposed' if min(fractions)>=.4 else 'buried' if max(fractions)<=.1 else 'intermediate'}
    return variants


def start(source,out):
    require_compute()
    import biotite.structure as struc
    from .prep import read_cif
    from .expanded import prepare,assemble,monitor
    source=Path(source); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((source/'plan.json').read_text()); selected=[]
    # Two references per interface type; complete first, smaller references next.
    for kind in ('homomer','heteromer','antibody'):
        candidates=sorted((r for r in parent['references'] if r['interface_type']==kind),key=lambda r:(r['original_missing_residues']+r['original_incomplete_residues'],r['structure']['n_residues'],r['complex_id']))
        taken=0
        for r in candidates:
            atoms=read_cif(Path(r['parent_campaign'])/'structures'/r['complex_id']/'AB.cif')
            starts=struc.get_residue_starts(atoms)
            if not any(sum(atoms.chain_id[starts]==c)>=80 for c in set(atoms.chain_id)): continue
            selected.append(r); taken+=1
            if taken==2: break
    if not selected: raise ValueError('No eligible long-tail reference')
    atomic_json(out/'plan.json',{'references':selected,'requested':6,'mode':'long_tails',
        'selection':'Pilot: up to two existing near-complete references per interface type, complete/smaller first. First sorted chain >=80 residues; delete N/C 20/30, leaving >=50 residues. Ordered-coordinate deletion, not evidence that native tails are disordered. Fresh baseline and same-seed repeat; no structure reprediction.',
        'source_plan_sha256':digest(source/'plan.json'),'implementation_sha256':digest(Path(__file__)),'production_allowed':False})
    prepare(out,0,1); assemble(out); monitor(out)

