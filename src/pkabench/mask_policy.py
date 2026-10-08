"""Versioned downstream masks; never mutate frozen teacher inputs."""
import json
from pathlib import Path
import numpy as np
import biotite.structure as struc
from .annotate import SITE_ATOMS
from .supervision import clearance
from .runtime import atomic_json, digest, require_compute

POLICY={'version':'missing-residue-train10-eval20-v1','training_radius_A':10.,'evaluation_radius_A':20.,
    'distance_method':'lower-bound clearance from conservative contour-reach missing-residue envelope; not measured missing-atom distance',
    'defect_kinds':['terminal_gap','internal_gap'],'repaired_atoms_eligible':True,'adaptive_radii':False,
    'comparison':'>=','imputation':False}


def select_sites(campaign,overlay,split):
    """Explicit downstream selector; split assignment and teacher validity remain separate."""
    import pyarrow.parquet as pq
    from .schema import read_table
    campaign=Path(campaign); overlay=Path(overlay)
    field={'train':'train_mask','validation':'eval_mask','test':'eval_mask'}[split]
    manifest=json.loads((overlay/'manifest.json').read_text())
    if manifest['policy']!=POLICY or Path(manifest['source_campaign']).resolve()!=campaign.resolve(): raise ValueError('mask policy/source mismatch')
    if digest(overlay/'site_masks.parquet')!=manifest['output_sha256']: raise ValueError('mask table hash mismatch')
    for cid,files in manifest['sources'].items():
        for name,sha in files.items():
            if digest(campaign/'structures'/cid/name)!=sha: raise ValueError('mask input hash mismatch')
    key=lambda r:tuple(r[k] for k in ('complex_id','chain','resnum','icode','group'))
    allowed={key(r) for r in pq.read_table(overlay/'site_masks.parquet').to_pylist() if r[field]}
    return [r for r in read_table(campaign/'sites.parquet') if key(r) in allowed]


def masks(sites,atoms,evidence):
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True); residues={}
    for s,e in zip(starts[:-1],starts[1:]):
        a=atoms[s:e]; residues[str(a.chain_id[0]),int(a.res_id[0]),str(a.ins_code[0]).strip()]=a
    gaps=[d for d in evidence['defects'] if d['kind'] in POLICY['defect_kinds']]
    known=bool(evidence.get('sequences')) and all(s['sequence_source']=='entity_poly_canonical' for s in evidence['sequences'])
    result=[]
    for site in sites:
        atom=residues[site['chain'],site['resnum'],site['icode']]
        points=atom.coord[np.isin(atom.atom_name,SITE_ATOMS[site['group']])]
        distance=min((clearance(points,d) for d in gaps),default=None)
        eligible=bool(known and site['functional_atoms_complete'] and not site['is_break_terminus'])
        result.append({k:site[k] for k in ('complex_id','chain','resnum','icode','group')} | {
            'missing_residue_clearance_A':distance,'coverage_known':known,'native_site_eligible':eligible,
            'train_mask':bool(eligible and (distance is None or distance>=10)),
            'eval_mask':bool(eligible and (distance is None or distance>=20)),
            'interface':site['residue_delta_sasa']>10})
    return result


def apply(campaign,out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .schema import read_table
    from .prep import read_cif
    require_compute(); campaign=Path(campaign); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    rows=[]; sources={}; complexes=[]
    for structure in read_table(campaign/'structures.parquet'):
        cid=structure['complex_id']; root=campaign/'structures'/cid
        evidence=json.loads((root/'input_atom_mask.json').read_text())
        rr=masks(read_table(root/'sites.parquet'),read_cif(root/'AB.cif'),evidence); rows.extend(rr)
        sources[cid]={n:digest(root/n) for n in ('sites.parquet','AB.cif','input_atom_mask.json')}
        complexes.append({'complex_id':cid,'pdb_id':structure['pdb_id'],'sites':len(rr),
            'eligible':sum(r['native_site_eligible'] for r in rr),'training':sum(r['train_mask'] for r in rr),
            'evaluation':sum(r['eval_mask'] for r in rr)})
    pq.write_table(pa.Table.from_pylist(rows),out/'site_masks.parquet')
    atomic_json(out/'manifest.json',{'policy':POLICY,'source_campaign':str(campaign),'sources':sources,
        'implementation_sha256':digest(Path(__file__)),'output_sha256':digest(out/'site_masks.parquet')})
    report={'complexes':len(complexes),'native_eligible_sites':sum(r['native_site_eligible'] for r in rows),
        'unknown_coverage_sites':sum(not r['coverage_known'] for r in rows),
        'training_sites':sum(r['train_mask'] for r in rows),'evaluation_sites':sum(r['eval_mask'] for r in rows),
        'training_interface_sites':sum(r['train_mask'] and r['interface'] for r in rows),
        'evaluation_interface_sites':sum(r['eval_mask'] and r['interface'] for r in rows),
        'complexes_with_training':sum(r['training']>0 for r in complexes),'complexes_with_evaluation':sum(r['evaluation']>0 for r in complexes),
        'note':'Structural masks only; intersect with teacher coverage and sequence-disjoint split membership. Natural-gap envelope exclusions are more conservative than known-deleted-atom distances.',
        'per_complex':complexes}
    atomic_json(out/'report.json',report); print(json.dumps({k:v for k,v in report.items() if k!='per_complex'},indent=2))
