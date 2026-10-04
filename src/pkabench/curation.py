"""Revised, explicitly scoped partner preparation with masked supervision."""
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree
from .prep import CANONICAL, Rejection, prepare_pair, read_cif, write_cif
from .supervision import inventory, annotate_masks, classify_exposure, RADII, TAIL_RADII
from .conformers import resolve
from .runtime import atomic_json, digest, config_hash, require_compute
from .schema import write_table, read_table

POLICY={'version':'pdb2pqr-inclusive-v1','assembly_residue_cap':1500,'partner_scope':'explicit selected partners; omitted protein context recorded',
    'neutral_additives':['GOL','EDO','PEG'],'additive_bond_distance':1.9,'additive_bridge_distance':4.,
    'interface_zone':10.,'radii':list(RADII),'default_radius':None,'radius_status':'diagnostic only; no distance exclusion until measured',
    'missing_sidechain_extent':8.,'missing_segment_extent':'CA contour length 3.8 A per residue plus 8 A heavy-atom extent',
    'exposed_tail_radii':{str(k):v for k,v in TAIL_RADII.items()},'altloc':'first deposited positive-occupancy alt ID per residue; shared atoms retained; alternatives are provenance only; same reference in AB/A/B',
    'teacher_input':'PDB2PQR-prepared observed structure; no manual dropping of incomplete residues; physical segment termini explicit',
    'missing_coordinate_gate':'PDB2PQR success with preserved observed heavy atoms; no interface-gap or missing-sidechain pre-rejection',
    'production_allowed':False}


def validate_partners(atoms, partners, cap=1500):
    if set(partners)!={'A','B'} or any(not v for v in partners.values()): raise Rejection('multi_partner','two nonempty partner sets required','selection')
    selected=list(partners['A'])+list(partners['B'])
    if len(selected)!=len(set(selected)): raise Rejection('multi_partner','overlapping or duplicate chain assignments','selection')
    protein=atoms[np.isin(atoms.res_name,list(CANONICAL))]
    if not set(selected)<=set(protein.chain_id): raise Rejection('multi_partner','unknown selected protein chain','selection')
    size=len(struc.get_residue_starts(protein))
    if size>cap: raise Rejection('size_cap',f'full observed assembly has {size} canonical residues','selection')
    return {'assembly_observed_residues':size,'omitted_protein_chains':sorted(set(protein.chain_id)-set(selected))}


def prepare_revised(source, row, out, *, diagnostic_perturbation=False):
    from .audit import component_inventory
    from .dataset_audit import filtered_cif
    source=Path(source); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    partners={p:row[f'partner_{p}_chains'] for p in ('A','B')}
    cif,conformers=resolve(pdbx.CIFFile.read(source),partners['A']+partners['B'])
    resolved=out/'resolved-source.cif'; cif.write(resolved)
    atomic_json(out/'conformers.json',conformers)
    atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
    context=validate_partners(atoms,partners,POLICY['assembly_residue_cap'])
    evidence=inventory(cif,partners); selected=partners['A']+partners['B']
    evidence['defects'].extend(conformers['defects'])
    evidence=classify_exposure(cif,partners,evidence)
    components=component_inventory(atoms,cif,selected)
    cat=cif.block['atom_site']; chain_ids=cat['label_asym_id'].as_array(str); names=cat['label_comp_id'].as_array(str)
    positions=cat['label_seq_id'].as_array(str); keep=np.ones(len(chain_ids),bool); removals=[]; protected=set()
    conn=cif.block.get('struct_conn')
    if conn is not None:
        for i,kind in enumerate(conn['conn_type_id'].as_array(str)):
            if str(kind).lower().startswith(('covale','metalc')):
                for p in ('ptnr1','ptnr2'):
                    c,n=f'{p}_label_asym_id',f'{p}_label_comp_id'
                    if c in conn and n in conn: protected.add((str(conn[c].as_array(str)[i]),str(conn[n].as_array(str)[i])))
    trees={p:cKDTree(atoms.coord[np.isin(atoms.chain_id,chains)]) for p,chains in partners.items()}
    declared_connections=set(protected)
    # A label chain can contain repeated component names. Filtering by chain/name
    # is allowed only when every such instance passes the removal checks.
    for component in components:
        coords=atoms.coord[component['start']:component['end']]
        distances=[float(tree.query(coords)[0].min()) for tree in trees.values()]
        if max(distances)<=POLICY['additive_bridge_distance'] or min(distances)<POLICY['additive_bond_distance']:
            protected.add((component['chain'],component['name']))
    for component in components:
        name=component['name']; c=component['chain']; coords=atoms.coord[component['start']:component['end']]
        distances={p:float(tree.query(coords)[0].min()) for p,tree in trees.items()}
        bridge=all(d<=POLICY['additive_bridge_distance'] for d in distances.values())
        remove=name in POLICY['neutral_additives'] and component['code']=='ligand' and (c,name) not in protected and not bridge and min(distances.values())>=POLICY['additive_bond_distance']
        decision={**component,'partner_distances':distances,'bridges_partners':bridge,'declared_connection':(c,name) in declared_connections,'protected_from_removal':(c,name) in protected,'removed':remove}
        removals.append(decision)
        if remove:
            keep &= ~((chain_ids==c)&(names==name))
            evidence['defects'].append({'kind':'removed_additive','centres':coords.astype(float).tolist(),'extent':0.,'name':name,'chain':c})
    filtered=filtered_cif(resolved,keep)
    structure,sites=prepare_pair(filtered,row,out,str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/pypka/bin/pdb2pqr30'),
        audit_allow_subcomplex=True,pdb2pqr_acceptance=True,enforce_geometry=not diagnostic_perturbation)
    # Input geometry is observed, not silently replaced by the teacher's completed geometry.
    author=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=True)
    original=atoms.copy(); original.res_id=author.res_id.copy(); original.ins_code=author.ins_code.copy()
    original=original[np.isin(original.chain_id,selected)&np.isin(original.res_name,list(CANONICAL))&~np.isin(np.char.upper(original.element),['H','D'])]
    original.coord=np.round(original.coord,3); write_cif(out/'student_AB.cif',original)
    sites=annotate_masks(sites,read_cif(out/'AB.cif'),evidence,15)
    for site in sites:
        site['supervision_mask']=bool(site['functional_atoms_complete'] and not site['is_break_terminus'])
    atomic_json(out/'input_atom_mask.json',evidence)
    provenance=json.loads((out/'provenance.json').read_text())
    provenance.update(curation_policy=POLICY,source_context=context,additive_review=removals,
        alternate_conformations_sha256=digest(out/'conformers.json'),altloc=conformers['selection'],
        dropped_incomplete_backbone_keys=[],supervision_radius=None,
        atom_mask_sha256=digest(out/'input_atom_mask.json'),student_input_sha256=digest(out/'student_AB.cif'),
        label_scope='paired response of the explicitly selected prepared partners, not full assembly binding')
    atomic_json(out/'provenance.json',provenance); structure['provenance']=json.dumps(provenance)
    return structure,sites,{'context':context,'additive_review':removals,'defects':len(evidence['defects']),
        'alternate_residues':len(conformers['records']),'exposed_tail_candidates':sum(d.get('smaller_tail_radius_candidate',False) for d in evidence['defects']),
        'exposure_classes':dict(Counter(d.get('exposure_class','unknown') for d in evidence['defects']))}


def initialise(audit, out):
    require_compute(); audit=Path(audit); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    index=json.loads((audit/'index.json').read_text())
    frozen={'candidates':index['candidates'],'source_audit':str(audit),'policy':POLICY,
        'source_index_sha256':digest(audit/'index.json'),'production_allowed':False,
        'implementation':{name:digest(Path(__file__).with_name(name)) for name in ('curation.py','supervision.py','conformers.py','prep.py','annotate.py','schema.py')}}
    atomic_json(out/'manifest.json',frozen)


def scan(campaign,shard,shards):
    require_compute(); campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text())
    if manifest['policy']!=POLICY or any(digest(Path(__file__).with_name(name))!=sha for name,sha in manifest['implementation'].items()):
        raise ValueError('frozen curation implementation differs; initialise a new campaign')
    source_root=Path(manifest['source_audit'])/'sources'
    for row in manifest['candidates'][shard::shards]:
        cid=row['complex_id']; receipt=campaign/'rows'/f'{cid}.json'
        if receipt.exists(): continue
        work=campaign/'structures'/cid; work.mkdir(parents=True,exist_ok=True)
        try:
            if digest(source_root/f"{row['pdb_id']}.cif")!=row['source_sha256']: raise ValueError('source hash mismatch')
            structure,sites,extra=prepare_revised(source_root/f"{row['pdb_id']}.cif",row,work)
            write_table(work/'sites.parquet','sites',sites)
            result={'status':'accepted','structure':structure,'extra':extra,
                'sites_by_radius':{str(r):sum(s[f'supervision_mask_{r}'] for s in sites) for r in RADII},
                'interface_sites_by_radius':{str(r):sum(s[f'supervision_mask_{r}'] and s['residue_delta_sasa']>10 for s in sites) for r in RADII},'sites':len(sites)}
        except Rejection as exc: result={'status':'rejected','code':exc.code,'stage':exc.stage,'detail':str(exc)}
        except Exception as exc:
            import traceback
            result={'status':'pipeline_error','detail':str(exc),'traceback':traceback.format_exc()}
        atomic_json(receipt,{**row,**result,'job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME']})
    atomic_json(campaign/'shards'/f'{shard}.json',{'shard':shard,'shards':shards,'status':'complete'})


def collect(campaign):
    require_compute(); campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text()); rows=[]
    for candidate in manifest['candidates']: rows.append(json.loads((campaign/'rows'/f"{candidate['complex_id']}.json").read_text()))
    accepted=[r for r in rows if r['status']=='accepted']; sites=[]
    for r in accepted: sites.extend(read_table(campaign/'structures'/r['complex_id']/'sites.parquet'))
    write_table(campaign/'structures.parquet','structures',[r['structure'] for r in accepted]); write_table(campaign/'sites.parquet','sites',sites)
    write_table(campaign/'rejections.parquet','rejections',[{'candidate_id':r['complex_id'],'code':r['code'],'detail':r['detail'],'stage':r['stage']} for r in rows if r['status']=='rejected'])
    report={'candidates':len(rows),'accepted':len(accepted),'acceptance':len(accepted)/len(rows),'rejected':sum(r['status']=='rejected' for r in rows),
        'pipeline_errors':[r for r in rows if r['status']=='pipeline_error'],'progression_allowed':False,'production_allowed':False,
        'histogram':dict(Counter(r.get('code',r['status']) for r in rows)),
        'radii':{str(radius):{'sites':sum(s[f'supervision_mask_{radius}'] for s in sites),
            'interface_sites':sum(s[f'supervision_mask_{radius}'] and s['residue_delta_sasa']>10 for s in sites),
            'complexes_with_sites':sum(r['sites_by_radius'][str(radius)]>0 for r in accepted),
            'complexes_with_interface_sites':sum(r['interface_sites_by_radius'][str(radius)]>0 for r in accepted)} for radius in RADII},
        'total_sites':len(sites),'note':'Counts are structural supervision eligibility; teacher coverage and sensitivity have not been applied.'}
    report['inclusive']={'sites':sum(s['supervision_mask'] for s in sites),
        'interface_sites':sum(s['supervision_mask'] and s['residue_delta_sasa']>10 for s in sites),
        'complexes_with_sites':len({s['complex_id'] for s in sites if s['supervision_mask']}),
        'complexes_with_interface_sites':len({s['complex_id'] for s in sites if s['supervision_mask'] and s['residue_delta_sasa']>10})}
    atomic_json(campaign/'curation_report.json',report); print(json.dumps(report,indent=2))
