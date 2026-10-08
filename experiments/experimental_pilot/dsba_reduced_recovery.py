"""State-corrected DsbA Cys30 recovery using reduced PDB 1A2L."""
import json
import os
import sys
import urllib.request
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
runtime=Path(os.environ['PKABENCH_RUNTIME'])
source=runtime/'experimental/pkadr-full-v1'
baseline=runtime/'experimental/pkadr-baselines-v1'
out=runtime/'experimental/pkadr-dsba-reduced-v1'
methods=('propka','pkai','pkai_plus','jaxka','pypka')
task_id='dsba-reduced-1a2l'

def prepare():
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from pkabench.conformers import resolve
    from pkabench.prep import CANONICAL, complete, topology, write_cif
    from pkabench.supervision import inventory, clearance

    out.mkdir(parents=True,exist_ok=False)
    work=out/'structure'; work.mkdir()
    source_path=out/'1A2L.cif'
    source_url='https://files.rcsb.org/download/1A2L.cif'
    source_path.write_bytes(urllib.request.urlopen(source_url,timeout=60).read())
    cif=pdbx.CIFFile.read(source_path); cat=cif.block['atom_site']
    polymer=(cat['label_seq_id'].as_array(str)!='.')&(cat['label_seq_id'].as_array(str)!='?')
    selected=sorted(set(cat['label_asym_id'].as_array(str)[polymer&(cat['auth_asym_id'].as_array(str)=='A')]))
    if len(selected)!=1: raise ValueError(f'author chain A maps to {selected}')
    chain=selected[0]
    models=cat['pdbx_PDB_model_num'].as_array(str); first=models[0]
    new=pdbx.CIFCategory()
    for name in cat: new[name]=pdbx.CIFColumn(cat[name].as_array(str)[models==first])
    cif.block['atom_site']=new
    cif,conformers=resolve(cif,[chain]); atomic_json(work/'conformers.json',conformers)
    evidence=inventory(cif,{'A':[chain],'B':[]}); atomic_json(work/'input_atom_mask.json',evidence)
    atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
    author=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=True)
    asym=cif.block['struct_asym']; poly=cif.block['entity_poly']
    entity=dict(zip(asym['id'].as_array(str),asym['entity_id'].as_array(str)))[chain]
    sequence=''.join(dict(zip(poly['entity_id'].as_array(str),poly['pdbx_seq_one_letter_code_can'].as_array(str)))[entity].split())
    old=json.loads((source/'rows/695e99fa7c29ed19.json').read_text())
    if sequence != old['sequence']: raise ValueError('1A2L and 1DSB deposited sequences differ')
    atoms.res_id=author.res_id.copy(); atoms.ins_code=author.ins_code.copy()
    atoms=atoms[(atoms.chain_id==chain)&np.isin(atoms.res_name,list(CANONICAL))&~np.isin(np.char.upper(atoms.element),['H','D'])]
    write_cif(work/'observed.cif',atoms)
    fixed=complete(atoms,str(runtime/'envs/pypka/bin/pdb2pqr30'),work)
    top=topology(fixed); write_cif(work/'prepared.cif',fixed)
    if top.metadata['disulfide_pairs']:
        raise ValueError(f'reduced structure contains a detected disulfide: {top.metadata["disulfide_pairs"]}')
    hits=[(i,k) for i,k in enumerate(top.keys) if k.chain==chain and k.number==30 and k.insertion=='']
    if len(hits)!=1: raise ValueError(f'Cys30 mapping is {hits}')
    i,key=hits[0]; residue=top.residue(i)
    if str(residue.res_name[0])!='CYS' or 'SG' not in set(residue.atom_name): raise ValueError('Cys30 thiol site incomplete')
    c30=residue.coord[residue.atom_name=='SG'][0]
    c33=[top.residue(j).coord[top.residue(j).atom_name=='SG'][0] for j,k in enumerate(top.keys)
         if k.chain==chain and k.number==33 and k.insertion=='' and 'SG' in set(top.residue(j).atom_name)]
    sg_distance=float(np.linalg.norm(c30-c33[0])) if len(c33)==1 else None
    defect_clearance=min((clearance(residue.coord,d) for d in evidence['defects']
                          if d['kind'] in ('terminal_gap','internal_gap')),default=None)
    area=struc.sasa(fixed,probe_radius=1.4,point_number=1000)
    mask=(fixed.chain_id==chain)&(fixed.res_id==30)&(fixed.ins_code=='')
    site=dict(complex_id=task_id,chain=chain,resnum=30,icode='',group='CYS')
    atomic_json(out/'manifest.json',dict(
        version='pkadr-dsba-reduced-v1',task_id=task_id,pdb='1A2L',author_chain='A',label_chain=chain,
        source_url=source_url,source_sha256=digest(source_path),prepared_sha256=digest(work/'prepared.cif'),
        original_record_id='142',experimental_pka=3.5,original_pdb='1DSB',sequence=sequence,
        sequence_matches_1DSB=True,redox_state='reduced',disulfide_pairs=top.metadata['disulfide_pairs'],
        cys30_cys33_sg_distance_A=sg_distance,cys30_sasa_A2=float(np.nansum(area[mask])),
        cys30_defect_clearance_A=defect_clearance,site=site,methods=list(methods),
        model_fit=False,experimental_training=False,job=os.environ['SLURM_JOB_ID'],
        code_sha256=digest(Path(__file__))))
    print(json.dumps(json.loads((out/'manifest.json').read_text()),indent=2),flush=True)

def run(index):
    from pkabench.adapters.base import Adapter
    from pkabench.schema import write_table
    manifest=json.loads((out/'manifest.json').read_text()); method=methods[index]
    structure=out/'structure/prepared.cif'
    if digest(structure)!=manifest['prepared_sha256']: raise ValueError('prepared structure hash changed')
    dest=out/'predictions'/method; dest.mkdir(parents=True,exist_ok=False)
    adapter=Adapter(method,[manifest['site']],runtime,timeout=2700)
    if method=='jaxka': adapter.config['steps']=1024
    rows=adapter.run({'AB':structure},dest)
    write_table(dest/'predictions.parquet','predictions',rows)
    receipt=dict(method=method,input_sha256=manifest['prepared_sha256'],prediction_sha256=digest(dest/'predictions.parquet'),
        errors=adapter.errors,timings=adapter.timings,extra=adapter.extras,job=os.environ['SLURM_JOB_ID'])
    atomic_json(dest/'receipt.json',receipt)
    print(json.dumps(dict(method=method,rows=[{k:r.get(k) for k in ('status','pka','intrinsic_pka')} for r in rows],
                               errors=adapter.errors,extra=adapter.extras),indent=2),flush=True)

def score():
    import numpy as np
    import biotite.structure as struc
    import pyarrow.parquet as pq
    from pkabench.anchor_tiers import terminal_radius
    from pkabench.annotate import SITE_ATOMS
    from pkabench.prep import read_cif
    from pkabench.supervision import clearance
    manifest=json.loads((out/'manifest.json').read_text()); result=[]
    atoms=read_cif(out/'structure/prepared.cif')
    evidence=json.loads((out/'structure/input_atom_mask.json').read_text())
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    residues={(str(atoms.chain_id[s]),int(atoms.res_id[s]),str(atoms.ins_code[s]).strip()):atoms[s:e]
              for s,e in zip(starts[:-1],starts[1:])}
    bylabel={(r['key'][0],r['label_seq_id']):tuple(r['key']) for r in evidence['atoms']}
    target=residues[(manifest['label_chain'],30,'')]
    points=target.coord[np.isin(target.atom_name,SITE_ATOMS['CYS'])]
    details=[]; near=False; unknown=False; clean=True
    for d in evidence['defects']:
        if d['kind'] not in ('terminal_gap','internal_gap'): continue
        internal=d['kind']=='internal_gap'
        radius=(15. if d['length']<=3 else None) if internal else terminal_radius(d['length'])
        keys=[bylabel[manifest['label_chain'],p] for p in (d['start']-1,d['end']+1)
              if (manifest['label_chain'],p) in bylabel]
        anchors=[]
        if radius is not None and len(keys)==(2 if internal else 1):
            for key in keys:
                residue=residues.get(key)
                inventory=next((r for r in evidence['atoms'] if tuple(r['key'])==key),None)
                if residue is None or inventory is None:
                    radius=None; break
                visible=set(('N','CA','C','O'))-set(inventory['missing_atoms'])
                backbone=residue.coord[np.isin(residue.atom_name,list(visible))]
                if len(backbone)!=4:
                    radius=None; break
                anchors.extend(backbone)
        else:
            radius=None
        lower=clearance(points,d); clean &= lower>=20
        distance=float(np.linalg.norm(points[:,None]-np.asarray(anchors)[None,:],axis=-1).min()) \
            if radius is not None and len(anchors) else None
        near |= distance is not None and distance<radius
        unknown |= radius is None and lower<20
        details.append(dict(chain=d['chain'],start=d['start'],end=d['end'],kind=d['kind'],length=d['length'],
                            anchor_distance_A=distance,radius_A=radius,envelope_clearance_A=lower))
    tier='near_gap' if near else 'uncalibrated' if unknown else 'clean' if clean else 'uncertain'
    gap_annotation=dict(site=manifest['site'],natural_gap_tier=tier,gap_details=details,
        structural_train_mask=tier in ('clean','uncertain'),structural_eval_mask=tier in ('clean','uncertain'),
        policy='Anchor-aware natural-gap policy calibrated by radial-error-v2-nearcomplete/anchor-analysis-v2.')
    atomic_json(out/'gap_annotation.json',gap_annotation)
    old_task=baseline/'tasks/695e99fa7c29ed19'
    for method in methods:
        dest=out/'predictions'/method; receipt=json.loads((dest/'receipt.json').read_text())
        if receipt['prediction_sha256']!=digest(dest/'predictions.parquet'): raise ValueError(f'{method} output hash changed')
        new=pq.read_table(dest/'predictions.parquet').to_pylist()[0]
        old=pq.read_table(old_task/method/'predictions.parquet').to_pylist()[0]
        result.append(dict(method=method,oxidized_pdb='1DSB',oxidized_status=old['status'],oxidized_pka=old['pka'],
                           reduced_pdb='1A2L',reduced_status=new['status'],reduced_pka=new['pka'],
                           experimental_pka=3.5,absolute_error=None if new['pka'] is None else abs(new['pka']-3.5)))
    atomic_json(out/'comparison.json',result)
    names={'propka':'PROPKA','pkai':'pKAI','pkai_plus':'pKAI+','jaxka':'JAX-Ka','pypka':'PypKa'}
    def val(x): return '—' if x is None else f'{x:.3f}'
    lines=['# Reduced-DsbA Cys30 state recovery','',
      '**This is a single-record state-correction diagnostic, not a method ranking or a fitted model.**','',
      f"PKAD-R record 142 reports reduced Cys30 pKa = {manifest['experimental_pka']:.1f}. Its original mapping, 1DSB, is oxidized; 1A2L is the sequence-identical reduced structure. The prepared 1A2L structure has no detected disulfide and its Cys30–Cys33 sulfur distance is {manifest['cys30_cys33_sg_distance_A']:.2f} Å.",'',
      '| Method | Oxidized 1DSB | Reduced 1A2L | |error| vs 3.5 |','|---|---:|---:|---:|']
    for r in result:
        old=val(r['oxidized_pka']) if r['oxidized_status']=='ok' else r['oxidized_status']
        new=val(r['reduced_pka']) if r['reduced_status']=='ok' else r['reduced_status']
        lines.append(f"| {names[r['method']]} | {old} | {new} | {val(r['absolute_error'])} |")
    lines += ['', 'The oxidized structure cannot carry a reduced-thiol pKa label. Only the reduced column is chemically aligned with the experiment. Missing predictions are retained as statuses and are not imputed.','',
      f"Cys30 SASA in the prepared reduced structure is {manifest['cys30_sasa_A2']:.2f} Å². Its anchor-aware natural-gap tier is **{tier}**; the nearest conservative deposited-gap lower bound is {val(manifest['cys30_defect_clearance_A'])} Å.",'',
      '| Gap | Length | Anchor distance | Calibrated radius | Conservative envelope clearance |','|---|---:|---:|---:|---:|']
    for d in details:
        lines.append(f"| {d['kind']} {d['start']}–{d['end']} | {d['length']} | {val(d['anchor_distance_A'])} | {val(d['radius_A'])} | {val(d['envelope_clearance_A'])} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'release.json',dict(manifest_sha256=digest(out/'manifest.json'),comparison_sha256=digest(out/'comparison.json'),
        gap_annotation_sha256=digest(out/'gap_annotation.json'),
        report_sha256=digest(out/'report.md'),code_sha256=digest(Path(__file__)),job=os.environ['SLURM_JOB_ID'],
        model_fit=False,experimental_training=False))
    print((out/'report.md').read_text(),flush=True)

if sys.argv[1]=='prepare': prepare()
elif sys.argv[1]=='run': run(int(os.environ['SLURM_ARRAY_TASK_ID']))
elif sys.argv[1]=='score': score()
else: raise ValueError(sys.argv[1])
