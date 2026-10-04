"""Small controlled defect pilot; provisional within-configuration PB sensitivity."""
import json
import os
from pathlib import Path
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree
from .curation import prepare_revised
from .dataset_audit import filtered_cif
from .prep import CANONICAL, Rejection
from .runtime import atomic_json, digest, require_compute
from .schema import read_table, write_table


def perturbations(source,row):
    cif=pdbx.CIFFile.read(source)
    atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
    cat=cif.block['atom_site']; chains=cat['label_asym_id'].as_array(str)
    seq=cat['label_seq_id'].as_array(str); names=cat['label_atom_id'].as_array(str)
    side=None; terminal=None
    for partner,other in (('A','B'),('B','A')):
        tree=cKDTree(atoms.coord[np.isin(atoms.chain_id,row[f'partner_{other}_chains'])])
        for chain in row[f'partner_{partner}_chains']:
            a=atoms[(atoms.chain_id==chain)&np.isin(atoms.res_name,list(CANONICAL))]
            starts=struc.get_residue_starts(a,add_exclusive_stop=True)
            residues=[a[s:e] for s,e in zip(starts[:-1],starts[1:])]
            for r in residues:
                # Remove one terminal ionisable atom, leaving a known complete reference.
                target={'LYS':'NZ','ASP':'OD2','GLU':'OE2','TYR':'OH'}.get(str(r.res_name[0]))
                if side is None and target in set(r.atom_name) and float(tree.query(r.coord)[0].min())>12:
                    position=str(int(r.res_id[0])); keep=~((chains==chain)&(seq==position)&(names==target))
                    side=(keep,{'kind':'sidechain_atom_deleted','chain':chain,'label_seq_id':position,'atom':target})
            if terminal is None and len(residues)>20:
                for rr in (residues[:2],residues[-2:]):
                    if all(float(tree.query(r.coord)[0].min())>12 for r in rr):
                        positions=[str(int(r.res_id[0])) for r in rr]
                        keep=~((chains==chain)&np.isin(seq,positions))
                        terminal=(keep,{'kind':'two_terminal_residues_deleted','chain':chain,'label_seq_ids':positions})
                        break
    return {'sidechain':side,'terminal':terminal}


def initialise(campaign,out):
    require_compute(); campaign=Path(campaign); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((campaign/'manifest.json').read_text()); source_root=Path(manifest['source_audit'])/'sources'
    candidates=[json.loads(p.read_text()) for p in (campaign/'rows').glob('*.json')]
    candidates=[r for r in candidates if r['status']=='accepted' and 80<=r['structure']['n_residues']<=350 and r['interface_sites_by_radius']['10']>0]
    candidates.sort(key=lambda r:(r['extra']['defects'],r['structure']['n_residues'],r['complex_id']))
    cases=[]; skipped=[]; tasks=[]
    for row in candidates:
        source=source_root/f"{row['pdb_id']}.cif"; variants=perturbations(source,row)
        if any(v is None for v in variants.values()): continue
        cid=row['complex_id']; made=[]
        try:
            for variant in ('baseline','repeat','sidechain','terminal'):
                root=out/cid/variant; work=root/'structures'/cid; work.mkdir(parents=True)
                modified=root/'source.cif'
                if variant in variants: filtered_cif(source,variants[variant][0]).write(modified)
                else: modified.write_bytes(source.read_bytes())
                item={k:row[k] for k in ('complex_id','pdb_id','partner_A_chains','partner_B_chains')}
                item['source_sha256']=digest(modified)
                structure,sites,extra=prepare_revised(modified,item,work)
                write_table(work/'sites.parquet','sites',sites)
                write_table(root/'structures.parquet','structures',[structure]); write_table(root/'sites.parquet','sites',sites)
                atomic_json(root/'manifest.json',{'candidates':[item],'variant':variant,'parent_campaign':str(campaign),
                    'perturbation':variants[variant][1] if variant in variants else None,'production_allowed':False})
                made.append((str(root),cid))
        except Rejection as exc:
            skipped.append({'complex_id':cid,'code':exc.code,'detail':str(exc)}); continue
        cases.append({'complex_id':cid,'pdb_id':row['pdb_id'],'n_residues':row['structure']['n_residues'],
            'perturbations':{k:v[1] for k,v in variants.items()}}); tasks.extend(made)
        if len(cases)==3: break
    atomic_json(out/'manifest.json',{'cases':cases,'skipped':skipped,'tasks':tasks,
        'purpose':'Compare paired PB shifts after known distal coordinate deletion/rebuilding against the intact reference; repeat baseline measures stochastic variation.',
        'production_allowed':False,'job':os.environ['SLURM_JOB_ID']})
    (out/'tasks.tsv').write_text(''.join(f'{root}\t{cid}\n' for root,cid in tasks))
    print(json.dumps({'cases':cases,'skipped':skipped,'jobs':len(tasks)},indent=2))
    if len(cases)<3: raise ValueError('fewer than three eligible controlled perturbation cases')


def paired_values(rows):
    index={(r['chain'],r['resnum'],r['icode'],r['group'],r['state']):r for r in rows}
    result={}
    for key,r in index.items():
        if key[-1]!='AB' or r['status']!='ok': continue
        free=next((index.get((*key[:-1],p)) for p in ('A','B') if index.get((*key[:-1],p)) is not None),None)
        if free and free['status']=='ok': result[key[:-1]]=r['pka']-free['pka']
    return result


def initialise_altloc(campaign,out):
    """One local deposited alternative at a time, without global A/B assumptions."""
    require_compute(); campaign=Path(campaign); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((campaign/'manifest.json').read_text()); source_root=Path(parent['source_audit'])/'sources'
    rows=[json.loads(p.read_text()) for p in (campaign/'rows').glob('*.json')]
    rows=sorted((r for r in rows if r['status']=='accepted' and r['extra'].get('alternate_residues') and r['structure']['n_residues']<=500),key=lambda r:(r['structure']['n_residues'],r['complex_id']))
    cases=[]; tasks=[]; skipped=[]
    for row in rows:
        cid=row['complex_id']; work=campaign/'structures'/cid
        records=json.loads((work/'conformers.json').read_text())['records']
        choice=None
        for record in records:
            variants=record['variants']
            if len(variants)<2 or any(v['missing_heavy_atoms'] for v in variants): continue
            variant=next(v for v in variants if v['alt_id']!=record['selected_alt_id'])
            if any(not set(v['atoms'])==set(variant['atoms']) for v in variants): continue
            choice=(record,variant); break
        if choice is None: continue
        record,variant=choice; source=source_root/f"{row['pdb_id']}.cif"
        cat=pdbx.CIFFile.read(source).block['atom_site']; keep=np.ones(cat.row_count,bool)
        all_ids={i for v in record['variants'] for i in v['source_row_indices']}
        chosen=set(variant['source_row_indices']); keep[list(all_ids-chosen)]=False
        made=[]
        try:
            for name in ('baseline','repeat','altloc'):
                root=out/cid/name; work=root/'structures'/cid; work.mkdir(parents=True)
                modified=root/'source.cif'
                if name=='altloc': filtered_cif(source,keep).write(modified)
                else: modified.write_bytes(source.read_bytes())
                item={k:row[k] for k in ('complex_id','pdb_id','partner_A_chains','partner_B_chains')}; item['source_sha256']=digest(modified)
                structure,sites,extra=prepare_revised(modified,item,work)
                write_table(work/'sites.parquet','sites',sites); write_table(root/'sites.parquet','sites',sites)
                write_table(root/'structures.parquet','structures',[structure])
                atomic_json(root/'manifest.json',{'candidates':[item],'variant':name,'parent_campaign':str(campaign),
                    'local_altloc_key':record['key'],'alternative':variant['alt_id'],'production_allowed':False})
                made.append((str(root),cid))
        except Rejection as exc:
            skipped.append({'complex_id':cid,'code':exc.code,'detail':str(exc)}); continue
        cases.append({'complex_id':cid,'pdb_id':row['pdb_id'],'variants':['baseline','repeat','altloc'],
            'local_key':record['key'],'reference_alt_id':record['selected_alt_id'],'alternative_alt_id':variant['alt_id']})
        tasks.extend(made)
        if len(cases)==3: break
    atomic_json(out/'manifest.json',{'cases':cases,'skipped':skipped,'tasks':tasks,'production_allowed':False,
        'purpose':'Bounded local-altloc sensitivity; no assertion of global conformer compatibility or ensemble probabilities.'})
    (out/'tasks.tsv').write_text(''.join(f'{root}\t{cid}\n' for root,cid in tasks))
    print(json.dumps({'cases':cases,'skipped':skipped,'jobs':len(tasks)},indent=2))


def collect(out):
    require_compute(); out=Path(out); manifest=json.loads((out/'manifest.json').read_text()); results=[]; failures=[]
    for case in manifest['cases']:
        cid=case['complex_id']; root=out/cid; reference=paired_values(read_table(root/'baseline/jobs/pypka'/f'{cid}.parquet'))
        variants=case.get('variants',['baseline','repeat','sidechain','terminal'])
        case['prepared_defects']={}
        for variant in variants:
            side=json.loads((root/variant/'jobs/pypka'/f'{cid}.json').read_text())
            from .jobs import inputs
            if side['inputs']!=inputs(root/variant,cid,'pypka') or side['output_sha256']!=digest(root/variant/'jobs/pypka'/f'{cid}.parquet'):
                raise ValueError('sensitivity shard provenance mismatch')
            if side['status']!='complete': failures.append({'complex_id':cid,'variant':variant,'errors':side['errors']})
            evidence=json.loads((root/variant/'structures'/cid/'input_atom_mask.json').read_text())
            case['prepared_defects'][variant]=[{k:d.get(k) for k in ('kind','key','chain','length','exposure_class','smaller_tail_radius_candidate')} for d in evidence['defects']]
        for variant in variants[1:]:
            values=paired_values(read_table(root/variant/'jobs/pypka'/f'{cid}.parquet'))
            sites=read_table(root/variant/'sites.parquet')
            for policy,radius in ((p,r) for p in ('strict','adaptive') for r in (10,15,20)):
                field=f'supervision_mask_{radius}' if policy=='strict' else f'supervision_mask_adaptive_{radius}'
                eligible=[s for s in sites if s[field] and s['residue_delta_sasa']>10]
                keys=[(s['chain'],s['resnum'],s['icode'],s['group']) for s in eligible]
                differences=[abs(values[k]-reference[k]) for k in keys if k in reference and k in values]
                results.append({'complex_id':cid,'pdb_id':case['pdb_id'],'variant':variant,'radius':radius,'policy':policy,
                    'eligible_interface_sites':len(keys),'paired_pb_sites':len(differences),
                    'median_absolute_delta_pka_change':float(np.median(differences)) if differences else None,
                    'max_absolute_delta_pka_change':max(differences,default=None),
                    'sites_over_0.1':sum(d>.1 for d in differences)})
            common=reference.keys()&values.keys()
            differences=[abs(values[k]-reference[k]) for k in common]
            results.append({'complex_id':cid,'pdb_id':case['pdb_id'],'variant':variant,'policy':'unmasked_diagnostic',
                'paired_pb_sites':len(common),'max_absolute_delta_pka_change':max(differences,default=None),
                'sites_over_0.1':sum(d>.1 for d in differences),
                'note':'Includes uncertain sites for sensitivity measurement only; not training eligibility.'})
    result={'cases':manifest['cases'],'results':results,'failures':failures,'production_allowed':False,
        'analysis_sha256':digest(Path(__file__)),
        'radius_status':'provisional: three controlled cases cannot establish a universal safe distance',
        'limits':'Within current teacher configuration, not historical pKPDB equivalence or experimental truth. ARG coverage remains absent. Artificial deletion does not represent all genuine disorder. No missing-residue pKa is imputed.'}
    atomic_json(out/'report.json',result); print(json.dumps(result,indent=2))
