"""Full PKAD-R structural audit. All computation is restricted to Slurm nodes."""
import json, os, re, sys, urllib.request, traceback
from pathlib import Path
from collections import Counter, defaultdict
from pkabench.runtime import require_compute, atomic_json, digest, config_hash

R=Path(os.environ['PKABENCH_RUNTIME'])
OUT=R/'experimental/pkadr-full-v1'
SOURCE=R/'experimental/pilot-v1/sources/PKAD-R-250211.json'

def initialise():
    OUT.mkdir(parents=True,exist_ok=False)
    raw=json.loads(SOURCE.read_text()); tasks={}; records=[]
    for row in raw:
        pdb=row['PDB'].strip().upper(); chain=row['Chain'].strip()
        task_id=config_hash([pdb,chain])[:16]
        task=tasks.setdefault(task_id,dict(task_id=task_id,pdb=pdb,chain=chain,record_ids=[]))
        task['record_ids'].append(str(row['Index']))
        value=str(row['Expt_pKa']).strip()
        numeric=bool(re.fullmatch(r'-?\d+(?:\.\d+)?',value))
        kind='point' if numeric else 'censored' if '<' in value or '>' in value else 'approximate_or_other'
        records.append(dict(record_id=str(row['Index']),task_id=task_id,label_kind=kind,
            value=float(value) if numeric else None,raw=row,training_eligible=False))
    assert len({r['record_id'] for r in records})==len(records)==1024
    atomic_json(OUT/'records.json',records); atomic_json(OUT/'tasks.json',list(tasks.values()))
    atomic_json(OUT/'manifest.json',dict(source=str(SOURCE),source_sha256=digest(SOURCE),
        code_sha256=digest(Path(__file__)),records=len(records),tasks=len(tasks),
        policy='Provisional isolated-chain geometry; preserve all labels/conditions; no model fitting or split mutation'))
    print(json.dumps(dict(records=len(records),tasks=len(tasks),label_kinds=dict(Counter(r['label_kind'] for r in records))),indent=2),flush=True)

def run_task(task,records):
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from pkabench.prep import CANONICAL, complete, topology, write_cif, Rejection
    from pkabench.conformers import resolve
    from pkabench.supervision import inventory, clearance
    from pkabench.glycan_buffer_policy import classify_components
    from pkabench.anchor_tiers import terminal_radius
    from pkabench.annotate import SITE_ATOMS
    root=OUT/'structures'/task['task_id']; root.mkdir(parents=True,exist_ok=True)
    result=dict(task,status='unprocessed',records=[])
    try:
        if not re.fullmatch(r'[0-9][A-Z0-9]{3}',task['pdb']):
            raise Rejection('non_pdb_identifier','A model/compound identifier requires explicit structure mapping')
        path=OUT/'sources'/f"{task['pdb']}.cif"; path.parent.mkdir(exist_ok=True)
        if not path.exists():
            cached=R/'experimental/pilot-v1/sources'/f"{task['pdb']}-cif.raw"
            data=cached.read_bytes() if cached.exists() else urllib.request.urlopen(f"https://files.rcsb.org/download/{task['pdb']}.cif",timeout=60).read()
            tmp=path.with_suffix('.'+task['task_id']+'.tmp'); tmp.write_bytes(data); tmp.replace(path)
        result['source_sha256']=digest(path)
        cif=pdbx.CIFFile.read(path); cat=cif.block['atom_site']
        polymer=(cat['label_seq_id'].as_array(str)!='.')&(cat['label_seq_id'].as_array(str)!='?')
        selected=sorted(set(cat['label_asym_id'].as_array(str)[polymer&(cat['auth_asym_id'].as_array(str)==task['chain'])]))
        if len(selected)!=1: raise Rejection('chain_mapping',f'Author chain maps to {selected}')
        chain=selected[0]; result['label_chain']=chain
        asym=cif.block['struct_asym']; poly=cif.block['entity_poly']
        entity=dict(zip(asym['id'].as_array(str),asym['entity_id'].as_array(str)))[chain]
        seqs=dict(zip(poly['entity_id'].as_array(str),poly['pdbx_seq_one_letter_code_can'].as_array(str)))
        result['sequence']=''.join(seqs[entity].split())
        result['polymer_entities']=poly.row_count
        # Explicit model selection makes this an auditable candidate, not an
        # implicit ensemble average. All model IDs remain in provenance.
        models=cat['pdbx_PDB_model_num'].as_array(str)
        result['deposited_models']=list(dict.fromkeys(models)); first=models[0]
        new=pdbx.CIFCategory()
        for name in cat: new[name]=pdbx.CIFColumn(cat[name].as_array(str)[models==first])
        cif.block['atom_site']=new
        cif,conformers=resolve(cif,selected); atomic_json(root/'conformers.json',conformers)
        cif.write(root/'resolved.cif')
        atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
        author=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=True)
        evidence=inventory(cif,{'A':selected,'B':[]}); atomic_json(root/'input_atom_mask.json',evidence)
        result['missing_residue_count']=sum(d['length'] for d in evidence['defects'] if d['kind'] in ('terminal_gap','internal_gap'))
        result['alternate_residue_count']=len(conformers['records'])
        protein_chains=sorted(set(atoms.chain_id[np.isin(atoms.res_name,list(CANONICAL))]))
        result['omitted_protein_chains']=[c for c in protein_chains if c!=chain]
        if len(atoms)>50000: raise Rejection('context_size_cap','Deposited context >50000 atoms; requires assembly-specific handling')
        # Shared component chemistry. For monomer buffer annotations the empty
        # partner tree gives infinity, replaced below by an explicit null field.
        removals,_=classify_components(atoms,cif,{'A':selected,'B':[]})
        for c in removals:
            if c['policy_class']=='buffer':
                c.update(train_radius_A=15.,eval_radius_A=20.,bridges_partners=None,exposed_nonbridging_annotation=None)
                c['partner_distances_A']={'A':c['partner_distances_A']['A']}
        atomic_json(root/'removed_components.json',removals)
        atoms.res_id=author.res_id.copy(); atoms.ins_code=author.ins_code.copy()
        atoms=atoms[(atoms.chain_id==chain)&np.isin(atoms.res_name,list(CANONICAL))&~np.isin(np.char.upper(atoms.element),['H','D'])]
        if len(struc.get_residue_starts(atoms))>1500: raise Rejection('size_cap','Selected chain >1500 residues')
        write_cif(root/'observed.cif',atoms)
        fixed=complete(atoms,str(R/'envs/pypka/bin/pdb2pqr30'),root); t=topology(fixed)
        write_cif(root/'prepared.cif',fixed)
        result['prepared_sha256']=digest(root/'prepared.cif')
        residues={(k.chain,k.number,k.insertion):t.residue(i) for i,k in enumerate(t.keys)}
        bylabel={(r['key'][0],r['label_seq_id']):tuple(r['key']) for r in evidence['atoms']}
        gap_data=[]
        for d in evidence['defects']:
            if d['kind'] not in ('terminal_gap','internal_gap'): continue
            internal=d['kind']=='internal_gap'
            radius=(15. if d['length']<=3 else None) if internal else terminal_radius(d['length'])
            keys=[bylabel[chain,p] for p in (d['start']-1,d['end']+1) if (chain,p) in bylabel]
            anchors=[]
            if radius is not None and len(keys)==(2 if internal else 1):
                for k in keys:
                    rr=atoms[(atoms.chain_id==k[0])&(atoms.res_id==k[1])&(atoms.ins_code==k[2])]
                    bb=rr.coord[np.isin(rr.atom_name,['N','CA','C','O'])]
                    if len(bb)!=4: radius=None
                    anchors.extend(bb)
            else: radius=None
            gap_data.append((d,radius,np.asarray(anchors)))
        # SASA is observational metadata, not a new admission threshold.
        area=struc.sasa(fixed,probe_radius=1.4,point_number=1000)
        for record in records:
            raw=record['raw']; group={'C-term':'CTERM','N-term':'NTERM'}.get(raw['ResName'],raw['ResName'])
            match=re.fullmatch(r'(-?\d+)([A-Za-z]?)',str(raw['ResID_in_PDB']).strip())
            rr=dict(record_id=record['record_id'],mapped=False,structural_train_mask=False,structural_eval_mask=False)
            if not match or group not in SITE_ATOMS:
                rr['reason']='unsupported_site_identifier'; result['records'].append(rr); continue
            num=int(match[1]); ins=match[2]; key=(chain,num,ins); residue=residues.get(key)
            if residue is None or (group not in ('NTERM','CTERM') and str(residue.res_name[0])!=group):
                rr['reason']='residue_absent_or_identity_mismatch'; result['records'].append(rr); continue
            points=residue.coord[np.isin(residue.atom_name,SITE_ATOMS[group])]
            complete_site=set(SITE_ATOMS[group])<=set(residue.atom_name)
            terminal=group in ('NTERM','CTERM')
            artificial=terminal and list(key) in evidence['artificial_terminal_keys']
            # Check true terminal position using deposited polymer numbering.
            labelpos=next((a['label_seq_id'] for a in evidence['atoms'] if tuple(a['key'])==key),None)
            true_terminal=not terminal or labelpos==(1 if group=='NTERM' else len(result['sequence']))
            eligible=bool(complete_site and not artificial and true_terminal)
            details=[]; near=False; unknown=False; clean=True
            for d,radius,anchors in gap_data:
                lower=clearance(points,d); clean &= lower>=20
                distance=float(np.linalg.norm(points[:,None]-anchors[None,:],axis=-1).min()) if radius is not None and len(points) else None
                near |= distance is not None and distance<radius
                unknown |= radius is None and lower<20
                details.append(dict(kind=d['kind'],length=d['length'],anchor_distance_A=distance,radius_A=radius,lower_bound_A=lower))
            tier='ineligible' if not eligible else 'near_gap' if near else 'uncalibrated' if unknown else 'clean' if clean else 'uncertain'
            distances=[float(np.linalg.norm(points[:,None]-np.asarray(c['coordinates'])[None,:],axis=-1).min()) if len(points) else 0. for c in removals]
            mask=tier in ('clean','uncertain')
            target=(fixed.chain_id==chain)&(fixed.res_id==num)&(fixed.ins_code==ins)
            rr.update(mapped=True,chain=chain,resnum=num,icode=ins,group=group,natural_gap_tier=tier,gap_details=details,
                structural_train_mask=bool(mask and all(d>=c['train_radius_A'] for d,c in zip(distances,removals))),
                structural_eval_mask=bool(mask and all(d>=c['eval_radius_A'] for d,c in zip(distances,removals))),
                nearest_component_A=min(distances,default=None),residue_sasa_A2=float(np.nansum(area[target])),
                functional_atoms_complete=complete_site,artificial_terminus=artificial,
                reason='provisional_structural_candidate')
            result['records'].append(rr)
        result.update(status='prepared',component_count=len(removals),topology_gaps=t.metadata['gaps'])
    except Exception as exc:
        result.update(status='held',reason=getattr(exc,'code',type(exc).__name__),detail=str(exc))
        (root/'error.txt').write_text(traceback.format_exc())
    atomic_json(OUT/'rows'/f"{task['task_id']}.json",result)
    print(task['pdb'],task['chain'],result['status'],result.get('reason',''),flush=True)

def shard(number,total):
    records=json.loads((OUT/'records.json').read_text()); tasks=json.loads((OUT/'tasks.json').read_text())
    for task in tasks[number::total]:
        run_task(task,[r for r in records if r['task_id']==task['task_id']])

if __name__=='__main__':
    require_compute()
    if sys.argv[1]=='init': initialise()
    else: shard(int(sys.argv[2]),int(sys.argv[3]))
