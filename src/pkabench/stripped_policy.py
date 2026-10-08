"""Versioned stripped-protein reference with component uncertainty masks."""
import json
import os
from pathlib import Path
from collections import Counter
from .runtime import require_compute,atomic_json,digest

POLICY={'version':'stripped-components-v1','ligand_train_A':15.,'ligand_eval_A':25.,'metal_radius_A':25.,
    'metal_min_exposed_fraction':.7,'metal_donor_cutoff_A':3.,'ligand_short_contact_A':1.9,
    'metal_rule':'Only monatomic metals/ions; >=70% exposed versus isolated, no declared covalent/metal connection and no nonwater N/O/S/Se within 3 A in source assembly.',
    'ligand_rule':'Noncovalent nonprotein ligands removed with 15/25 A uncertainty masks. Declared covalent/metal-linked or <1.9 A protein contacts excluded. Glycans and nonstandard residues remain separate exclusions.',
    'scope':'Stripped protein-only reference; operational masks, not validated physical error bounds. Production split/teacher validity still required.'}


def initialise(source,out):
    require_compute(); source=Path(source); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    index=json.loads((source/'index.json').read_text())
    atomic_json(out/'manifest.json',{'candidates':index['candidates'],'source_audit':str(source),'policy':POLICY,
        'implementation_sha256':digest(Path(__file__)),'source_index_sha256':digest(source/'index.json'),'production_allowed':False})


def scan(campaign,shard,shards):
    require_compute()
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from scipy.spatial import cKDTree
    from .prep import CANONICAL,Rejection,read_cif
    from .conformers import resolve
    from .curation import prepare_revised,validate_partners
    from .audit import component_inventory
    from .dataset_audit import filtered_cif
    from .schema import write_table
    from .annotate import SITE_ATOMS
    campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text())
    if manifest['policy']!=POLICY or manifest['implementation_sha256']!=digest(Path(__file__)): raise ValueError('Frozen stripped policy changed')
    for row in manifest['candidates'][shard::shards]:
        cid=row['complex_id']; receipt=campaign/'rows'/f'{cid}.json'
        if receipt.exists(): continue
        root=campaign/'structures'/cid; root.mkdir(parents=True,exist_ok=True)
        source=Path(manifest['source_audit'])/'sources'/f"{row['pdb_id']}.cif"; result=dict(row)
        try:
            if digest(source)!=row['source_sha256']: raise ValueError('Source hash mismatch')
            partners={p:row[f'partner_{p}_chains'] for p in ('A','B')}; selected=partners['A']+partners['B']
            cif,conformers=resolve(pdbx.CIFFile.read(source),selected); resolved=root/'original-resolved.cif'; cif.write(resolved)
            atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
            validate_partners(atoms,partners)
            heavy=~np.isin(np.char.upper(atoms.element),['H','D']); water=np.isin(atoms.res_name,['HOH','WAT','H2O','DOD'])
            protein=np.isin(atoms.res_name,list(CANONICAL))&heavy
            tree=cKDTree(atoms.coord[protein]); components=component_inventory(atoms,cif,selected)
            # The legacy cleaner silently drops these ions: inventory them here.
            starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
            for s,e in zip(starts[:-1],starts[1:]):
                if e-s==1 and str(atoms.res_name[s]) in ('NA','K','CL'):
                    components.append({'chain':str(atoms.chain_id[s]),'resnum':int(atoms.res_id[s]),'name':str(atoms.res_name[s]),'code':'metal','start':int(s),'end':int(e)})
            protected=set(); conn=cif.block.get('struct_conn')
            if conn is not None:
                for i,kind in enumerate(conn['conn_type_id'].as_array(str)):
                    if str(kind).lower().startswith(('covale','metalc')):
                        for p in ('ptnr1','ptnr2'):
                            c,n=p+'_label_asym_id',p+'_label_comp_id'
                            if c in conn and n in conn: protected.add((str(conn[c].as_array(str)[i]),str(conn[n].as_array(str)[i])))
            removals=[]
            for c in components:
                if c['code'] not in ('ligand','metal'): raise Rejection(c['code'],f"component {c['name']} remains outside stripped policy")
                a=atoms[c['start']:c['end']]; a=a[~np.isin(np.char.upper(a.element),['H','D'])]
                declared=(c['chain'],c['name']) in protected
                if declared: raise Rejection('connected_component',f"{c['name']} has a declared covalent/metal connection")
                d=float(tree.query(a.coord)[0].min())
                rec=dict(c); rec.update(coordinates=a.coord.astype(float).tolist(),min_protein_distance_A=d,declared_connection=declared)
                if c['code']=='metal':
                    if len(a)!=1: raise Rejection('metal_complex',f"{c['name']} is not a monatomic ion")
                    donor=heavy&~water&np.isin(np.char.upper(atoms.element),['N','O','S','SE']); donor[c['start']:c['end']]=False
                    donor_distance=float(np.linalg.norm(atoms.coord[donor]-a.coord[0],axis=1).min()) if donor.any() else None
                    if donor_distance is not None and donor_distance<=3: raise Rejection('coordinated_metal',f"{c['name']} donor distance {donor_distance:.3f} A")
                    environment=atoms[heavy&~water]; isolated=float(struc.sasa(a,ignore_ions=False,vdw_radii='Single').sum())
                    # Evaluate the same ion in the full nonwater environment.
                    indexes=np.flatnonzero(heavy&~water); positions=np.flatnonzero((indexes>=c['start'])&(indexes<c['end']))
                    try: exposed=float(struc.sasa(environment,ignore_ions=False,vdw_radii='Single')[positions].sum())/isolated
                    except (KeyError,ValueError): raise Rejection('metal_sasa_unavailable',c['name'])
                    if not np.isfinite(exposed) or exposed<.7: raise Rejection('buried_metal',f"{c['name']} exposed fraction {exposed}")
                    rec.update(exposed_fraction=exposed,nearest_donor_A=donor_distance,train_radius_A=25.,eval_radius_A=25.)
                else:
                    if d<1.9: raise Rejection('connected_component',f"{c['name']} short protein contact {d:.3f} A")
                    rec.update(train_radius_A=15.,eval_radius_A=25.)
                removals.append(rec)
            cat=cif.block['atom_site']; cc=cat['label_asym_id'].as_array(str); nn=cat['label_comp_id'].as_array(str)
            remove_keys={(c['chain'],c['name']) for c in removals}
            keep=np.array([(str(c),str(n)) not in remove_keys for c,n in zip(cc,nn)])
            modified=root/'stripped-source.cif'; filtered_cif(resolved,keep).write(modified)
            item={**row,'source_sha256':digest(modified)}
            structure,sites,extra=prepare_revised(modified,item,root)
            fixed=read_cif(root/'AB.cif'); ss=struc.get_residue_starts(fixed,add_exclusive_stop=True)
            residues={(str(fixed.chain_id[s]),int(fixed.res_id[s]),str(fixed.ins_code[s]).strip()):fixed[s:e] for s,e in zip(ss[:-1],ss[1:])}
            masks=[]
            for site in sites:
                a=residues[site['chain'],site['resnum'],site['icode']]; points=a.coord[np.isin(a.atom_name,SITE_ATOMS[site['group']])]
                ds=[float(np.linalg.norm(points[:,None]-np.array(c['coordinates'])[None,:],axis=-1).min()) if len(points) else 0 for c in removals]
                eligible=site['functional_atoms_complete'] and not site['is_break_terminus']
                masks.append({k:site[k] for k in ('complex_id','chain','resnum','icode','group')}|{'component_train_mask':bool(eligible and all(d>=c['train_radius_A'] for d,c in zip(ds,removals))),
                    'component_eval_mask':bool(eligible and all(d>=c['eval_radius_A'] for d,c in zip(ds,removals))),
                    'nearest_removed_A':min(ds,default=None)})
            atomic_json(root/'removed_components.json',{'policy':POLICY,'source_sha256':row['source_sha256'],'resolved_sha256':digest(resolved),'components':removals})
            atomic_json(root/'component_masks.json',masks)
            provenance=json.loads(structure['provenance']); provenance.update(stripping_policy=POLICY,removed_components_sha256=digest(root/'removed_components.json'),label_scope='stripped protein-only reference',original_source_sha256=row['source_sha256'])
            structure['provenance']=json.dumps(provenance); atomic_json(root/'provenance.json',provenance)
            write_table(root/'sites.parquet','sites',sites)
            result.update(status='accepted',structure=structure,sites=len(sites),components_removed=len(removals))
        except Rejection as exc: result.update(status='rejected',code=exc.code,detail=str(exc),stage=exc.stage)
        except Exception as exc:
            import traceback
            result.update(status='pipeline_error',detail=str(exc),traceback=traceback.format_exc())
        atomic_json(receipt,result)
    print(json.dumps({'shard':shard,'status':'complete'}))


def collect(campaign):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .schema import read_table,write_table,KEY
    from .anchor_tiers import apply
    campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text())
    rows=[json.loads((campaign/'rows'/f"{r['complex_id']}.json").read_text()) for r in manifest['candidates']]
    accepted=[r for r in rows if r['status']=='accepted']; sites=[s for r in accepted for s in read_table(campaign/'structures'/r['complex_id']/'sites.parquet')]
    write_table(campaign/'structures.parquet','structures',[r['structure'] for r in accepted]); write_table(campaign/'sites.parquet','sites',sites)
    natural=campaign/'natural-gap-tiers'; apply(campaign,natural)
    key=lambda r:tuple(r[k] for k in KEY)
    base={key(r):r for r in pq.read_table(natural/'site_tiers.parquet').to_pylist()}; final=[]
    for r in accepted:
        for s in json.loads((campaign/'structures'/r['complex_id']/'component_masks.json').read_text()):
            gap=base[key(s)]; final.append(s|{'natural_gap_tier':gap['tier'],'interface':gap['interface'],
                'train_mask':s['component_train_mask'] and gap['provisional_retained'],
                'eval_mask':s['component_eval_mask'] and gap['provisional_retained']})
    pq.write_table(pa.Table.from_pylist(final),campaign/'site_masks.parquet')
    report={'candidates':len(rows),'accepted':len(accepted),'histogram':dict(Counter(r.get('code',r['status']) for r in rows)),
        'pipeline_errors':[{'complex_id':r['complex_id'],'detail':r['detail']} for r in rows if r['status']=='pipeline_error'],
        'training_sites':sum(r['train_mask'] for r in final),'evaluation_sites':sum(r['eval_mask'] for r in final),
        'training_interface_sites':sum(r['train_mask'] and r['interface'] for r in final),'evaluation_interface_sites':sum(r['eval_mask'] and r['interface'] for r in final),
        'complexes_with_training':len({r['complex_id'] for r in final if r['train_mask']}),'mask_sha256':digest(campaign/'site_masks.parquet'),
        'limits':'Structural masks before teacher coverage and split assignment. Natural-gap tiers retain clean+uncertain for both splits; component radii distinguish train/eval. Long gaps uncalibrated, conservative fallback retained. Production labels are stripped protein-only.'}
    atomic_json(campaign/'report.json',report); print(json.dumps(report,indent=2))
