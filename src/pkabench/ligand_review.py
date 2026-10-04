"""Diagnostic preparation after explicit candidate-component removal."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from .runtime import require_compute,atomic_json,digest


def run(audit,out,shard,shards):
    require_compute()
    import numpy as np
    from biotite.structure.io import pdbx
    from .dataset_audit import filtered_cif
    from .curation import prepare_revised
    from .prep import Rejection
    from .schema import write_table
    audit=Path(audit); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((audit/'manifest.json').read_text()); campaign=Path(provenance['campaign'])
    pairs=[p for p in json.loads((audit/'pairs.json').read_text()) if p['remote_or_additives_only']]
    components=defaultdict(list)
    with (audit/'components.csv').open() as f:
        for r in csv.DictReader(f): components[r['complex_id']].append(r)
    for pair in pairs[shard::shards]:
        cid=pair['complex_id']; work=out/cid; work.mkdir(exist_ok=False)
        source=campaign/'structures'/cid/'resolved-source.cif'
        if digest(source)!=provenance['sources'][cid]: raise ValueError('Audit source changed')
        row=json.loads((campaign/'rows'/f'{cid}.json').read_text())
        removals=[r for r in components[cid] if r['already_removable']!='True']
        assert removals and all(r['category'] in ('remote_gt10A','additive_candidate_near') and r['protected']=='False' for r in removals)
        cif=pdbx.CIFFile.read(source); cat=cif.block['atom_site']; chains=cat['label_asym_id'].as_array(str); names=cat['label_comp_id'].as_array(str)
        chem=cif.block.get('chem_comp'); metadata={}
        if chem is not None:
            for i,name in enumerate(chem['id'].as_array(str)):
                metadata[str(name)]={k:str(chem[k].as_array(str)[i]) for k in ('name','type','pdbx_formal_charge') if k in chem}
        for r in removals:
            r['ccd']=metadata.get(r['name'],{})
            r['review_class']='neutral_additive_candidate' if r['name'] in {'GOL','EDO','PEG','PGE','PG4','1PE','MPD','DMS'} else 'buffer_ion_or_other_review' if r['additive_candidate']=='True' else 'remote_other_ligand'
        keys={(r['chain'],r['name']) for r in removals}
        keep=np.array([(str(c),str(n)) not in keys for c,n in zip(chains,names)])
        assert keep.any() and (~keep).any()
        # Inventory includes every instance, so chain/name deletion cannot silently
        # remove a protected copy with the same identifier.
        assert all(r['protected']=='False' for r in components[cid] if (r['chain'],r['name']) in keys)
        modified=work/'counterfactual.cif'; filtered_cif(source,keep).write(modified)
        record={'complex_id':cid,'pdb_id':pair['pdb_id'],'source_sha256':digest(source),
            'modified_sha256':digest(modified),'removed':removals,'removed_atom_rows':int((~keep).sum()),
            'review_classes':sorted({r['review_class'] for r in removals}),'production_allowed':False}
        try:
            item={k:row[k] for k in ('complex_id','pdb_id','partner_A_chains','partner_B_chains')}; item['source_sha256']=digest(modified)
            structure,sites,extra=prepare_revised(modified,item,work/'prepared')
            write_table(work/'sites.parquet','sites',sites)
            record.update(status='prepared',n_sites=len(sites),n_residues=structure['n_residues'])
        except Rejection as exc: record.update(status='rejected',code=exc.code,detail=str(exc))
        except Exception as exc: record.update(status='pipeline_error',detail=str(exc))
        atomic_json(work/'result.json',record)
    print(json.dumps({'shard':shard,'attempted':len(pairs[shard::shards])}))


def sasa(audit,out):
    require_compute()
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from .prep import CANONICAL
    from .audit import component_inventory
    audit=Path(audit); out=Path(out); dest=out/'buffer-sasa'; dest.mkdir(exist_ok=False)
    campaign=Path(json.loads((audit/'manifest.json').read_text())['campaign']); rows=[]
    for path in sorted(out.glob('*/result.json')):
        result=json.loads(path.read_text()); requested={(r['chain'],r['name']) for r in result['removed'] if r['review_class']=='buffer_ion_or_other_review'}
        if not requested: continue
        cid=result['complex_id']; source=campaign/'structures'/cid/'resolved-source.cif'
        assert digest(source)==result['source_sha256']
        selection=json.loads((campaign/'rows'/f'{cid}.json').read_text()); chains=selection['partner_A_chains']+selection['partner_B_chains']
        cif=pdbx.CIFFile.read(source); atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
        protein=atoms[np.isin(atoms.chain_id,chains)&np.isin(atoms.res_name,list(CANONICAL))&~np.isin(np.char.upper(atoms.element),['H','D'])]
        for component in component_inventory(atoms,cif,chains):
            if (component['chain'],component['name']) not in requested: continue
            ligand=atoms[component['start']:component['end']]; ligand=ligand[~np.isin(np.char.upper(ligand.element),['H','D'])]
            isolated=float(np.sum(struc.sasa(ligand,probe_radius=1.4,point_number=1000,ignore_ions=False,vdw_radii='Single')))
            together=protein+ligand
            bound=float(np.sum(struc.sasa(together,probe_radius=1.4,point_number=1000,ignore_ions=False,vdw_radii='Single')[-len(ligand):]))
            if not np.isfinite(bound+isolated) or isolated<=0: raise ValueError('Invalid SASA')
            rows.append({'complex_id':cid,'pdb_id':result['pdb_id'],'chain':component['chain'],'resnum':component['resnum'],'name':component['name'],
                'preparation_status':result['status'],'sasa_with_selected_proteins_A2':bound,'isolated_sasa_A2':isolated,
                'buried_by_selected_proteins_A2':isolated-bound,'exposed_fraction':bound/isolated})
    with (dest/'components.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for name in sorted({r['name'] for r in rows}):
        rr=[r for r in rows if r['name']==name]; bound=[r['sasa_with_selected_proteins_A2'] for r in rr]; frac=[r['exposed_fraction'] for r in rr]
        summary.append({'name':name,'pair_component_observations':len(rr),'pairs':len({r['complex_id'] for r in rr}),
            'sasa_median_A2':float(np.median(bound)),'sasa_range_A2':[min(bound),max(bound)],
            'exposed_fraction_median':float(np.median(frac)),'exposed_fraction_range':[min(frac),max(frac)]})
    report={'by_component':summary,'method':'Heavy atoms; 1.4 A probe, 1000 Fibonacci points, element-specific Single radii, ions included. SASA of each component with selected protein partners only versus same isolated component. Other species, omitted proteins and water excluded. Observations repeat across pairs. Exposure is not a pKa perturbation estimate.',
        'source_audit':str(audit),'code_sha256':digest(Path(__file__)),'pkai_limitation':'Installed protein.py only parses ATOM records; normal HETATM buffers/ligands ignored.'}
    atomic_json(dest/'report.json',report); print(json.dumps(report,indent=2))


def grid(audit,out):
    require_compute()
    import numpy as np
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from .prep import read_cif
    from .schema import read_table
    from .annotate import SITE_ATOMS
    from .anchor_tiers import classify
    audit=Path(audit); out=Path(out); dest=out/'buffer-threshold-grid-v1'; dest.mkdir(exist_ok=False)
    campaign=Path(json.loads((audit/'manifest.json').read_text())['campaign'])
    with (out/'buffer-sasa/components.csv').open() as f: exposure=list(csv.DictReader(f))
    metrics=[]; sites_out=[]; pairdata=[]; skipped=[]
    for path in sorted(out.glob('*/result.json')):
        result=json.loads(path.read_text()); cid=result['complex_id']
        if 'buffer_ion_or_other_review' not in result['review_classes']: continue
        if result['status']!='prepared': skipped.append({'complex_id':cid,'reason':result.get('code',result['status'])}); continue
        root=path.parent; atoms=read_cif(root/'prepared/AB.cif')
        starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
        residues={(str(atoms.chain_id[s]),int(atoms.res_id[s]),str(atoms.ins_code[s]).strip()):atoms[s:e] for s,e in zip(starts[:-1],starts[1:])}
        sites=[s for s in read_table(root/'sites.parquet') if s['functional_atoms_complete'] and not s['is_break_terminus']]
        points=[]
        for site in sites:
            a=residues[site['chain'],site['resnum'],site['icode']]; p=a.coord[np.isin(a.atom_name,SITE_ATOMS[site['group']])]
            assert len(p); points.append(p)
        source=campaign/'structures'/cid/'resolved-source.cif'; assert digest(source)==result['source_sha256']
        original=pdbx.get_structure(pdbx.CIFFile.read(source),model=1,altloc='occupancy',use_author_fields=False)
        component_rows=[]; distances=[]
        seen=set()
        for r in result['removed']:
            if r['review_class']!='buffer_ion_or_other_review': continue
            key=(r['chain'],int(r['resnum']),r['name'])
            if key in seen: continue
            seen.add(key)
            xyz=original.coord[(original.chain_id==key[0])&(original.res_id==key[1])&(original.res_name==key[2])&~np.isin(np.char.upper(original.element),['H','D'])]
            assert len(xyz)
            match=[e for e in exposure if e['complex_id']==cid and (e['chain'],int(e['resnum']),e['name'])==key]
            assert len(match)==1
            dd=np.array([float(np.linalg.norm(p[:,None]-xyz[None,:],axis=-1).min()) for p in points]); distances.append(dd)
            rec={'complex_id':cid,'pdb_id':result['pdb_id'],'chain':key[0],'resnum':key[1],'name':key[2],
                'exposed_fraction':float(match[0]['exposed_fraction']),'sasa_A2':float(match[0]['sasa_with_selected_proteins_A2']),
                'nearest_native_titratable_site_A':float(dd.min()) if len(dd) else None,'protected':r['protected']=='True',
                'ccd_formal_charge':r.get('ccd',{}).get('pdbx_formal_charge','unknown')}
            metrics.append(rec); component_rows.append(rec)
        if not component_rows: continue
        distance=np.min(distances,axis=0)
        # Require every newly removed species to be in this measured buffer group;
        # other diagnostic removals cannot be silently approved by this grid.
        buffer_only=result['review_classes']==['buffer_ion_or_other_review']
        for s,d in zip(sites,distance):
            sites_out.append({k:s[k] for k in ('complex_id','chain','resnum','icode','group')}|{'nearest_removed_buffer_A':float(d),'mask_10A':bool(d>=10),'mask_20A':bool(d>=20),'interface':s['residue_delta_sasa']>10})
        pairdata.append({'complex_id':cid,'buffer_only':buffer_only,'components':component_rows,'distances':distance,'interface':np.array([s['residue_delta_sasa']>10 for s in sites])})
    rows=[]
    for fraction in (.7,.8,.9,.95):
        for cutoff in (5.,10.,15.,20.):
            selected=[p for p in pairdata if p['buffer_only'] and all(not c['protected'] and c['exposed_fraction']>=fraction and c['nearest_native_titratable_site_A'] is not None and c['nearest_native_titratable_site_A']>=cutoff for c in p['components'])]
            passing=[c for c in metrics if not c['protected'] and c['exposed_fraction']>=fraction and c['nearest_native_titratable_site_A'] is not None and c['nearest_native_titratable_site_A']>=cutoff]
            for radius in (10.,20.):
                rows.append({'min_exposed_fraction':fraction,'min_titratable_distance_A':cutoff,'uncertainty_radius_A':radius,
                    'passing_component_observations':len(passing),'eligible_pairs':len(selected),
                    'native_sites_before_buffer_mask':sum(len(p['distances']) for p in selected),
                    'retained_sites':sum(int((p['distances']>=radius).sum()) for p in selected),
                    'retained_interface_sites':sum(int(((p['distances']>=radius)&p['interface']).sum()) for p in selected),
                    'complex_ids':'|'.join(p['complex_id'] for p in selected)})
    for name,data in [('components',metrics),('site_masks',sites_out),('threshold_grid',rows)]:
        if data:
            with (dest/f'{name}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    report={'prepared_pairs_with_buffer_candidates':len(pairdata),'prepared_buffer_only_pairs':sum(p['buffer_only'] for p in pairdata),
        'excluded_preparation_failures':skipped,'component_observations':len(metrics),'grid':rows,
        'limits':'Only buffer candidates in the 48-pair removal pilot; requires all newly removed components to pass and disallows mixed-species removal. Distances use prepared native titratable functional atoms, including completed atoms, ARG and native termini. Missing residues are not represented. Site counts precede missing-residue masks, teacher coverage and splits. 10/20 A masks are sensitivity scenarios, not validated ligand-error cutoffs. No production admission or policy change.',
        'code_sha256':digest(Path(__file__)),'source_sasa_sha256':digest(out/'buffer-sasa/components.csv')}
    atomic_json(dest/'report.json',report); print(json.dumps(report,indent=2))


def collect(audit,out):
    require_compute(); audit=Path(audit); out=Path(out)
    pairs=[p for p in json.loads((audit/'pairs.json').read_text()) if p['remote_or_additives_only']]
    rows=[json.loads((out/p['complex_id']/'result.json').read_text()) for p in pairs]
    classes=defaultdict(Counter); names=defaultdict(set)
    for r in rows:
        classes[' + '.join(r['review_classes'])][r['status']]+=1
        for c in r['removed']: names[c['name']].add(r['complex_id'])
    report={'pairs':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
        'rejection_codes':dict(Counter(r['code'] for r in rows if r['status']=='rejected')),
        'review_classes':{k:dict(v) for k,v in classes.items()},
        'removed_names_by_pairs':{k:len(v) for k,v in sorted(names.items(),key=lambda t:-len(t[1]))},
        'pipeline_errors':[r for r in rows if r['status']=='pipeline_error'],
        'limits':'Preparation feasibility only. No pKa equivalence test, no production admission or removal approval. Remote components can affect electrostatics. CCD formal charge retained where supplied; absence is not neutrality. Neutral-additive grouping is a review shortlist. Original coordinates preserved separately.',
        'production_policy_changed':False}
    atomic_json(out/'report.json',report)
    atomic_json(out/'manifest.json',{'audit':str(audit),'audit_manifest_sha256':digest(audit/'manifest.json'),
        'code_sha256':digest(Path(__file__)),'results':{r['complex_id']:digest(out/r['complex_id']/'result.json') for r in rows}})
    print(json.dumps(report,indent=2))
