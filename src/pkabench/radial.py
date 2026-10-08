"""Measured radial PB error after terminal and buried whole-residue deletion."""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree
from .annotate import SITE_ATOMS
from .curation import prepare_revised, POLICY
from .dataset_audit import filtered_cif
from .prep import read_cif, Rejection
from .runtime import atomic_json, digest, require_compute
from .schema import read_table, write_table

BINS=(0,5,10,15,20,30,float('inf'))


def residue_key(residue):
    return (str(residue.chain_id[0]),int(residue.res_id[0]),str(residue.ins_code[0]).strip())


def deletion_plan(atoms, *, expanded=False, partners=None):
    """Exposure measured before deleting coordinates; no inferred tail position."""
    sasa=np.asarray(struc.sasa(atoms,probe_radius=1.4,point_number=1000),float)
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    chains=defaultdict(list)
    for s,e in zip(starts[:-1],starts[1:]):
        residue=atoms[s:e]; denom=float(np.nansum(struc.sasa(residue,probe_radius=1.4,point_number=1000)))
        fraction=float(np.nansum(sasa[s:e]))/denom if denom else 0.
        chains[str(residue.chain_id[0])].append({'key':list(residue_key(residue)),
            'name':str(residue.res_name[0]),'exposure_fraction':fraction,'coordinates':residue.coord.astype(float).tolist()})
    variants={}; buried=[]
    # One deterministic chain for terminal dose series, avoiding multiplicity bias.
    chain=next((c for c in sorted(chains) if len(chains[c])>=20),None)
    if chain:
        rr=chains[chain]
        for end in ('N','C'):
            for length in ((1,3,5,10) if expanded else (1,3)):
                removed=rr[:length] if end=='N' else rr[-length:]
                variants[f'{end}{length}']={'kind':'terminal','end':end,'length':length,'residues':removed}
    for chain,rr in chains.items():
        for i in range(3,len(rr)-3):
            if rr[i]['name']=='CYS': continue
            if rr[i]['exposure_fraction']<=.1:
                buried.append((rr[i]['exposure_fraction'],chain,i,rr))
    if buried:
        _,chain,i,rr=min(buried,key=lambda r:(r[0],r[1],r[2]))
        variants['buried1']={'kind':'buried','length':1,'residues':[rr[i]]}
        triples=[b for b in buried if all(r['exposure_fraction']<=.2 and r['name']!='CYS' for r in b[3][b[2]-1:b[2]+2])]
        if triples:
            _,chain,i,rr=min(triples,key=lambda r:(r[0],r[1],r[2]))
            variants['buried3']={'kind':'buried','length':3,'residues':rr[i-1:i+2]}
    if expanded:
        if not partners: raise ValueError('expanded deletions require explicit partners')
        partner_trees={p:cKDTree(atoms.coord[np.isin(atoms.chain_id,cc)]) for p,cc in partners.items()}
        for chain,rr in chains.items():
            other='B' if chain in partners['A'] else 'A'
            for residue in rr:
                residue['partner_distance']=float(partner_trees[other].query(residue['coordinates'])[0].min())
        charged=[b for b in buried if b[3][b[2]]['name'] in ('ASP','GLU','HIS','LYS','ARG')]
        if charged:
            _,chain,i,rr=min(charged,key=lambda b:(b[0],b[1],b[2]))
            variants['buried_charged1']={'kind':'buried_charged','length':1,'residues':[rr[i]]}
        # Windows stay internal and avoid CYS to avoid conflating disulfide removal.
        for location in ('near','remote'):
            windows=[]
            for chain,rr in chains.items():
                for i in range(3,len(rr)-5):
                    window=rr[i:i+3]
                    if any(r['name']=='CYS' for r in window): continue
                    distance=min(r['partner_distance'] for r in window)
                    if (location=='near' and distance<=5) or (location=='remote' and distance>=15):
                        windows.append((sum(r['exposure_fraction'] for r in window),chain,i,window))
            if windows:
                window=min(windows,key=lambda w:(w[0],w[1],w[2]))[3]
                variants[f'internal_{location}3']={'kind':f'internal_{location}','length':3,'residues':window}
    for variant in variants.values():
        variant['deleted_ionisable_residues']=sum(r['name'] in ('ASP','GLU','HIS','LYS','ARG','CYS','TYR') for r in variant['residues'])
        variant['removed_coordinates']=[xyz for r in variant['residues'] for xyz in r['coordinates']]
        variant['reference_exposure_mean']=float(np.mean([r['exposure_fraction'] for r in variant['residues']]))
        fractions=[r['exposure_fraction'] for r in variant['residues']]
        variant['reference_exposure_class']='exposed' if min(fractions)>=.4 else 'buried' if max(fractions)<=.1 else 'intermediate'
        if expanded:
            variant['minimum_partner_distance']=min(r['partner_distance'] for r in variant['residues'])
    return variants


def deletion_mask(cif,keys):
    cat=cif.block['atom_site']; keys={tuple(k) for k in keys}
    chains=cat['label_asym_id'].as_array(str); nums=cat['auth_seq_id'].as_array(str)
    ins=cat['pdbx_PDB_ins_code'].as_array(str)
    return np.asarray([(c,int(n),'' if i in ('?','.',' ') else i) not in keys for c,n,i in zip(chains,nums,ins)])


def initialise(campaign,out,count=3):
    require_compute(); campaign=Path(campaign); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((campaign/'manifest.json').read_text()); source_root=Path(parent['source_audit'])/'sources'
    rows=[json.loads(p.read_text()) for p in (campaign/'rows').glob('*.json')]
    eligible=[]
    for row in rows:
        if row['status']!='accepted' or not 80<=row['structure']['n_residues']<=350: continue
        work=campaign/'structures'/row['complex_id']; evidence=json.loads((work/'input_atom_mask.json').read_text())
        gaps=[d for d in evidence['defects'] if d['kind'] in ('terminal_gap','internal_gap')]
        missing=sum(d['length'] for d in gaps)
        incomplete=sum(d['kind']=='missing_atoms' for d in evidence['defects'])
        if any(d['kind']=='internal_gap' for d in gaps) or missing>2 or incomplete>2: continue
        if any(d['kind']=='removed_additive' for d in evidence['defects']): continue
        eligible.append((missing,incomplete,row['structure']['n_residues'],row))
    eligible.sort(key=lambda r:(r[0],r[1],r[2],r[3]['complex_id']))
    cases=[]; tasks=[]; failures=[]
    for missing,incomplete,n,row in eligible:
        cid=row['complex_id']; reference=read_cif(campaign/'structures'/cid/'AB.cif'); variants=deletion_plan(reference)
        if 'buried1' not in variants: continue
        source=source_root/f"{row['pdb_id']}.cif"
        case={'complex_id':cid,'pdb_id':row['pdb_id'],'n_residues':n,'original_missing_residues':missing,
            'original_incomplete_residues':incomplete,'variants':{},'unavailable_variants':[] if 'buried3' in variants else ['buried3']}
        for name,spec in [('baseline',None),('repeat',None),*variants.items()]:
            root=out/cid/name; work=root/'structures'/cid; work.mkdir(parents=True)
            modified=root/'source.cif'
            if spec:
                original=pdbx.CIFFile.read(source)
                filtered_cif(source,deletion_mask(original,[r['key'] for r in spec['residues']])).write(modified)
            else: modified.write_bytes(source.read_bytes())
            item={k:row[k] for k in ('complex_id','pdb_id','partner_A_chains','partner_B_chains')}; item['source_sha256']=digest(modified)
            try:
                structure,sites,extra=prepare_revised(modified,item,work,diagnostic_perturbation=spec is not None)
                write_table(work/'sites.parquet','sites',sites); write_table(root/'sites.parquet','sites',sites)
                write_table(root/'structures.parquet','structures',[structure])
                atomic_json(root/'manifest.json',{'candidates':[item],'variant':name,'perturbation':spec,
                    'parent_campaign':str(campaign),'policy':POLICY,'production_allowed':False})
                case['variants'][name]={'status':'prepared','perturbation':spec}
                tasks.append((str(root),cid))
            except Rejection as exc:
                case['variants'][name]={'status':'rejected','code':exc.code,'detail':str(exc),'perturbation':spec}
                failures.append({'complex_id':cid,'variant':name,'code':exc.code,'detail':str(exc)})
            except Exception:
                # An implementation failure must stop initialisation, not become
                # evidence that a biological deletion is unpreparable.
                raise
        cases.append(case)
        if len(cases)>=count: break
    result={'cases':cases,'tasks':tasks,'prep_failures':failures,'requested_cases':count,
        'selection':'Smallest near-complete references (<=2 missing terminal residues, <=2 incomplete residues, no internal gap/additive removal), ordered by completeness first.',
        'distance':'minimum distance from retained target functional-group atoms to deleted heavy atoms in the intact prepared reference',
        'metrics':'signed/absolute AB pKa error, free pKa error, and paired shift error; radial shells and outside-radius summaries; no preset distance mask',
        'scope':'Whole-residue deletions, matched perturbation in AB and corresponding free state; explicit new segment termini, no relaxation. Conditional on the prepared reference.',
        'production_allowed':False,'job':os.environ['SLURM_JOB_ID'],'implementation_sha256':digest(Path(__file__))}
    atomic_json(out/'manifest.json',result)
    (out/'tasks.tsv').write_text(''.join(f'{root}\t{cid}\n' for root,cid in tasks))
    print(json.dumps({'cases':[{k:v for k,v in c.items() if k!='variants'} for c in cases],'jobs':len(tasks),'prep_failures':failures},indent=2))
    if not cases: raise ValueError('no eligible reference structures')


def statistics(rows):
    errors=np.asarray([r['delta_pka_error'] for r in rows if r['status']=='ok'],float)
    result={'eligible_sites':len(rows),'paired_sites':len(errors),'complexes':len({r['complex_id'] for r in rows if r['status']=='ok'})}
    if not len(errors): return result
    absolute=np.abs(errors)
    result.update(mae=float(absolute.mean()),rmse=float(np.sqrt(np.mean(errors**2))),median=float(np.median(absolute)),
        p95=float(np.quantile(absolute,.95)),maximum=float(absolute.max()),
        fraction_over_0_1=float(np.mean(absolute>.1)),fraction_over_0_5=float(np.mean(absolute>.5)),
        ab_mae=float(np.mean([abs(r['ab_pka_error']) for r in rows if r['status']=='ok'])),
        free_mae=float(np.mean([abs(r['free_pka_error']) for r in rows if r['status']=='ok'])))
    by_complex=defaultdict(list)
    for row in rows:
        if row['status']=='ok': by_complex[row['complex_id']].append(abs(row['delta_pka_error']))
    result['mean_complex_mae']=float(np.mean([np.mean(v) for v in by_complex.values()]))
    return result


def radial_bin(distance):
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        if lo<=distance<hi: return f'{lo}-{hi}' if np.isfinite(hi) else f'{lo}+'
    raise ValueError('invalid radial distance')


def collect(out,completed_only=False):
    from .jobs import inputs
    require_compute(); out=Path(out); manifest=json.loads((out/'manifest.json').read_text()); results=[]; failures=[]; coverage=[]
    cases=manifest['cases']; output=out
    if completed_only:
        cases=[c for c in cases if all((out/c['complex_id']/v/'jobs/pypka'/f"{c['complex_id']}.json").exists()
            and (out/c['complex_id']/v/'jobs/pypka'/f"{c['complex_id']}.parquet").exists()
            for v,s in c['variants'].items() if s['status']=='prepared')]
        output=out/'partial'; output.mkdir(exist_ok=True)
    for case in cases:
        cid=case['complex_id']; root=out/cid; predictions={}
        for name,variant in case['variants'].items():
            if variant['status']!='prepared': continue
            parquet=root/name/'jobs/pypka'/f'{cid}.parquet'; receipt=parquet.with_suffix('.json')
            if not receipt.exists() or not parquet.exists(): raise ValueError(f'unfinished task {cid}/{name}')
            side=json.loads(receipt.read_text())
            if side['inputs']!=inputs(root/name,cid,'pypka') or side['output_sha256']!=digest(parquet): raise ValueError('radial provenance mismatch')
            if side['status']!='complete': failures.append({'complex_id':cid,'variant':name,'errors':side['errors']})
            predictions[name]={(r['chain'],r['resnum'],r['icode'],r['group'],r['state']):r for r in read_table(parquet)}
        if 'baseline' not in predictions: continue
        reference=predictions['baseline']; atoms=read_cif(root/'baseline/structures'/cid/'AB.cif')
        starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
        residues={residue_key(atoms[s:e]):atoms[s:e] for s,e in zip(starts[:-1],starts[1:])}
        sites=read_table(root/'baseline/sites.parquet')
        for site in sites:
            if site['is_break_terminus'] or not site['functional_atoms_complete']: continue
            key=tuple(site[k] for k in ('chain','resnum','icode','group'))
            pair=[reference.get((*key,state)) for state in ('AB',site['partner'])]
            coverage.append({'complex_id':cid,'group':site['group'],'paired':all(p and p['status']=='ok' for p in pair),
                'statuses':'|'.join(p['status'] if p else 'absent' for p in pair)})
        for name,predicted in predictions.items():
            if name=='baseline': continue
            spec=case['variants'][name]['perturbation']; removed={tuple(r['key']) for r in spec['residues']} if spec else set()
            tree=cKDTree(spec['removed_coordinates']) if spec else None
            variant_sites={tuple(s[k] for k in ('chain','resnum','icode','group')):s for s in read_table(root/name/'sites.parquet')}
            for site in sites:
                key=tuple(site[k] for k in ('chain','resnum','icode','group'))
                if key[:3] in removed or site['is_break_terminus']: continue
                target=variant_sites.get(key)
                if target and target['is_break_terminus']: continue
                if not site['functional_atoms_complete']: continue
                residue=residues[key[:3]]; points=residue.coord[np.isin(residue.atom_name,SITE_ATOMS[site['group']])]
                distance=float(tree.query(points)[0].min()) if tree else None
                r={**{k:site[k] for k in ('complex_id','chain','resnum','icode','group')},'pdb_id':case['pdb_id'],
                    'variant':name,'kind':spec['kind'] if spec else 'repeat','deleted_residues':spec['length'] if spec else 0,
                    'reference_size':case['n_residues'],'reference_interface_type':case.get('interface_type','unclassified'),
                    'reference_missing_residues':case['original_missing_residues'],
                    'reference_incomplete_residues':case['original_incomplete_residues'],
                    'deleted_ionisable_residues':spec['deleted_ionisable_residues'] if spec else 0,
                    'deleted_exposure_mean':spec['reference_exposure_mean'] if spec else None,
                    'deleted_exposure_class':spec['reference_exposure_class'] if spec else 'repeat',
                    'interface':site['residue_delta_sasa']>10,'distance':distance,'shell':radial_bin(distance) if tree else 'repeat',
                    'status':'coverage_failure','ab_pka_error':None,'free_pka_error':None,'delta_pka_error':None}
                quartet=[reference.get((*key,state)) for state in ('AB',site['partner'])]+[predicted.get((*key,state)) for state in ('AB',site['partner'])]
                r['prediction_statuses']='|'.join(q['status'] if q else 'absent' for q in quartet)
                if all(q and q['status']=='ok' for q in quartet):
                    a,b,c,d=[q['pka'] for q in quartet]
                    r.update(status='ok',ab_pka_error=c-a,free_pka_error=d-b,delta_pka_error=(c-d)-(a-b))
                results.append(r)
    if results:
        with (output/'site_errors.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    shells=[]; outside=[]
    for kind in sorted({r['kind'] for r in results if r['kind']!='repeat'}):
        for length in sorted({r['deleted_residues'] for r in results if r['kind']==kind}):
            selected=[r for r in results if r['kind']==kind and r['deleted_residues']==length]
            for lo,hi in zip(BINS[:-1],BINS[1:]):
                shell=radial_bin(lo)
                shells.append({'kind':kind,'deleted_residues':length,'shell':shell,**statistics([r for r in selected if r['shell']==shell])})
            for radius in (0,5,10,15,20,30):
                outside.append({'kind':kind,'deleted_residues':length,'outside_radius':radius,**statistics([r for r in selected if r['distance']>=radius])})
    per_variant=[{'complex_id':case['complex_id'],'variant':name,**statistics([r for r in results if r['complex_id']==case['complex_id'] and r['variant']==name])} for case in cases for name in case['variants'] if name!='baseline']
    exposure_shells=[]
    for kind,exposure in sorted({(r['kind'],r['deleted_exposure_class']) for r in results if r['kind']!='repeat'}):
        for lo in BINS[:-1]:
            shell=radial_bin(lo)
            exposure_shells.append({'kind':kind,'exposure_class':exposure,'shell':shell,
                **statistics([r for r in results if r['kind']==kind and r['deleted_exposure_class']==exposure and r['shell']==shell])})
    report={'cases':len(cases),'planned_cases':len(manifest['cases']),'partial':completed_only,
        'prep_failures':manifest['prep_failures'],'teacher_failures':failures,
        'repeat_reproducibility':statistics([r for r in results if r['kind']=='repeat']),
        'repeat_note':'Same teacher settings and default MC seed 1234567: this checks reproducibility, not independent-seed Monte Carlo variance.',
        'radial_shells':shells,'radial_shells_by_exposure':exposure_shells,'outside_radius':outside,'per_variant':per_variant,
        'production_allowed':False,'radius_selected':None,'analysis_sha256':digest(Path(__file__)),
        'limits':'Pilot relative to prepared intact/near-complete references, not experimental truth. Error is measured on retained native sites; deleted sites and artificial termini are not targets. Teacher coverage failures included. Pooled sites are correlated; no population safe-radius inference.'}
    from collections import Counter
    report['baseline_teacher_coverage']={'eligible_sites':len(coverage),'paired_sites':sum(r['paired'] for r in coverage),
        'statuses':dict(Counter(r['statuses'] for r in coverage)),
        'by_group':{g:{'eligible':sum(r['group']==g for r in coverage),'paired':sum(r['group']==g and r['paired'] for r in coverage)} for g in sorted({r['group'] for r in coverage})}}
    strata=[]
    for field in ('reference_interface_type','reference_missing_residues','reference_incomplete_residues','deleted_ionisable_residues'):
        for value in sorted({r[field] for r in results}):
            for radius in (0,5,10,15,20,30):
                strata.append({'field':field,'value':value,'outside_radius':radius,
                    **statistics([r for r in results if r['kind']!='repeat' and r[field]==value and r['distance']>=radius])})
    report['stratified_outside_radius']=strata
    for name,table in (('radial_shells',shells),('outside_radius',outside),('per_variant',per_variant),('radial_shells_by_exposure',exposure_shells),('stratified_outside_radius',strata)):
        fields=list(dict.fromkeys(k for row in table for k in row))
        with (output/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(table)
    atomic_json(output/'report.json',report)
    print(json.dumps({'cases':report['cases'],'prep_failures':report['prep_failures'],'teacher_failures':failures,'repeat_reproducibility':report['repeat_reproducibility'],'per_variant':per_variant},indent=2))
