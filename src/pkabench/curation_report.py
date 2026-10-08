"""Sequence diversity and cleanup accounting for the revised local audit."""
import json
import copy
from collections import Counter, defaultdict
from pathlib import Path
from .runtime import atomic_json, config_hash, require_compute
from .dataset_audit import load_rows, diversity_summary
from .schema import read_table


def report(campaign):
    from .curation import collect
    require_compute(); campaign=Path(campaign); collect(campaign)
    manifest=json.loads((campaign/'manifest.json').read_text())
    audit=Path(manifest['source_audit']); old=load_rows(audit)
    rows={r['complex_id']:json.loads((campaign/'rows'/f"{r['complex_id']}.json").read_text()) for r in old}
    pairs={r['complex_id']:['s'+config_hash(c['sequence'])[:20] for c in r['chains']] for r in old}
    links=[]
    for line in (audit/'sequence/hits.tsv').read_text().splitlines():
        a,b,identity,qcov,tcov=line.split('\t')
        if min(float(qcov),float(tcov))>=.8: links.append((a,b,float(identity)))
    result=json.loads((campaign/'curation_report.json').read_text())
    result['sequence_diversity']={}
    for radius in (10,15,20):
        annotated=[{'complex_id':cid,'scenarios':{'strict':{'status':'accepted' if r['status']=='accepted' and r['interface_sites_by_radius'][str(radius)]>0 else 'rejected'}}} for cid,r in rows.items()]
        result['sequence_diversity'][str(radius)]={str(cutoff):diversity_summary(annotated,pairs,links,cutoff)['scenarios']['strict'] for cutoff in (.3,.5,.9)}
    result['sequence_note']='All 279 original candidates contribute graph edges, including rejected bridges. Components group examples for leakage control; no sequence-based example deletion.'
    accepted=[r for r in rows.values() if r['status']=='accepted']
    tables={r['complex_id']:read_table(campaign/'structures'/r['complex_id']/'sites.parquet') for r in accepted}
    inclusive_ids={cid for cid,table in tables.items() if any(s['supervision_mask'] and s['residue_delta_sasa']>10 for s in table)}
    inclusive_rows=[{'complex_id':cid,'scenarios':{'strict':{'status':'accepted' if cid in inclusive_ids else 'rejected'}}} for cid in rows]
    result['inclusive']['sequence_components_30pct']=diversity_summary(inclusive_rows,pairs,links,.3)['scenarios']['strict']['independent_components']
    result['adaptive_radii']={}; strata=defaultdict(lambda:Counter(total=0,eligible=0))
    for radius in (10,15,20):
        field=f'supervision_mask_adaptive_{radius}'
        if not all(field in s for table in tables.values() for s in table): continue
        ids={cid for cid,table in tables.items() if any(s[field] and s['residue_delta_sasa']>10 for s in table)}
        annotated=[{'complex_id':cid,'scenarios':{'strict':{'status':'accepted' if cid in ids else 'rejected'}}} for cid in rows]
        result['adaptive_radii'][str(radius)]={'sites':sum(s[field] for table in tables.values() for s in table),
            'interface_sites':sum(s[field] and s['residue_delta_sasa']>10 for table in tables.values() for s in table),
            'complexes_with_interface_sites':len(ids),
            'sequence_components_30pct':diversity_summary(annotated,pairs,links,.3)['scenarios']['strict']['independent_components']}
        for r in accepted:
            n=r['structure']['n_residues']; size='<200' if n<200 else '200-499' if n<500 else '>=500'
            for s in tables[r['complex_id']]:
                distance=s['min_partner_distance']; shell='0-10' if distance<=10 else '10-20' if distance<=20 else '>20'
                for category in (f"group:{s['group']}",f'size:{size}',f'distance:{shell}'):
                    strata[f'{radius}/{category}'].update(total=1,eligible=int(s[field]))
    result['adaptive_strata']={k:dict(v) for k,v in strata.items()}
    result['alternate_residues']=sum(r['extra'].get('alternate_residues',0) for r in accepted)
    result['exposed_tail_candidates']=sum(r['extra'].get('exposed_tail_candidates',0) for r in accepted)
    result['exposure_classes']=dict(sum((Counter(r['extra'].get('exposure_classes',{})) for r in accepted),Counter()))
    # A deposited alternative is observed geometry, not an absent atom. Report
    # conformer-conditioned eligibility separately from robustness to alternatives.
    from .supervision import annotate_masks
    from .prep import read_cif
    conditioned={}
    for r in accepted:
        cid=r['complex_id']; work=campaign/'structures'/cid
        evidence=json.loads((work/'input_atom_mask.json').read_text())
        evidence['defects']=[d for d in evidence['defects'] if d['kind']!='alternate_conformation']
        conditioned[cid]=annotate_masks(copy.deepcopy(tables[cid]),read_cif(work/'AB.cif'),evidence)
    result['conformer_conditioned']={}
    for radius in (10,15,20):
        field=f'supervision_mask_adaptive_{radius}'
        ids={cid for cid,table in conditioned.items() if any(s[field] and s['residue_delta_sasa']>10 for s in table)}
        annotated=[{'complex_id':cid,'scenarios':{'strict':{'status':'accepted' if cid in ids else 'rejected'}}} for cid in rows]
        result['conformer_conditioned'][str(radius)]={'sites':sum(s[field] for table in conditioned.values() for s in table),
            'interface_sites':sum(s[field] and s['residue_delta_sasa']>10 for table in conditioned.values() for s in table),
            'complexes_with_interface_sites':len(ids),
            'sequence_components_30pct':diversity_summary(annotated,pairs,links,.3)['scenarios']['strict']['independent_components']}
    result['conformer_conditioned_note']='Diagnostic eligibility conditional on the selected observed conformation, retaining missing-coordinate masks. Not an ensemble-robust label or teacher-coverage count. Frozen default masks remain conservative; no production labels released.'
    result['additive_review']=[]
    for r in old:
        if r['scenarios']['strict']['status']=='accepted' or r['scenarios']['all_additives']['status']!='accepted': continue
        current=rows[r['complex_id']]
        result['additive_review'].append({'pdb_id':r['pdb_id'],'complex_id':r['complex_id'],
            'current_status':current['status'],'current_rejection':current.get('code'),
            'components':[{k:c.get(k) for k in ('name','partner_distances','bridges_partners_4A','declared_connection','formal_charge')} for c in r['components']]})
    atomic_json(campaign/'curation_report.json',result)
    print(json.dumps({k:result[k] for k in ('inclusive','radii','adaptive_radii','conformer_conditioned','alternate_residues','exposed_tail_candidates','exposure_classes')},indent=2))
