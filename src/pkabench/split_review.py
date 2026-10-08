"""Representation review of an immutable proposed split; no new geometry."""
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from .runtime import require_compute, atomic_json, digest


def run(root):
    require_compute()
    import numpy as np
    import pyarrow.parquet as pq
    root=Path(root); source=root/'usable-proposal-v2'; out=source/'representation-review'; out.mkdir(exist_ok=False)
    report=json.loads((source/'report.json').read_text())
    if digest(source/'proposal.parquet')!=report['proposal_sha256']: raise ValueError('Proposal hash changed')
    proposal=pq.read_table(source/'proposal.parquet').to_pylist(); byid={r['complex_id']:r for r in proposal}
    index=json.loads((root/'index.json').read_text()); candidates={r['complex_id']:r for r in index['candidates']}
    structures={}; sitecounts=defaultdict(Counter); masks=defaultdict(Counter); hashes=[]
    for entry in index['sources']:
        campaign=Path(entry['campaign'])
        if digest(campaign/'site_masks.parquet')!=entry['mask_sha256']: raise ValueError('Source mask changed')
        for name in ('structures.parquet','sites.parquet','site_masks.parquet'):
            hashes.append({'path':str(campaign/name),'sha256':digest(campaign/name)})
        for s in pq.read_table(campaign/'structures.parquet').to_pylist(): structures[s['complex_id']]=s
        for s in pq.read_table(campaign/'sites.parquet').to_pylist():
            count=sitecounts[s['complex_id']]; count['all_sites']+=1
            native=s['functional_atoms_complete'] and not s['is_break_terminus']
            count['native_sites']+=int(native)
            count['native_interface_sites']+=int(native and s['residue_delta_sasa']>10)
        for s in pq.read_table(campaign/'site_masks.parquet').to_pylist():
            count=masks[s['complex_id']]
            for mode in ('train','eval'):
                if s[mode+'_mask']:
                    count[mode+'_'+s['natural_gap_tier']]+=1
                    count[mode+'_group_'+s['group']]+=1
    records=[]
    for r in proposal:
        cid=r['complex_id']; mode='train' if r['split']=='train' else 'eval'
        if not r['benchmark_eligible'] or r[mode+'_interface_sites']<1: continue
        s=structures[cid]; p=json.loads(s['provenance']); annotation=p['annotation']; count=sitecounts[cid]
        rec={'complex_id':cid,'component_id':r['component_id'],'pdb_id':candidates[cid]['pdb_id'], 'split':r['split'],
             'class':'antibody_antigen' if r['role']=='antibody_antigen' else 'homomer' if s['homomeric'] else 'heteromer',
             'n_residues':s['n_residues'],'buried_area_A2':annotation['half_sum_buried_area'],
             'interface_residues':annotation['interface_residues'],'retained_sites':r[mode+'_sites'],
             'retained_interface_sites':r[mode+'_interface_sites'],'eval_interface_sites':r['eval_interface_sites'],
             'site_retention_fraction':r[mode+'_sites']/max(1,count['native_sites']),
             'interface_retention_fraction':r[mode+'_interface_sites']/max(1,count['native_interface_sites']),
             'uncertain_site_fraction':masks[cid][mode+'_uncertain']/max(1,r[mode+'_sites']),
             'reference_reserved':r['reference_reserved']}
        if rec['site_retention_fraction']>1 or rec['interface_retention_fraction']>1: raise ValueError('Mask denominator mismatch')
        records.append(rec)
    def quantiles(values):
        return dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(values,[0,.1,.5,.9,1])))) if values else None
    summaries={}
    metrics=('n_residues','buried_area_A2','interface_residues','retained_sites','retained_interface_sites','site_retention_fraction','interface_retention_fraction','uncertain_site_fraction')
    for split in ('train','val','test'):
        rs=[r for r in records if r['split']==split]; groups=Counter(r['component_id'] for r in rs)
        size=sum(groups.values()); top=[]
        for g,n in groups.most_common(8):
            member=[r for r in rs if r['component_id']==g]; descriptions=Counter(c.get('description','') for r in member for c in candidates[r['complex_id']]['chains'])
            top.append({'component_id':g,'pairs':n,'share':n/size,'descriptions':descriptions.most_common(5)})
        mode='train' if split=='train' else 'eval'; group_sites=Counter()
        for r in rs:
            group_sites.update({k.removeprefix(mode+'_group_'):v for k,v in masks[r['complex_id']].items() if k.startswith(mode+'_group_')})
        summaries[split]={'pairs':len(rs),'unique_assemblies':len({r['pdb_id'] for r in rs}), 'components':len(groups),
            'effective_component_count_by_pair_weights':size**2/sum(n*n for n in groups.values()) if groups else 0,
            'classes':dict(Counter(r['class'] for r in rs)), 'metrics':{m:quantiles([r[m] for r in rs]) for m in metrics},
            'top_components':top,'reference_reserved_pairs':sum(r['reference_reserved'] for r in rs),
            'retained_site_groups':dict(group_sites), 'pairs_with_only_1_or_2_interface_sites':sum(r['retained_interface_sites']<=2 for r in rs)}
        assert len(rs)==report['splits'][split]['interface_eligible_pairs']
    # Compare splits under one mask to separate mask-policy effects from selection.
    common={}
    for split in ('train','val','test'):
        rs=[r for r in records if r['split']==split and r['eval_interface_sites']>0]
        common[split]={'eval_interface_eligible_pairs':len(rs),'eval_interface_sites':quantiles([r['eval_interface_sites'] for r in rs])}
    warnings=[]
    for split,s in summaries.items():
        if s['top_components'] and s['top_components'][0]['share']>.1: warnings.append(f'{split}: largest usable component exceeds 10% of pairs; report family-stratified results and component bootstrap.')
        if s['pairs_with_only_1_or_2_interface_sites']>s['pairs']*.25: warnings.append(f'{split}: over 25% of pairs have only one or two usable interface sites.')
    trainfrac=summaries['train']['classes'].get('antibody_antigen',0)/summaries['train']['pairs']
    testfrac=summaries['test']['classes'].get('antibody_antigen',0)/summaries['test']['pairs']
    if abs(trainfrac-testfrac)>.05: warnings.append('Antibody share differs by more than five percentage points between train and test; report class-stratified scores.')
    result={'proposal_sha256':report['proposal_sha256'],'source_hashes':hashes,'splits':summaries,'common_evaluation_mask_comparison':common,
        'warnings':warnings,'frozen':False,'selection_limits':'Target-count allocator favors preservation of training pairs; test is not a random sample of the PDB universe. Class and family strata must be reported.',
        'remaining_freeze_checks':['Unresolved antibody roles remain excluded from quotas; no claim all general entries are antibody-free.',
            'Experimental set-2 completeness and PKAD-3 inventory coverage remain unfinished.',
            'No teacher predictions or teacher-success selection used.'],
        'metric_notes':'Interface sites require residue delta SASA >10 A^2. Native denominator requires complete functional atoms and no artificial terminus. Uncertain is the retained natural-gap tier, not a measured error probability. Effective component count is 1/sum(pair-share^2), not independent sample size.'}
    atomic_json(out/'report.json',result)
    with (out/'pairs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps({'splits':{k:{x:v[x] for x in ('pairs','unique_assemblies','components','effective_component_count_by_pair_weights','classes','pairs_with_only_1_or_2_interface_sites')} for k,v in summaries.items()},'warnings':warnings,'common_mask':common},indent=2),flush=True)
