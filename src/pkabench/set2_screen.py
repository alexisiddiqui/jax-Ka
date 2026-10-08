"""Prospective experimental system sequence overlaps; no split mutation."""
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from .runtime import require_compute,atomic_json,digest,config_hash


def run(manifest,out):
    require_compute()
    import pyarrow.parquet as pq
    from .split_targets import references
    from .download import fetch
    manifest=Path(manifest); out=Path(out); out.mkdir(exist_ok=False)
    runtime=Path(os.environ['PKABENCH_RUNTIME']); root=runtime/'universe/combined-split-v1'; source=root/'usable-proposal-v2'
    systems=json.loads(manifest.read_text())['systems']; work=out/'usable-proposal-v2'; work.mkdir()
    (out/'sequence').mkdir(); (out/'sequence/sequences.fasta').symlink_to(root/'sequence/sequences.fasta')
    refs={}; owners=defaultdict(set)
    for system in systems:
        for seed in system['structure_seeds']:
            for chain in seed['chains']:
                key=(seed['pdb_id'],chain)
                refs[key]={'pdb_id':key[0],'chain':chain,'uniprot_ids':[]}
                owners['r'+config_hash(key)[:20]].add(system['id'])
    atomic_json(work/'reference-inventory.json',{'rows':list(refs.values()),'lead_manifest_sha256':digest(manifest)})
    atomic_json(out/'screened-leads.json',json.loads(manifest.read_text()))
    references(out)
    ref=json.loads((work/'reference-sequences.json').read_text())
    if ref['failed']: raise ValueError('Prospective reservation sequence unresolved')
    hits=defaultdict(set); identities={}
    for line in (work/'reference-hits.tsv').read_text().splitlines():
        a,b,i,q,t=line.split('\t')
        if float(i)>=.3 and min(float(q),float(t))>=.8:
            hits[b].update(owners[a]); identities[a,b]=[float(i),float(q),float(t)]
    proposal=pq.read_table(source/'proposal.parquet').to_pylist(); byid={r['complex_id']:r for r in proposal}
    old=json.loads((source/'report.json').read_text())
    if digest(source/'proposal.parquet')!=old['proposal_sha256']: raise ValueError('Split changed')
    matches=[]
    for r in json.loads((root/'index.json').read_text())['candidates']:
        for chain in r['chains']:
            n='s'+config_hash(chain['sequence'])[:20]
            for system in hits[n]:
                p=byid[r['complex_id']]
                matches.append({'system':system,'complex_id':r['complex_id'],'pdb_id':r['pdb_id'],'chain':chain['chain'],
                                **{k:p[k] for k in ('split','component_id','benchmark_eligible','train_interface_sites','eval_interface_sites')}})
    summary={}
    for s in systems:
        matched={r['complex_id']:r for r in matches if r['system']==s['id']}
        affected={r['component_id'] for r in matched.values() if r['split']!='test'}
        summary[s['id']]={'status':s['status'],'matched_pairs_by_split':dict(Counter(r['split'] for r in matched.values())),
                          'non_test_components':len(affected),
                          'direct_usable_conflicts':{split:sum(r['split']==split and r['benchmark_eligible'] and r[('train' if split=='train' else 'eval')+'_interface_sites']>0 for r in matched.values()) for split in ('train','val')},
                          'whole_component_usable_impact':{split:sum(r['split']==split and r['component_id'] in affected and r['benchmark_eligible'] and r[('train' if split=='train' else 'eval')+'_interface_sites']>0 for r in proposal) for split in ('train','val')},
                          'reservation_already_satisfied':not affected}
    atomic_json(out/'matched-candidate-chains.json',matches)
    result={'systems':summary,'proposal_sha256':old['proposal_sha256'],'manifest_sha256':digest(manifest),
            'resolved_seed_chains':len(ref['resolved']),'split_changed':False,'accepted_quantitative_labels':0,
            'scope':'Prospective reservation from structure-proxy sequences at 30% identity/80% bidirectional coverage. Does not verify exact experimental constructs, whole-domain containment, or structural preparation acceptance.',
            'limits':'FcRn is on hold; its overlaps are diagnostic, not adopted reservations. No new exception to experimental independence is assumed.'}
    assert digest(source/'proposal.parquet')==old['proposal_sha256']
    atomic_json(out/'report.json',result); print(json.dumps(result,indent=2),flush=True)
