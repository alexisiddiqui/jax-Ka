"""Audit all candidate chains against available experimental-reference hits."""
import json
from pathlib import Path
from collections import Counter
from .runtime import require_compute, atomic_json, digest, config_hash


def run(root, scoped=False):
    require_compute()
    import pyarrow.parquet as pq
    root=Path(root); source=root/'usable-proposal-v2'
    report=json.loads((source/'report.json').read_text())
    if digest(source/'proposal.parquet')!=report['proposal_sha256']: raise ValueError('Proposal changed')
    proposal={r['complex_id']:r for r in pq.read_table(source/'proposal.parquet').to_pylist()}
    index=json.loads((root/'index.json').read_text())
    ref=json.loads((source/'reference-sequences.json').read_text())
    excluded=set()
    if scoped:
        seeds=[r for r in ref['resolved'] if r['pdb_id'].lower()=='1axt' and r['chain']=='H']
        if len(seeds)!=1: raise ValueError('Expected exactly one approved 1AXT H reference')
        seed=seeds[0]
        excluded={r['id'] for r in ref['resolved'] if r['id']==seed['id'] or 'P01865' in r['uniprot_ids'] or r['sequence']==seed['sequence']}
        scope={'version':'independent-experimental-v1','user_approved_exception':'Exclude the overlapping 1AXT H antibody reference family from independent PKAD evaluation claims; retain the antigen-held-out split.',
               'excluded_references':[{k:r[k] for k in ('id','pdb_id','chain','uniprot_ids')} for r in ref['resolved'] if r['id'] in excluded],
               'reference_eligibility':[{'reference_id':r['id'],'independent_evaluation_eligible':r['id'] not in excluded} for r in ref['resolved']],
               'family_rule':'Current inventory: 1AXT H, P01865 accession, or identical reference sequence. Newly added related antibodies require family review before any independent experimental claim.',
               'proposal_sha256':report['proposal_sha256'],'reference_sequences_sha256':digest(source/'reference-sequences.json'),
               'consumer_requirement':'Independent experimental scoring must join reference_eligibility by reference ID; unknown IDs fail closed pending review. Excluded references may only appear as explicitly non-independent diagnostics.',
               'complete_experimental_inventory':False}
        atomic_json(source/'independent-experimental-scope.json',scope)
    receipt=json.loads((source/'reference-report.json').read_text())
    if digest(source/'reference-hits.tsv')!=receipt['hits_sha256']: raise ValueError('Reference hits changed')
    parents={r['uniprot_id'] for r in ref.get('parent_proxies',[]) if 'sequence' in r}
    unmapped=[r for r in ref['failed'] if not set(r['uniprot_ids']) & parents]
    hits={}
    for line in (source/'reference-hits.tsv').read_text().splitlines():
        a,b,i,q,t=line.split('\t')
        if a not in excluded and float(i)>=.3 and min(float(q),float(t))>=.8: hits.setdefault(b,[]).append(a)
    matched=[]; conflicts=[]
    for candidate in index['candidates']:
        cid=candidate['complex_id']; r=proposal[cid]
        chains=[{'chain':c['chain'],'reference_ids':hits['s'+config_hash(c['sequence'])[:20]]} for c in candidate['chains'] if 's'+config_hash(c['sequence'])[:20] in hits]
        if not chains: continue
        row={'complex_id':cid,'pdb_id':candidate['pdb_id'],'split':r['split'],'role':r['role'],'chains':chains,
             'benchmark_eligible':r['benchmark_eligible'],'train_interface_sites':r['train_interface_sites'],'eval_interface_sites':r['eval_interface_sites']}
        matched.append(row)
        if r['split']!='test': conflicts.append(row)
    affected={r['split']:sum(x['benchmark_eligible'] and x[('train' if r['split']=='train' else 'eval')+'_interface_sites']>0 for x in conflicts if x['split']==r['split']) for r in conflicts}
    identifiers=Counter(a for r in conflicts for c in r['chains'] for a in set(c['reference_ids']))
    refs={r['id']:{k:r[k] for k in ('pdb_id','chain','uniprot_ids')} for r in ref['resolved']}
    moved={proposal[r['complex_id']]['component_id'] for r in conflicts}
    remaining={s:sum(r['benchmark_eligible'] and r[('train' if s=='train' else 'eval')+'_interface_sites']>0 and r['component_id'] not in moved for r in proposal.values() if r['split']==s) for s in ('train','val')}
    summary={'proposal_sha256':report['proposal_sha256'],'reference_hits_sha256':receipt['hits_sha256'],
             'exact_structure_reference_chains':len(ref['resolved']), 'parent_sequence_proxy_identifiers':len(ref['failed'])-len(unmapped),
             'unmapped_reference_identifiers':unmapped,'matched_candidate_pairs':len(matched),'matched_by_split':dict(Counter(r['split'] for r in matched)),
             'non_test_conflicts':conflicts,'affected_interface_pairs':affected,'conflicts_by_role':dict(Counter(r['role'] for r in conflicts)),
             'top_conflict_references':[{'reference_id':a,'matched_chains':n,**refs.get(a,{})} for a,n in identifiers.most_common(10)],
             'move_all_flagged_components_to_test_counterfactual':{'components_to_move':len(moved),'remaining_interface_pairs':remaining,'applied':False},
             'checks_available_reference_inventory_pass':not conflicts and not unmapped,
             'complete_experimental_inventory':False,'frozen':False,
             'scope':'All candidate chains, including antibody chains omitted from antigen-only graph; >=30% identity and >=80% coverage of both sequences.',
             'limitations':['Checks only the locally released KaML reference inventory plus barnase/barstar seed; not independently verified as complete PKAD-3.',
                           'Parent sequences provide family-level mutant reservations, not exact construct verification.',
                           'Remaining experimental set-2 systems have not been curated; their absence cannot be certified.',
                           'Uses heuristic sequence hits and fixed coverage threshold; no claim of excluding every remote homologue.']}
    summary['excluded_reference_ids']=sorted(excluded)
    if scoped:
        summary['experimental_scope_sha256']=digest(source/'independent-experimental-scope.json')
        summary['scope']='All candidate chains checked against the available reference inventory after the user-approved independent-evaluation exception; >=30% identity and >=80% bidirectional coverage.'
    assert digest(source/'proposal.parquet')==report['proposal_sha256']
    atomic_json(source/('leakage-audit-scoped.json' if scoped else 'leakage-audit.json'),summary)
    atomic_json(source/('reference-matched-pairs-scoped.json' if scoped else 'reference-matched-pairs.json'),matched)
    print(json.dumps({k:v for k,v in summary.items() if k!='non_test_conflicts'},indent=2),flush=True)
