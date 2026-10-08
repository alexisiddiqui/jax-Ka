"""Freeze scoped structural split and masks after experimental-family reservation."""
import json
import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from .runtime import require_compute, atomic_json, digest, config_hash


def run(out):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .antigen_split import Groups
    from .schema import KEY
    runtime=Path(os.environ['PKABENCH_RUNTIME']); root=runtime/'universe/combined-split-v1'; previous=root/'usable-proposal-v2'
    source=runtime/'universe/buffer-15-20-v3-recount'; out=Path(out); out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((source/'report.json').read_text()); assert receipt['verification']['passed']
    source_assignment=source/'eligibility-with-existing-assignments.parquet'
    assignments=pq.read_table(source_assignment).to_pylist(); before={r['complex_id']:dict(r) for r in assignments}
    candidates=json.loads((root/'index.json').read_text())['candidates']; roles={r['complex_id']:r for r in json.loads((previous/'roles.json').read_text())['rows']}
    assert set(before)=={r['complex_id'] for r in candidates}
    matches_path=runtime/'audits/set2-screen-v1/matched-candidate-chains.json'; matches=json.loads(matches_path.read_text())
    reserved={before[r['complex_id']]['component_id'] for r in matches if r['system']=='protein_g_fc'}
    assert 'c4293009afb208f2d2b0f' in reserved
    moved=[]
    for r in assignments:
        if r['component_id'] in reserved:
            if r['split']!='test': moved.append(r['complex_id'])
            r['split']='test'; r['reference_reserved']=True
        r['set2_protein_g_fc_reserved']=r['component_id'] in reserved
    assigned={r['complex_id']:r for r in assignments}
    group_splits=defaultdict(set)
    for r in assignments: group_splits[r['component_id']].add(r['split'])
    assert all(len(s)==1 for s in group_splits.values())
    # Rebuild the accepted graph convention: antibody antigens; all other partners.
    nodes={r['complex_id']:['s'+config_hash(c['sequence'])[:20] for c in r['chains'] if roles[r['complex_id']]['role']!='antibody_antigen' or c['chain'] in roles[r['complex_id']]['antigen_chains']] for r in candidates}
    node_splits=defaultdict(set); node_groups=defaultdict(set)
    for cid,ns in nodes.items():
        assert ns
        for n in ns: node_splits[n].add(assigned[cid]['split']); node_groups[n].add(assigned[cid]['component_id'])
    assert all(len(s)==1 for s in node_splits.values()) and all(len(g)==1 for g in node_groups.values())
    cdr_nodes={}
    for r in candidates:
        if r['stratum']!='antibody': continue
        fields=['CDR-H1','CDR-H2','CDR-H3']+(['CDR-L1','CDR-L2','CDR-L3'] if len(r['partner_A_chains'])==2 else [])
        parts=[''.join(r['sabdab'].get(f,'').split()).upper() for f in fields]
        if all(p and p not in ('NA','NAN','NONE') and set(p)<=set('ACDEFGHIKLMNPQRSTVWY') for p in parts): cdr_nodes[r['complex_id']]='s'+config_hash(''.join(parts))[:20]
    cdr_set=set(cdr_nodes.values()); graphs={c:Groups(cdr_set) for c in (30,50,70,90)}; edges=0
    with (root/'sequence/hits.tsv').open() as f:
        for line in f:
            a,b,i,q,t=line.split(); i,q,t=map(float,(i,q,t))
            if min(q,t)<.8: continue
            if i>=.3 and a in node_splits and b in node_splits:
                assert node_splits[a]==node_splits[b] and node_groups[a]==node_groups[b]; edges+=1
            if a in cdr_set and b in cdr_set:
                for c,g in graphs.items():
                    if i>=c/100: g.join(a,b)
    novelty={}
    for c,g in graphs.items():
        train={g.find(n) for cid,n in cdr_nodes.items() if assigned[cid]['split']=='train'}; counts=Counter()
        for r in assignments:
            label=None
            if r['role']=='antibody_antigen' and r['split']=='test':
                n=cdr_nodes.get(r['complex_id']); label='unknown' if n is None else 'seen_antibody' if g.find(n) in train else 'both_unseen'
                if r['benchmark_eligible'] and r['eval_interface_sites']>0: counts[label]+=1
            r[f'novelty_cdr_{c}']=label
        novelty[str(c)]=dict(counts)
    # Current all-chain reference inventory, including official PKAD supplement.
    scope_path=previous/'independent-experimental-scope-v2.json'; scope=json.loads(scope_path.read_text())
    assert scope['pkad_release_reservation_check_pass']
    reference_files=[previous/'reference-matched-pairs-scoped.json',runtime/'audits/experimental-inventory-v1/official-delta-matches.json']
    reference_checks=0
    for path in reference_files:
        for m in json.loads(path.read_text()):
            assert assigned[m['complex_id']]['split']=='test'; reference_checks+=1
    assert all(assigned[m['complex_id']]['split']=='test' for m in matches if m['system']=='protein_g_fc')
    # Pin actual scientific inputs as well as campaign-level files.
    final_masks=[]; structures=[]; inputs=[]; total_counts=defaultdict(Counter)
    (out/'campaigns').mkdir()
    for record in receipt['source_campaigns']:
        campaign=Path(record['campaign']); report=json.loads((campaign/'report.json').read_text())
        assert report['checks_passed'] and not report['pipeline_errors']
        assert digest(campaign/'manifest.json')==record['manifest_sha256']
        assert digest(campaign/'site_masks.parquet')==record['mask_sha256']
        dest=out/'campaigns'/campaign.name; dest.mkdir()
        for name in ('manifest.json','report.json','structures.parquet','site_masks.parquet'):
            shutil.copyfile(campaign/name,dest/name)
        for r in pq.read_table(campaign/'structures.parquet').to_pylist():
            cid=r['complex_id']; folder=campaign/'structures'/cid
            states={state:digest(folder/f'{state}.cif') for state in ('AB','A','B')}
            assert config_hash(states)==r['content_sha256']
            files={name:digest(folder/name) for name in ('student_AB.cif','sites.parquet','removed_components.json','input_atom_mask.json','provenance.json')}
            inputs.append({'complex_id':cid,'structure_root':str(folder),'state_sha256':states,'files_sha256':files})
            r.update(split=assigned[cid]['split'],component_id=assigned[cid]['component_id']); structures.append(r)
        for r in pq.read_table(campaign/'site_masks.parquet').to_pylist():
            a=assigned[r['complex_id']]
            assert not r['eval_mask'] or r['train_mask']
            for role in ('train','eval'):
                total_counts[r['complex_id']][role+'_sites']+=int(r[role+'_mask'])
                total_counts[r['complex_id']][role+'_interface_sites']+=int(r[role+'_mask'] and r['interface'])
            final_masks.append(r|{'split':a['split'],'component_id':a['component_id'],'benchmark_eligible':a['benchmark_eligible'],
                'training_eligible':bool(a['benchmark_eligible'] and a['split']=='train' and r['train_mask']),
                'evaluation_eligible':bool(a['benchmark_eligible'] and a['split'] in ('val','test') and r['eval_mask'])})
    assert len(final_masks)==len({tuple(r[k] for k in KEY) for r in final_masks})
    for r in assignments:
        for k in ('train_sites','eval_sites','train_interface_sites','eval_interface_sites'): assert r[k]==total_counts[r['complex_id']][k]
        assert r['component_id']==before[r['complex_id']]['component_id']
        if r['complex_id'] not in moved: assert r['split']==before[r['complex_id']]['split']
    pq.write_table(pa.Table.from_pylist(assignments),out/'assignments.parquet')
    pq.write_table(pa.Table.from_pylist(final_masks),out/'site_masks.parquet')
    pq.write_table(pa.Table.from_pylist(structures),out/'structures.parquet')
    atomic_json(out/'input-files.json',inputs)
    # Snapshot reference evidence. Future experimental IDs fail closed by contract.
    (out/'references').mkdir()
    evidence=[scope_path,*reference_files,matches_path,root/'index.json',previous/'roles.json',root/'sequence/hits.tsv',source/'report.json',source_assignment,
        runtime/'audits/experimental-inventory-v1/official-delta-report.json']
    source_hashes={str(p):digest(p) for p in evidence}
    for p in (scope_path,*reference_files,matches_path): shutil.copyfile(p,out/'references'/p.name)
    summary={}
    for split in ('train','val','test'):
        rr=[r for r in assignments if r['split']==split and r['benchmark_eligible']]; role='train' if split=='train' else 'eval'
        summary[split]={'interface_pairs':sum(r[role+'_interface_sites']>0 for r in rr),'interface_sites':sum(r[role+'_interface_sites'] for r in rr),'sites':sum(r[role+'_sites'] for r in rr),'usable_groups':len({r['component_id'] for r in rr if r[role+'_interface_sites']>0})}
    verification={'passed':True,'sequence_edges_checked':edges,'reference_matched_rows_checked':reference_checks,'protein_preparations_hashed':len(inputs),'site_masks_checked':len(final_masks),'groups_never_cross_splits':True,'novelty_recomputed':True,'mask_counts_match_recount':True}
    atomic_json(out/'verification.json',verification)
    policy={'freeze_scope':'Versioned structural dataset, assignments, current masks and current reference inventory. Teacher coverage and method validation are not implied.',
        'future_experimental_references':'Unknown reference IDs are ineligible for independent scoring until all-chain overlap is checked against this frozen training set. Overlapping systems must be excluded from independent claims or require a new split version before training.',
        'set2_complete':False,'independent_pkad_exception':scope['user_approved_exception'],'mask_consumer':'Use frozen site_masks.parquet training_eligible/evaluation_eligible plus interface when required; never sites.supervision_mask.',
        'production_ready':False,'remaining_before_production':['Reproducible preparation gate revision','50-pair end-to-end method/mask smoke benchmark','Production scorer with group-aware bootstrap'],'group_policy':'30% identity, 80% bidirectional coverage; general partners disjoint, antibodies antigen-held-out with independent CDR novelty annotations.'}
    report={'frozen':True,'production_ready':False,'splits':summary,'reserved_protein_g_fc_groups':sorted(reserved),'moved_pair_ids':moved,'moved_candidate_pairs':len(moved),'lost_train_interface_pairs':sum(before[c]['benchmark_eligible'] and before[c]['train_interface_sites']>0 for c in moved),'antibody_test_interface_novelty':novelty,'verification':verification,'policy':policy}
    atomic_json(out/'report.json',report)
    artifact_hashes={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    atomic_json(out/'manifest.json',{'version':'structural-freeze-v1','frozen':True,'source_sha256':source_hashes,'artifacts_sha256':artifact_hashes,'implementation_sha256':digest(Path(__file__)),'policy':policy})
    assert all(digest(Path(p))==sha for p,sha in source_hashes.items())
    print(json.dumps(report,indent=2),flush=True)


def verify(out):
    require_compute()
    import pyarrow.parquet as pq
    out=Path(out); manifest=json.loads((out/'manifest.json').read_text())
    for name,sha in manifest['artifacts_sha256'].items(): assert digest(out/name)==sha
    for name,sha in manifest['source_sha256'].items(): assert digest(Path(name))==sha
    assignments={r['complex_id']:r for r in pq.read_table(out/'assignments.parquet').to_pylist()}
    reserved={cid for cid,r in assignments.items() if r['set2_protein_g_fc_reserved']}
    assert all(assignments[cid]['split']=='test' for cid in reserved)
    masks=pq.read_table(out/'site_masks.parquet').to_pylist()
    for r in masks:
        a=assignments[r['complex_id']]
        assert r['split']==a['split'] and r['component_id']==a['component_id']
        assert r['training_eligible']==bool(a['benchmark_eligible'] and a['split']=='train' and r['train_mask'])
        assert r['evaluation_eligible']==bool(a['benchmark_eligible'] and a['split'] in ('val','test') and r['eval_mask'])
        if r['complex_id'] in reserved: assert not r['training_eligible']
    # External receipt avoids modifying the content-addressed frozen bundle.
    result={'passed':True,'manifest_sha256':digest(out/'manifest.json'),'artifacts_checked':len(manifest['artifacts_sha256']),'masks_checked':len(masks),'reserved_candidates':len(reserved),'reserved_training_sites':0}
    atomic_json(out.parent/(out.name+'-verification.json'),result)
    print(json.dumps(result),flush=True)
