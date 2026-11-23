"""Combine verified exposure stages into inactive, reviewable v7 decisions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import gzip
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest, require_compute
from pkabench.source_policy import SourcePolicy
from sabdab_v7_exposure import read, skey


def global_hits(folder):
    summary=read(folder/'summary.json')
    if summary['status']!='global_antigen_alignment_complete':raise ValueError('Incomplete '+str(folder))
    result={}
    with gzip.open(folder/'hits.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['sequence_key'] in result:raise ValueError('Duplicate global query result')
            result[r['sequence_key']]=r['hits']
    if len(result)!=summary['hit_queries']:raise ValueError('Global hit receipt differs')
    return result


def run(args):
    require_compute(threads=args.workers,allow_comp1400=True)
    out=args.output
    if out.exists():raise FileExistsError('Choose a new report directory')
    out.mkdir(parents=True)
    inventory=read(args.inventory/'inventory-manifest.json')
    policy=SourcePolicy.load(args.inventory/'source-policy.json')
    if read(args.screen/'numbering-verification.json')['status']!='passed':
        raise ValueError('Numbering evidence did not reproduce')
    units=pq.read_table(args.inventory/'units.parquet').to_pylist();by_id={u['unit_id']:u for u in units}
    evidence=pq.read_table(args.screen/'unit-exposure.parquet').to_pylist()
    if len(evidence)!=len(units):raise ValueError('Incomplete unit screen')
    member=pq.read_table(args.antigen/'antigen-sequence-membership.parquet').to_pylist()
    unit_antigen=defaultdict(set)
    for r in member:unit_antigen[r['unit_id']].add(r['sequence_key'])
    gh={scope:global_hits(args.antigen/scope) for scope in ['test','external','validation']}
    validation_ag_components={c for hh in gh['validation'].values() for h in hh for c in h['components']}
    seeds=read(args.screen/'component-seeds.json')
    validation_components=set(seeds['validation'])|validation_ag_components
    external_ag_components=set()
    for u in units:
        if u['source']=='sabdab' and unit_antigen[u['unit_id']]&gh['external'].keys():
            external_ag_components.add(u['sabdab_component'])
    external_components=set(seeds['heldout'])|external_ag_components
    # Map test-reference sequences back to published components for the explicit
    # preserve-pKPDB counterfactual and checkpoint exposure annotations.
    test_by_sequence=defaultdict(set)
    with Path(inventory['input_pins']['sabdab_csv']['path']).open() as f:
        for r in csv.DictReader(f):
            if r['ab_ag_split']!='test':continue
            for name in ['Hseq','Lseq','Hseq_expected','Lseq_expected','VH_numerable_seq','VL_numerable_seq']:
                if r[name]:test_by_sequence[skey(r[name])].add(r['ab_ag_cluster'])
    for r in pq.read_table(args.inventory/'sabdab-antigen-targets.parquet').to_pylist():
        if r['original_role']=='test':test_by_sequence[r['sequence_key']].add(r['component'])
    test_by_query=defaultdict(set)
    with gzip.open(args.screen/'sequence-CDR-evidence.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['role']!='H':continue  # Light similarity alone is not the legacy H gate.
            for k in r['matched_chains']['sabdab_test']:test_by_query[r['sequence_key']].update(test_by_sequence[k])
    generic_path=args.screen/'generic-sabdab_test/hits.tsv'
    if generic_path.exists():
        for line in generic_path.read_text().splitlines():
            q,t,ident,qc,tc=line.split('\t')
            if float(ident)>=.7 and min(float(qc),float(tc))>=.8:test_by_query[q].update(test_by_sequence[t])
    for k,cc in test_by_sequence.items():test_by_query[k].update(cc)
    strict=[];alternative=[];counts=defaultdict(Counter);alt_counts=defaultdict(Counter)
    extra_reason_counts=Counter();past_train=set();past_val=set();preserved_components=set();preserved_pkpdb=[]
    alternative_policy=replace(policy,evaluation_policy='preserve_pkpdb_report_clean_test_subset')
    for r in evidence:
        u=by_id[r['unit_id']];agkeys=unit_antigen[r['unit_id']]
        components=set(r['official_components'])
        target_components=set(r['sabdab_test_matched_components'])
        for k in u['sequence_keys']:target_components.update(test_by_query[k])
        for k in agkeys:
            for h in gh['test'].get(k,[]):target_components.update(h['components'])
        if u['source']!='sabdab':
            (past_train if u['original_role']=='train' else past_val).update(target_components)
        reasons=set(r['exclusion_reasons']);new=[]
        global_test=bool(agkeys&gh['test'].keys());global_external=bool(agkeys&gh['external'].keys())
        val_component=bool(components&validation_components);ext_component=bool(components&external_components)
        if u['original_role']=='train':
            if global_test:new.append('sabdab_test:antigen_global_40_80')
            if global_external:new.append('heldout:antigen_global_40_80')
            if val_component and not r['validation_component_hit']:new.append('validation:component_from_antigen')
            if ext_component and not r['heldout_component_hit']:new.append('heldout:component_from_antigen')
        reasons.update(new);extra_reason_counts.update(new)
        row=dict(r,exclusion_reasons=sorted(reasons),global_test_antigen_hit=global_test,
                 global_external_antigen_hit=global_external,validation_component_with_antigens=val_component,
                 external_component_with_antigens=ext_component,matched_test_components=sorted(target_components),
                 source_precedence=list(policy.source_precedence),policy_sha256=policy.sha256,
                 prepared_key=None,teacher_context_id=None,release_active=False)
        row['review_flags']=[f for f in row['review_flags'] if f!='full_antigen_global_homology_pending']
        if u['original_role']=='train':row['proposed_action']='exclude_training' if reasons else 'candidate_pending_context_and_structure'
        strict.append(row)
        counts[u['source']+'/'+u['original_role']][row['proposed_action']]+=1
        alt=dict(row,policy_sha256=alternative_policy.sha256)
        if u['original_role']=='train' and reasons and all(x.startswith('sabdab_test:') for x in reasons):
            if alternative_policy.training_action(u['source'],'sabdab_test')=='keep_training_exclude_affected_test_component':
                alt['proposed_action']='candidate_retain_pkpdb_exclude_affected_test_components'
                preserved_pkpdb.append(u['source_native_id']);preserved_components.update(target_components)
        alternative.append(alt);alt_counts[u['source']+'/'+u['original_role']][alt['proposed_action']]+=1
    test_loss=Counter();past_exposure=Counter()
    for rows in [strict,alternative]:
        for r in rows:
            if r['source']=='sabdab' and r['original_role']=='test':
                comp=r['published_component']
                r['v6c_training_membership_overlap']=comp in past_train
                r['v6c_validation_membership_overlap']=comp in past_val
                if rows is alternative and comp in preserved_components:
                    r['proposed_action']='exclude_from_clean_test_subset_pkpdb_precedence'
                    test_loss['all_antigens']+=1
                    if r['protein_partner']:test_loss['protein_peptide']+=1
                if rows is strict and r['protein_partner']:
                    if comp in past_train:past_exposure['protein_test_v6c_training_component_overlap']+=1
                    if comp in past_val:past_exposure['protein_test_v6c_validation_component_overlap']+=1
    pq.write_table(pa.Table.from_pylist(strict),out/'strict-proposed-decisions.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(alternative),out/'preserve-pkpdb-counterfactual.parquet',compression='zstd')
    sab_train=[r for r in strict if r['source']=='sabdab' and r['original_role']=='train' and r['protein_partner']]
    candidates=[r for r in sab_train if r['proposed_action']=='candidate_pending_context_and_structure']
    existing_pdbs={s:{u['pdb_id'] for u in units if u['source']==s} for s in ['pkpdb','pinder']}
    candidate_parent_overlap={s:sum(by_id[r['unit_id']]['pdb_id'] in pp for r in candidates) for s,pp in existing_pdbs.items()}
    # First-party catalogues retain source-native identity even while prepared
    # geometry/ownership is deliberately unresolved. These are not train pools.
    catalogues=out/'source-catalogues';catalogues.mkdir()
    for source in ['pkpdb','sabdab','pinder']:
        pq.write_table(pa.Table.from_pylist([u for u in units if u['source']==source]),
                       catalogues/(source+'.parquet'),compression='zstd')
    sb_counts=Counter(r['proposed_action'] for r in sab_train)
    summary=dict(status='full_sequence_and_antigen_audit_complete_not_a_release',
        strict_source_roles={k:dict(v) for k,v in counts.items()},
        protein_sabdab_train=dict(total=len(sab_train),actions=dict(sb_counts),
            candidates=len(candidates),candidate_interface_unknown=sum(not r['interface_mapped'] for r in candidates),
            candidate_type_review=sum(any('VNAR' in f or 'standalone_light' in f for f in r['review_flags']) for r in candidates),
            candidate_parent_PDB_overlap_review_only=candidate_parent_overlap),
        new_global_antigen_exclusions=dict(extra_reason_counts),
        preserve_pkpdb_counterfactual=dict(training_units=len(preserved_pkpdb),
            excluded_test_components=len(preserved_components),excluded_test_instances=dict(test_loss),
            policy_sha256=alternative_policy.sha256,training_ids=preserved_pkpdb),
        potential_checkpoint_exposure_from_v6c_membership=dict(past_exposure),
        source_policy=policy.to_dict(),source_policy_sha256=policy.sha256,
        no_active_pool_or_label_changes=True,
        outstanding=['Prepared-structure identity and duplicate ownership remain unproven.',
                     'Unmapped candidate interfaces need structural checks.',
                     'The 65 PINDER antigen-context flags remain excluded/reviewed, not declared preparation errors.',
                     'Source/role retirement and checkpoint exposure must be retained in the release manifest.',
                     'No teacher or feature build has started.'],
        inputs_sha256={str(p):digest(p) for p in [args.inventory/'inventory-manifest.json',args.screen/'summary.json',
                       args.screen/'unit-exposure.parquet',*[args.antigen/s/'summary.json' for s in gh]]},
        code_sha256=digest(__file__),slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(out/'summary.json',summary)
    lines=['# Full v7 source exposure audit','',
      'Policy: pKPDB > SAbDab > PINDER; protect the official SAbDab test. These are inactive proposed decisions. No current release, labels or features changed.','',
      '| Existing source | Training entries | Proposed exclusions | Remaining before context/identity review |',
      '|---|---:|---:|---:|']
    for source in ['pkpdb','pinder']:
        c=counts[source+'/train'];n=sum(c.values())
        lines.append(f'| {source} | {n:,} | {c["exclude_training"]:,} | {n-c["exclude_training"]:,} |')
    lines+=['',f'Of {len(sab_train):,} protein/peptide SAbDab training instances, **{len(candidates):,}** remain candidates after sequence, published-component and global antigen checks. These are not approved additions; {sum(not r["interface_mapped"] for r in candidates):,} lack a PINDER interface mapping.',
      '',f'The explicit preserve-pKPDB counterfactual retains {len(preserved_pkpdb):,} additional pKPDB training entries but excludes {test_loss["protein_peptide"]:,} protein/peptide SAbDab test instances from clean evaluation. It is **not the selected default**.',
      '', 'Antigen comparisons used exhaustive global alignments after a safe length bound, with the published match/mismatch/gap parameters and 40% protein / 80% peptide thresholds. Resolved and expected antigen sequences were both checked conservatively; this is stronger context protection than assuming the cropped ML inputs alone are used.',
      '', 'The official-test protection inventory also records apo-only test antibody sequences as reference metadata; no apo structures were newly annotated or added as supervised inputs.',
      '', 'The per-instance flags record v6c membership overlap, not proof that a particular checkpoint saw every eligible record. Actual checkpoint/run memberships need their own exposure audit. Repartitioning cannot erase any actual prior training or model-selection use.',
      '', 'Prepared-unit keys and teacher context IDs remain null deliberately: sequence similarity and half-Fv lineage cannot authorise deduplication or cache reuse.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='inputs_sha256'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','screen','antigen','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
