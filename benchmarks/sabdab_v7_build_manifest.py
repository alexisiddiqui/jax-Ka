"""Inactive, per-unit v7 native build actions and measured resource budgets."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.source_registry import SourceRegistry


def read(path):return json.loads(Path(path).read_text())


def rows(path):return pq.read_table(path).to_pylist()


def write_rows(path,data):
    columns=sorted({k for r in data for k in r})
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in columns} for r in data]),path,compression='zstd')


def surface_estimate(n,bins):
    for b in bins:
        if b['lo']<=n<=b['hi'] and b['successes']:
            return dict(median_s=b['annotation_s']['median'],p90_s=b['annotation_s']['p90'],
                median_bytes=b['compressed_bytes']['median'],p90_bytes=b['compressed_bytes']['p90'],
                reference_surfaces=b['successes'],lo=b['lo'],hi=b['hi'])
    raise ValueError('No measured size stratum for native surface: '+str(n))


def budget(tasks,bins):
    estimates=[surface_estimate(t['n_res'],bins) for t in tasks]
    median_s=sum(t['median_s'] for t in estimates);p90_s=sum(t['p90_s'] for t in estimates)
    med_bytes=sum(t['median_bytes'] for t in estimates);high_bytes=sum(t['p90_bytes'] for t in estimates)
    result=dict(surfaces=len(tasks),size_bin_counts=dict(Counter(str((t['lo'],t['hi'])) for t in estimates)),
        sparse_stratum_surfaces=sum(t['reference_surfaces']<30 for t in estimates),
        median_work_hours=median_s/3600,p90_sum_work_hours=p90_s/3600,
        estimated_compressed_bytes_median=round(med_bytes),estimated_compressed_bytes_p90_sum=round(high_bytes),
        packed_files_median=math.ceil(med_bytes/(2*1024**3))+8,
        packed_files_p90_sum=math.ceil(high_bytes/(2*1024**3))+8,
        worker_scenarios=[])
    for workers in [4,8,16,23]:
        # Historical throughput is mostly batched/warm. Native pilot batch=1
        # includes a cold restore of about 30--50 s per view. Include it rather
        # than implying that the warm benchmark is a measured production ETA.
        result['worker_scenarios'].append(dict(workers=workers,cpus=workers*2,
            planning_low_h=(median_s+30*len(tasks))*1.25/(3600*workers),
            planning_high_h=(p90_s+50*len(tasks))*1.5/(3600*workers),
            assumptions='2 CPUs/worker; reserved startup and retries; fixed-throughput projection, high-concurrency GPU contention unmeasured'))
    return result


def run(args):
    require_compute(threads=2,allow_comp1400=True)
    root=args.root;out=args.output or root/'build-manifest'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    authority=read(args.authority);precedence=args.precedence.split(',')
    registry=SourceRegistry.load(args.registry);registry.validate_precedence(precedence)
    units={r['unit_id']:r for r in rows(args.audit/'units.parquet')}
    assignments={r['unit_id']:r for r in rows(root/'rescope/proposed-assignments.parquet')}
    native={r['unit_id']:r for r in rows(root/'existing-native/units.parquet')}
    sabdab={r['unit_id']:r for r in rows(root/'sabdab/units.parquet')}
    scopes={r['unit_id']:r for r in rows(root/'rescope/sabdab-scope-checks.parquet')}
    pilot={r['unit_id']:r for r in read(root/'reuse/pilot-reuse.json')['verified']}
    context_path=root/'reuse/teacher-context.json'
    teacher_context=read(context_path)
    compatible={r['unit_id']:r for r in teacher_context['records'] if r['teacher_reuse_approved']}
    aliases={};ownership=[];context_review=defaultdict(set)
    for pair in rows(root/'reuse/context-comparisons.parquet'):
        if pair['status']=='rigid_heavy_atom_identity' and pair['candidate_parent'].startswith('sabdab:'):
            context_review[pair['unit_id']].add(pair['candidate_parent'])
            context_review[pair['candidate_parent']].add(pair['unit_id'])
    for uid,proof in compatible.items():
        candidates=[]
        for key in [uid,proof['candidate_parent']]:
            u=units[key];a=assignments[key];eligible=a['operational_role'] in ('train','validation')
            if u['source']=='sabdab':
                mask='train_mask_interface' if a['operational_role']=='train' else 'eval_mask_interface'
                eligible &= (sabdab[key]['eligibility_status']=='potential_paired_interface' and
                    scopes[key]['status']=='published_retained_scope_verified' and proof['counts'][mask]>0)
            else:eligible &= native[key]['status']=='checked'
            candidates.append(dict(unit_id=key,source=u['source'],source_native_id=u['source_native_id'],
                eligible=bool(eligible),operational_role=a['operational_role'],
                prepared_key=proof['teacher_state_signatures']['AB'],teacher_context_id=proof['teacher_context_id']))
        owner=registry.choose_owner(candidates,precedence)
        for candidate in candidates:
            if candidate['eligible'] and candidate['unit_id']!=owner['unit_id']:
                aliases[candidate['unit_id']]=owner['unit_id']
        ownership.append(dict(owner=owner['unit_id'],aliases=[c['unit_id'] for c in candidates if c['eligible'] and c['unit_id']!=owner['unit_id']],
            teacher_context_id=proof['teacher_context_id'],operational_role=owner['operational_role']))
    assets=read(root/'timing-and-assets.json');bins=assets['size_bins']
    cache=args.surface_release;manifest=read(cache/'manifest.json');index=read(cache/'index.json')['records']
    for name in ['coverage.json','policy.json','index.json','verification.json']:
        if digest(cache/name)!=manifest['files'][name]:raise ValueError('Published surface metadata pin changed: '+name)
    coverage={(r['dataset'],r['cid']):r for r in read(cache/'coverage.json')['parents']}
    by_id={r['id']:r for r in index}
    ab_inventory=read(args.antibody_inventory)
    ab_parents={p['id']:p for p in ab_inventory['parents']}
    ab_units=defaultdict(list)
    for unit in ab_inventory['units']:ab_units[unit['cid']].append(unit)
    # Trust the prior exhaustive source/geometry/field qualification, whose
    # receipt is explicitly part of this immutable release. We do not silently
    # convert its mixed, documented preparation profiles to a new profile.
    policy=read(cache/'policy.json')
    if policy['name']!='masif-native-explicit-profile-complete-assembly-v1':raise ValueError('Unrecognised native cache policy')
    proposed=[];surface_tasks=[];optional_tasks=[];reuse_ids=set();label_build=[]
    pilot_surfaces=args.pilot/'masif-qualified-v1'
    pilot_index={r['id']:r for r in read(pilot_surfaces/'index.json')['records']}
    label_pilot=read(args.pilot/'pilot-report-v2.json')
    for uid,u in sorted(units.items()):
        a=assignments[uid];role=a['operational_role'];operational=role in ('train','validation')
        prep=sabdab.get(uid) if u['source']=='sabdab' else native.get(uid)
        result=dict(unit_id=uid,source=u['source'],source_native_id=u['source_native_id'],
            original_source_split=u['native_split'],operational_role=role,
            input_scope_id=a['input_scope_id'],data_kind=registry.get(u['source']).data_kind,
            reason_codes=a['reason_codes'],release_active=False,source_owner=uid,
            owner_status='source_native_record_no_cross_source_teacher_equivalence_claim',
            ownership_ready=uid not in context_review,duplicate_context_candidates=sorted(context_review.get(uid,[])),
            eligible_for_build=False,n_res=prep.get('n_res') if prep else None)
        if uid in context_review:result['owner_status']='duplicate_context_review_required_before_membership_activation'
        if uid in aliases:
            result.update(source_owner=aliases[uid],ownership_ready=True,owner_status='verified_compatible_context_alias',
                preparation_action='reuse_verified_native_context',teacher_action='reuse_canonical_owner_teacher',
                surface_action='alias_canonical_owner_no_duplicate_build',graph_action='alias_no_additional_sampling_weight')
        elif not operational:
            result.update(preparation_action='retain_audit_only',teacher_action='withhold_nonoperational_role',
                surface_action='withhold_nonoperational_role',graph_action='withhold_nonoperational_role')
        elif u['source']=='sabdab':
            status=prep['status'];elig=prep['eligibility_status'];result['preteacher_eligibility']=elig
            result['preparation_action']='reuse_packed_native_preparation' if status=='prepared' else 'withhold_preparation_failure'
            result['preparation_error']=prep.get('error')
            scope_ok=scopes.get(uid,{}).get('status')=='published_retained_scope_verified'
            ready=status=='prepared' and scope_ok and elig=='potential_paired_interface'
            if ready:
                teacher_action=('reuse_verified_same_native_pilot' if uid in pilot else
                    'reuse_verified_compatible_pinder_teacher' if uid in compatible else 'build_native_pkai_pkai_plus_AB_A_B')
                result.update(eligible_for_build=True,teacher_action=teacher_action,
                    surface_action='build_missing_native_biological_partner_surfaces',
                    graph_action='qualify_teacher_correspondence_then_build_paired_graph')
                if uid not in pilot and uid not in compatible:label_build.append(result)
                if uid in compatible:
                    result.update(teacher_context_receipt_sha256=digest(context_path),teacher_source_entry=compatible[uid]['native_teacher_entry'],
                        owner_status='verified_compatible_context_owner',ownership_ready=True)
                chain_sides={x['chain']:x['side'] for x in prep['side_by_chain']}
                side_sizes={s:sum(len(c['sequence']) for c in prep['chains'] if chain_sides[c['chain']]==s) for s in ['A','B']}
                required=[];optional=[]
                for state,n in [('A',side_sizes['A']),('B',side_sizes['B']),('AB',prep['n_res'])]:
                    sid='sabdab/'+u['source_native_id']+'/'+state
                    cached=pilot_index.get(sid) if uid in pilot else None
                    task=dict(unit_id=uid,surface_id=sid,state=state,n_res=n,
                        action='reuse_verified_native_pilot' if cached else 'build',
                        biological_scope='joint_antibody' if state=='A' else 'joint_antigen' if state=='B' else 'bound_complex')
                    if state=='AB':optional.append(task);optional_tasks.extend([task] if not cached else [])
                    else:required.append(task);surface_tasks.extend([task] if not cached else [])
                    if cached:reuse_ids.add(('pilot',sid))
                result.update(required_surfaces=required,optional_bound_complex_surface=optional,
                    post_teacher_eligibility='verified_finite_corresponding_targets' if uid in compatible or uid in pilot else 'pending_finite_corresponding_targets')
            else:
                reason=('scope_unassessed' if status=='prepared' and not scope_ok else elig)
                result.update(teacher_action='withhold_'+reason,surface_action='defer_'+reason,graph_action='withhold_'+reason)
        else:
            checked=prep is not None and prep['status']=='checked'
            result['preparation_action']='reuse_verified_frozen_native' if checked else 'withhold_native_scope_unassessed'
            if not checked:
                result.update(teacher_action='withhold_native_scope_unassessed',surface_action='withhold_native_scope_unassessed',graph_action='withhold_native_scope_unassessed')
            else:
                old_role={'train':'train','val':'validation','test':'validation'}.get(u['native_split'],u['native_split'])
                result.update(eligible_for_build=True,teacher_action='reuse_frozen_source_labels_no_cross_source_substitution',
                    graph_action='rebuild_query_masks_for_new_role' if role!=old_role else 'reuse_graph_arrays_relink_v7_membership',
                    teacher_scope_note=prep['teacher_context_status'])
                cov=coverage.get((u['source'],u['source_native_id']))
                cached=(cov is not None and cov['status']=='covered' and cov['canonical_residues']==prep['n_res']
                    and all(x['id'] in by_id for x in cov['required_units']))
                if cached:
                    required=[]
                    for spec in cov['required_units']:
                        ix=by_id[spec['id']];reuse_ids.add(('native',spec['id']))
                        required.append(dict(surface_id=spec['id'],state=spec['state'],action='reuse_prior_fully_audited_native_frame',
                            n_res=ix['summary']['n'],biological_scope=spec['role']))
                    result.update(surface_action='reuse_approved_native_profiles_repack_role_metadata',required_surfaces=required,
                        prior_surface_role=cov['split'],prior_cache_policy=policy['name'])
                else:
                    if u['source']=='pkpdb':
                        parent=ab_parents.get(u['source_native_id'])
                        if parent and parent['status']=='accepted':
                            specs=ab_units[u['source_native_id']]
                            if set().union(*(set(s['canonical_residue_ids']) for s in specs))!=set(range(prep['n_res'])):
                                raise ValueError('Antibody inventory does not cover full native assembly: '+uid)
                            required=[]
                            for spec in specs:
                                labels=[prep['chains'][i]['chain_id'] for i in spec['chain_indices']]
                                if labels!=spec['source_chain_labels']:raise ValueError('Antibody physical-chain inventory changed')
                                required.append(dict(unit_id=uid,surface_id=spec['id'],state=spec['state'],
                                    n_res=spec['n'],action='build',biological_scope='joint_'+spec['role']))
                        elif parent and parent['status']=='antibody_only':
                            required=[dict(unit_id=uid,surface_id=uid+'/Ab',state='Ab',n_res=prep['n_res'],
                                action='build',biological_scope='native_antibody_only_no_antigen_in_source')]
                        else:
                            required=[dict(unit_id=uid,surface_id=uid+'/chain-'+str(i),state='chain-'+str(i),
                                n_res=len(c['sequence']),action='build',biological_scope='physical_protein_chain') for i,c in enumerate(prep['chains'])]
                        result['surface_action']='build_missing_existing_native_policy'
                    else:
                        sides={x['chain']:x['side'] for x in (prep.get('side_by_chain') or [])}
                        if not prep.get('side_chain_partition_unambiguous') or not sides:
                            required=[];result['surface_action']='withhold_ambiguous_partner_partition'
                        else:
                            required=[dict(unit_id=uid,surface_id=uid+'/'+state,state=state,
                                n_res=sum(len(c['sequence']) for c in prep['chains'] if sides[c['chain_id']]==state),
                                action='build',biological_scope='whole_native_biological_partner') for state in ['A','B']]
                            result['surface_action']='build_missing_native_biological_partner_surfaces'
                    result['required_surfaces']=required
                    surface_tasks.extend(required)
                if u['source']=='pinder':
                    optional=dict(unit_id=uid,surface_id=uid+'/AB',state='AB',n_res=prep['n_res'],action='build',biological_scope='bound_complex')
                    result['optional_bound_complex_surface']=[optional];optional_tasks.append(optional)
        proposed.append(result)
    write_rows(out/'units.parquet',proposed);write_rows(out/'new-required-surfaces.parquet',surface_tasks)
    write_rows(out/'optional-bound-surfaces.parquet',optional_tasks)
    # Separate >1536 surfaces: no extrapolated timing invents a measured bin.
    supported=[t for t in surface_tasks if t['n_res']<=1536];oversize=[t for t in surface_tasks if t['n_res']>1536]
    optional_supported=[t for t in optional_tasks if t['n_res']<=1536]
    budgets={s:budget([t for t in supported if t['unit_id'].startswith(s+':')],bins) for s in ['pkpdb','pinder','sabdab']}
    reused_native_bytes=sum(by_id[sid]['bytes'] for origin,sid in reuse_ids if origin=='native')
    reused_pilot_bytes=sum(pilot_index[sid]['bytes'] for origin,sid in reuse_ids if origin=='pilot')
    graph_report=read(args.pilot.parent/'sabdab-v7-antibody-scope-fix-v1/loader-v1/report.json')
    graph_examples=read(args.pilot.parent/'sabdab-v7-antibody-scope-fix-v1/labels-v1/summary.json')['records']
    accepted={r['unit_id'] for r in proposed if r['source']=='sabdab' and r['eligible_for_build']}
    residues=sum(r['n_res'] for r in proposed if r['unit_id'] in accepted)
    pilot_graph_n=sum(r['unit']['n'] for r in graph_examples if r['unit']['unit_id'] in accepted)
    graph_byte_estimate=round(graph_report['packed_bytes']*residues/pilot_graph_n)
    # CPU pKAI cold process timings are measured on five selected examples,
    # so give a broad linear/quadratic size envelope, not a precise ETA.
    examples=label_pilot['examples'];rates=[r['pkai_total_wall_s']/r['residues'] for r in examples]
    quadratic=[r['pkai_total_wall_s']/(r['residues']**2) for r in examples]
    label_low=sum(max(20.,min(rates)*r['n_res']) for r in label_build)/3600
    label_high=sum(max(25.,max(quadratic)*(r['n_res']**2)) for r in label_build)/3600
    pins=[args.authority,args.registry,args.antibody_inventory,args.audit/'inputs.json',root/'sabdab/summary.json',root/'existing-native/summary.json',
        root/'rescope/summary.json',root/'reuse/summary.json',context_path,root/'timing-and-assets.json',cache/'manifest.json']
    totals=defaultdict(Counter)
    for r in proposed:
        for column in ['operational_role','preparation_action','teacher_action','surface_action','graph_action']:
            totals[r['source']][column+':'+r[column]]+=1
    report=dict(status='inactive_v7_full_preparation_reuse_build_manifest',units=len(proposed),
        by_source={s:dict(c) for s,c in totals.items()},source_precedence=precedence,split_authority=authority['policy_id'],
        sabdab_preparation=read(root/'sabdab/summary.json'),native_preparation=read(root/'existing-native/summary.json'),
        observed_scope_roles=read(root/'rescope/summary.json'),reuse=read(root/'reuse/summary.json'),
        compatible_context_ownership=ownership,teacher_context_checks=dict(approved=teacher_context['approved'],receipt_sha256=digest(context_path)),
        pending_duplicate_context_review_units=sum(not r['ownership_ready'] for r in proposed),
        buildable_pending_duplicate_context_review_units=sum(r['eligible_for_build'] and not r['ownership_ready'] for r in proposed),
        required_surface_policy='biological A and B partition for paired units; approved existing native policy for site-only proteins',
        bound_AB_surface_policy='optional diagnostics; not part of the assembly-partition loader requirement',
        new_required_surfaces=len(surface_tasks),unestimated_oversize_surfaces=len(oversize),
        optional_bound_surfaces=len(optional_tasks),optional_oversize_surfaces=len(optional_tasks)-len(optional_supported),
        budgets=budgets,optional_bound_budget=budget(optional_supported,bins),
        verified_native_cache=dict(frames=sum(origin=='native' for origin,sid in reuse_ids),bytes=reused_native_bytes,
            policy=policy['name'],requires_role_metadata_repack=True,full_payload_rescan_repeated=False),
        verified_pilot_cache=dict(frames=sum(origin=='pilot' for origin,sid in reuse_ids),bytes=reused_pilot_bytes),
        new_teacher=dict(complexes=len(label_build),cpu_work_hours_low=label_low,cpu_work_hours_high=label_high,
            at_16_workers_h=[label_low/16,label_high/16],at_32_workers_h=[label_low/32,label_high/32],
            estimate_basis='five actual cold-process native pKAI+pKAI+ AB/A/B examples; broad linear/quadratic size envelope, no new apo labels'),
        sabdab_graph_storage=dict(records_upper_bound=len(accepted),compressed_bytes_point_estimate=graph_byte_estimate,
            planning_bytes_range=[round(graph_byte_estimate*.6),round(graph_byte_estimate*2.)],zstd_level=1,
            reader_files=5,layout='existing ProductionStore: one records.bin plus index.npz/records.json/metadata.json/verification.json',
            caveat='Residue-scaled from three real paired graphs; query/neighbor density and teacher availability affect final size.'),
        packing=dict(zstd_level=1,prepared_and_surface_shard_bytes=2*1024**3,per_structure_permanent_files=False,
            temporary_work='bounded scratch directories per active worker, removed after packing',
            caveat='Reader files are shards plus small manifests/indexes; build checkpoints, reports and temporary scratch are separate.'),
        graph_payload_inventory=assets['graph'],input_pins={str(p):digest(p) for p in pins},
        output_pins={p.name:digest(p) for p in out.glob('*.parquet')},release_active=False,teacher_or_surfaces_built=False,
        remaining_gates=['actual finite teacher correspondence and post-teacher graph eligibility',
            'resolve any matching prepared-context candidates before duplicate ownership',
            'confirm existing graph payload location and repack v7 roles/min_fraction values',
            'qualify new surfaces and complete-assembly coverage',
            'independent novel-test freeze and production recipe activation'],
        limitations=['Warm historical timing bins plus explicit cold-start/retry allowances are projections, not measured concurrent ETA.',
            'Only two historical successful surfaces exceed 1024 residues; larger units need a tail benchmark.',
            'Sum of bin p90 values is a conservative scenario, not a 90% confidence interval.',
            'No source is treated as equivalent from PDB ID, sequence or rigid geometry alone.',
            'Existing site-only pKPDB labels remain their frozen source protocol; paired pKAI teacher labels are separate.'])
    atomic_json(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],units=len(proposed),by_source=report['by_source'],
        new_required_surfaces=len(surface_tasks),unestimated_oversize_surfaces=len(oversize),budgets=budgets,
        new_teacher=report['new_teacher'],sabdab_graph_storage=report['sabdab_graph_storage']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','audit','authority','registry','surface-release','pilot','antibody-inventory']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--precedence',default='pkpdb,sabdab,pinder');run(p.parse_args())
