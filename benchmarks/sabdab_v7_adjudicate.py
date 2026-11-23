"""Explain v2 quarantines and compare inactive group policies using cached evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import os
import shutil
import time

import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, config_hash, digest, require_compute
from pkabench.role_engine import Assignment, Evidence, Unit, assign, quarantine_group_conflicts
from pkabench.group_adjudication import ACTIVE, GroupPolicy, assert_boundaries, make_groups, propagate_family_anchors, resolve_groups
from pkabench.source_registry import SourceRegistry
from pkabench.split_authority import SplitAuthority
from sabdab_v7_exposure import read
from sabdab_v7_roles import collect, general_holdout_guard, write_rows


def assignment(row):
    data = {k: row[k] for k in Assignment.__dataclass_fields__}
    for name in ['matched_components', 'reason_codes']:
        data[name] = tuple(data[name])
    return Assignment(**data)


def fingerprint(aa):
    return config_hash([asdict(a) for a in sorted(aa, key=lambda a: a.unit_id)])


def checked_context(root):
    summary = read(root / 'report/summary.json')
    receipt = read(root / 'report/verification.json')
    if receipt['status'] != 'passed' or digest(root / 'report/summary.json') != receipt['summary_sha256']:
        raise ValueError('Unverified v2 audit')
    for name, expected in receipt['artifact_pins'].items():
        if digest(root / 'report' / name) != expected:
            raise ValueError('Changed v2 artifact: ' + name)
    data = read(root / 'inputs.json')
    if digest(root / 'inputs.json') != summary['inputs_sha256']:
        raise ValueError('Changed audit input')
    for path, expected in data['input_pins'].items():
        if digest(path) != expected:
            raise ValueError('Changed pinned input: ' + path)
    for pin in data['original_input_pins'].values():
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Historical input changed: ' + pin['path'])
    units = {u['unit_id']: u for u in pq.read_table(root / 'units.parquet').to_pylist()}
    members = pq.read_table(root / 'scope-membership.parquet').to_pylist()
    baseline = [assignment(a) for a in pq.read_table(root / 'report/component-assignments.parquet').to_pylist()]
    old = [assignment(a) for a in pq.read_table(root / 'report/proposed-assignments.parquet').to_pylist()]
    inventory = Path(data['inventory'])
    numbered = read(inventory / 'numbering-cached.json')['numbered']
    numbered.update(read(inventory / 'numbering-new.json')['numbered'])
    manifest = read(inventory / 'inventory-manifest.json')
    executable = Path(manifest['runtime']) / 'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
    return dict(root=root, summary=summary, receipt=receipt, data=data, units=units,
        members=members, baseline=baseline, old=old, numbered=numbered, executable=executable)


def counterfactual_baseline(ctx, policy, union_hits=None):
    if policy.antigen_reference_scope == 'resolved':
        return ctx['baseline']
    root = ctx['root']; data = ctx['data']
    if union_hits is None:
        union_hits, _ = collect(root, 'union', digest(root / 'inputs.json'))
    ag = defaultdict(set)
    for m in ctx['members']:
        if ctx['units'][m['unit_id']]['source'] == 'sabdab':
            continue  # Authoritative published input/role stays fixed.
        for h in union_hits.get(m['sequence_key'], []):
            ag[m['unit_id']].update(h['components'])
    evidence = []
    for row in pq.read_table(root / 'report/published-evidence.parquet').to_pylist():
        if row['axis'] == 'antigen' and ctx['units'][row['unit_id']]['source'] != 'sabdab':
            row = dict(row, components=sorted(ag[row['unit_id']]),
                       status='hit' if ag[row['unit_id']] else 'no_hit')
        row['components'] = tuple(row['components'])
        evidence.append(Evidence(**row))
    units = [Unit(**{k: u[k] for k in Unit.__dataclass_fields__}) for u in ctx['units'].values()]
    return assign(units, evidence, data['component_splits'], SplitAuthority.load(root / 'authority.json'), SourceRegistry.load(root / 'registry.json'))


def run_case(ctx, output, policy, workers, *, historical=False, union_hits=None):
    output.mkdir()
    started = time.monotonic()
    baseline = ctx['baseline'] if historical else counterfactual_baseline(ctx, policy, union_hits)
    propagated, propagation = (propagate_family_anchors(baseline, ctx['units'])
        if policy.anchor_propagation else (baseline, []))
    groups = make_groups(ctx['units'], ctx['members'], propagated,
        exempt_mapped=policy.mapped_family_mode == 'published_components')
    if historical:
        grouped, log = quarantine_group_conflicts(propagated, groups, explain=True)
    else:
        grouped, log = resolve_groups(propagated, groups, preferred_removal_source=policy.preferred_removal_source)
    final, generic = general_holdout_guard(grouped, ctx['units'], ctx['members'], ctx['numbered'],
        ctx['data']['sequences'], ctx['executable'], output / 'general-holdout', workers)
    assert_boundaries(final, groups)
    original = {a.unit_id: a for a in ctx['old']}
    component_roles = {c: {'train': 'train', 'test': 'validation'}[s] for c, s in ctx['data']['component_splits'].items()}
    for a in final:
        if a.native_source_split != original[a.unit_id].native_source_split:
            raise ValueError('Native split provenance changed')
        if a.source == 'sabdab' and a.operational_role != original[a.unit_id].operational_role:
            raise ValueError('Authoritative SAbDab role changed')
        if a.operational_role in ACTIVE and a.matched_components and {component_roles[c] for c in a.matched_components} != {a.operational_role}:
            raise ValueError('Published component crosses operational roles')
    if historical:
        if final != ctx['old']:
            raise ValueError('CF0 did not reproduce v2 assignments')
        shutil.copy2(ctx['root'] / 'report/proposed-assignments.parquet', output / 'proposed-assignments.parquet')
        assert digest(output / 'proposed-assignments.parquet') == ctx['receipt']['artifact_pins']['proposed-assignments.parquet']
    else:
        write_rows(output / 'proposed-assignments.parquet', [dict(asdict(a), release_active=False,
            adjudication_policy_sha256=policy.sha256, prepared_context_status='unknown_not_established') for a in final])
    atomic_json(output / 'group-decisions.json', log)
    atomic_json(output / 'anchor-propagation.json', propagation)
    if generic:
        write_rows(output / 'general-holdout-evidence.parquet', generic)
    transitions = Counter(); changes = []; counts = defaultdict(Counter)
    for a in final:
        counts[a.source][a.operational_role] += 1
        before = original[a.unit_id]
        if a.operational_role != before.operational_role:
            transitions[before.operational_role + '->' + a.operational_role] += 1
            changes.append(dict(unit_id=a.unit_id, source=a.source, native_source_split=a.native_source_split,
                v2_role=before.operational_role, proposed_role=a.operational_role,
                previous_reasons=list(before.reason_codes), reason_codes=list(a.reason_codes),
                matched_components=list(a.matched_components), prepared_context_status='unknown_not_established'))
    if changes:
        write_rows(output / 'membership-diff.parquet', changes)
    report = dict(policy=asdict(policy), policy_sha256=policy.sha256,
        status='inactive_counterfactual_complete', release_ready=False,
        assignment_sha256=fingerprint(final), counts={k: dict(v) for k, v in counts.items()},
        transitions_from_v2=dict(transitions), changed_units=len(changes),
        recoveries=sum(before.operational_role == 'quarantine' and after.operational_role in ACTIVE
                       for before, after in zip(ctx['old'], final)),
        old_input_pins_verified=True, authoritative_roles_preserved=True,
        retained_family_and_exact_boundaries_verified=True, generic_holdout_rerun=True,
        generic_hits_sha256=digest(output / 'general-holdout/hits.tsv'),
        v2_assignments_sha256=ctx['receipt']['artifact_pins']['proposed-assignments.parquet'],
        code_sha256=digest(__file__), wall_s=time.monotonic()-started,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output / 'summary.json', report)
    print(json.dumps(dict(case=policy.policy_id, counts=report['counts'], changes=len(changes), transitions=dict(transitions))), flush=True)
    return final, log, grouped


def ledger(ctx, out, group_log, grouped):
    old = {a.unit_id: a for a in ctx['old']}; base = {a.unit_id: a for a in ctx['baseline']}
    causes = defaultdict(list); neighbours = set()
    for event in group_log:
        for uid in event['removed_members']:
            causes[uid].append(event['group_id'])
        neighbours.update(event['train_members']); neighbours.update(event['validation_members'])
    focus = {a.unit_id for a in ctx['old'] if a.operational_role in {'quarantine', 'pending'}} | neighbours
    rows, classes = [], Counter()
    for uid in sorted(focus):
        a, b = old[uid], base[uid]
        kind = 'conflict_neighbour_retained'
        if a.operational_role == 'pending':
            kind = 'unmapped_antibody_pending'
        elif 'bridges_authoritative_train_validation' in a.reason_codes:
            kind = 'component_bridge_at_inventory_scope'
        elif uid in causes:
            kinds = {g.split(':', 1)[0] for g in causes[uid]}
            kind = 'group_' + ('family_and_exact' if len(kinds) > 1 else next(iter(kinds)))
        elif 'general_protein_holdout_70_80' in a.reason_codes:
            kind = 'general_protein_holdout'
        if a.operational_role in {'quarantine', 'pending'}:
            classes[kind] += 1
        rows.append(dict(unit_id=uid, source=a.source, native_source_split=a.native_source_split,
            baseline_role=b.operational_role, v2_role=a.operational_role, conflict_class=kind,
            causing_groups=sorted(causes[uid]), reason_codes=list(a.reason_codes),
            matched_components=list(a.matched_components), input_scope_id=a.input_scope_id,
            prepared_context_identity='not_established'))
    write_rows(out / 'unit-conflict-ledger.parquet', rows)
    memberships = [dict(m, input_scope_id=ctx['units'][m['unit_id']]['input_scope_id'])
                   for m in ctx['members'] if m['unit_id'] in focus]
    write_rows(out / 'focus-sequence-membership.parquet', memberships)
    group_rows = []
    for event in group_log:
        for role in ['train', 'validation']:
            for uid in event[role + '_members']:
                group_rows.append(dict(unit_id=uid, group_id=event['group_id'],
                    group_type=event['group_id'].split(':', 1)[0], input_scope_id=base[uid].input_scope_id,
                    role_before_removal=role, removed=uid in event['removed_members'],
                    rule=event['rule'], chain_id=None, annotation_role='whole_unit_group_membership'))
    write_rows(out / 'conflict-group-membership.parquet', group_rows)
    summary = dict(status='conflict_ledger_complete_metadata_scope', classes=dict(classes), focus_units=len(focus),
        group_conflicts=sum(v for k, v in classes.items() if k.startswith('group_')),
        group_events=len(group_log), native_scope_verified=False,
        source_assignment_sha256=fingerprint(ctx['old']),
        limitations=['Family memberships are whole-unit context evidence; they do not identify a chain by themselves.',
            'Exact sequence sharing is not prepared-context equivalence.',
            'A bridge at inventory scope must still be checked against actual prepared inputs.'])
    if summary['group_conflicts'] != ctx['summary']['family_or_exact_guard_quarantined']:
        raise ValueError('Ledger does not reconcile with v2 group exclusions')
    atomic_json(out / 'ledger-summary.json', summary)
    return focus


def run(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    if args.output.exists():
        raise FileExistsError('Choose a new adjudication output root')
    args.output.mkdir(parents=True)
    ctx = checked_context(args.root)
    settings = read(args.policy)
    if set(settings) != {'schema_version', 'status', 'authority_policy', 'profiles'} or settings['schema_version'] != 1 or settings['status'] != 'inactive_counterfactuals_only':
        raise ValueError('Unsupported adjudication configuration')
    if settings['authority_policy'] != ctx['summary']['authority']['policy_id']:
        raise ValueError('Wrong authority policy')
    atomic_json(args.output / 'adjudication-policy.json', settings)
    cf0, log, grouped = run_case(ctx, args.output / 'cf0-reproduction', GroupPolicy('cf0-reproduction'), args.workers, historical=True)
    focus = ledger(ctx, args.output, log, grouped)
    from sabdab_v7_conflict_edges import reconstruct_edges
    reconstruct_edges(ctx, args.output, focus, args.workers)
    summaries = []
    for p in settings['profiles']:
        policy = GroupPolicy(**p)
        final, _, _ = run_case(ctx, args.output / policy.policy_id, policy, args.workers)
        if policy.policy_id == 'cf1-staged-removal':
            old_removed = {a.unit_id for a in cf0 if 'mixed_family_or_exact_input_group' in a.reason_codes}
            new_removed = {a.unit_id for a in final if 'adjudicated_group_conflict' in a.reason_codes}
            if not new_removed <= old_removed:
                raise ValueError('Staged-only group removal expanded original exclusions')
        summaries.append(read(args.output / policy.policy_id / 'summary.json'))
    # Check pins once more after all counterfactual jobs. No historical outputs
    # or native input files may have changed during the run.
    checked_context(args.root)
    atomic_json(args.output / 'summary.json', dict(status='inactive_adjudication_sweep_complete',
        release_ready=False, historical_inputs_unchanged=True,
        v2_root=str(args.root), v2_reproduced_exactly=True,
        ledger=read(args.output / 'ledger-summary.json'), counterfactuals=summaries,
        policy_sha256=digest(args.policy), code_sha256=digest(__file__),
        slurm_job_id=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True); p.add_argument('--workers', type=int, default=32)
    run(p.parse_args())
