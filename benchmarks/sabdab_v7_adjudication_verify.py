"""Independent saved-artifact checks for inactive adjudication counterfactuals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest, require_compute
from sabdab_v7_exposure import fasta, read
from sabdab_v7_roles import collect
from sabdab_v7_adjudicate import checked_context


def run(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    if (args.output / 'verification.json').exists():
        raise FileExistsError('Verification already exists')
    ctx = checked_context(args.root); data = ctx['data']; old = {a.unit_id: a for a in ctx['old']}
    cached = read(args.root / 'report/sequence-component-evidence.json')
    union, _ = collect(args.root, 'union', digest(args.root / 'inputs.json'))
    roles = {c: {'train': 'train', 'test': 'validation'}[s] for c, s in data['component_splits'].items()}
    members = defaultdict(list)
    for m in ctx['members']:
        members[m['unit_id']].append(m)
    summaries = []
    for folder in sorted(p for p in args.output.iterdir() if p.is_dir() and (p / 'summary.json').exists()):
        meta = read(folder / 'summary.json'); policy = meta['policy']
        rows = pq.read_table(folder / 'proposed-assignments.parquet').to_pylist()
        aa = {a['unit_id']: a for a in rows}
        assert aa.keys() == old.keys()
        expected_components = {}
        for uid, u in ctx['units'].items():
            kk = {m['sequence_key'] for m in members[uid]}
            if u['source'] == 'sabdab':
                comp = {u['native_component']}
            else:
                comp = {c for k in kk for c in cached['cdr_components'].get(k, [])}
                if policy['antigen_reference_scope'] == 'resolved':
                    comp.update(c for k in kk for c in cached['antigen_components'].get(k, []))
                else:
                    comp.update(c for k in kk for h in union.get(k, []) for c in h['components'])
            expected_components[uid] = comp
            a = aa[uid]
            assert set(a['matched_components']) == comp
            assert a['native_source_split'] == old[uid].native_source_split
            assert a['source'] == old[uid].source
            assert a['release_active'] is False
            if a['source'] == 'sabdab':
                assert a['operational_role'] == old[uid].operational_role
            if a['operational_role'] in {'train', 'validation'} and comp:
                assert {roles[c] for c in comp} == {a['operational_role']}
        # Check the retained group boundaries directly from source metadata,
        # without calling either quarantine/removal implementation.
        protected = defaultdict(set)
        for uid, a in aa.items():
            if a['operational_role'] not in {'train', 'validation'}:
                continue
            if a['source'] != 'sabdab' and not (policy['mapped_family_mode'] == 'published_components' and expected_components[uid]):
                for g in ctx['units'][uid]['source_groups']:
                    protected[('family', a['source'], g)].add(a['operational_role'])
            for m in members[uid]:
                protected[('exact', m['sequence_key'])].add(a['operational_role'])
        assert all(len(v) == 1 for v in protected.values())
        # Verify logged group causes refer to real membership and opposing
        # pre-removal roles. A cause is not accepted just because a final set
        # happens to be clean.
        for event in read(folder / 'group-decisions.json'):
            g = event['group_id']; uu = event['train_members'] + event['validation_members']
            assert event['train_members'] and event['validation_members']
            assert set(event['removed_members']) <= set(uu)
            for uid in uu:
                if g.startswith('exact:'):
                    assert g[6:] in {m['sequence_key'] for m in members[uid]}
                else:
                    _, src, name = g.split(':', 2)
                    assert ctx['units'][uid]['source'] == src
                    assert name in ctx['units'][uid]['source_groups']
                    if policy['mapped_family_mode'] == 'published_components':
                        assert not expected_components[uid]
            assert all(ctx['units'][uid]['source'] != 'sabdab' for uid in event['removed_members'])
        source_counts = defaultdict(Counter); changes = Counter(); recovered = 0
        for uid, a in aa.items():
            source_counts[a['source']][a['operational_role']] += 1
            previous = old[uid].operational_role
            if previous != a['operational_role']:
                changes[previous + '->' + a['operational_role']] += 1
            recovered += previous == 'quarantine' and a['operational_role'] in {'train', 'validation'}
        assert {k: dict(v) for k, v in source_counts.items()} == meta['counts']
        assert dict(changes) == meta['transitions_from_v2']
        assert recovered == meta['recoveries']
        if policy['policy_id'] == 'cf0-reproduction':
            assert digest(folder / 'proposed-assignments.parquet') == ctx['receipt']['artifact_pins']['proposed-assignments.parquet']
        # Every searched sequence must match the pinned source, including newly
        # restored queries/targets. MMseqs remains the declared heuristic guard.
        for name in ['q.fasta', 't.fasta']:
            for key, sequence in fasta(folder / 'general-holdout' / name).items():
                assert sequence == data['sequences'][key]
        summaries.append(dict(case=policy['policy_id'], units=len(rows),
            assignments_sha256=digest(folder / 'proposed-assignments.parquet'),
            published_components_verified=True, native_splits_preserved=True,
            retained_family_and_exact_boundaries_verified=True,
            group_causes_verified=True, counts_and_transitions_verified=True))
    ledger = read(args.output / 'ledger-summary.json')
    classes = ledger['classes']
    assert classes['component_bridge_at_inventory_scope'] == 46
    assert classes['general_protein_holdout'] == 66
    assert classes['unmapped_antibody_pending'] == 62
    assert sum(v for k, v in classes.items() if k.startswith('group_')) == 891
    edge = read(args.output / 'edge-verification.json')
    assert edge['status'] == 'passed' and edge['full_antigen_alignments_recomputed'] is False
    checked_context(args.root)
    atomic_json(args.output / 'verification.json', dict(status='passed', release_ready=False,
        counterfactuals=summaries, ledger_counts_reconciled=True,
        component_edge_reconstruction=edge, historical_inputs_unchanged=True,
        code_sha256=digest(__file__)))
    print('Verified', len(summaries), 'counterfactuals; historical inputs unchanged.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    run(p.parse_args())
