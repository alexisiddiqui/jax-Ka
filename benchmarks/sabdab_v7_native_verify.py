"""Verify bounded native checks and summarize recorded overlap evidence per unit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest, require_compute
from sabdab_v7_exposure import read
from sabdab_v7_roles import write_rows


def run(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    target = args.remap / 'verification.json'
    if target.exists():
        raise FileExistsError('Native verification already exists')
    summary = read(args.scope / 'summary.json'); mapped = read(args.remap / 'summary.json')
    scope_file = args.scope / 'native-scope-checks.parquet'
    assert digest(scope_file) == summary['scope_receipt_sha256'] == mapped['actual_native_scope_sha256']
    assert not summary['structures_modified'] and not summary['labels_or_features_built']
    source_pins = {}; native = pq.read_table(scope_file).to_pylist()
    for r in native:
        assert r['status'] == 'checked'
        assert sum(c['residues'] for c in r['chains']) == r['n_res']
        for c in r['chains']:
            assert len(c['sequence']) == c['residues'] == len(c['residue_keys'])
            keys = {(k['chain'], k['number'], k['insertion']) for k in c['residue_keys']}
            assert len(keys) == c['residues']
        for pin in r['input_pins']:
            if pin['path'] in source_pins:
                assert source_pins[pin['path']] == pin['sha256']
            source_pins[pin['path']] = pin['sha256']
    for path, sha in source_pins.items():
        assert digest(path) == sha
    for path, sha in mapped['numbering_pins'].items():
        assert digest(path) == sha
    aa = pq.read_table(args.remap / 'observed-component-checks.parquet').to_pylist()
    assert len(aa) == len(native) == len({r['unit_id'] for r in aa})
    assert all(r['release_active'] is False and r['full_pool_screen_complete'] is False for r in aa)
    original = pq.read_table(args.root / 'report/proposed-assignments.parquet').to_pylist()
    removed = {r['unit_id'] for r in original if 'general_protein_holdout_70_80' in r['reason_codes']}
    pairs = pq.read_table(args.remap / 'observed-generic-pair-checks.parquet').to_pylist()
    by_unit = defaultdict(list)
    for row in pairs:
        for uid in {row['train_unit'], row['validation_unit']} & removed:
            by_unit[uid].append(row['status'])
    assert by_unit.keys() == removed
    rows = []
    for uid, statuses in sorted(by_unit.items()):
        if 'sequence_overlap_persists_at_observed_scope' in statuses:
            status = 'recorded_generic_overlap_persists'
        elif 'counterpart_native_scope_unassessed' in statuses:
            status = 'counterpart_scope_unassessed'
        else:
            status = 'recorded_links_not_reproduced_requires_full_pool_rescreen'
        rows.append(dict(unit_id=uid, status=status, recorded_pair_edges=len(statuses), release_active=False))
    write_rows(args.remap / 'original-generic-unit-findings.parquet', rows)
    reasons = defaultdict(Counter)
    for row in aa:
        for reason in row['original_selection_reasons']:
            if not reason.startswith('native_scope_reuse_pilot'):
                reasons[reason][row['finding']] += 1
    assert {k: dict(v) for k, v in reasons.items()} == mapped['by_original_selection_reason']
    report = dict(status='passed', release_ready=False, native_units=len(native),
        native_input_pins_verified=len(source_pins), coordinate_inputs_unchanged=True,
        observed_component_checks_sha256=digest(args.remap / 'observed-component-checks.parquet'),
        observed_generic_pair_checks_sha256=digest(args.remap / 'observed-generic-pair-checks.parquet'),
        original_generic_unit_findings=dict(Counter(r['status'] for r in rows)),
        inventory_bridges_at_observed_scope=reasons['bridges_authoritative_train_validation'],
        scope_summary_sha256=digest(args.scope / 'summary.json'),
        remap_summary_sha256=digest(args.remap / 'summary.json'), code_sha256=digest(__file__),
        no_release_or_teacher_activation=True)
    atomic_json(target, report)
    print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'scope', 'remap']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    run(p.parse_args())
