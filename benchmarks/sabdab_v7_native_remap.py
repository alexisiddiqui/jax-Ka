"""Check component and recorded generic-overlap evidence at observed native scope.

This is a bounded diagnostic, not an activation or a replacement whole-pool
screen. A disappearing inventory edge does not automatically release a unit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest, require_compute
from pkabench.role_engine import Evidence, PUBLISHED, Unit, assign
from pkabench.source_registry import SourceRegistry
from pkabench.split_authority import SplitAuthority
from sabdab_v7_exposure import read
from sabdab_v7_roles import cdr_components, numbering_assessed, write_rows
from sabdab_v7_antigen import compare_chunk, initialise
from sabdab_v7_screen import mmsearch
from sabdab_v7_adjudicate import checked_context


def run(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    if args.output.exists():
        raise FileExistsError('Choose a new native remapping output')
    args.output.mkdir(parents=True)
    ctx = checked_context(args.root); data = ctx['data']
    scope_summary = read(args.scope / 'summary.json')
    path = args.scope / 'native-scope-checks.parquet'
    if digest(path) != scope_summary['scope_receipt_sha256']:
        raise ValueError('Native scope receipt changed')
    native = [r for r in pq.read_table(path).to_pylist() if r['status'] == 'checked']
    observed = read(args.scope / 'observed-sequences.json')
    sequences = dict(data['sequences'])
    for k, seq in observed.items():
        if k in sequences and sequences[k] != seq:
            raise ValueError('Observed sequence digest collision')
        sequences[k] = seq
    raw = dict(ctx['numbered']); attempted = set()
    inventory = Path(data['inventory'])
    attempted.update(read(inventory / 'numbering-cached.json')['attempted'])
    attempted.update(read(inventory / 'numbering-new.json')['numbered'])
    number_pins = {}
    for number_file in args.numbering:
        extra = read(number_file)
        raw.update(extra['numbered']); attempted.update(extra['numbered'])
        number_pins[str(number_file)] = digest(number_file)
    if not observed.keys() <= attempted:
        raise ValueError('Observed chain numbering is incomplete')
    original_members = defaultdict(dict)
    for m in ctx['members']:
        original_members[m['unit_id']][m['chain_id']] = m['annotation_role']
    memberships = defaultdict(list); query_keys = set()
    for r in native:
        uid = r['unit_id']; fv = r['input_scope_id'] == 'v6c-native-observed-Fv-coordinates'
        for c in r['chains']:
            chain, k = c['chain_id'], c['sequence_key']
            annotation = ('antibody_' + chain if chain in {'H', 'L'} else 'antigen') if fv else original_members[uid].get(chain, 'protein_context_role_unresolved')
            memberships[uid].append(dict(chain_id=chain, sequence_key=k, annotation_role=annotation))
            if not annotation.startswith('antibody') and (not raw.get(k) or annotation in {'antigen', 'antigen_context'}):
                query_keys.add(k)
    cd = cdr_components(raw, set(observed), data['official'], args.workers)
    chunks = [sorted(query_keys)[i:i+16] for i in range(0, len(query_keys), 16)]
    gh = {}; pairs = 0; started = time.monotonic()
    with ProcessPoolExecutor(args.workers, initializer=initialise, initargs=(sequences, data['targets'])) as pool:
        for result in pool.map(compare_chunk, chunks):
            pairs += result['pairs']
            gh.update((h['sequence_key'], h['hits']) for h in result['hits'])
    with gzip.open(args.output / 'observed-antigen-hits.jsonl.gz', 'wt') as f:
        for k, hits in sorted(gh.items()):
            f.write(json.dumps(dict(sequence_key=k, hits=hits)) + '\n')
    specs, evidence = [], []
    for r in native:
        uid = r['unit_id']; u = ctx['units'][uid]
        ab, ag = set(), set(); complete = True
        for m in memberships[uid]:
            k = m['sequence_key']; ab.update(cd.get(k, []))
            if k in query_keys:
                ag.update(c for h in gh.get(k, []) for c in h['components'])
            complete &= numbering_assessed(k, m['annotation_role'], raw, attempted)
        specs.append(Unit(uid, u['source'], u['native_split'], 'verified_model_observed_coordinates',
            u['native_component'], u['supervised'], u['declared_general']))
        for axis, comp in [('antibody_CDR', ab), ('antigen', ag)]:
            evidence.append(Evidence(uid, PUBLISHED, axis, 'observed_native_scope',
                'hit' if comp else 'no_hit', tuple(sorted(comp))))
        if not complete:
            evidence.append(Evidence(uid, PUBLISHED, 'antibody_CDR', 'partial_observed_domain', 'unassessed'))
    aa = assign(specs, evidence, data['component_splits'], SplitAuthority.load(args.root / 'authority.json'), SourceRegistry.load(args.root / 'registry.json'))
    old_base = {a.unit_id: a for a in ctx['baseline']}; native_by_id = {r['unit_id']: r for r in native}
    component_roles = {c: {'train': 'train', 'test': 'validation'}[s] for c, s in data['component_splits'].items()}
    rows, by_reason = [], defaultdict(Counter)
    for a in aa:
        before = old_base[a.unit_id]
        old_roles = {component_roles[c] for c in before.matched_components}
        new_roles = {component_roles[c] for c in a.matched_components}
        if len(new_roles) > 1:
            finding = 'component_bridge_in_observed_scope'
        elif a.operational_role == 'pending':
            finding = 'observed_mapping_pending'
        elif len(old_roles) > 1:
            finding = 'single_component_role_at_observed_scope_requires_pool_rescreen'
        elif set(before.matched_components) == set(a.matched_components):
            finding = 'component_memberships_unchanged'
        else:
            finding = 'component_memberships_changed_requires_pool_rescreen'
        row = dict(asdict(a), inventory_component_role=before.operational_role,
            inventory_components=list(before.matched_components), finding=finding,
            original_selection_reasons=native_by_id[a.unit_id]['selected_reasons'],
            release_active=False, full_pool_screen_complete=False)
        rows.append(row)
        for reason in row['original_selection_reasons']:
            if not reason.startswith('native_scope_reuse_pilot'):
                by_reason[reason][finding] += 1
    write_rows(args.output / 'observed-component-checks.parquet', rows)
    # Revisit the recorded generic conflicts using the actual opposing native
    # contexts, keeping the original pair direction and declared MMseqs rule.
    old_final = {a.unit_id: a for a in ctx['old']}
    general_removed = {uid for uid, a in old_final.items() if 'general_protein_holdout_70_80' in a.reason_codes}
    pairs_to_check = [p for p in pq.read_table(args.root / 'report/general-holdout-evidence.parquet').to_pylist()
                      if {p['train_unit'], p['validation_unit']} & general_removed]
    eligible = {uid: {m['sequence_key'] for m in mm if not raw.get(m['sequence_key']) and not m['annotation_role'].startswith('antibody')}
                for uid, mm in memberships.items()}
    qkeys = {k for p in pairs_to_check for k in eligible.get(p['train_unit'], set())}
    tkeys = {k for p in pairs_to_check for k in eligible.get(p['validation_unit'], set())}
    hits = mmsearch(args.output / 'observed-generic', {k: sequences[k] for k in qkeys},
        {k: sequences[k] for k in tkeys}, ctx['executable'], args.workers)
    pair_rows = []
    for pair in pairs_to_check:
        tr, va = pair['train_unit'], pair['validation_unit']
        if tr not in memberships or va not in memberships:
            status = 'counterpart_native_scope_unassessed'
        else:
            present = any(hits.get(k, set()) & eligible[va] for k in eligible[tr])
            status = 'sequence_overlap_persists_at_observed_scope' if present else 'old_overlap_not_reproduced_at_observed_scope_requires_pool_rescreen'
        pair_rows.append(dict(train_unit=tr, validation_unit=va, status=status,
            query_sequence_keys=sorted(eligible.get(tr, set())), target_sequence_keys=sorted(eligible.get(va, set()))))
    write_rows(args.output / 'observed-generic-pair-checks.parquet', pair_rows)
    for r in native:
        for pin in r['input_pins']:
            if digest(pin['path']) != pin['sha256']:
                raise ValueError('Native input changed: ' + pin['path'])
    atomic_json(args.output / 'summary.json', dict(status='bounded_native_scope_remapping_complete', release_ready=False,
        native_units_checked=len(native), observed_unique_sequences=len(observed), antigen_queries=len(query_keys),
        global_pairs=pairs, by_original_selection_reason={k: dict(v) for k, v in by_reason.items()},
        original_generic_pair_findings=dict(Counter(r['status'] for r in pair_rows)),
        actual_native_scope_sha256=digest(path), numbering_pins=number_pins,
        labels_or_features_built=False, input_structures_changed=False, code_sha256=digest(__file__),
        limitations=['This covers selected contexts, not a whole-pool native-scope screen.',
            'Observed scope changes never establish novel hard-test eligibility or new teacher compatibility.',
            'A disappearing original generic edge is not a full screen against the final development corpus.'],
        wall_s=time.monotonic()-started))
    print(json.dumps(read(args.output / 'summary.json'), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'scope', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--numbering', type=Path, nargs='+', required=True); p.add_argument('--workers', type=int, default=32)
    run(p.parse_args())
