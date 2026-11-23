"""Bounded native-scope screen; a matching PDB ID never authorizes cache reuse."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest, require_compute
from sabdab_v7_native_scope import check_one


def read(path):
    return json.loads(Path(path).read_text())


def run(args):
    require_compute(threads=args.workers, allow_comp1400=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    pilot = pq.read_table(args.pilot / 'preparation-results.parquet').to_pylist()
    prepared = {r['unit_id']: r for r in read(args.pilot / 'prepared-units.json')['units']}
    candidates = sorted({u for r in pilot if r['status']=='prepared' for u in r['existing_parent_candidates']})
    selected = candidates[:args.max_parents]
    cache = {r['unit_id']: r for r in pq.read_table(args.scope_cache).to_pylist()}
    units = {r['unit_id']: r for r in pq.read_table(args.audit / 'units.parquet').to_pylist()}
    inventory = Path(read(args.audit / 'inputs.json')['inventory'])
    manifest = read(inventory / 'inventory-manifest.json')
    tasks = [(manifest['runtime'], dict(units[uid], _baseline_root=manifest['baseline']), [], ['sabdab-native-reuse-pilot'])
             for uid in selected if uid not in cache]
    with ProcessPoolExecutor(args.workers) as pool:
        checks = list(pool.map(check_one, tasks))
    cache.update({r['unit_id']: r for r in checks})
    atomic_json(args.output / 'new-native-scope-checks.json', checks)
    rows = []
    for row in pilot:
        if row['unit_id'] not in prepared:
            continue
        meta = read(Path(prepared[row['unit_id']]['entry']) / 'structure.json')
        sequences = Counter(c['observed_sequence'] for c in meta['chains'])
        for peer in row['existing_parent_candidates']:
            scope = cache.get(peer) if peer in selected else None
            status = 'not_assessed_parent_limit'
            if scope is not None:
                if scope['status'] != 'checked':
                    status = 'native_scope_unresolved'
                elif Counter(c['sequence'] for c in scope['chains']) != sequences:
                    status = 'different_observed_assembly_scope'
                else:
                    status = 'matching_chain_sequences_requires_atom_partition_and_teacher_checks'
            rows.append(dict(unit_id=row['unit_id'], candidate_parent=peer, scope_status=status,
                wholesale_reuse_approved=False,
                paired_teacher_status='historical_site_labels_do_not_supply_paired_pkai_targets'
                    if peer.startswith('pkpdb:') else 'teacher_equivalence_not_established'))
    atomic_json(args.output / 'reuse-screen.json', dict(status='screen_complete_not_reuse_authorization',
        records=rows, candidate_pairs=len(rows), unique_parent_candidates=len(candidates),
        parent_limit=args.max_parents, parents_selected=len(selected), parents_newly_checked=len(checks),
        scope_status_counts=dict(Counter(r['scope_status'] for r in rows)), approved_reuses=0,
        pilot_sha256=digest(args.pilot / 'prepared-units.json'),
        cached_scope_sha256=digest(args.scope_cache), code_sha256=digest(__file__),
        limitation='A matching sequence multiset is necessary but insufficient for wholesale cache reuse; no coordinate or teacher identity is inferred.',
        active_release_changed=False))
    print(json.dumps(dict(candidate_pairs=len(rows), scope_status_counts=dict(Counter(r['scope_status'] for r in rows)),
                          approved_reuses=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['pilot', 'audit', 'scope-cache', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-parents', type=int, default=32)
    run(parser.parse_args())
