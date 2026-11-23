"""Verify inactive native training manifests, all SAbDab targets, and real batches."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain import production_graphs as pg, production_loading as pl


def read(path):
    return json.loads(Path(path).read_text())


def manifests(root):
    directory = root / 'candidate-manifests-v2'
    index = read(directory / 'candidate.json')
    assert index['release_active'] is False
    result = {}
    for entry in index['sources']:
        assert digest(entry['manifest']) == entry['manifest_sha256']
        result[entry['dataset']] = read(entry['manifest'])
    return result


def semantic(args):
    import pyarrow.parquet as pq
    from jaxpropka.parameters import GROUPS, ALPHABET
    from pkatrain.augmentation_store import AugmentationStore
    from pkabench.training_pools import min_fractions, SEED

    started = time.monotonic()
    root = args.root
    views = manifests(root)
    members = pq.read_table(root / 'pools-v2/membership.parquet').to_pylist()
    expected = {r['unit_id']: r for r in members}
    assert len(expected) == len(members)
    train = [r for r in members if r['operational_role'] == 'train']
    validation = [r for r in members if r['operational_role'] == 'validation']
    assert not ({r['group'] for r in train} & {r['group'] for r in validation})
    ranks = min_fractions({r['group']: r['stratum'] for r in train}, seed=SEED, version='pool-v7-native-v1')
    for r in train:
        assert abs(r['min_fraction'] - ranks[r['group']]) <= 5.1e-10
    found = set()
    norms = {}
    for dataset, view in views.items():
        assert view['release_active'] is False
        for row in view['records']:
            uid = dataset + ':' + row['id']
            assert uid not in found
            found.add(uid)
            original = expected[uid]
            assert row['operational_role'] == original['operational_role']
            assert row['split'] == ('train' if original['operational_role'] == 'train' else 'val')
            if row['split'] == 'train':
                assert row['min_fraction'] == original['min_fraction']
            if dataset != 'pkpdb':
                assert len(row['residue_side']) == row['n'] and set(row['residue_side']) == {0, 1}
        if pl.manifest_kind(view) == 'paired':
            norms[dataset] = {}
            for fraction in [.01, .1, .25, .5, 1.]:
                selected = pl.select(view, 'train', fraction)
                if not selected:
                    continue
                calculated = pl.normalization(view, fraction)
                assert all(np.isfinite(v) and v > 0 for v in calculated.values())
                norms[dataset][str(fraction)] = dict(records=len(selected), **calculated)
    assert found == expected.keys()
    prepared = AugmentationStore(root.parent / 'sabdab-v7-build-audit-v1/sabdab/prepared')
    teachers = AugmentationStore(root.parent / 'sabdab-v7-pkai-build-v1/full/labels')
    store = pg.ProductionStore(views['sabdab']['store'])
    canary = pg.ProductionStore(root / 'graph-canary-v2/training/gqt-production-v2/sabdab/store-v2')
    records = {r['id']: r for r in read(store.path / 'records.json')}

    def check(row):
        uid = 'sabdab:' + row['id']
        frame = prepared.raw(uid)
        meta = frame['metadata']['structure']
        label = teachers.raw(uid)['metadata']
        raw = store.raw(row['id'])
        assert pg.arrays_digest(raw, store.fields) == records[row['id']]['arrays_sha256']
        assert row['teacher_frame_sha256'] == teachers.by_id[uid]['sha256']
        assert label['prepared_frame_sha256'] == prepared.by_id[uid]['sha256']
        assert row['native_coordinate_sha256'] == meta['native_coordinate_sha256']
        assert row['native_source_split'] == meta['native_source_split']
        assert row['split'] == {'train': 'train', 'test': 'val'}[meta['native_source_split']]
        mapping = meta['residue_map']
        assert len(mapping) == row['n']
        side = np.asarray([0 if m['side'] == 'A' else 1 for m in mapping], np.int8)
        assert np.array_equal(side, row['residue_side'])
        sequence = ''.join(c['observed_sequence'] for c in meta['chains'])
        assert np.array_equal(raw['nodes'][:, :20], np.eye(20, dtype=np.float32)[[ALPHABET.index(a) for a in sequence]])
        lookup = {(m['chain'], m['number'], m['insertion']): i for i, m in enumerate(mapping)}
        values = {state: {tuple(v[:4]): v[4] for v in rows} for state, rows in label['labels']['pkai'].items()}
        chosen = [s for s in label['sites'] if s['train_mask' if row['split'] == 'train' else 'eval_mask']]
        assert len(chosen) == row['q'] and any(s['interface'] for s in chosen)
        real = {(v['chain'], v['number'], v['insertion'], v['group']) for v in label['true_termini']}
        for i, site in enumerate(chosen):
            key = (site['chain'], site['resnum'], site['icode'], site['group'])
            assert key[3] != 'ARG'
            if key[3] in ('NTERM', 'CTERM'):
                assert key in real
            assert raw['query_residue'][i] == lookup[key[:3]]
            assert GROUPS[int(raw['query_group'][i])] == key[3]
            wanted = np.asarray([values['AB'][key], values[site['partner']][key]], np.float32)
            assert np.array_equal(raw['targets'][:, i], wanted)
        for field in ['w_burial', 'w_interface', 'interface', 'partner_distance_A', 'rsa_free']:
            wanted = np.asarray([s[field] for s in chosen], dtype=raw[field].dtype)
            assert np.array_equal(raw[field], wanted, equal_nan=True)
        edge = raw['edge_mask'] & (side[raw['neighbors']] == side[:, None])
        ss = side[raw['site_residue']]
        site_edge = raw['site_edge_mask'] & (ss[raw['site_neighbors']] == ss[:, None])
        assert np.array_equal(raw['branch_edge_mask'][0], raw['edge_mask'])
        assert np.array_equal(raw['branch_edge_mask'][1], edge)
        assert np.array_equal(raw['branch_site_edge_mask'][0], raw['site_edge_mask'])
        assert np.array_equal(raw['branch_site_edge_mask'][1], site_edge)
        assert np.isfinite(raw['targets']).all()
        if row['id'] in canary.by_id:
            assert pg.arrays_digest(canary.raw(row['id']), canary.fields) == pg.arrays_digest(raw, store.fields)
        return row['q']

    try:
        with ThreadPoolExecutor(args.workers) as pool:
            sites = sum(pool.map(check, views['sabdab']['records']))
    finally:
        for handle in [store, canary, prepared, teachers]:
            handle.close()
    previous = read(root / 'native-patches/summary.json')
    current = read(root / 'native-patches-v2/summary.json')
    assert set(previous['withheld_ids']) == set(current['withheld_ids'])
    report = dict(status='passed', all_catalogue_memberships_checked=len(found),
        sabdab_graphs_checked=len(views['sabdab']['records']), sabdab_teacher_queries_checked=sites,
        canary_graphs_checked=len(canary.by_id), normalization_train_only=norms,
        published_split_and_biological_side_checks=True, masks_and_targets_unchanged=True,
        nested_fraction_ranks_verified=True, operational_groups_disjoint=True,
        native_admission_sha256=digest(root / 'native-patches-v2/summary.json'),
        membership_sha256=digest(root / 'pools-v2/membership.parquet'),
        candidate_index_sha256=digest(root / 'candidate-manifests-v2/candidate.json'),
        wall_s=time.monotonic() - started, release_active=False, code_sha256=digest(__file__))
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


def loader(args):
    from pkatrain.loading import LoaderConfig, BucketPolicy
    root = args.root
    views = manifests(root)
    reports = {}
    for dataset in ['pkpdb', 'pinder', 'sabdab']:
        view = views[dataset] if dataset == 'sabdab' else read(root / 'cache-smoke-v2' / (dataset + '.json'))
        paired = pl.manifest_kind(view) == 'paired'
        extra = dict(norms=pl.normalization(views[dataset])) if paired else dict(mask='train_mask')
        source = pl.source_from_manifest(view, config=LoaderConfig(workers=args.workers), **extra)
        if dataset == 'sabdab':
            spread = defaultdict(list)
            for row in view['records']:
                spread[(row['split'], source.policy.bucket(row['n']))].append(row)
            rows = []
            for values in spread.values():
                values.sort(key=lambda r: (r['n'], r['id']))
                rows.extend([values[0], values[-1]] if len(values) > 1 else values)
        else:
            rows = view['records']
        timings = []
        try:
            for repeat in range(2):
                elapsed = []
                for row in rows:
                    began = time.perf_counter()
                    batch = source.load([row['id']])
                    elapsed.append(time.perf_counter() - began)
                    raw = source.store.raw(row['id'])
                    n, q = row['n'], row['q']
                    graph = batch[0]
                    nodes = graph['nodes'][0, 0] if paired else graph['nodes'][0]
                    assert np.array_equal(nodes[:n], raw['nodes']) and not nodes[n:].any()
                    expected = raw['targets'] if paired else raw['labels']
                    actual = batch[1][0, :, :q] if paired else batch[1][0, :q]
                    assert np.array_equal(actual, expected)
                    eligible = np.ones(q, bool) if paired else raw['train_mask']
                    assert np.array_equal(batch[2][0, :q], eligible) and not batch[2][0, q:].any()
                    assert batch[-1].sum() == 1 and batch[-1][0]
                    assert not batch[2][1:].any() and not batch[1][1:].any()
                    if paired:
                        for key, branch in [('edge_mask', 'branch_edge_mask'), ('site_edge_mask', 'branch_site_edge_mask')]:
                            size = raw[branch].shape
                            assert np.array_equal(graph[key][0, :size[0], :size[1], :size[2]], raw[branch])
                timings.append(dict(pass_number=repeat + 1, records=len(rows), wall_s=sum(elapsed),
                    median_batch_ms=float(np.median(elapsed) * 1000), p95_batch_ms=float(np.percentile(elapsed, 95) * 1000),
                    structures_per_s=len(rows) / sum(elapsed)))
            plans = BucketPolicy.from_json(view['bucket_policy']).plans(rows, np.random.default_rng(17))
            # Exercise populated batches as well as one-record padding in the exact candidate capacities.
            for ids in plans:
                batch = source.load(ids)
                assert batch[-1].sum() == len(ids)
                for i, cid in enumerate(ids):
                    raw = source.store.raw(cid)
                    q = source.by_id[cid]['q']
                    actual = batch[1][i, :, :q] if paired else batch[1][i, :q]
                    assert np.array_equal(actual, raw['targets'] if paired else raw['labels'])
            reports[dataset] = dict(source_class=type(source).__name__, sampled_records=len(rows),
                sampled_roles=dict(Counter(r['split'] for r in rows)), populated_batches=len(plans), timings=timings,
                native_overlay_records=sum(r['id'] in getattr(source.store, 'routes', {}) for r in rows),
                correct_native_targets_masks_padding_and_branch_edges=True)
        finally:
            source.close()
    report = dict(status='passed', datasets=reports, workers=args.workers, cpu_only=True,
        cost_scope='Local host batch assembly in candidate capacities; cache is not evicted between passes. No model, GPU or target-filesystem benchmark.',
        candidate_index_sha256=digest(root / 'candidate-manifests-v2/candidate.json'), code_sha256=digest(__file__),
        release_active=False)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['semantic', 'loader'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    require_compute(threads=args.workers + 1)
    if args.output.exists():
        raise FileExistsError(args.output)
    {'semantic': semantic, 'loader': loader}[args.mode](args)
