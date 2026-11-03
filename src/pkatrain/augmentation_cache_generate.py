"""Resumable production feature build from an immutable coordinate release.

Every process owns its packed writers; large graphs never enter an IPC queue.
The outer manifest is the publication marker. Native graphs are not duplicated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import heapq
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from . import augmentation_cache_pilot as pilot
from .augmentation_store import (AugmentationStore, ReleaseWriter, canonical,
                                finalize_workers, fingerprint)

DTYPES = {name: ('|b1' if name.endswith('mask') else '<f4' if name in
    ('nodes', 'edge', 'switch', 'site_edge', 'site_switch') else '<i4')
    for name in pilot.feature_shapes(1, 1, 1, 1, 1)}
RECIPE_FILES = (*pilot.FEATURE_SOURCE_FILES, 'pkatrain/augmentation_cache_generate.py',
    'pkatrain/augmentation_cache_pilot.py', 'pkatrain/augmentation_store.py',
    'pkatrain/augmentation_sources.py', 'pkatrain/augmentation_graphs.py')


def assign_workers(items, count):
    """Stable longest-first assignment bounds each process's retained inventory."""
    if count < 1: raise ValueError('workers must be positive')
    bins = [[] for _ in range(count)]; heap = [(0, i) for i in range(count)]
    heapq.heapify(heap)
    for item in sorted(items, key=lambda x: (-x['summary']['n'], x['parent'])):
        cost, slot = heapq.heappop(heap); bins[slot].append(item)
        heapq.heappush(heap, (cost + (item['summary']['n'] + 50)**2, slot))
    return bins


def validate_feature(record, row):
    m, arrays = record['metadata'], record['arrays']; dims = m['unit_dims']
    shapes = pilot.feature_shapes(*dims)
    if set(arrays) != set(shapes): raise ValueError('feature fields differ')
    for key, shape in shapes.items():
        if arrays[key].shape != shape or arrays[key].dtype.str != DTYPES[key]:
            raise ValueError('feature layout differs: ' + key)
    if any(m[k] != row['summary'][k] for k in row['summary']):
        raise ValueError('feature header/index metadata differ')
    n, _, q, s, _ = dims
    for field, limit, length in [('residue_ids', m['dims'][0], n),
                                  ('site_ids', m['dims'][3], s),
                                  ('query_ids', m['dims'][2], q)]:
        a = arrays[field]
        if len(a) != length or np.any(a < 0) or np.any(a >= limit) or np.any(np.diff(a) <= 0):
            raise ValueError('invalid canonical map: ' + field)
    for field, mask, limit in [('neighbors', 'edge_mask', n),
                                ('site_neighbors', 'site_edge_mask', s)]:
        a = arrays[field][arrays[mask]]
        if np.any(a < 0) or np.any(a >= limit): raise ValueError('invalid local neighbor')
    for field, limit in [('query_residue', n), ('site_residue', n)]:
        if np.any(arrays[field] < 0) or np.any(arrays[field] >= limit):
            raise ValueError('invalid local residue index')
    if np.any(arrays['query_site'] < -1) or np.any(arrays['query_site'] >= s):
        raise ValueError('invalid local query site')


def array_hashes(arrays):
    return {name: hashlib.sha256(canonical(dict(shape=list(value.shape), dtype=value.dtype.str))
        + np.ascontiguousarray(value).tobytes()).hexdigest() for name, value in sorted(arrays.items())}


@contextmanager
def writable_or_published(path, provenance):
    """A worker can resume after either of its two manifests has been published."""
    if (path/'manifest.json').exists():
        with AugmentationStore(path) as store:
            if store.manifest['provenance'] != provenance: raise ValueError('published provenance differs')
            yield None, store.by_id
    else:
        with ReleaseWriter(path, provenance, zstd_level=1) as writer:
            yield writer, writer.ids


def run_worker(task):
    output, coordinates, runtime, slot, cpu, items, provenance = task
    os.sched_setaffinity(0, {cpu}); started = time.monotonic()
    root = Path(output); suffix = f'{slot:04d}'
    feature_path = root/'features/workers'/suffix; parent_path = root/'parents/workers'/suffix
    worker_provenance = dict(provenance, worker=slot,
        parents_sha256=fingerprint([x['parent'] for x in items]))
    expected_features = []; expected_parents = [x['parent'] for x in items]
    status_path = root/f'progress/worker-{suffix}.json'; counts = Counter()
    with AugmentationStore(coordinates) as source, \
            writable_or_published(feature_path, worker_provenance) as (features, feature_ids), \
            writable_or_published(parent_path, worker_provenance) as (parents, parent_ids):
        pilot.COORDINATE_STORE = source
        # Parent checkpoints are committed only after their feature checkpoints.
        old = {r['id']: r['summary'] for r in parents.rows} if parents else {
            key: value['summary'] for key, value in parent_ids.items()}
        try:
            for position, item in enumerate(items):
                parent = item['parent']; n_selected = item['summary']['selected']
                if parent in parent_ids:
                    info = old[parent]
                else:
                    result = pilot._build_parent((coordinates, runtime, item))
                    if result['status'] == 'feature_failed':
                        atomic_json(status_path, dict(status='failed', parent=parent,
                            error=result['error'], traceback=result['traceback'], completed=position))
                        raise RuntimeError(parent + ': ' + result['error'])
                    info = dict(source_parent=parent, dataset=item['summary']['dataset'],
                        split=item['summary']['split'], state=item['summary']['state'],
                        source_status=item['summary']['status'], status=result['status'],
                        coordinate_record_sha256=source.by_id[parent]['sha256'],
                        candidates=[], wall_s=result['wall_s'])
                    native_arrays = {}; metadata = dict(info)
                    if result['status'] == 'built':
                        control = result['control']; key = result['control_key']; n = result['n']
                        native_dims = [n, control['neighbors'].shape[1], len(control['query_residue']),
                            len(control['site_residue']), control['site_neighbors'].shape[1]]
                        hashes = array_hashes(control)
                        info.update(control_key=key, native_dims=native_dims,
                            native_fingerprint=fingerprint(hashes),
                            candidates=[v['candidate'] for v in result['views']],
                            peak_rss_GiB=result['peak_rss_GiB'])
                        metadata = dict(info, native_field_sha256=hashes)
                        native_arrays = dict(partners=control['partners'])
                        for view in result['views']:
                            c = view['candidate']; rid = parent + f'/c{c:02d}'; g = view['arrays']
                            unit_dims = [len(g['nodes']), g['neighbors'].shape[1], len(g['query_residue']),
                                len(g['site_residue']), g['site_neighbors'].shape[1]]
                            dims = [max(a, b) if i in (1, 4) else a
                                for i, (a, b) in enumerate(zip(native_dims, unit_dims))]
                            meta = dict(source_parent=parent, control_key=key, dataset=info['dataset'],
                                split=info['split'], state=info['state'], candidate=c,
                                kind='zero' if c == 0 else 'augmented', dims=dims, unit_dims=unit_dims,
                                weight=1., stress_only=False, stratum=item['stratum'], geometry_labels_reused=False)
                            validate_feature(dict(metadata=meta, arrays=g), dict(summary=meta))
                            if rid not in feature_ids:
                                if features is None: raise ValueError('published worker is missing a view')
                                features.add(rid, meta, g, summary=meta)
                    expected_count = n_selected + 1 if n_selected else 0
                    if len(info['candidates']) != expected_count:
                        raise ValueError('coordinate/feature view count differs: ' + parent)
                    if parents is None: raise ValueError('published worker is missing a parent')
                    parents.add(parent, metadata, native_arrays, summary=info)
                ids = [parent + f'/c{c:02d}' for c in info['candidates']]
                if any(rid not in feature_ids for rid in ids): raise ValueError('committed parent lost a view')
                expected_features.extend(ids); counts[info['status']] += 1
                if (position + 1) % 16 == 0:
                    if features: features.checkpoint()
                    if parents: parents.checkpoint()
                    atomic_json(status_path, dict(status='building', completed=position+1,
                        total=len(items), records=len(expected_features), counts=dict(counts),
                        wall_s=time.monotonic()-started))
            if features: features.finish(expected_ids=expected_features)
            elif set(feature_ids) != set(expected_features): raise ValueError('published worker inventory differs')
            if parents: parents.finish(expected_ids=expected_parents)
            elif set(parent_ids) != set(expected_parents): raise ValueError('published parent inventory differs')
        finally:
            pilot.COORDINATE_STORE = None
    result = dict(status='complete', worker=slot, parents=len(items), records=len(expected_features),
        counts=dict(counts), wall_s=time.monotonic()-started)
    atomic_json(status_path, result)
    return result


def publish_workers(root, kind, assignments, provenance, expected, *, check_record=None):
    path = root/kind
    if (path/'manifest.json').exists():
        with AugmentationStore(path) as store:
            if store.manifest['provenance'] != provenance or set(store.by_id) != set(expected):
                raise ValueError('published consolidated inventory/provenance differs')
            store.verify_files()
            return store.manifest
    workers = [path/'workers'/f'{i:04d}' for i in range(len(assignments))]
    for i, worker in enumerate(workers):
        m = json.loads((worker/'manifest.json').read_text())
        expected_provenance = dict(provenance, worker=i,
            parents_sha256=fingerprint([x['parent'] for x in assignments[i]]))
        if m['provenance'] != expected_provenance: raise ValueError('worker provenance differs')
    return finalize_workers(path, workers, provenance, expected_ids=expected, check_record=check_record)


def _add_capacity(capacities, dataset, dims):
    n = dims[0]; bucket = next((b for b in pilot.BOUNDS if n <= b), None)
    if bucket is None: raise ValueError('native complex exceeds supported training buckets')
    key = dataset + ':' + str(bucket); d = [bucket, *dims[1:]]
    capacities[key] = [max(a, b) for a, b in zip(capacities.get(key, d), d)]


def consolidate(root, assignments, provenance, coordinate_rows, *, canary_reference=None):
    """Coverage, compact native partitions, one full payload verification, marker last."""
    inventory = sorted((x for group in assignments for x in group), key=lambda x:x['parent'])
    parent_ids = [x['parent'] for x in inventory]
    publish_workers(root, 'parents', assignments, provenance, parent_ids)
    coordinate_index = {r['id']: r for r in coordinate_rows}; expected = []; counts = Counter()
    native_capacities = {}; capacities = {}; native_signatures = {}; native_ids = []
    with AugmentationStore(root/'parents') as parents, \
            writable_or_published(root/'partitions', provenance) as (partitions, partition_ids):
        for item in inventory:
            row = parents.by_id[item['parent']]; info = row['summary']; cs = info['candidates']
            src = coordinate_index[item['parent']]; selected = src['summary']['selected']
            if info['coordinate_record_sha256'] != src['sha256'] or info['source_status'] != src['summary']['status']:
                raise ValueError('parent source lineage differs')
            if len(cs) != (selected+1 if selected else 0) or len(set(cs)) != len(cs) or (cs and (cs[0] != 0 or any(c <= 0 for c in cs[1:]))):
                raise ValueError('parent candidate coverage differs')
            expected.extend(item['parent'] + f'/c{c:02d}' for c in cs); counts[info['status']] += 1
            if not cs:
                if info['status'] != 'coordinate_unavailable': raise ValueError('unexpected absent features')
                continue
            if info['status'] != 'built': raise ValueError('unexpected feature failure')
            _add_capacity(native_capacities, info['dataset'], info['native_dims'])
            key = info['control_key']; signature = info['native_fingerprint']
            if key in native_signatures:
                if native_signatures[key] != signature: raise ValueError('partners disagree on native complex')
                continue
            native_signatures[key] = signature; native_ids.append(key)
            if key not in partition_ids:
                record = parents.raw(item['parent']); m = record['metadata']
                if partitions is None: raise ValueError('published partition missing')
                meta = {k:m[k] for k in ('dataset','split','control_key','native_dims','native_fingerprint','native_field_sha256')}
                partitions.add(key, meta, record['arrays'], summary={k:meta[k] for k in ('dataset','split','native_dims','native_fingerprint')})
        if partitions: partitions.finish(expected_ids=native_ids)
        elif set(partition_ids) != set(native_ids): raise ValueError('native partition inventory differs')
    atomic_json(root/'phase.json', dict(phase='verify-features', expected_records=len(expected),
        counts=dict(counts), training_enabled=False))
    manifest = publish_workers(root, 'features', assignments, provenance, expected, check_record=validate_feature)
    # Emit fast-decoder layouts from checked index metadata, without rereading payloads.
    with AugmentationStore(root/'features') as features:
        dimensions = np.empty((len(features.rows), 5), np.int32)
        for i, row in enumerate(features.rows):
            m = row['summary']; dimensions[i] = m['unit_dims']
            _add_capacity(capacities, m['dataset'], m['dims'])
        np.save(root/'feature-layouts.npy', dimensions)
    atomic_json(root/'feature-layouts.json', dict(
        feature_manifest_sha256=digest(root/'features/manifest.json'),
        layout_sha256=digest(root/'feature-layouts.npy'), records=len(expected), dtypes=DTYPES,
        shape_definition='base-A free partner, sorted arrays in KAAR1; validated header/body and canonical maps'))
    summary = dict(passed=True, parents=len(inventory), counts=dict(counts), records=len(expected),
        native_complexes=len(native_ids), zstd_level=1, shard_bytes=2*1024**3,
        feature_bytes=manifest['compressed_bytes'], feature_manifest_sha256=digest(root/'features/manifest.json'),
        feature_source_sha256=provenance['source_sha256'], capacities=capacities,
        native_capacities=native_capacities, training_enabled=False,
        storage='free partner views plus compact native partitions; live native graphs required',
        accepted_pilot_exception=provenance['accepted_pilot_exception'])
    atomic_json(root/'build-summary.json', summary)
    if canary_reference: check_canary(root, canary_reference)
    files = {}
    for kind in ('features','parents','partitions'):
        m = json.loads((root/kind/'manifest.json').read_text())
        for name, sha in m['files'].items(): files[kind+'/'+name] = sha
        files[kind+'/manifest.json'] = digest(root/kind/'manifest.json')
    for name in ('inventory.json','feature-layouts.npy','feature-layouts.json','build-summary.json'):
        files[name] = digest(root/name)
    if canary_reference: files['canary-verification.json'] = digest(root/'canary-verification.json')
    inodes = sum(1 for _ in root.rglob('*'))
    if inodes > 5000: raise ValueError('working release exceeds 5,000 inode allowance')
    result = dict(format='augmentation-feature-cache-v1', verified=True, provenance=provenance,
        files=files, records=len(expected), parents=len(inventory), reader_files=len(files)+1,
        working_inodes_at_publish=inodes, training_enabled=False)
    atomic_json(root/'manifest.json', result)
    return result


def canary_items(rows, sample_path):
    """Small pilot-overlapping cohort covers size extremes, partners and failures."""
    sample = json.loads(Path(sample_path).read_text())['parents']; groups = {}
    for item in sample:
        s = item['summary']; key = (s['dataset'], s['split'], s['state'])
        groups.setdefault(key, []).append(item)
    chosen = {}
    for group in groups.values():
        eligible = [x for x in group if x['summary']['selected']]
        if eligible:
            for item in (min(eligible, key=lambda x:x['summary']['n']), max(eligible, key=lambda x:x['summary']['n'])):
                chosen[item['parent']] = item
    for item in sample:
        if item['summary']['status'] != 'usable': chosen[item['parent']] = item
    return [r for r in rows if r['id'] in chosen]


def check_canary(root, reference):
    checked = 0
    with AugmentationStore(root/'features') as new, AugmentationStore(Path(reference)/'features') as old:
        for row in new.rows:
            a, b = new.raw(row['id'])['arrays'], old.raw(row['id'])['arrays']
            if set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in a):
                raise ValueError('canary differs from independently published pilot: ' + row['id'])
            checked += 1
    result = dict(passed=True, records_bitwise_equal=checked, reference_feature_manifest_sha256=
        digest(Path(reference)/'features/manifest.json'))
    atomic_json(root/'canary-verification.json', result)
    return result


def build(args):
    cpus = int(os.environ['SLURM_CPUS_PER_TASK']); require_compute(threads=cpus, allow_comp1400=True)
    if os.environ['SLURMD_NODENAME'].split('.')[0] != 'comp1400': raise RuntimeError('this launch is restricted to comp1400')
    if not 1 <= args.workers <= cpus: raise ValueError('workers exceed CPU allocation')
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    with (root/'.build.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'manifest.json').exists():
            m = json.loads((root/'manifest.json').read_text())
            for name, sha in m['files'].items():
                if digest(root/name) != sha: raise ValueError('published release changed')
            print(json.dumps(dict(status='already-published', records=m['records'])), flush=True)
            return
        started = time.monotonic(); source_root = Path(__file__).parents[1]
        with AugmentationStore(args.coordinates) as coordinates:
            coordinate_rows = coordinates.rows
            rows = canary_items(coordinate_rows, args.canary_reference/'sample.json') if args.canary_reference else coordinate_rows
            items = [dict(parent=r['id'], summary=r['summary'], stratum=pilot.stratum(r),
                          weight=1., stress_only=False) for r in rows]
        assignments = assign_workers(items, args.workers)
        inventory = dict(coordinate_manifest_sha256=digest(args.coordinates/'manifest.json'),
            workers=args.workers, assignments=assignments, scope='canary' if args.canary_reference else 'PINDER+pKPDB train+validation')
        inventory_path = root/'inventory.json'
        if inventory_path.exists():
            if json.loads(inventory_path.read_text()) != inventory: raise ValueError('resume inventory differs')
        else: atomic_json(inventory_path, inventory)
        provenance = dict(kind='free-partner-feature-cache-v1',
            coordinate_manifest_sha256=inventory['coordinate_manifest_sha256'],
            inventory_sha256=digest(inventory_path),
            source_sha256={f:digest(source_root/f) for f in pilot.FEATURE_SOURCE_FILES},
            recipe_sha256={f:digest(source_root/f) for f in RECIPE_FILES},
            storage='unit-only KAAR1 zstd-level-1 2GiB; compact native partitions; no native control graphs',
            accepted_pilot_exception='User approved proceeding after pilot: smallest PINDER bucket retains 87% versus 90% target; weighted 98%',
            training_enabled=False)
        estimate = 1.86*10**12 if not args.canary_reference else 10*1024**3
        if shutil.disk_usage(root).free < estimate: raise RuntimeError('insufficient storage headroom')
        atomic_json(root/'phase.json', dict(phase='build-workers', workers=args.workers, parents=len(items), training_enabled=False))
        allowed = sorted(os.sched_getaffinity(0))
        tasks = [(root, args.coordinates, args.runtime, i, allowed[i], group, provenance)
            for i, group in enumerate(assignments)]
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_worker, task) for task in tasks]
            for future in as_completed(futures): print(json.dumps(future.result()), flush=True)
        result = consolidate(root, assignments, provenance, coordinate_rows, canary_reference=args.canary_reference)
        atomic_json(root/'phase.json', dict(phase='complete', wall_s=time.monotonic()-started,
            records=result['records'], training_enabled=False))
        print(json.dumps(dict(status='published', records=result['records'], parents=result['parents'],
            reader_files=result['reader_files'], wall_s=time.monotonic()-started)), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--coordinates', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--canary-reference', type=Path)
    args = p.parse_args(); build(args)


if __name__ == '__main__': main()
