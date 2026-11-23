"""Source routing and split guards, using explicitly synthetic tensor fixtures."""
import copy
import json

import numpy as np
import pytest

from pkatrain import production_graphs as pg
from pkatrain import production_loading as pl
from pkatrain.loading import BucketPolicy, LoaderConfig


def store_fixture(tmp_path):
    fields = pg.FIELDS['pinder']
    dims = dict(n=4, k=2, q=1, s=2, sk=1)
    arrays = {}
    indices = {'neighbors', 'query_residue', 'query_group', 'site_residue', 'site_type', 'site_neighbors', 'query_site'}
    for field in fields:
        dtype = np.int32 if field in indices else bool if field.endswith('mask') or field == 'interface' else np.float32
        arrays[field] = np.ones(pg.shape(field, *dims.values()), dtype=dtype)
    arrays['targets'][:] = [[4.1], [4.7]]
    out = pg.output(tmp_path, 'sabdab'); out.mkdir(parents=True)
    records = [dict(id=cid, split=split, **dims, arrays_sha256=pg.arrays_digest(arrays, fields))
               for cid, split in [('fixture-train', 'train'), ('fixture-heldout', 'val')]]
    (out / 'ids.json').write_text(json.dumps(dict(ids=[[r['id'], r['split']] for r in records])))
    pg._write_store(out, 'sabdab', records, {f: arrays[f].dtype.str for f in fields},
                    [pg.encode(arrays, fields, level=1) for _ in records], {}, 1,
                    fields=fields, data_kind='paired', zstd_level=1)
    pool = tmp_path / 'pool.tsv'; pool.write_text('id\tmin_fraction\nfixture-train\t0.25\n')
    provenance = {r['id']: dict(source='sabdab', source_native_id=r['id'],
        native_source_split='train' if r['split'] == 'train' else 'test',
        native_component='component-' + r['id'], operational_role='train' if r['split']=='train' else 'validation',
        input_scope_id='synthetic-fixture-not-native', teacher_fingerprint='synthetic-fixture') for r in records}
    return arrays, pool, provenance


def test_native_identity_and_heldout_role_survive_packing_and_batching(tmp_path):
    arrays, pool, provenance = store_fixture(tmp_path)
    manifest = pl.build_manifest(tmp_path, 'sabdab', policy=BucketPolicy((8,), (2,)), workers=1,
        pool_path=pool, data_kind='paired', record_provenance=provenance)
    assert [r['id'] for r in pl.select(manifest, 'train')] == ['fixture-train']
    assert not pl.select(manifest, 'train', .1)
    assert pl.select(manifest, 'val')[0]['native_source_split'] == 'test'
    assert manifest['records'][0]['teacher_fingerprint'] == 'synthetic-fixture'
    source = pl.source_from_manifest(manifest, config=LoaderConfig(workers=1))
    try:
        assert type(source) is pl.PairedSource
        batch = source.load(['fixture-train'])
        assert batch[-1].tolist() == [True, False]
        assert np.array_equal(batch[1][0, :, :1], arrays['targets'])
        assert batch[2][0, 0] and not batch[2][1].any()
        assert not batch[1][1].any()
        assert source.provenance()['dataset'] == 'sabdab'
        assert source.store.metadata['zstd_level'] == 1
    finally:
        source.close()


@pytest.mark.parametrize('defect', ['wrong-role', 'wrong-published-split', 'overwrite-split', 'missing', 'extra', 'heldout-in-pool', 'bad-fraction', 'wrong-kind'])
def test_reject_inconsistent_source_manifest(tmp_path, defect):
    _, pool, provenance = store_fixture(tmp_path)
    provenance = copy.deepcopy(provenance)
    kind = 'paired'
    if defect == 'wrong-role': provenance['fixture-heldout']['operational_role'] = 'train'
    elif defect == 'wrong-published-split': provenance['fixture-train']['native_source_split'] = 'test'
    elif defect == 'overwrite-split': provenance['fixture-train']['split'] = 'val'
    elif defect == 'missing': del provenance['fixture-train']
    elif defect == 'extra': provenance['foreign'] = provenance['fixture-train']
    elif defect == 'heldout-in-pool': pool.write_text(pool.read_text() + 'fixture-heldout\t0.25\n')
    elif defect == 'bad-fraction': pool.write_text('id\tmin_fraction\nfixture-train\tnan\n')
    elif defect == 'wrong-kind': kind = 'site_only'
    with pytest.raises(ValueError):
        pl.build_manifest(tmp_path, 'sabdab', workers=1, pool_path=pool,
                          data_kind=kind, record_provenance=provenance)
    assert not (pg.output(tmp_path, 'sabdab') / pl.MANIFEST).exists()


def test_unknown_catalogue_requires_explicit_capability_and_pool():
    assert pl.manifest_kind(dict(dataset='pinder')) == 'paired'
    assert pl.manifest_kind(dict(dataset='pkpdb')) == 'site_only'
    with pytest.raises(ValueError): pl.manifest_kind(dict(dataset='sabdab'))
    with pytest.raises(ValueError): pl.build_manifest('/unused', 'sabdab', data_kind='paired')
