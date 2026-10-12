import json
import pytest
from pkabench import experimental_ph as ep


def record(points=(), ranges=(), notes='', nmr=(), em=()):
    return {'exptl_crystal_grow': [{'pH': p, 'pdbx_details': notes} for p in points] +
            [{'pdbx_pH_range': r, 'pdbx_details': notes} for r in ranges] +
            ([{'pdbx_details': notes}] if notes and not points and not ranges else []),
            'pdbx_nmr_exptl_sample_conditions': list(nmr), 'em_buffer_ph': list(em)}


@pytest.mark.parametrize('raw', [record(points=[7.5]), record(points=[7.5,7.5]),
                                record(ranges=['6,5-6,5']), record(em=['7.2'])])
def test_structured_points_can_be_normalized(raw):
    a = ep.audit_record(raw)
    r = ep.quality_record({'id':'x','pdb_id':'1abc'}, a, 2)
    assert r['ph_mask'] and r['ph_quality_weight']==1 and r['parent_weight']==.5


@pytest.mark.parametrize('raw', [record(points=[7,8]), record(points=[0]),
    record(points=[7],notes='0.1 M acetate pH 5.5'), record(ranges=['5-8']),
    record(ranges=['7.0-9.o']), record(notes='buffer pH 7.5'),
    record(nmr=[{'pH':'7','pH_units':None}]), record(nmr=[{'pH':'7','pH_units':'pD'}]),
    record(points=[7],ranges=['4-6']), record(points=[15]), record()])
def test_ambiguous_or_incomplete_metadata_never_enters_loss(raw):
    r = ep.quality_record({'id':'x','pdb_id':'1abc'},ep.audit_record(raw),1)
    assert not r['ph_mask'] and r['ph_target'] is None and r['ph_quality_weight']==0


def test_explicit_ids_align_targets_and_similarity_filter():
    good=ep.quality_record({'id':'a','pdb_id':'1abc','chain_70_to_validation':'1'},ep.audit_record(record(points=[7.5])),1)
    missing=ep.quality_record({'id':'b','pdb_id':'2abc'},ep.audit_record(record()),1)
    out=ep.targets_for_records({'a':good,'b':missing},['b','a'])
    assert out['ph_target'].tolist()==[0,7.5] and out['ph_mask'].tolist()==[False,True]
    assert not ep.targets_for_records({'a':good},['a'],exclude_validation_similar=True)['ph_mask'][0]
    with pytest.raises(KeyError):ep.targets_for_records({'a':good},['unknown'])


def test_build_verify_is_immutable_and_detects_stale_pool(tmp_path):
    source=tmp_path/'audits/ph-coverage';source.mkdir(parents=True)
    (source/'metadata.json').write_text(json.dumps({'1ABC':record(points=[7.5])}))
    (source/'em-metadata.json').write_text('{}')
    pool=tmp_path/ep.DATASETS['pkpdb']/'pool-v3.tsv';pool.parent.mkdir(parents=True)
    pool.write_text('id\tpdb_id\n1abc\t1abc\n')
    first=ep.build(tmp_path,'pkpdb');assert first['counts']['train']=={'entries':1,'eligible':1}
    assert ep.build(tmp_path,'pkpdb')==first
    assert ep.load_annotations(tmp_path,'pkpdb')['1abc']['ph_target']==7.5
    pool.write_text('id\tpdb_id\n2abc\t2abc\n')
    with pytest.raises(ValueError,match='pool hash'):ep.load_annotations(tmp_path,'pkpdb')


def test_transfer_bundles_include_annotation_sidecars(tmp_path):
    from test_dataset_transfer import _fake_pkpdb
    from pkabench import dataset_transfer as dt
    _fake_pkpdb(tmp_path)
    folder=ep.annotation_dir(tmp_path,'pkpdb');folder.mkdir(parents=True)
    for name in ('manifest.json','records.jsonl','pdb-evidence.jsonl'):(folder/name).write_text('{}')
    files=dt.dataset_files(tmp_path,'pkpdb')
    assert all(str((folder/n).relative_to(tmp_path)) in files['core'] for n in ('manifest.json','records.jsonl','pdb-evidence.jsonl'))


def test_feature_names_enforce_split_and_record_order():
    good=ep.quality_record({'id':'a','pdb_id':'1abc','split':'val'},ep.audit_record(record(points=[7.5])),1)
    assert ep.targets_for_feature_records({'a':good},['val-a'])['ph_target'].tolist()==[7.5]
    with pytest.raises(KeyError):ep.targets_for_feature_records({'a':good},['train-a'])
