from pathlib import Path
from types import SimpleNamespace
import json
import zstandard
from pkatrain.augmentation_generate import parent_seed,parent_id,child
from pkatrain.augmentation_store import decode


def test_seed_is_parent_specific_and_independent_of_worker_order():
    row=dict(dataset='pinder',id='x',state='A',index=1,worker=0)
    other=dict(row,index=999,worker=99)
    assert parent_seed(parent_id(row))==parent_seed(parent_id(other))
    assert parent_seed('pinder/x/A')!=parent_seed('pinder/x/B')
    assert 0<=parent_seed('pinder/x/A')<2147483647


def test_failed_source_keeps_identity_and_never_publishes_partial_arrays(tmp_path,monkeypatch):
    import pkatrain.augmentation_generate as generation
    row=dict(dataset='pkpdb',id='x',state='assembly')
    source=tmp_path/'task.json';source.write_text(json.dumps(row));result=tmp_path/'result.zst'
    monkeypatch.setattr(generation,'guard',lambda args:None)
    def failed(args,row):raise ValueError('bad backbone')
    monkeypatch.setattr(generation,'produce',failed)
    child(SimpleNamespace(input=source,result=result))
    raw=result.read_bytes();record=decode(raw,max_output_size=zstandard.frame_content_size(raw))
    assert record['metadata']['parent']=='pkpdb/x/assembly'
    assert record['metadata']['status']=='preparation_failed'
    assert record['metadata']['selected']==[] and not record['metadata']['quota_met']
    assert record['arrays']=={}


def test_missing_backend_is_environment_failure_not_a_bad_protein(tmp_path,monkeypatch):
    import pkatrain.augmentation_generate as generation
    source=tmp_path/'task.json';source.write_text(json.dumps(dict(dataset='pkpdb',id='x',state='assembly')))
    result=tmp_path/'result.zst';monkeypatch.setattr(generation,'guard',lambda args:None)
    def failed(args,row):raise ModuleNotFoundError('backend unavailable')
    monkeypatch.setattr(generation,'produce',failed)
    child(SimpleNamespace(input=source,result=result))
    raw=result.read_bytes();record=decode(raw,max_output_size=zstandard.frame_content_size(raw))
    assert record['metadata']['status']=='environment_failed'
