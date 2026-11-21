import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_release_pack import pack,canonical_runs
from pkatrain.augmentation_store import encode
from pkatrain.masif_store import MaSIFStore


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();plan=tmp_path/'plan';plan.mkdir()
    profile=dict(checkpoint_sha256={'test':'checkpoint'},paper_runtime_sha256='runtime')
    entries=[];chunks=[];offset=0
    for rid,ids,role in [('intact-Ab',[0,1],'Ab'),('Ag',[2],'Ag')]:
        arrays=dict(vertices=np.eye(3,dtype=np.float32),faces=np.array([[0,1,2]],np.int32),normals=np.ones((3,3),np.float32),
            descriptors=np.ones((3,80),np.float32),raw_features=np.ones((3,4),np.float32),owner=np.arange(3,dtype=np.int32)%len(ids),canonical_residue_ids=np.array(ids,np.int32))
        meta=dict(parent=rid,dataset='pinder',cid='complex',split='train',receipt=dict(status='complete',inference_backend='tensorflow-cpu',**profile))
        data,size=encode(meta,arrays,zstd_level=1);chunks.append(data)
        entries.append(dict(id=rid,path=str(source/'frames.bin'),offset=offset,bytes=len(data),raw_bytes=size,sha256=hashlib.sha256(data).hexdigest(),
            summary=dict(dataset='pinder',cid='complex',split='train',role=role,state=role,n=len(ids),surface_policy='original-v1')))
        offset+=len(data)
    (source/'frames.bin').write_bytes(b''.join(chunks));(source/'manifest.json').write_text('{}')
    parent=dict(dataset='pinder',cid='complex',split='train',group='g',status='covered',canonical_residues=3,
        required_units=[dict(id='intact-Ab',role='Ab',state='A'),dict(id='Ag',role='Ag',state='B')],missing_units=[],reasons=[])
    coverage=dict(format='masif-native-assembly-coverage-v1',parents=[parent]);(plan/'coverage.json').write_text(json.dumps(coverage))
    selection=dict(entries=entries,scientific_profile=profile,coverage_sha256=hashlib.sha256((plan/'coverage.json').read_bytes()).hexdigest(),
        source_manifests={str(source):hashlib.sha256(b'{}').hexdigest()},audit_results_sha256='audit',outlier_review_sha256='review')
    (plan/'selection.json').write_text(json.dumps(selection));(plan/'outlier-review.json').write_text('{}')
    return plan,source,b''.join(chunks)


def test_packing_preserves_frames_and_loader_keeps_intact_antibody_mapping(tmp_path):
    plan,source,data=fixture(tmp_path);out=tmp_path/'release'
    pack(plan,out,tmp_path/'work',workers=1)
    assert (out/'records-00000.bin').read_bytes()==data
    with MaSIFStore(out) as store:
        item=store.load_assembly('pinder','complex',split='train')
        ab=next(u for u in item['units'] if u['role']=='Ab')
        np.testing.assert_array_equal(ab['arrays']['canonical_residue_ids'],[0,1])
        assert len(store.assemblies(split='train'))==1
        with pytest.raises(ValueError,match='split'):store.load_assembly('pinder','complex',split='val')
    (out/'coverage.json').write_text('{}')
    with pytest.raises(ValueError,match='coverage metadata'):MaSIFStore(out)


def test_corrupt_source_frame_never_publishes_release_marker(tmp_path):
    plan,source,data=fixture(tmp_path);out=tmp_path/'release'
    (source/'frames.bin').write_bytes(b'bad'+data[3:])
    with pytest.raises(ValueError,match='frame hash'):pack(plan,out,tmp_path/'work',workers=1)
    assert not (out/'manifest.json').exists()


def test_canonical_runs_rejects_unsigned_wraparound_and_overlap():
    assert canonical_runs(np.array([0,1,4,5],np.int32))==[[0,2],[4,6]]
    for ids in [np.array([5,1],np.uint32),np.array([0,0],np.int32)]:
        with pytest.raises(ValueError):canonical_runs(ids)
