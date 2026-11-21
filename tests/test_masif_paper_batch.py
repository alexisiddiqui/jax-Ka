import json
from pathlib import Path
import sys
import os
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_annotation_pilot import select_units,load_cohort
from pkabench.runtime import digest
from masif_paper_runtime import verify,sha,PROFILE
from masif_gpu_queue import initialize,submit,validate_names
import masif_gpu_queue
from masif_paper_batch import matlab_literal,worker,worker_provenance
from pkatrain.augmentation_store import ReleaseWriter,AugmentationStore


def test_training_chain_inventory_keeps_assembly_identity_and_excludes_validation(tmp_path):
    table=tmp_path/'training/pool-v4';table.mkdir(parents=True)
    (table/'pkpdb.tsv').write_text('id\tn_chains\ntrain\t2\nmono\t1\n')
    (table/'pkpdb-val.tsv').write_text('id\tn_chains\nval\t3\n')
    units=[dict(id='pkpdb/'+cid+'/assembly',dataset='pkpdb',cid=cid,split=split,n=n,family='pkpdb_assembly')
        for cid,split,n in [('train','train',300),('mono','train',100),('val','val',300)]]
    selected=select_units(units,tmp_path,dataset='pkpdb',split='train',separate_chains=True)
    assert [x['id'] for x in selected]==['pkpdb/train/chain-0000','pkpdb/train/chain-0001','pkpdb/mono/chain-0000']
    assert [x['chain_index'] for x in selected]==[0,1,0]
    assert selected[0]['expected_chains']==2 and all(x['split']=='train' for x in selected)


def test_pinned_node_cohort_excludes_nonprotein_without_renumbering(tmp_path):
    table=tmp_path/'training/pool-v4';table.mkdir(parents=True)
    (table/'pkpdb.tsv').write_text('id\tn_chains\ntrain\t3\n')
    units=[dict(id=f'pkpdb/train/chain-{i:04d}',dataset='pkpdb',split='train',chain_index=i,n=100) for i in range(3)]
    data=dict(format='masif-native-cohort-v1',dataset='pkpdb',split='train',coordinate_manifest_sha256='coordinates',
              training_tsv_sha256=digest(table/'pkpdb.tsv'),global_cohort_sha256='global',
              units=[dict(id=units[2]['id'],n=215,source_chain_label='C',scheduling_cost=1000)])
    path=tmp_path/'cohort.json';path.write_text(json.dumps(data))
    selected,provenance=load_cohort(units,path,tmp_path,'coordinates')
    assert len(selected)==1 and selected[0]['chain_index']==2 and selected[0]['n']==215
    assert selected[0]['source_chain_label']=='C' and provenance['cohort_manifest_sha256']==digest(path)
    with pytest.raises(ValueError,match='source inventory changed'):
        load_cohort(units,path,tmp_path,'changed')
    for invalid in [dict(data,units=data['units']*2),dict(data,units=[dict(data['units'][0],id='pkpdb/val/chain-0000')]),
                    dict(data,units=[dict(data['units'][0],n=0)])]:
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):load_cohort(units,path,tmp_path,'coordinates')
    path.write_text(json.dumps(data));(table/'pkpdb.tsv').write_text('changed membership')
    with pytest.raises(ValueError,match='source inventory changed'):
        load_cohort(units,path,tmp_path,'coordinates')


def test_runtime_detects_modified_scientific_source_and_path_escape(tmp_path):
    (tmp_path/'source.m').write_text('original')
    manifest=dict(profile=PROFILE,files={'source.m':sha(tmp_path/'source.m')})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    verify(tmp_path)
    (tmp_path/'source.m').write_text('different')
    with pytest.raises(ValueError,match='runtime changed'):verify(tmp_path)
    manifest['files']={'../escape':'ignored'};(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='runtime changed'):verify(tmp_path)


def test_gpu_queue_rejects_model_mixing_before_writing_payload(tmp_path):
    initialize(tmp_path,{'data':'one'})
    with pytest.raises(ValueError,match='model mismatch'):
        submit(tmp_path,dict(features=np.zeros((1,200,5))),{'data':'two'})
    assert sorted(p.name for p in tmp_path.iterdir())==['config.json']
    with pytest.raises(ValueError,match='configuration mismatch'):initialize(tmp_path,{'data':'two'})


@pytest.mark.parametrize('status',['complete','failed'])
@pytest.mark.parametrize('stale_request_exists',[False,True])
def test_gpu_queue_polls_existing_pending_mailbox_and_cleans_reply(tmp_path,monkeypatch,status,stale_request_exists):
    checkpoint={'data':'one'};initialize(tmp_path,checkpoint)
    def reply(_):
        request_path=next(tmp_path.glob('*.request'))
        request=json.loads(request_path.read_text());response=tmp_path/request['response']
        assert json.loads(response.read_text())=={'status':'pending'}
        np.save(tmp_path/request['output'],np.ones((1,80),np.float32))
        masif_gpu_queue.atomic(response,dict(status=status,checkpoint_sha256=checkpoint,error='consumer failure'))
        request_path.unlink()
    monkeypatch.setattr(masif_gpu_queue.time,'sleep',reply)
    if stale_request_exists:
        real_exists=Path.exists
        monkeypatch.setattr(Path,'exists',lambda path:True if path.suffix=='.request' else real_exists(path))
    if status=='complete':
        np.testing.assert_array_equal(submit(tmp_path,dict(features=np.zeros((1,200,5))),checkpoint),np.ones((1,80)))
    else:
        with pytest.raises(RuntimeError,match='consumer failure'):
            submit(tmp_path,dict(features=np.zeros((1,200,5))),checkpoint)
    assert [p.name for p in tmp_path.iterdir()]==['config.json']


def test_matlab_paths_are_quoted_as_data():
    assert matlab_literal("/tmp/a'b") == "'/tmp/a''b'"


def test_queue_cannot_overwrite_config_or_escape_spool():
    request=dict(input='unit.npz',output='unit.output.npy',response='unit.response')
    validate_names(request,'unit')
    for name in ['../outside.npy','config.json','other.response']:
        with pytest.raises(ValueError,match='unsafe queue'):
            validate_names(dict(request,response=name),'unit')


def test_resume_discards_uncommitted_tail_and_ahead_metadata(tmp_path):
    # Crash scenario: metadata reached disk before its frame checkpoint.
    (tmp_path/'sample.json').write_text('{}')
    args=SimpleNamespace(output=tmp_path,checkpoint_sha256={'data':'model'},paper_runtime_sha256='profile',
        gpu_spool=None,resume=True,batch_size=2)
    packed=tmp_path/'fingerprints/workers/0000'
    with ReleaseWriter(packed,worker_provenance(args),zstd_level=1) as writer:
        writer.add('committed',{'id':'committed'},{'descriptors':np.ones((2,80),np.float32)})
        writer.checkpoint()
    with (packed/'records-00000.bin').open('ab') as stream:stream.write(b'uncommitted tail')
    state=dict(worker=0,parents=[dict(id='committed',status='complete'),dict(id='ahead',status='complete')],
        peak_tree_rss_bytes=1,wall_s=0.)
    (tmp_path/'worker-00.json').write_text(json.dumps(state))
    result=worker((args,0,sorted(os.sched_getaffinity(0)),[dict(id='committed')]))
    assert [x['id'] for x in result['parents']]==['committed']
    with AugmentationStore(packed) as store:
        assert list(store.by_id)==['committed']
        np.testing.assert_array_equal(store.raw('committed')['arrays']['descriptors'],np.ones((2,80),np.float32))
