from pkatrain.compilation_cache import MAX_BYTES,configure
from pathlib import Path

import numpy as np

from pkabench.runtime import digest
from pkatrain.graph_batches import BatchLoader,LoaderTelemetry
from pkatrain.graph_mmap import GraphMMap,build_bundle


def test_loader_telemetry_isolated_snapshot():
    telemetry=LoaderTelemetry();telemetry.structure({'x':1});telemetry.batch({'y':2});telemetry.prefetch_wait({'z':3})
    snapshot=telemetry.snapshot()
    assert snapshot=={'structures':[{'x':1}],'batches':[{'y':2}],'prefetch_waits':[{'z':3}]}
    snapshot['structures'].append({'x':4})
    assert len(telemetry.snapshot()['structures'])==1


def test_compilation_cache_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv('PKATRAIN_COMPILATION_CACHE_DIR',raising=False)
    assert configure({}, {})=={'enabled':False}
    assert MAX_BYTES==5*1024**3


def test_mmap_bundle_and_batch_are_exact_and_read_only(tmp_path):
    out=tmp_path/'experiment';records=[]
    for number,cid in enumerate(('one','two')):
        folder=out/'data'/cid;folder.mkdir(parents=True)
        n,k,q=2+number,2,1+number
        graph=dict(
            nodes=np.arange(n*24,dtype=np.float32).reshape(n,24)+number,
            node_mask=np.ones(n,bool),
            neighbors=np.tile(np.arange(k,dtype=np.int32),(n,1))%n,
            edge=np.arange(n*k*20,dtype=np.float32).reshape(n,k,20)/100,
            edge_mask=np.ones((n,k),bool),switch=np.ones((n,k),np.float32),
            query_residue=np.arange(q,dtype=np.int32),query_group=np.arange(q,dtype=np.int32),
        )
        labels=np.arange(q,dtype=np.float32)+3.5
        np.savez_compressed(folder/'graph.npz',**graph,labels=labels)
        records.append(dict(complex_id=cid,sha256=digest(folder/'graph.npz'),n=n,k=k,q=q,split='train'))
    manifest=dict(records=records,capacities={'384':[32,32,32]},config={})
    bundle=build_bundle(out,manifest)
    store=GraphMMap(bundle,records,verify_files=True)
    for row in records:
        raw,labels=store.raw(row['complex_id'])
        assert not labels.flags.writeable
        with np.load(out/'data'/row['complex_id']/'graph.npz',allow_pickle=False) as source:
            for name,value in raw.items():np.testing.assert_array_equal(value,source[name])
            np.testing.assert_array_equal(labels,source['labels'])
    store.close()
    native=BatchLoader(out,manifest,2,backend='npz');mapped=BatchLoader(out,manifest,2,backend='mmap')
    assert native.provenance()['backend']=='npz'
    assert mapped.provenance()['backend']=='mmap' and mapped.provenance()['read_only']
    a=native.load(['one','two']);b=mapped.load(['one','two'])
    for left,right in zip(a,b):
        if isinstance(left,dict):
            for name in left:np.testing.assert_array_equal(left[name],right[name])
        else:np.testing.assert_array_equal(left,right)
    tight=mapped.load(['one','two'],[4,4,4])
    assert tight[0]['neighbors'].shape==(2,4,4)
    planned=list(mapped.iterate([{'cids':['one','two'],'capacities':[4,4,4]}]))
    assert planned[0][0]==['one','two'] and planned[0][1][0]['neighbors'].shape==(2,4,4)
    native.close();mapped.close()
