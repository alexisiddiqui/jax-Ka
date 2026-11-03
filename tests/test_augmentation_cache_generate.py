import json
from pathlib import Path

import numpy as np
import pytest

from pkatrain import augmentation_cache_generate as generate
from pkatrain.augmentation_cache_pilot import unit_graph
from pkatrain.augmentation_graphs import backbone_table
from pkatrain.augmentation_store import AugmentationStore, ReleaseWriter, fingerprint, finalize_workers
from pkabench.runtime import atomic_json
from test_augmentation_release import native_fixture


def test_worker_assignment_is_stable_complete_and_spreads_largest():
    items=[dict(parent=str(i),summary=dict(n=20+100*i)) for i in range(20)]
    bins=generate.assign_workers(items,4)
    assert bins==generate.assign_workers(list(reversed(items)),4)
    assert sorted(x['parent'] for b in bins for x in b)==sorted(x['parent'] for x in items)
    assert len({b[0]['parent'] for b in bins})==4
    costs=[sum((x['summary']['n']+50)**2 for x in b) for b in bins]
    assert max(costs)/min(costs)<1.1


def test_full_worker_resume_coverage_partition_dedup_and_fast_layout(tmp_path,monkeypatch):
    coordinates=tmp_path/'coordinates';root=tmp_path/'cache';root.mkdir()
    native,bb,chain=native_fixture();seq=np.array([0,1,0,1]);items=[]
    control={k:v.copy() for k,v in native.items()}
    control['nodes']=np.concatenate((native['nodes'],backbone_table(bb,chain,seq)),axis=-1)
    control.update(partners=chain,free_edge_mask=native['edge_mask'] & (chain[native['neighbors']]==chain[:,None]),
        free_site_edge_mask=native['site_edge_mask'] & (chain[native['site_residue']][native['site_neighbors']]==chain[native['site_residue']][:,None]))
    with ReleaseWriter(coordinates,{}) as writer:
        for state in ('A','B','failed'):
            s=dict(dataset='pinder',split='train',state=state,n=2,selected=2 if state!='failed' else 0,
                status='usable' if state!='failed' else 'preparation_failed',quota_met=state!='failed')
            parent='pinder/x/'+state
            writer.add(parent,{},dict(x=np.ones(2)),summary=s)
            items.append(dict(parent=parent,summary=s,stratum='fixture',weight=1.,stress_only=False))
        writer.finish()
    def fake_build(task):
        item=task[2];state=item['summary']['state']
        if state=='failed':return dict(status='coordinate_unavailable',wall_s=.01)
        ids=np.array([0,1] if state=='A' else [2,3],np.int32);views=[]
        for c in (0,1,2):
            moved=bb[ids].copy();moved[-1]+=[.2*c,0,0]
            views.append(dict(candidate=c,arrays=unit_graph(native,moved,chain[ids],seq[ids],ids)))
        return dict(status='built',control=control,control_key='pinder/x',n=4,views=views,wall_s=.01,peak_rss_GiB=.1)
    monkeypatch.setattr(generate.pilot,'_build_parent',fake_build)
    monkeypatch.setattr(generate.os,'sched_setaffinity',lambda *args:None)
    assignments=generate.assign_workers(items,2);provenance=dict(source_sha256={},accepted_pilot_exception='fixture')
    task=(root,coordinates,tmp_path,0,0,assignments[0],provenance)
    # A parent commits only after its feature checkpoint; simulate a durable
    # feature-only tail, as can occur between the two checkpoint writes.
    worker_provenance=dict(provenance,worker=0,parents_sha256=fingerprint([x['parent'] for x in assignments[0]]))
    first=assignments[0][0];result=fake_build((None,None,first));view=result['views'][0];g=view['arrays']
    with ReleaseWriter(root/'features/workers/0000',worker_provenance,zstd_level=1) as writer:
        d=[len(g['nodes']),g['neighbors'].shape[1],len(g['query_residue']),len(g['site_residue']),g['site_neighbors'].shape[1]]
        dims=[4,max(native['neighbors'].shape[1],d[1]),len(native['query_residue']),len(native['site_residue']),max(native['site_neighbors'].shape[1],d[4])]
        meta=dict(source_parent=first['parent'],control_key='pinder/x',dataset='pinder',split='train',state=first['summary']['state'],
            candidate=0,kind='zero',dims=dims,unit_dims=d,weight=1.,stress_only=False,stratum='fixture',geometry_labels_reused=False)
        writer.add(first['parent']+'/c00',meta,g,summary=meta);writer.checkpoint()
    generate.run_worker(task)
    generate.run_worker((root,coordinates,tmp_path,1,0,assignments[1],provenance))
    # Published workers are immutable and reusable on restart.
    before=(root/'features/workers/0000/manifest.json').read_bytes()
    generate.run_worker(task)
    assert (root/'features/workers/0000/manifest.json').read_bytes()==before
    atomic_json(root/'inventory.json',dict(fixture=True))
    with AugmentationStore(coordinates) as source:
        result=generate.consolidate(root,assignments,provenance,source.rows)
    assert result['records']==6 and result['parents']==3 and result['verified']
    assert not (root/'controls').exists()
    with AugmentationStore(root/'partitions') as partitions:
        assert len(partitions.rows)==1
        np.testing.assert_array_equal(partitions.raw('pinder/x')['arrays']['partners'],chain)
    from pkatrain.augmentation_cache_pilot import FeatureSource
    with pytest.raises(ValueError,match='live native'):
        FeatureSource(root,1,1)
    layouts=np.load(root/'feature-layouts.npy')
    with AugmentationStore(root/'features') as store:
        assert layouts.shape==(6,5)
        for row,d in zip(store.rows,layouts):
            graph=store.raw(row['id'])['arrays']
            assert tuple(d)==(len(graph['nodes']),graph['neighbors'].shape[1],len(graph['query_residue']),len(graph['site_residue']),graph['site_neighbors'].shape[1])
    # Source-derived coverage, rather than whatever workers happened to emit,
    # determines whether the outer publication marker may be written.
    (root/'manifest.json').unlink()
    with AugmentationStore(coordinates) as source: bad=json.loads(json.dumps(source.rows))
    bad[0]['summary']['selected']+=1
    with pytest.raises(ValueError,match='coverage'):
        generate.consolidate(root,assignments,provenance,bad)
    assert not (root/'manifest.json').exists()


def test_finalizer_verifies_against_worker_digest_and_runs_domain_checks(tmp_path):
    worker=tmp_path/'workers/one'
    with ReleaseWriter(worker,{}) as writer:
        writer.add('x',{},dict(xyz=np.ones((3,3),np.float32)));writer.finish()
    checks=[]
    finalize_workers(tmp_path,[worker],{},expected_ids=['x'],check_record=lambda record,row:checks.append(row['id']))
    assert checks==['x']
    # Corruption must still fail even though redundant worker verification was removed.
    (tmp_path/'manifest.json').unlink()
    payload=worker/'records-00000.bin';data=bytearray(payload.read_bytes());data[-1]^=1;payload.write_bytes(data)
    with pytest.raises(ValueError,match='digest'):
        finalize_workers(tmp_path,[worker],{},expected_ids=['x'])
