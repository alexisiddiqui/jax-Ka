import copy
import numpy as np
import pytest
from pkabench.runtime import digest
from pkatrain import production_graphs as pg,production_loading as pl
from pkatrain.production_native import PatchedStore,augmentation_guard,declared_store_kind
from test_source_loading import store_fixture
from pkatrain.loading import BucketPolicy,LoaderConfig


def test_verified_record_override_routes_targets_and_rejects_bad_pin(tmp_path):
    arrays,_,_=store_fixture(tmp_path/'base')
    base=pg.ProductionStore(pg.output(tmp_path/'base','sabdab')/'store-v2')
    out=pg.output(tmp_path/'patch','sabdab');out.mkdir(parents=True)
    changed={k:v.copy() for k,v in arrays.items()};changed['targets'][:]=9.
    row=dict(id='fixture-train',split='train',n=4,k=2,q=1,s=2,sk=1,arrays_sha256=pg.arrays_digest(changed,pg.FIELDS['pinder']))
    (out/'ids.json').write_text('{"ids": [["fixture-train", "train"]]}')
    pg._write_store(out,'sabdab',[row],{k:v.dtype.str for k,v in changed.items()},[pg.encode(changed,pg.FIELDS['pinder'],level=1)],{},1,
        fields=pg.FIELDS['pinder'],data_kind='paired',zstd_level=1)
    reference=dict(store=str(out/'store-v2'),store_verification_sha256=digest(out/'store-v2/verification.json'))
    store=PatchedStore(base,[reference],{row['id']:row},'sabdab')
    assert np.array_equal(store.raw('fixture-train')['targets'],changed['targets'])
    assert np.array_equal(store.raw('fixture-heldout')['targets'],arrays['targets'])
    store.close()
    base=pg.ProductionStore(pg.output(tmp_path/'base','sabdab')/'store-v2')
    with pytest.raises(ValueError,match='verification changed'):
        PatchedStore(base,[dict(reference,store_verification_sha256='bad')],{row['id']:row},'sabdab')


def test_augmentation_receipt_checks_roles_context_and_missing_donors():
    row=dict(id='native',operational_role='train',native_coordinate_sha256='coordinates')
    route=dict(recipient_unit_id='sabdab:native',recipient_context_sha256='coordinates',donor_unit_ids=['pinder:donor'])
    policy=dict(schema_version=1,records={'views':{'native':route}},unit_roles={'pinder:donor':'train'})
    manifest=dict(dataset='sabdab',augmentation_role_policy=policy)
    assert augmentation_guard(manifest,row)('views','native',None,{}) is None
    policy['unit_roles']['pinder:donor']='validation'
    assert augmentation_guard(manifest,row)('views','native',None,{})=='augmentation_donor_role_mismatch'
    policy['unit_roles'].clear()
    assert augmentation_guard(manifest,row)('views','native',None,{})=='unassessed_augmentation_donors'
    assert augmentation_guard(manifest,row)('unknown','native',None,{})=='unassessed_augmentation_donors'
    assert augmentation_guard(dict(dataset='pinder'),row) is None


def test_augmented_free_branch_retains_joint_antibody_edges():
    from pkatrain.production_alt import free_branch
    n=3;neighbors=np.tile(np.arange(n,dtype=np.int32),(n,1))
    edge=np.zeros((n,n,20),np.float32);edge[...,19]=np.eye(n)
    record=dict(nodes=np.zeros((n,24),np.float32),node_mask=np.ones(n,bool),neighbors=neighbors,
        edge=edge,edge_mask=np.ones((n,n),bool),switch=np.ones((n,n),np.float32),
        site_residue=np.arange(n,dtype=np.int32),site_type=np.zeros(n,np.int32),site_mask=np.ones(n,bool),
        site_neighbors=neighbors,site_edge=np.zeros((n,n,1),np.float32),site_edge_mask=np.ones((n,n),bool),
        site_switch=np.ones((n,n),np.float32))
    graph={k:np.stack([v,v]) for k,v in record.items()}
    legacy,why=free_branch(graph,record,n,(n,n,1,n,n))
    assert why is None
    assert np.array_equal(legacy['edge_mask'][1],np.eye(n,dtype=bool))
    paired,why=free_branch(graph,record,n,(n,n,1,n,n),biological_side=[0,0,1])
    assert why is None
    expected=np.array([[1,1,0],[1,1,0],[0,0,1]],bool)
    assert np.array_equal(paired['edge_mask'][1],expected)
    assert np.array_equal(paired['site_edge_mask'][1],expected)
    assert paired['edge_mask'][0].all() and graph['edge_mask'].all()


def test_legacy_native_kind_requires_registered_source_and_metadata_pin(tmp_path):
    from types import SimpleNamespace
    metadata=tmp_path/'metadata.json';metadata.write_text('{"legacy":true}')
    store=SimpleNamespace(path=tmp_path,metadata=dict(dataset='pinder'),fields=pg.FIELDS['pinder'])
    manifest=dict(dataset='pinder',legacy_store_metadata_sha256=digest(metadata))
    assert declared_store_kind(manifest,store)=='paired'
    assert declared_store_kind(dict(manifest,legacy_store_metadata_sha256='bad'),store) is None
    assert declared_store_kind(dict(manifest,dataset='sabdab'),store) is None


def test_real_loader_hook_blocks_heldout_donor_without_changing_native_targets(tmp_path):
    from pkatrain.production_alt import AltConfig,MemoryStore
    arrays,pool,provenance=store_fixture(tmp_path)
    for cid,value in provenance.items():value['native_coordinate_sha256']='context-'+cid
    manifest=pl.build_manifest(tmp_path,'sabdab',policy=BucketPolicy((8,),(2,)),workers=1,
        pool_path=pool,data_kind='paired',record_provenance=provenance)
    manifest['augmentation_role_policy']=dict(schema_version=1,records={'memory':{'fixture-train':dict(
        recipient_unit_id='sabdab:fixture-train',recipient_context_sha256='context-fixture-train',donor_unit_ids=['pinder:donor'])}},
        unit_roles={'pinder:donor':'validation'})
    alternate={k:v.copy() for k,v in arrays.items()};alternate['alt_ok']=np.ones(2,bool)
    source=pl.source_from_manifest(manifest,config=LoaderConfig(workers=1))
    source.alt=AltConfig(MemoryStore({'fixture-train':alternate}),1.)
    try:
        blocked=source.load(['fixture-train'])
        assert not blocked[0]['use_alt'].any()
        assert np.array_equal(blocked[1][0,:,:1],arrays['targets'])
        assert source.alt.counts['augmentation_donor_role_mismatch']==1
        manifest['augmentation_role_policy']['unit_roles']['pinder:donor']='train'
        allowed=source.load(['fixture-train'])
        assert allowed[0]['use_alt'][0].all()
        assert np.array_equal(allowed[1][0,:,:1],arrays['targets'])
    finally:source.close()
