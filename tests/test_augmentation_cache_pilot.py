import numpy as np
from pkatrain.augmentation_cache_pilot import choose_sample,unit_graph,replace_partner
from pkatrain.augmentation_graphs import backbone_table,same_padded,free_graph
from test_augmentation_release import native_fixture


def test_stratum_weights_recover_population_and_stress_has_zero_weight():
    rows=[]
    for dataset in ('pkpdb','pinder'):
        for i in range(30):
            rows.append(dict(id=f'{dataset}/{i}/A',summary=dict(dataset=dataset,split='train',state='A',n=100+i,
                status='usable' if i%3 else 'preparation_failed',quota_met=bool(i%3))))
    sample=choose_sample(rows,12,2)
    assert np.isclose(sum(x['weight'] for x in sample),len(rows))
    assert sum(not x['stress_only'] for x in sample)==12
    assert len({x['parent'] for x in sample})==len(sample)
    assert sample==choose_sample(list(reversed(rows)),12,2)
    assert all(x['weight']==0 for x in sample if x['stress_only'])


def test_partner_cache_matches_independent_graph_without_changing_bound():
    native,bb,chain=native_fixture();sequence=np.array([0,1,0,1]);residues=np.array([0,1],np.int32)
    control={k:v.copy() for k,v in native.items()}
    control['nodes']=np.concatenate((native['nodes'],backbone_table(bb,chain,sequence)),axis=-1)
    control.update(partners=chain,free_edge_mask=native['edge_mask']&(chain[native['neighbors']]==chain[:,None]),
        free_site_edge_mask=native['site_edge_mask']&(chain[native['site_residue']][native['site_neighbors']]==chain[native['site_residue']][:,None]))
    before={k:v.copy() for k,v in control.items()};moved=bb.copy();moved[1]+=[35,0,0]
    variant=unit_graph(native,moved[residues],chain[residues],sequence[residues],residues)
    result=replace_partner(control,variant);reference=free_graph(native,moved,chain,chain)
    for k in ('neighbors','edge','edge_mask','switch'):
        assert same_padded(result[k][residues],reference[k][residues])
    np.testing.assert_array_equal(result['nodes'][residues,-14:-2],moved[residues].reshape(2,12))
    np.testing.assert_array_equal(result['nodes'][2:],control['nodes'][2:])
    for k,v in before.items():np.testing.assert_array_equal(control[k],v)
    assert variant['nodes'].shape==(2,38)
    assert not set(variant)&{'labels','targets','rsa','w_burial','w_interface'}


def test_packed_loader_uses_single_pkpdb_branch_and_preserves_pinder_bound(tmp_path,monkeypatch):
    import json
    from pkatrain.augmentation_cache_pilot import FeatureSource,build_layouts
    from pkatrain.augmentation_store import ReleaseWriter
    native,bb,chain=native_fixture();sequence=np.array([0,1,0,1]);capacities={}
    with ReleaseWriter(tmp_path/'features',{},zstd_level=1) as features,ReleaseWriter(tmp_path/'controls',{},zstd_level=1) as controls:
        for dataset in ('pkpdb','pinder'):
            key=dataset+'/x';parent=key+'/A';partners=chain if dataset=='pinder' else np.zeros(4,int)
            control={k:v.copy() for k,v in native.items()}
            control['nodes']=np.concatenate((native['nodes'],backbone_table(bb,chain,sequence)),axis=-1)
            control.update(partners=partners,free_edge_mask=native['edge_mask']&(partners[native['neighbors']]==partners[:,None]),
                free_site_edge_mask=native['site_edge_mask']&(partners[native['site_residue']][native['site_neighbors']]==partners[native['site_residue']][:,None]))
            controls.add(key,{},control,summary=dict(dataset=dataset))
            ids=np.array([0,1],np.int32) if dataset=='pinder' else np.arange(4,dtype=np.int32)
            for c in (0,1):
                moved=bb[ids].copy();moved[-1]+=[10*c,0,0]
                graph=unit_graph(native,moved,chain[ids],sequence[ids],ids)
                dims=[4,max(native['neighbors'].shape[1],graph['neighbors'].shape[1]),len(native['query_residue']),
                    len(native['site_residue']),max(native['site_neighbors'].shape[1],graph['site_neighbors'].shape[1])]
                meta=dict(dataset=dataset,source_parent=parent,control_key=key,candidate=c,dims=dims)
                features.add(parent+f'/c{c:02d}',meta,graph,summary=meta)
                capacities[dataset+':128']=[128,*dims[1:]]
        features.finish();controls.finish()
    (tmp_path/'build-summary.json').write_text(json.dumps(dict(capacities=capacities,native_capacities=capacities)))
    build_layouts(tmp_path)
    for dataset,branches in [('pkpdb',1),('pinder',2)]:
        source=FeatureSource(tmp_path,2,2,dataset=dataset)
        try:
            for row in source.store.rows:
                if row['summary']['dataset']!=dataset:continue
                fast=source._variant(row['id']);reference=source.store.raw(row['id'])['arrays']
                for name in reference:np.testing.assert_array_equal(fast[name],reference[name])
            native_batch=source.load(source.plans('128','native',1)[0])[0]
            augmented=source.load(source.plans('128','augmented',1)[0])[0]
            assert augmented['nodes'].shape==(2,branches,128,38)
            if dataset=='pinder':
                for k in native_batch:np.testing.assert_array_equal(augmented[k][:,0],native_batch[k][:,0])
                np.testing.assert_array_equal(augmented['nodes'][:,1,2:4],native_batch['nodes'][:,1,2:4])
                reference=replace_partner(source.controls.raw('pinder/x')['arrays'],source.store.raw('pinder/x/A/c01')['arrays'])
                for name,value in reference.items():
                    if name not in augmented:continue
                    if name=='query_site':value=np.maximum(value,0)
                    np.testing.assert_array_equal(augmented[name][(0,1,*(slice(0,n) for n in value.shape))],value)
            assert not np.array_equal(augmented['nodes'],native_batch['nodes'])
        finally:source.close()
    # A small pilot must still pad to the real production manifest, not its own smaller maxima.
    import pkatrain.production_graphs as production
    import pkatrain.production_inputs as inputs
    class LiveStore:
        def __init__(self,path):pass
        def raw(self,cid):
            return dict(native,branch_edge_mask=np.stack((native['edge_mask'],control['free_edge_mask'])),
                branch_site_edge_mask=np.stack((native['site_edge_mask'],control['free_site_edge_mask'])))
        def close(self):pass
    table=backbone_table(bb,chain,sequence)
    monkeypatch.setattr(production,'ProductionStore',LiveStore)
    monkeypatch.setattr(production,'output',lambda root,dataset:root/dataset)
    monkeypatch.setattr(inputs,'BackboneTable',lambda root,dataset:lambda cid:table)
    live_capacity=[128,native['neighbors'].shape[1]+7,len(native['query_residue'])+3,
                   len(native['site_residue'])+5,native['site_neighbors'].shape[1]+9]
    (tmp_path/'pinder').mkdir()
    (tmp_path/'pinder/manifest-v1.json').write_text(json.dumps(dict(capacities={'128':live_capacity})))
    source=FeatureSource(tmp_path,2,2,dataset='pinder',runtime=tmp_path)
    try:
        assert source.native_capacities['128']==live_capacity
        assert source.capacities['128']==live_capacity
        graph,mask,_=source.load(source.plans('128','augmented',1)[0])
        assert graph['neighbors'].shape[-1]==live_capacity[1]
        assert graph['site_neighbors'].shape[-1]==live_capacity[4]
        assert mask.shape[-1]==live_capacity[2]
        assert not graph['edge'][...,native['neighbors'].shape[1]:,:].any()
        assert not graph['site_edge'][...,native['site_neighbors'].shape[1]:,:].any()
    finally:source.close()
    import pytest
    layout=tmp_path/'feature-layouts.npy';original=layout.read_bytes();layout.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='layout provenance'):
        FeatureSource(tmp_path,2,2,dataset='pinder')
