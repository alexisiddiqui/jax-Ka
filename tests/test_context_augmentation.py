import numpy as np
from pkatrain.context_augmentation import residue_mask,epoch_mask,mask_graph,encode_candidates


def test_mask_reproducibility_protection_and_epoch_refresh():
    first=residue_mask(500,[1,2,3],seed=17,epoch=1,complex_id='abc')
    np.testing.assert_array_equal(first,residue_mask(500,[1,2,3],seed=17,epoch=1,complex_id='abc'))
    assert first.any() and not first[[1,2,3]].any()
    assert not np.array_equal(first,residue_mask(500,[1,2,3],seed=17,epoch=2,complex_id='abc'))
    plan=dict(residues=500,structures={'abc':dict(start=0,stop=500,protected=[1,2,3])})
    np.testing.assert_array_equal(first,epoch_mask(plan,17,1))


def test_graph_mask_removes_incoming_and_outgoing_geometry():
    nodes=np.arange(456,dtype=np.float32).reshape(3,152);identity=nodes[:,:20].copy()
    g=dict(nodes=nodes,neighbors=np.tile(np.arange(3),(3,1)),edge=np.ones((3,3,20),np.float32),
        edge_mask=np.ones((3,3),bool),switch=np.ones((3,3),np.float32),query_residue=np.array([0,2]))
    mask_graph(g,np.array([False,True,False]))
    assert not g['edge_mask'][1].any() and not g['edge_mask'][:,1].any()
    assert not g['edge'][1,:,:19].any() and not g['edge'][:,1,:19].any()
    assert g['edge_mask'][0,2] and g['nodes'][1,23]==0
    assert not g['nodes'][1,24:].any()
    assert g['nodes'][0,24:].any() and g['nodes'][2,24:].any()
    np.testing.assert_array_equal(g['nodes'][:,:20],identity)


def test_pkai_mask_refills_from_beyond_original_250():
    classes=np.arange(260)%16;values=np.arange(260,dtype=np.float32)+1
    owners=np.arange(260);masked=np.zeros(260,bool);masked[:10]=True
    onehot=np.eye(8,dtype=np.float32)[3]
    x=encode_candidates(classes,values,owners,masked,onehot)
    assert np.count_nonzero(x[:4000])==250
    np.testing.assert_array_equal(x[:4000].reshape(250,16).sum(-1),values[10:])
    np.testing.assert_array_equal(x[4000:],onehot)
    empty=encode_candidates(classes,values,owners,np.ones(260,bool),onehot)
    assert not empty[:4000].any()
