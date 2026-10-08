import numpy as np
from pkabench.gqt_learning_diagnostics import calibration,edge_distance,perturb_batch


def toy():
    nodes=np.zeros((1,3,24),np.float32);nodes[0,:,0]=1;nodes[0,1,3]=1;nodes[0,1,0]=0
    neighbors=np.array([[[0,1,2],[0,1,2],[0,1,2]]]);edge=np.zeros((1,3,3,20),np.float32)
    centers=np.linspace(0,20,16)
    for i,d in enumerate((0,5,12)):edge[0,:,i,:16]=np.exp(-((d-centers)/1.5)**2)
    return dict(nodes=nodes,node_mask=np.ones((1,3),bool),neighbors=neighbors,edge=edge,edge_mask=np.ones((1,3,3),bool),switch=np.ones((1,3,3),np.float32),query_residue=np.array([[0,0]]),query_group=np.array([[0,0]]))


def test_distance_and_perturbations_preserve_self():
    graph=toy();eligible=np.array([[True,False]])
    # The non-negative RBF grid biases the reconstructed zero-distance edge
    # slightly upward; self edges are identified from indices, not this value.
    np.testing.assert_allclose(edge_distance(graph['edge'])[0,0],[0,5,12],atol=.5)
    local=perturb_batch(graph,eligible,'local_only');assert local['edge_mask'].sum()==3
    shell=perturb_batch(graph,eligible,'remove_0_6A');assert 3<=shell['edge_mask'].sum()<9
    assert np.all(np.diagonal(shell['edge_mask'][0]))
    identity=perturb_batch(graph,eligible,'remove_identity');assert identity['nodes'][0,0,:20].sum()==1 and not identity['nodes'][0,1:,:20].any()


def test_calibration_detects_compression():
    target=np.array([2.,4.,6.]);group=np.zeros(3,int);base=3.8;pred=base+.5*(target-base)
    result=calibration(target,pred,group);np.testing.assert_allclose(result['slope'],.5);np.testing.assert_allclose(result['variance_ratio'],.5)
