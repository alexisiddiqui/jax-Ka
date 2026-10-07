import numpy as np
from pkabench.gqt_alignment_audit import interaction_strength,graph_hops,site_distance


def test_reference_state_interaction_strength_is_gauge_invariant():
    core=np.array([[2.,-1.],[.5,3.]])
    row=np.array([4.,-2.,1.])[:,None];col=np.array([3.,-5.,2.])[None,:]
    w=np.zeros((3,3));w[:2,:2]=core
    expected=interaction_strength(w)
    np.testing.assert_allclose(interaction_strength(w+row+col),expected)
    assert expected==3.


def test_graph_hops_direct_multihop_missing_and_collapsed():
    graph=[{0,1},{0,1,2},{1,2,3},{2,3}]
    assert graph_hops(graph,1,1)==0
    assert graph_hops(graph,0,1)==1
    assert graph_hops(graph,0,2)==2
    assert graph_hops(graph,0,3)==3
    assert graph_hops(graph,None,3) is None


def test_functional_atom_minimum_distance():
    a=np.array([[0.,0.,0.],[5.,0.,0.]])
    b=np.array([[2.,0.,0.],[9.,0.,0.]])
    assert site_distance(a,b)==2.
    assert site_distance(np.empty((0,3)),b) is None
