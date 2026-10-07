import numpy as np
from pkabench.gqt_partner_accuracy import ablate_edges,hl_cut,metrics


def test_ablate_edges_changes_only_mask_and_switch():
    graph=dict(nodes=np.ones((2,24),np.float32),neighbors=np.array([[0,1],[0,1]]),
        edge=np.ones((2,2,20),np.float32),edge_mask=np.ones((2,2),bool),switch=np.ones((2,2),np.float32),
        node_mask=np.ones(2,bool),query_residue=np.array([0]),query_group=np.array([0]))
    cut=np.array([[False,True],[True,False]]);out=ablate_edges(graph,cut)
    assert not out["edge_mask"][cut].any() and not out["switch"][cut].any()
    np.testing.assert_array_equal(out["nodes"],graph["nodes"])
    assert graph["edge_mask"].all() and graph["switch"].all()


def test_hl_cut_is_bidirectional_and_specific():
    chains=np.array(["H","L","A"]);neighbors=np.array([[0,1,2],[0,1,2],[0,1,2]])
    cut=hl_cut(chains,neighbors,np.ones((3,3),bool),"H","L")
    np.testing.assert_array_equal(cut,[[False,True,False],[True,False,False],[False,False,False]])


def test_shift_metrics_and_error_cancellation():
    rows=[dict(teacher_shift=1.,model_shift=.8,ab_error=.4,free_error=.2),dict(teacher_shift=-1.,model_shift=-.6,ab_error=-.3,free_error=-.1)]
    got=metrics(rows,"model_shift")
    assert np.isclose(got["mae"],.3) and np.isclose(got["rmse"],np.sqrt(.1))
    assert got["sign_accuracy"]==1 and got["error_cancellation"]>0
