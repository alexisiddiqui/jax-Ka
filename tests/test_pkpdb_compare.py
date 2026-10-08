import numpy as np
from pkatrain.pkpdb_compare_data import select_queries


def test_cleaning_changes_supervision_not_environment():
    data=dict(nodes=np.arange(24).reshape(3,8),neighbors=np.array([[1],[2],[0]]),
              query_residue=np.array([0,1,2]),query_group=np.array([0,2,1]),labels=np.array([3.,4.,5.]))
    selected=select_queries(data,np.array([True,False,True]))
    assert selected['nodes'] is data['nodes']
    assert selected['neighbors'] is data['neighbors']
    np.testing.assert_array_equal(selected['query_residue'],[0,2])
    np.testing.assert_array_equal(selected['query_group'],[0,1])
    np.testing.assert_array_equal(selected['labels'],[3.,5.])
