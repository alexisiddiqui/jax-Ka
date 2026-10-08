import numpy as np
from pkabench.gqt_overfit_attention import attention_metrics,sample_train


def test_component_unique_sample_is_deterministic():
    records=[dict(complex_id=str(i),component_id=f'c{i//2}',split='train') for i in range(10)]
    a=sample_train(records,n=5,seed=3);b=sample_train(records,n=5,seed=3)
    assert [x['complex_id'] for x in a]==[x['complex_id'] for x in b]
    assert len({x['component_id'] for x in a})==5


def test_attention_categories_partition_distance_mass():
    graph=dict(neighbors=np.array([[0,1,2]]),edge_mask=np.ones((1,3),bool),
        edge=np.zeros((1,3,20),np.float32),nodes=np.zeros((3,24),np.float32))
    graph['edge'][0,:,0]=1;graph['edge'][0,:,19]=[1,1,0];graph['nodes'][:,0]=1
    weights=np.array([[[.2,.1],[.3,.4],[.5,.5]]])
    result=attention_metrics(weights,graph,np.array([0]),1)
    np.testing.assert_allclose(result['mass_self'][0],[.2,.1])
    np.testing.assert_allclose(result['mass_0_6'][0],[.8,.9])
    np.testing.assert_allclose(result['mass_cross_chain'][0],[.5,.5])
