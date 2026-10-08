import numpy as np
import jax
from pkanet.model import initialize,predict
from pkatrain.trainer import ScalarEngine
from test_pkanet import graph


def test_dropout_keys_eval_and_batched_gradients():
    p=initialize(jax.random.PRNGKey(17),width=44,ff=88);g,_=graph()
    baseline=predict(p,g)
    np.testing.assert_array_equal(baseline,predict(p,g,key=jax.random.PRNGKey(1),dropout_rate=0.))
    one=predict(p,g,key=jax.random.PRNGKey(1),dropout_rate=.1)
    two=predict(p,g,key=jax.random.PRNGKey(2),dropout_rate=.1)
    assert not np.array_equal(one,two)
    np.testing.assert_array_equal(one,predict(p,g,key=jax.random.PRNGKey(1),dropout_rate=.1))
    engine=ScalarEngine(predict,training_predict=lambda p,g,k:predict(p,g,key=k,dropout_rate=.1))
    batch=jax.tree.map(lambda x:np.stack([x,x]),g)
    y=np.array([[4.,5.,7.,8.]]*2,np.float32);mask=np.ones_like(y,bool);valid=np.ones(2,bool)
    key=jax.random.PRNGKey(4)
    loss,grad=engine.batch_value_grad(p,batch,y,mask,valid,key)
    assert np.isfinite(loss) and all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    again=engine.batch_value_grad(p,batch,y,mask,valid,key)
    for a,b in zip(jax.tree.leaves((loss,grad)),jax.tree.leaves(again)):
        np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-7)  # Parallel reductions need not be bitwise deterministic.
    np.testing.assert_allclose(engine.forward(p,g),baseline,rtol=1e-6,atol=1e-6)
