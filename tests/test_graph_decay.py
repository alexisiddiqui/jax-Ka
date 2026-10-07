import numpy as np
import jax
import jax.numpy as jnp
from pkatrain.graph_decay import ScheduledScalarEngine,continuation_lr
from pkatrain.trainer import ScalarEngine


def test_cosine_continuation_endpoints_and_monotonicity():
    values=[continuation_lr(epoch,1,10) for epoch in range(21,101)]
    assert values[0]<1e-3 and values[-1]>0.99e-5
    assert all(a>=b for a,b in zip(values,values[1:]))


def test_runtime_learning_rate_preserves_scalar_adam_state_and_update():
    predict=lambda p,x:x@p;p=jnp.array([.3,.7]);x=jnp.arange(24,dtype=jnp.float32).reshape(3,4,2)/20
    y=jnp.ones((3,4));mask=jnp.ones((3,4),bool);valid=jnp.ones(3,bool)
    old=ScalarEngine(predict,.001);new=ScheduledScalarEngine(predict);a=old.optimizer.init(p);b=new.optimizer.init(p)
    assert str(jax.tree.structure(a))==str(jax.tree.structure(b))
    po,ao,lo=old.audited_batch_update(p,a,x,y,mask,valid);pn,bn,ln=new.audited_batch_update(p,b,x,y,mask,valid,.001)
    np.testing.assert_allclose(lo,ln,rtol=1e-7)
    for left,right in zip(jax.tree.leaves((po,ao)),jax.tree.leaves((pn,bn))):np.testing.assert_allclose(left,right,rtol=2e-6,atol=2e-7)
