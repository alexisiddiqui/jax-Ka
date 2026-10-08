from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
from pkatrain.trainer import ScalarEngine
from pkatrain.graph_batches import epoch_batches,tightened_capacity


def test_batched_gradient_and_update_match_accumulated_structures():
    def predict(p,x):return x@p
    p=jnp.array([.3,.7]);engine=ScalarEngine(predict)
    x=jnp.arange(24,dtype=jnp.float32).reshape(3,4,2)/20
    y=jnp.array([[1.,2.,3.,4.],[1.,3.,0.,0.],[jnp.nan]*4])
    mask=jnp.array([[True]*4,[True,True,False,False],[False]*4]);valid=jnp.array([True,True,False])
    separate=[engine.audited_gradient(p,a,b,c) for a,b,c in zip(x[:2],y[:2],mask[:2])]
    loss,gradient=engine.batch_value_grad(p,x,y,mask,valid)
    np.testing.assert_allclose(gradient,np.mean([g for g,l in separate],axis=0),rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(loss,np.mean([l for g,l in separate]),rtol=1e-6)
    state=engine.optimizer.init(p)
    expected=engine.update(p,state,[g for g,l in separate])
    actual=engine.audited_batch_update(p,state,x,y,mask,valid)[:2]
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)


def test_bucketing_preserves_sampled_membership_and_tail():
    records={str(i):dict(n=n) for i,n in enumerate([100,200,600,900,120,800,450])}
    order=['0','1','0','2','3','4','5','6','2']
    batches=epoch_batches(order,records,np.random.default_rng(17),3)
    assert Counter(x for batch in batches for x in batch)==Counter(order)
    assert all(1<=len(batch)<=3 for batch in batches)


def test_tightened_capacity_is_rounded_and_capped_by_legacy_bucket():
    records={
        'a':dict(n=129,k=65,q=7),
        'b':dict(n=250,k=70,q=9),
    }
    manifest=dict(capacities={'384':[384,160,32]},config=dict(capacity_rounding=[128,64,None]))
    assert tightened_capacity(['a','b'],records,manifest)==[256,128,32]
    records['b'].update(n=380,k=159)
    assert tightened_capacity(['a','b'],records,manifest)==[384,160,32]


def test_no_tightened_policy_preserves_legacy_loader_path():
    records={'a':dict(n=12,k=8,q=3)}
    manifest=dict(capacities={'384':[32,32,32]},config={})
    assert tightened_capacity(['a'],records,manifest) is None
