import jax
import jax.numpy as jnp
import numpy as np
from pkatrain.production_train import AuxEngine
from pkatrain.production_shift_weight import site_weights
# Isolate loss arithmetic from architecture: differentiable, deterministic site embeddings.
p={'embed':jnp.array([[.2,.1],[.3,-.2]]),'head':{'w':jnp.array([[.1],[.2]]),'b':jnp.zeros(1)},'aux_burial':{'w':jnp.ones((2,1))*.1,'b':jnp.zeros(1)}}
def emb(p,g,**kw):return g['nodes']@p['embed'],jnp.arange(2)
g={'nodes':jnp.array([[[[1.,0],[0,1]],[[.5,0],[0,.5]]]]),'query_group':jnp.zeros((1,2,2),jnp.int32)}
t=jnp.array([[[4.,7.],[4.,4.]]]);m=jnp.ones((1,2),bool);rsa=jnp.ones((1,2))*.5;c=jnp.zeros((1,2));v=jnp.ones(1,bool);key=jax.random.PRNGKey(17)
a=AuxEngine(p,1.,heads="burial",site_embeddings=emb)
b=AuxEngine(p,1.,heads="burial",site_embeddings=emb,shift_weights={'weights':[1.]*7})
x=a.paired_value_grad(p,g,t,m,rsa,c,v,key);y=b.paired_value_grad(p,g,t,m,rsa,c,v,key)
for aa,bb in zip(jax.tree.leaves(x),jax.tree.leaves(y)): np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6)
w={'weights':[.6,2.3,2.8,3.2,4,5,5]};z=AuxEngine(p,1.,heads="burial",site_embeddings=emb,shift_weights=w)
(zval,zaux),zg=z.paired_value_grad(p,g,t,m,rsa,c,v,key)
pred=a.predictions(p,g);err=((pred[:,0]-pred[:,1])-(t[:,0]-t[:,1]))**2
expected=jnp.mean(err*site_weights(t,w))
np.testing.assert_allclose(zaux[1],expected,rtol=1e-6)
np.testing.assert_allclose(zaux[0],x[0][1][0],rtol=1e-6)
# Padding a whole complex must not alter loss or gradients.
g2=jax.tree.map(lambda x:jnp.concatenate([x,x]),g);t2=jnp.concatenate([t,t]);m2=jnp.concatenate([m,jnp.zeros_like(m)])
z2=z.paired_value_grad(p,g2,t2,m2,jnp.concatenate([rsa,rsa]),jnp.concatenate([c,c]),jnp.array([True,False]),key)
for aa,bb in zip(jax.tree.leaves(((zval,zaux),zg)),jax.tree.leaves(z2)): np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6)
assert all(np.isfinite(q).all() for q in jax.tree.leaves(zg))
print('PASS: uniform-weight equivalence, exact weighted paired loss, unchanged state loss, padding invariance, finite gradients',flush=True)
