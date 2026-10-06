"""Small graph encoder + titratable-site cross-attention, scalar pKa pretraining head.

The scalar head predicts an output midpoint, NEVER a binary intrinsic energy.
Later LocalTerms heads may reuse encode() without changing training ownership.
"""
import jax
import jax.numpy as jnp
from jaxpropka.parameters import MODEL_PKA

D=32
HEADS=4
FF=72


def initialize(key, width=D, ff=FF):
    assert width % HEADS == 0
    keys=iter(jax.random.split(key,64))
    def linear(a,b):return dict(w=jax.random.normal(next(keys),(a,b),dtype=jnp.float32)/jnp.sqrt(float(a)),b=jnp.zeros(b,jnp.float32))
    def block():return dict(q=linear(width,width),k=linear(width,width),v=linear(width,width),o=linear(width,width),
        edge=linear(20,HEADS),up=linear(width,ff),down=linear(ff,width),
        norm1=jnp.stack((jnp.ones(width),jnp.zeros(width))),norm2=jnp.stack((jnp.ones(width),jnp.zeros(width))))
    p=dict(embed=linear(24,width),blocks=[block(),block()],query=block(),
           groups=jax.random.normal(next(keys),(9,width),dtype=jnp.float32)*.1,head=linear(width,1))
    # Near model-value baseline, but nonzero head lets the encoder receive gradients.
    p['head']['w']*=.01
    return p


def linear(p,x):return x@p['w']+p['b']
def norm(p,x):
    z=x-jnp.mean(x,axis=-1,keepdims=True)
    return z*jax.lax.rsqrt(jnp.mean(z*z,axis=-1,keepdims=True)+1e-5)*p[0]+p[1]


def attend(p,x,context,neighbors,edge,mask,switch):
    width=p['q']['w'].shape[0]
    q=linear(p['q'],norm(p['norm1'],x)).reshape((-1,HEADS,width//HEADS))
    c=norm(p['norm1'],context)
    k=linear(p['k'],c)[neighbors].reshape((*neighbors.shape,HEADS,width//HEADS))
    v=linear(p['v'],c)[neighbors].reshape((*neighbors.shape,HEADS,width//HEADS))
    logits=jnp.einsum('nhd,nkhd->nkh',q,k)/jnp.sqrt(float(width//HEADS))+linear(p['edge'],edge)
    logits=jnp.where(mask[...,None],logits,-1e9)
    weights=jax.nn.softmax(logits,axis=1)*mask[...,None]*switch[...,None]
    weights=weights/jnp.maximum(weights.sum(axis=1,keepdims=True),1e-8)
    message=jnp.einsum('nkh,nkhd->nhd',weights,v).reshape((-1,width))
    x=x+linear(p['o'],message)
    return x+linear(p['down'],jax.nn.gelu(linear(p['up'],norm(p['norm2'],x))))


def encode(p,g):
    h=linear(p['embed'],g['nodes'])
    for block in p['blocks']:
        h=attend(block,h,h,g['neighbors'],g['edge'],g['edge_mask'],g['switch'])
        h=h*g['node_mask'][:,None]
    return h


def predict(p,g):
    h=encode(p,g); i=g['query_residue']; group=g['query_group']
    query=h[i]+p['groups'][group]
    z=attend(p['query'],query,h,g['neighbors'][i],g['edge'][i],g['edge_mask'][i],g['switch'][i])
    return jnp.asarray(MODEL_PKA,jnp.float32)[group]+8*jnp.tanh(linear(p['head'],z)[:,0])
