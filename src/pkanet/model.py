"""Small graph encoder + titratable-site cross-attention, scalar pKa pretraining head.

The scalar head predicts an output midpoint, NEVER a binary intrinsic energy.
Later LocalTerms heads may reuse encode() without changing training ownership.
"""
import jax
import jax.numpy as jnp
from jaxpropka.parameters import MODEL_PKA

# Historical pKPDB model-compound constants used to define its deposited
# per-site shifts.  pKPDB did not deposit ARG labels, so ARG is deliberately
# undefined rather than silently borrowing JAX-Ka's physical-model constant.
PKPDB_PK_MOD = jnp.asarray(
    [3.79, 4.20, 6.74, 8.67, 9.59, 10.46, jnp.nan, 7.99, 2.90],
    dtype=jnp.float32,
)

D=32
HEADS=4
FF=72


def initialize(key, width=D, ff=FF, node_dim=24):
    assert width % HEADS == 0
    keys=iter(jax.random.split(key,64))
    def linear(a,b):return dict(w=jax.random.normal(next(keys),(a,b),dtype=jnp.float32)/jnp.sqrt(float(a)),b=jnp.zeros(b,jnp.float32))
    def block():return dict(q=linear(width,width),k=linear(width,width),v=linear(width,width),o=linear(width,width),
        edge=linear(20,HEADS),up=linear(width,ff),down=linear(ff,width),
        norm1=jnp.stack((jnp.ones(width),jnp.zeros(width))),norm2=jnp.stack((jnp.ones(width),jnp.zeros(width))))
    p=dict(embed=linear(node_dim,width),blocks=[block(),block()],query=block(),
           groups=jax.random.normal(next(keys),(9,width),dtype=jnp.float32)*.1,head=linear(width,1))
    # Near model-value baseline, but nonzero head lets the encoder receive gradients.
    p['head']['w']*=.01
    return p


def linear(p,x):return x@p['w']+p['b']
def norm(p,x):
    z=x-jnp.mean(x,axis=-1,keepdims=True)
    return z*jax.lax.rsqrt(jnp.mean(z*z,axis=-1,keepdims=True)+1e-5)*p[0]+p[1]


def dropout(x,key,rate):
    if rate==0:return x
    if key is None:raise ValueError('Training dropout requires an explicit RNG key')
    return jnp.where(jax.random.bernoulli(key,1-rate,x.shape),x/(1-rate),0.)


def attend(p,x,context,neighbors,edge,mask,switch,*,key=None,dropout_rate=0.,context_norm=None):
    width=p['q']['w'].shape[0]
    q=linear(p['q'],norm(p['norm1'],x)).reshape((-1,HEADS,width//HEADS))
    c=norm(p['norm1'] if context_norm is None else context_norm,context)
    k=linear(p['k'],c)[neighbors].reshape((*neighbors.shape,HEADS,width//HEADS))
    v=linear(p['v'],c)[neighbors].reshape((*neighbors.shape,HEADS,width//HEADS))
    logits=jnp.einsum('nhd,nkhd->nkh',q,k)/jnp.sqrt(float(width//HEADS))+linear(p['edge'],edge)
    logits=jnp.where(mask[...,None],logits,-1e9)
    weights=jax.nn.softmax(logits,axis=1)*mask[...,None]*switch[...,None]
    weights=weights/jnp.maximum(weights.sum(axis=1,keepdims=True),1e-8)
    message=jnp.einsum('nkh,nkhd->nhd',weights,v).reshape((-1,width))
    keys=jax.random.split(key,2) if dropout_rate else (None,None)
    x=x+dropout(linear(p['o'],message),keys[0],dropout_rate)
    return x+dropout(linear(p['down'],jax.nn.gelu(linear(p['up'],norm(p['norm2'],x)))),keys[1],dropout_rate)


def attend_indexed(p,x,context,neighbors,edge,mask,switch,*,key=None,dropout_rate=0.):
    """Attention using the optional float32 Triton indexed-message kernel."""
    from .triton_attention import indexed_attention_one
    width=p['q']['w'].shape[0]
    q=linear(p['q'],norm(p['norm1'],x)).reshape((-1,HEADS,width//HEADS))
    c=norm(p['norm1'],context)
    k=linear(p['k'],c).reshape((-1,HEADS,width//HEADS))
    v=linear(p['v'],c).reshape((-1,HEADS,width//HEADS))
    bias=linear(p['edge'],edge)
    message=indexed_attention_one(q,k,v,neighbors,bias,mask,switch).reshape((-1,width))
    keys=jax.random.split(key,2) if dropout_rate else (None,None)
    x=x+dropout(linear(p['o'],message),keys[0],dropout_rate)
    return x+dropout(linear(p['down'],jax.nn.gelu(linear(p['up'],norm(p['norm2'],x)))),keys[1],dropout_rate)


def attend_with_trace(p,x,context,neighbors,edge,mask,switch):
    """Deterministic attention with diagnostics; numerically matches ``attend``."""
    width=p['q']['w'].shape[0]
    q=linear(p['q'],norm(p['norm1'],x)).reshape((-1,HEADS,width//HEADS))
    c=norm(p['norm1'],context)
    k=linear(p['k'],c)[neighbors].reshape((*neighbors.shape,HEADS,width//HEADS))
    v=linear(p['v'],c)[neighbors].reshape((*neighbors.shape,HEADS,width//HEADS))
    logits=jnp.einsum('nhd,nkhd->nkh',q,k)/jnp.sqrt(float(width//HEADS))+linear(p['edge'],edge)
    logits=jnp.where(mask[...,None],logits,-1e9)
    softmax=jax.nn.softmax(logits,axis=1)*mask[...,None]
    weights=softmax*switch[...,None]
    weights=weights/jnp.maximum(weights.sum(axis=1,keepdims=True),1e-8)
    message=jnp.einsum('nkh,nkhd->nhd',weights,v).reshape((-1,width))
    x=x+linear(p['o'],message)
    output=x+linear(p['down'],jax.nn.gelu(linear(p['up'],norm(p['norm2'],x))))
    return output,dict(logits=logits,softmax=softmax,weights=weights)


def encode(p,g,*,key=None,dropout_rate=0.):
    h=linear(p['embed'],g['nodes'])
    keys=jax.random.split(key,len(p['blocks'])) if dropout_rate else [None]*len(p['blocks'])
    for block,k in zip(p['blocks'],keys):
        h=attend(block,h,h,g['neighbors'],g['edge'],g['edge_mask'],g['switch'],key=k,dropout_rate=dropout_rate)
        h=h*g['node_mask'][:,None]
    return h


def encode_indexed(p,g,*,key=None,dropout_rate=0.):
    """Encode residue nodes without materializing neighbor-expanded K/V."""
    h=linear(p['embed'],g['nodes'])
    keys=jax.random.split(key,len(p['blocks'])) if dropout_rate else [None]*len(p['blocks'])
    for block,k in zip(p['blocks'],keys):
        h=attend_indexed(block,h,h,g['neighbors'],g['edge'],g['edge_mask'],g['switch'],key=k,dropout_rate=dropout_rate)
        h=h*g['node_mask'][:,None]
    return h


def encode_with_trace(p,g):
    """Deterministic encoder plus per-layer, per-head attention tensors."""
    h=linear(p['embed'],g['nodes']);traces=[]
    for block in p['blocks']:
        h,trace=attend_with_trace(block,h,h,g['neighbors'],g['edge'],g['edge_mask'],g['switch'])
        h=h*g['node_mask'][:,None];traces.append(trace)
    return h,traces


def predict_shift(p,g,*,key=None,dropout_rate=0.):
    keys=jax.random.split(key,2) if dropout_rate else (None,None)
    h=encode(p,g,key=keys[0],dropout_rate=dropout_rate); i=g['query_residue']; group=g['query_group']
    query=h[i]+p['groups'][group]
    z=attend(p['query'],query,h,g['neighbors'][i],g['edge'][i],g['edge_mask'][i],g['switch'][i],key=keys[1],dropout_rate=dropout_rate)
    return 8*jnp.tanh(linear(p['head'],z)[:,0])


def predict_shift_indexed(p,g,*,key=None,dropout_rate=0.):
    """Triton encoder with the existing native, smaller query-attention layer."""
    keys=jax.random.split(key,2) if dropout_rate else (None,None)
    h=encode_indexed(p,g,key=keys[0],dropout_rate=dropout_rate);i=g['query_residue'];group=g['query_group']
    query=h[i]+p['groups'][group]
    z=attend(p['query'],query,h,g['neighbors'][i],g['edge'][i],g['edge_mask'][i],g['switch'][i],key=keys[1],dropout_rate=dropout_rate)
    return 8*jnp.tanh(linear(p['head'],z)[:,0])


def predict(p,g,*,key=None,dropout_rate=0.):
    """Legacy/JAX-Ka absolute prediction using its physical-model constants."""
    return (jnp.asarray(MODEL_PKA,jnp.float32)[g['query_group']]
            +predict_shift(p,g,key=key,dropout_rate=dropout_rate))


def predict_pkpdb(p,g,*,key=None,dropout_rate=0.):
    """Absolute prediction reconstructed from an explicitly learned pKPDB shift."""
    return PKPDB_PK_MOD[g['query_group']]+predict_shift(p,g,key=key,dropout_rate=dropout_rate)


def predict_pkpdb_indexed(p,g,*,key=None,dropout_rate=0.):
    """pKPDB absolute prediction using the optional indexed encoder."""
    return PKPDB_PK_MOD[g['query_group']]+predict_shift_indexed(p,g,key=key,dropout_rate=dropout_rate)


def predict_pkpdb_with_trace(p,g):
    """pKPDB prediction with encoder/query attention; inference only, no dropout."""
    h,encoder=encode_with_trace(p,g);i=g['query_residue'];group=g['query_group']
    query=h[i]+p['groups'][group]
    z,query_trace=attend_with_trace(
        p['query'],query,h,g['neighbors'][i],g['edge'][i],
        g['edge_mask'][i],g['switch'][i])
    shift=8*jnp.tanh(linear(p['head'],z)[:,0])
    return dict(predicted_shift=shift,predicted_pka=PKPDB_PK_MOD[group]+shift,
                encoder=tuple(encoder),query=query_trace)
