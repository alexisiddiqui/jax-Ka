"""Isolated base-A positional/relation ablations; frozen pkanet modules unchanged."""
import jax
import jax.numpy as jnp
from pkanet.model import linear, norm, dropout, HEADS
from pkanet.site_model import _site_tokens, attend_sites_indexed
from .production_inputs import orientation_edges

VARIANTS=('none','fourier','rope','relations','shared-projection','paired-delta')
SALT=13012026

def initialize(params,seed,variant):
    if variant=='none':return params
    width=params['head']['w'].shape[0]
    if width!=64:raise ValueError('registered positional ablations use width 64')
    key=jax.random.fold_in(jax.random.PRNGKey(seed),SALT)
    if variant=='fourier':
        params={**params,'position_projection':{'w':jax.random.normal(key,(32,width)) / jnp.sqrt(32.),'b':jnp.zeros(width)}}
    elif variant in ('relations','shared-projection'):
        count=6 if variant=='relations' else 1
        # Identity initializes the new map to the same existing attention function.
        blocks=[{**b,'relation_projection':{'w':jnp.broadcast_to(jnp.eye(width),(count,width,width)).copy(),'b':jnp.zeros((count,width))}} for b in params['blocks']]
        params={**params,'blocks':blocks}
    elif variant=='paired-delta':
        params={**params,'delta_head':{'up':{'w':jax.random.normal(key,(width*2,width))/jnp.sqrt(width*2.),'b':jnp.zeros(width)},
                                     'down':{'w':jnp.zeros((width,width)),'b':jnp.zeros(width)}}}
    return params

def delta_correction(head,a,b):
    # Symmetric gate times antisymmetric difference: swap negates; identical inputs give zero.
    context=jnp.concatenate((a+b,jnp.abs(a-b)),axis=-1)
    gate=linear(head['down'],jax.nn.gelu(linear(head['up'],context)))
    return 8*jnp.tanh(jnp.sum(gate*(a-b),axis=-1)/jnp.sqrt(float(a.shape[-1])))

def annotate(graph,variant):
    table=graph['nodes'][:,-14:];n=table.shape[0];row=jnp.arange(n)
    chain=table[:,12];seq=table[:,13];valid=graph['node_mask'].astype(bool)
    prev=jnp.maximum(row-1,0)
    connected=(row>0)&valid&valid[prev]&(chain==chain[prev])&(seq==seq[prev]+1)&(jnp.linalg.norm(table[prev,6:9]-table[:,0:3],axis=-1)<2.)
    segment=jnp.cumsum(~connected)
    graph={**graph,'position_chain':chain,'position_seq':seq,'position_segment':segment}
    if variant not in ('relations','shared-projection'):return graph
    # Explicit +/-1,+/-2 edges, deduplicated against the existing spatial graph.
    offset=jnp.array([-2,-1,1,2]);dest=jnp.clip(row[:,None]+offset,0,n-1)
    mask=(row[:,None]+offset>=0)&(row[:,None]+offset<n)&valid[:,None]&valid[dest]
    mask=mask&(chain[:,None]==chain[dest])&(segment[:,None]==segment[dest])
    duplicate=jnp.any((dest[:,:,None]==graph['neighbors'][:,None,:])&graph['edge_mask'][:,None,:],axis=-1)
    mask=mask&~duplicate
    ca=table[:,3:6];x=table[:,6:9]-ca;y=table[:,:3]-ca
    xn=jnp.linalg.norm(x,axis=-1,keepdims=True);x=x/jnp.maximum(xn,1e-6)
    y=y-jnp.sum(y*x,axis=-1,keepdims=True)*x;yn=jnp.linalg.norm(y,axis=-1,keepdims=True);y=y/jnp.maximum(yn,1e-6)
    frame=jnp.stack((x,y,jnp.cross(x,y)),axis=-1)
    delta=ca[dest]-ca[:,None];distance=jnp.sqrt(jnp.sum(delta**2,axis=-1)+1e-8)
    direction=jnp.einsum('nkj,njl->nkl',delta,frame)/distance[...,None]
    direction=direction*((xn[:,0]>1e-5)&(yn[:,0]>1e-5))[:,None,None]
    rbf=jnp.exp(-((distance[...,None]-jnp.linspace(0,20,16))/1.5)**2)
    extra=jnp.concatenate((rbf,direction,jnp.ones((*dest.shape,1))),axis=-1)
    return {**graph,'neighbors':jnp.concatenate((graph['neighbors'],dest),axis=1),
            'edge':jnp.concatenate((graph['edge'],extra),axis=1),
            'edge_mask':jnp.concatenate((graph['edge_mask'],mask),axis=1),
            'switch':jnp.concatenate((graph['switch'],mask.astype(jnp.float32)),axis=1)}

def rotate(x,pos):
    half=x.shape[-1]//2;freq=10000.**(-jnp.arange(half)/half)
    angle=pos[:,None,None]*freq[None,None,:];a,b=x[...,:half],x[...,half:]
    return jnp.concatenate((a*jnp.cos(angle)-b*jnp.sin(angle),a*jnp.sin(angle)+b*jnp.cos(angle)),axis=-1)

def relation_index(graph):
    nb=graph['neighbors'];chain=graph['position_chain'];pos=graph['position_seq'];seg=graph['position_segment']
    same=chain[:,None]==chain[nb];diff=pos[nb]-pos[:,None]
    rel=jnp.where(same,4,5)
    for k,offset in enumerate((-2,-1,1,2)):
        rel=jnp.where(same&(seg[:,None]==seg[nb])&(diff==offset),k,rel)
    return rel.astype(jnp.int32)

def attend(block,x,graph,variant,*,key=None,dropout_rate=0.):
    from pkanet.triton_attention import indexed_attention_one
    width=x.shape[-1];nb=graph['neighbors'];c=norm(block['norm1'],x)
    q=linear(block['q'],c).reshape((-1,HEADS,width//HEADS))
    k=linear(block['k'],c).reshape((-1,HEADS,width//HEADS));v=linear(block['v'],c).reshape(k.shape)
    bias=linear(block['edge'],graph['edge'])
    if variant=='rope':
        # Relative form of RoPE avoids cancellation from arbitrary position offsets.
        half=q.shape[-1]//2
        angle=(graph['position_seq'][nb]-graph['position_seq'][:,None])[...,None,None]*(10000.**(-jnp.arange(half)/half))
        gathered=k[nb];a,b=gathered[...,:half],gathered[...,half:]
        rk=jnp.concatenate((a*jnp.cos(angle)-b*jnp.sin(angle),a*jnp.sin(angle)+b*jnp.cos(angle)),axis=-1)
        correction=(jnp.einsum('nhd,nkhd->nkh',q,rk)-jnp.einsum('nhd,nkhd->nkh',q,gathered))/jnp.sqrt(float(width//HEADS))
        same=graph['position_chain'][:,None]==graph['position_chain'][nb]
        bias=bias+jnp.where(same[...,None],correction,0.)
    if variant in ('relations','shared-projection'):
        proj=block['relation_projection'];mapped=jnp.einsum('nw,rwv->rnv',c,proj['w'])+proj['b'][:,None,:]
        k=linear(block['k'],mapped).reshape((-1,HEADS,width//HEADS));v=linear(block['v'],mapped).reshape(k.shape)
        if variant=='relations':nb=nb+relation_index(graph)*x.shape[0]
    message=indexed_attention_one(q,k,v,nb,bias,graph['edge_mask'],graph['switch']).reshape(x.shape)
    keys=jax.random.split(key,2) if dropout_rate else (None,None)
    x=x+dropout(linear(block['o'],message),keys[0],dropout_rate)
    return x+dropout(linear(block['down'],jax.nn.gelu(linear(block['up'],norm(block['norm2'],x)))),keys[1],dropout_rate)

def functions(variant):
    def embeddings(params,graph,*,key=None,dropout_rate=0.):
        keys=jax.random.split(key,3) if dropout_rate else (None,None,None)
        h=linear(params['embed'],graph['nodes']);mask=graph['node_mask'][:,None]
        if variant=='fourier':
            angle=graph['position_seq'][:,None]*(10000.**(-jnp.arange(16)/16))[None,:]
            h=h+linear(params['position_projection'],jnp.concatenate((jnp.sin(angle),jnp.cos(angle)),axis=-1))
        h=h*mask
        bkeys=jax.random.split(keys[0],len(params['blocks'])) if dropout_rate else [None]*len(params['blocks'])
        for b,k in zip(params['blocks'],bkeys):h=attend(b,h,graph,variant,key=k,dropout_rate=dropout_rate)*mask
        tok=_site_tokens(params,graph,h,key=keys[1],dropout_rate=dropout_rate)
        tok=attend_sites_indexed(params['site'],tok,graph,key=keys[2],dropout_rate=dropout_rate)
        return tok,graph['query_site']
    def predict(params,graph,**kw):
        tok,q=embeddings(params,graph,**kw);return (8*jnp.tanh(linear(params['head'],tok)[:,0]))[q]
    return embeddings,predict
