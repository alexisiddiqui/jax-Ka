"""Global site-to-residue query attention only; local residue and site encoders unchanged.

All-to-all query edges retain the base-A 33 geometric features but no radius cutoff
or radial switch. The explicit loader branch flag gates cross-chain attention.
Queries are chunked and checkpointed to bound activation memory; no new parameters.
"""
import jax
import jax.numpy as jnp
from pkanet.model import HEADS,linear,norm,dropout,encode_indexed
from pkanet.site_model import attend_sites_indexed
from .production_inputs import orientation_edges
CHUNK=32

def allowed(graph,rows):
    table=graph['global_backbone'];chain=table[:,12]
    return graph['node_mask'][None,:].astype(bool)&(graph['allow_cross_chain']|(chain[rows,None]==chain[None,:]))

def edge_features(table,rows):
    n=table.shape[0];nb=jnp.broadcast_to(jnp.arange(n),(rows.shape[0],n))
    ca=table[:,3:6];x=table[:,6:9]-ca;y=table[:,:3]-ca
    xn=jnp.linalg.norm(x,axis=-1,keepdims=True);x=x/jnp.maximum(xn,1e-6)
    y=y-jnp.sum(y*x,axis=-1,keepdims=True)*x;yn=jnp.linalg.norm(y,axis=-1,keepdims=True);y=y/jnp.maximum(yn,1e-6)
    frames=jnp.stack((x,y,jnp.cross(x,y)),axis=-1);valid=(xn[:,0]>1e-5)&(yn[:,0]>1e-5)
    delta=ca[None,:]-ca[rows,None];distance=jnp.sqrt(jnp.sum(delta**2,axis=-1)+1e-8)
    direction=jnp.einsum('rnj,rjl->rnl',delta,frames[rows])/distance[...,None]*valid[rows,None,None]
    rbf=jnp.exp(-((distance[...,None]-jnp.linspace(0,20,16))/1.5)**2)
    same=(table[rows,12,None]==table[None,:,12])[...,None]
    return jnp.concatenate((rbf,direction,same,orientation_edges(table,nb,rows=rows)),axis=-1)

def query_tokens(params,graph,residue,*,key=None,dropout_rate=0.):
    block=params['query'];idx=graph['site_residue'];site_mask=graph['site_mask'];width=residue.shape[-1]
    tokens=(residue[idx]+params['groups'][graph['site_type']])*site_mask[:,None]
    q=linear(block['q'],norm(block['norm1'],tokens)).reshape((-1,HEADS,width//HEADS))
    c=norm(block['norm1'],residue);k=linear(block['k'],c).reshape((-1,HEADS,width//HEADS));v=linear(block['v'],c).reshape(k.shape)
    s=tokens.shape[0];extra=(-s)%CHUNK
    qq=jnp.pad(q,((0,extra),(0,0),(0,0))).reshape((-1,CHUNK,HEADS,width//HEADS))
    rr=jnp.pad(idx,(0,extra)).reshape((-1,CHUNK));sm=jnp.pad(site_mask,(0,extra)).reshape((-1,CHUNK))
    def chunk(inputs):
        query,rows,valid=inputs
        bias=linear(block['edge'],edge_features(graph['global_backbone'],rows))
        logits=jnp.einsum('rhd,nhd->rnh',query,k,precision=jax.lax.Precision.HIGHEST)/jnp.sqrt(float(width//HEADS))+bias
        mask=allowed(graph,rows)&valid[:,None].astype(bool)
        weights=jax.nn.softmax(jnp.where(mask[...,None],logits,-1e9),axis=1)*mask[...,None]
        weights=weights/jnp.maximum(weights.sum(axis=1,keepdims=True),1e-8)
        return jnp.einsum('rnh,nhd->rhd',weights,v,precision=jax.lax.Precision.HIGHEST)
    message=jax.lax.map(jax.checkpoint(chunk),(qq,rr,sm)).reshape((-1,width))[:s]
    keys=jax.random.split(key,2) if dropout_rate else (None,None)
    tokens=tokens+dropout(linear(block['o'],message),keys[0],dropout_rate)
    tokens=tokens+dropout(linear(block['down'],jax.nn.gelu(linear(block['up'],norm(block['norm2'],tokens)))),keys[1],dropout_rate)
    return tokens*site_mask[:,None]

def functions():
    def embeddings(params,graph,*,key=None,dropout_rate=0.):
        keys=jax.random.split(key,3) if dropout_rate else (None,None,None)
        residue=encode_indexed(params,graph,key=keys[0],dropout_rate=dropout_rate)
        tokens=query_tokens(params,graph,residue,key=keys[1],dropout_rate=dropout_rate)
        tokens=attend_sites_indexed(params['site'],tokens,graph,key=keys[2],dropout_rate=dropout_rate)
        return tokens,graph['query_site']
    def predict(params,graph,**kwargs):
        tokens,idx=embeddings(params,graph,**kwargs)
        return (8*jnp.tanh(linear(params['head'],tokens)[:,0]))[idx]
    return embeddings,predict
