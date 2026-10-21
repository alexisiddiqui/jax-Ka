"""Isolated base-A positional/relation ablations; frozen pkanet modules unchanged."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from pkanet.model import linear, norm, dropout, HEADS
from pkanet.site_model import _site_tokens, attend_sites_indexed
from .production_inputs import orientation_edges

VARIANTS=('none','fourier','rope','relations','shared-projection','paired-delta')
SALT=13012026
# relation-maps-v1 (2026-10-17): per-layer relation map counts, variant 'relations-A-B' (layer 1 A maps, layer 2 B).
# relation_index classes are -2,-1,+1,+2,same chain,other chain; REMAP folds them onto a layer's maps:
# 6 = unchanged, 4 = +-1 / +-2 / same / other (sign merged), 1 = one map for every neighbour (shared-projection).
REMAP={6:(0,1,2,3,4,5),4:(1,0,0,1,2,3),1:(0,0,0,0,0,0)}
# full-attention-v1 (2026-10-17, user: "test 4/1 with transformer layer 2"): suffix '-full' (variant relations-A-1-full)
# makes residue layer 2 attend to every residue of the structure with its single map. Graph pairs keep the 33-feature
# edge bias, every other pair a learned per-head constant far_bias, blended by the 18-20 A switch so the bias is
# continuous in the coordinates: bias = s * edge_bias + (1 - s) * far_bias. Cross-chain pairs are allowed only when the
# graph itself has a cross-chain edge (bound / multi-chain construct); the free branch stays chain-isolated.
FAR_INIT=-4.0  # graph-like start: at init a non-graph pair gets e^-4 of a near pair's weight
# site-layers-v1 (2026-10-17, user: "Yup" to a second site-to-site layer after the triplet analysis): flag '-site2'
# (e.g. relations-4-1-site2) adds a second site-to-site attention block 'site2' after the first, same structure, its own
# parameters (random init like the first, key fold_in(seed, SALT + 2)). Flags combine: relations-4-1-full-site2.
# all-residue-v1 (2026-10-17, user approved; termini simplified, titratable gate later): flag '-allres' makes every
# residue a site token: residue embedding + a learned amino-acid-type embedding (aa_types, 20 types), the query block over
# its residue neighbours, then the site-to-site block(s) over the residue graph itself (same 20 A CA radius as the old site
# graph; 34-dim site edges reassembled from the 33 residue-edge features plus a same-residue flag; pair_bias 20 x 20 by
# residue type). One token per residue with three outputs: side chain, N-terminus and C-terminus, the termini read
# through their own identity-initialised maps (terminus_maps) before the shared heads. Query groups 0-6 read the side
# chain output, 7 the N-terminus, 8 the C-terminus. Labels and masks are unchanged.
# token-cutoff-v1 (2026-10-17, user: "lets try 15A, compare the memory to 20A for other chain, 15 for same chain"): flag
# '-cutS' or '-cutSxC' (with -allres; e.g. relations-4-1-allres-cut15x20) restricts the all-residue token layers (query
# block and site-to-site blocks) to CA-CA <= S A for same-chain and <= C A (default S) for other-chain neighbours; the
# residue layers keep the 20 A graph. The cosine switch is redone at each pair's radius (full weight below r - 2 A), so
# the token layers stay continuous in the coordinates. To save memory the neighbour lists are compacted, nearest first,
# to TOKEN_WIDTHS[cut][K] + 4 slots: the largest kept count of any stored structure (PINDER, pKPDB, benchmark-val) whose
# padded model-visible width is K (token_widths.json, written by token_widths.py); a K not in the table keeps all K
# slots. No new parameters.
FLAGS=('full','site2','allres')
SITE2_SALT=SALT+2
ALLRES_SALT=SALT+3
_WIDTHS=Path(__file__).with_name('token_widths.json')
TOKEN_WIDTHS=json.loads(_WIDTHS.read_text())['widths'] if _WIDTHS.exists() else {}

def _flag(p):return p in FLAGS or (p.startswith('cut') and len(p)>3)

def variant_parts(variant):
    """('relations-4-1', {'site2'}) for 'relations-4-1-site2'."""
    parts=variant.split('-');return '-'.join(p for p in parts if not _flag(p)),{p for p in parts if _flag(p)}

def full_layer2(variant):return 'full' in variant_parts(variant)[1]

def site_blocks(variant):return 2 if 'site2' in variant_parts(variant)[1] else 1

def all_residue(variant):return 'allres' in variant_parts(variant)[1]

def token_cutoff(variant):
    """('15x20', 15.0, 20.0) for a '-cut15x20' flag ('-cut15': same radius for both chains), None without one."""
    cut=[p[3:] for p in variant_parts(variant)[1] if p.startswith('cut')]
    if not cut:return None
    if len(cut)>1 or not all_residue(variant):raise ValueError(f'one token cutoff, with -allres: {variant}')
    same,_,cross=cut[0].partition('x');same=float(same);cross=float(cross or same)
    if not (4.<same<=20. and 4.<cross<=20.):raise ValueError(f'token cutoff radii must be in (4, 20] A: {variant}')
    return cut[0],same,cross

def token_width(name,k):
    """Compacted slot count for token cutoff `name` at model-visible neighbour width k."""
    w=TOKEN_WIDTHS.get(name,{}).get(str(k))
    return k if w is None else min(k,int(w)+4)

def relation_maps(variant):
    """Per-layer map counts of the relation variants, None for the others."""
    variant=variant_parts(variant)[0]
    if variant=='relations':return (6,6)
    if variant=='shared-projection':return (1,1)
    if variant.startswith('relations-'):
        counts=tuple(int(c) for c in variant.split('-')[1:])
        if len(counts)!=2 or any(c not in REMAP for c in counts):raise ValueError(f'unknown relation maps {variant}')
        return counts
    return None

def initialize(params,seed,variant):
    if variant=='none':return params
    width=params['head']['w'].shape[0]
    if width!=64:raise ValueError('registered positional ablations use width 64')
    key=jax.random.fold_in(jax.random.PRNGKey(seed),SALT)
    counts=relation_maps(variant)
    if variant=='fourier':
        params={**params,'position_projection':{'w':jax.random.normal(key,(32,width)) / jnp.sqrt(32.),'b':jnp.zeros(width)}}
    elif counts:
        if len(counts)!=len(params['blocks']):raise ValueError('one relation map count per residue block')
        # Identity initializes the new map to the same existing attention function.
        blocks=[{**b,'relation_projection':{'w':jnp.broadcast_to(jnp.eye(width),(count,width,width)).copy(),'b':jnp.zeros((count,width))}}
                for b,count in zip(params['blocks'],counts)]
        if full_layer2(variant):
            if counts[1]!=1:raise ValueError('full attention in layer 2 uses a single map')
            blocks[1]={**blocks[1],'far_bias':jnp.full((HEADS,),FAR_INIT,jnp.float32)}
        params={**params,'blocks':blocks}
        if site_blocks(variant)==2:params={**params,'site2':new_site_block(params['site'],jax.random.fold_in(jax.random.PRNGKey(seed),SITE2_SALT))}
        if all_residue(variant):
            ka,kb,kc=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),ALLRES_SALT),3)
            pair=lambda k:jax.random.normal(k,(20,20,HEADS),dtype=jnp.float32)*0.01
            params={**params,'aa_types':jax.random.normal(ka,(20,width),dtype=jnp.float32)*.1,
                    'terminus_maps':{'w':jnp.broadcast_to(jnp.eye(width),(2,width,width)).copy(),'b':jnp.zeros((2,width))},
                    'site':{**params['site'],'pair_bias':pair(kb)}}
            if 'site2' in params:params={**params,'site2':{**params['site2'],'pair_bias':pair(kc)}}
    elif variant=='paired-delta':
        params={**params,'delta_head':{'up':{'w':jax.random.normal(key,(width*2,width))/jnp.sqrt(width*2.),'b':jnp.zeros(width)},
                                     'down':{'w':jnp.zeros((width,width)),'b':jnp.zeros(width)}}}
    return params

def new_site_block(site,key):
    """A fresh site-to-site block shaped like `site`, drawn as pkanet.site_model.initialize_site draws the first."""
    from pkanet.site_model import _linear
    width,ff=site['q']['w'].shape[0],site['up']['w'].shape[1];edge=site['edge_up']['w'].shape[0];types=site['pair_bias'].shape[0]
    keys=iter(jax.random.split(key,12))
    return {"q":_linear(next(keys),width,width),"k":_linear(next(keys),width,width),"v":_linear(next(keys),width,width),"o":_linear(next(keys),width,width),
            "edge_up":_linear(next(keys),edge,width),"edge_down":_linear(next(keys),width,HEADS),
            "pair_bias":jax.random.normal(next(keys),(types,types,HEADS),dtype=jnp.float32)*0.01,
            "up":_linear(next(keys),width,ff),"down":_linear(next(keys),ff,width),
            "norm1":jnp.stack((jnp.ones(width),jnp.zeros(width))),"norm2":jnp.stack((jnp.ones(width),jnp.zeros(width)))}

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
    if token_cutoff(variant):graph={**graph,'position_ca':table[:,3:6]}
    if relation_maps(variant) is None:return graph
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

def dense_bias(graph,bias,far):
    """(N, N, H) pair bias: switch-blended edge bias on graph pairs, far elsewhere."""
    n=graph['neighbors'].shape[0];nb=graph['neighbors'];s=graph['switch']*graph['edge_mask']
    rows=jnp.broadcast_to(jnp.arange(n)[:,None],nb.shape)
    delta=jnp.where(graph['edge_mask'][...,None],s[...,None]*(bias-far),0.)
    return jnp.broadcast_to(far,(n,n,far.shape[0])).at[rows,nb].add(delta)

def pair_mask(graph):
    """Real residue pairs; cross-chain only if the graph has a cross-chain edge (free branch: chain-isolated)."""
    nb=graph['neighbors'];chain=graph['position_chain'];cross=chain[:,None]!=chain[None,:]
    allow=jnp.any(graph['edge_mask']&(chain[:,None]!=chain[nb]))
    real=graph['node_mask'].astype(bool)
    return real[:,None]&real[None,:]&(allow|~cross)

def attend_all(q,k,v,graph,bias,far):
    """Dense attention of every residue to every allowed residue (float32 contractions)."""
    d=q.shape[-1];hi=jax.lax.Precision.HIGHEST;allowed=pair_mask(graph)[...,None]
    logits=jnp.einsum('nhd,mhd->nmh',q,k,precision=hi)/jnp.sqrt(float(d))+dense_bias(graph,bias,far)
    logits=jnp.where(allowed,logits,-1e9)
    weights=jax.nn.softmax(logits,axis=1)*allowed
    weights=weights/jnp.maximum(weights.sum(axis=1,keepdims=True),1e-8)
    return jnp.einsum('nmh,mhd->nhd',weights,v,precision=hi)

def attend(block,x,graph,variant,*,key=None,dropout_rate=0.):
    from pkanet.attention_backend import indexed_attention_one
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
    if relation_maps(variant):
        proj=block['relation_projection'];mapped=jnp.einsum('nw,rwv->rnv',c,proj['w'])+proj['b'][:,None,:]
        k=linear(block['k'],mapped).reshape((-1,HEADS,width//HEADS));v=linear(block['v'],mapped).reshape(k.shape)
        # This block's map count comes from its parameters; neighbours read the K/V rows of their folded relation.
        count=proj['w'].shape[0]
        if count>1:nb=nb+jnp.asarray(REMAP[count],jnp.int32)[relation_index(graph)]*x.shape[0]
    if 'far_bias' in block:message=attend_all(q,k,v,graph,bias,block['far_bias']).reshape(x.shape)
    else:message=indexed_attention_one(q,k,v,nb,bias,graph['edge_mask'],graph['switch']).reshape(x.shape)
    keys=jax.random.split(key,2) if dropout_rate else (None,None)
    x=x+dropout(linear(block['o'],message),keys[0],dropout_rate)
    return x+dropout(linear(block['down'],jax.nn.gelu(linear(block['up'],norm(block['norm2'],x)))),keys[1],dropout_rate)

def residue_site_graph(graph):
    """The residue graph as an all-residue site graph. Residue edges (after the orientation transform) are rbf 0:16,
    direction to j in i's frame 16:19, same chain 19, direction to i in j's frame 20:23, F_i^T F_j 23:32, separation 32;
    site edges (pkanet.site_model) are rbf, direction i, direction j, orientation, same chain, same residue, separation."""
    e=graph['edge'];nb=graph['neighbors']
    if e.shape[-1]!=33:raise ValueError('all-residue tokens need the 33 orientation residue-edge features')
    same=(nb==jnp.arange(nb.shape[0])[:,None])[...,None].astype(e.dtype)
    edge=jnp.concatenate((e[...,0:16],e[...,16:19],e[...,20:23],e[...,23:32],e[...,19:20],same,e[...,32:33]),axis=-1)
    aa=jnp.argmax(graph['nodes'][:,:20],axis=1).astype(jnp.int32)
    return {'site_neighbors':nb,'site_edge':edge,'site_edge_mask':graph['edge_mask'],'site_switch':graph['switch'],
            'site_type':aa,'site_mask':graph['node_mask'].astype(bool)}

def token_graph(graph,cut):
    """The residue graph for the token layers under cutoff cut = (name, same-chain A, other-chain A): edges beyond their
    pair's radius masked, the switch redone at that radius, slots compacted nearest first to token_width(name, K)."""
    name,same_r,cross_r=cut;nb=graph['neighbors'];ca=graph['position_ca'];chain=graph['position_chain']
    distance=jnp.sqrt(jnp.sum((ca[nb]-ca[:,None])**2,axis=-1)+1e-8)
    radius=jnp.where(chain[nb]==chain[:,None],same_r,cross_r)
    keep=graph['edge_mask']&(distance<=radius)
    order=jnp.argsort(jnp.where(keep,distance,jnp.inf),axis=1)[:,:token_width(name,nb.shape[1])]
    pick=lambda x:jnp.take_along_axis(x,order.reshape(order.shape+(1,)*(x.ndim-2)),axis=1)
    distance,radius=pick(distance),pick(radius)
    switch=jnp.where(distance<radius-2,1.,.5*(1+jnp.cos(jnp.pi*jnp.clip((distance-radius+2)/2,0,1))))
    return {**graph,'neighbors':pick(nb),'edge':pick(graph['edge']),'edge_mask':pick(keep),'switch':switch.astype(jnp.float32)}

def all_residue_tokens(params,graph,h,keys,dropout_rate,cut=None):
    """(3N, W) tokens (side chain, N-terminus, C-terminus outputs per residue) and each query's row."""
    from pkanet.model import attend_indexed
    if cut:graph=token_graph(graph,cut)
    site=residue_site_graph(graph);mask=graph['node_mask'][:,None];n=h.shape[0]
    tok=(h+params['aa_types'][site['site_type']])*mask
    tok=attend_indexed(params['query'],tok,h,graph['neighbors'],graph['edge'],graph['edge_mask'],graph['switch'],key=keys[1],dropout_rate=dropout_rate)*mask
    g={**graph,**site}
    tok=attend_sites_indexed(params['site'],tok,g,key=keys[2],dropout_rate=dropout_rate)
    if 'site2' in params:tok=attend_sites_indexed(params['site2'],tok,g,key=keys[3],dropout_rate=dropout_rate)
    tm=params['terminus_maps'];ends=jnp.einsum('nw,twv->tnv',tok,tm['w'])+tm['b'][:,None,:]
    group=graph['query_group']
    query=graph['query_residue']+n*(group==7)+2*n*(group==8)
    return jnp.concatenate((tok,ends[0]*mask,ends[1]*mask),axis=0),query.astype(jnp.int32)

def functions(variant):
    cut=token_cutoff(variant)
    def embeddings(params,graph,*,key=None,dropout_rate=0.):
        two='site2' in params
        keys=jax.random.split(key,4 if two else 3) if dropout_rate else (None,)*4
        h=linear(params['embed'],graph['nodes']);mask=graph['node_mask'][:,None]
        if variant=='fourier':
            angle=graph['position_seq'][:,None]*(10000.**(-jnp.arange(16)/16))[None,:]
            h=h+linear(params['position_projection'],jnp.concatenate((jnp.sin(angle),jnp.cos(angle)),axis=-1))
        h=h*mask
        bkeys=jax.random.split(keys[0],len(params['blocks'])) if dropout_rate else [None]*len(params['blocks'])
        for b,k in zip(params['blocks'],bkeys):h=attend(b,h,graph,variant,key=k,dropout_rate=dropout_rate)*mask
        if 'aa_types' in params:return all_residue_tokens(params,graph,h,keys,dropout_rate,cut)
        tok=_site_tokens(params,graph,h,key=keys[1],dropout_rate=dropout_rate)
        tok=attend_sites_indexed(params['site'],tok,graph,key=keys[2],dropout_rate=dropout_rate)
        if two:tok=attend_sites_indexed(params['site2'],tok,graph,key=keys[3],dropout_rate=dropout_rate)
        return tok,graph['query_site']
    def predict(params,graph,**kw):
        tok,q=embeddings(params,graph,**kw);return (8*jnp.tanh(linear(params['head'],tok)[:,0]))[q]
    return embeddings,predict
