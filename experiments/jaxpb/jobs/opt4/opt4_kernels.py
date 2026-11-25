"""Independent candidate arms; copied into src only after timing and exact gates."""
import jax
import jax.numpy as jnp
from jaxpb import maps as M, geometry as G

def compact_indices(start,count,rank):
    """Invert offset-major occupied-slot ranks (including runs of empty cells)."""
    end=jnp.cumsum(count,axis=1); lo=jnp.zeros(rank.shape,jnp.int32); hi=jnp.full(rank.shape,count.shape[1],jnp.int32)
    for _ in range(count.shape[1].bit_length()):
        mid=(lo+hi)//2
        edge=jnp.take_along_axis(end,jnp.minimum(mid,count.shape[1]-1),axis=1)
        right=(mid<count.shape[1]) & (rank>=edge)
        lo=jnp.where(right,mid+1,lo); hi=jnp.where(right,hi,mid)
    off=jnp.minimum(lo,count.shape[1]-1); prior=jnp.take_along_axis(end-count,off,axis=1)
    return off,rank-prior,rank<end[:,-1:]

def compact_query(q,P,w,table,lo,ncell,cell,offs,slab=128):
    c=jnp.floor((q-lo)/cell).astype(jnp.int32); cc=c[:,None]+offs[None]
    ok=jnp.all((cc>=0)&(cc<ncell),axis=2); cid=jnp.where(ok,(cc[...,0]*ncell+cc[...,1])*ncell+cc[...,2],0)
    if isinstance(table,tuple):
        order,start,stop,slots=table; st=start[cid]; count=jnp.where(ok,jnp.minimum(stop[cid]-st,len(slots)),0)
    else:
        st=jnp.zeros_like(cid); count=jnp.where(ok,jnp.sum(table[cid]>=0,axis=2,dtype=jnp.int32),0)
    init=(jnp.full(q.shape[0],jnp.inf,q.dtype),jnp.full(q.shape[0],-1,jnp.int32))
    def body(k,best):
        rank=jnp.broadcast_to(k*slab+jnp.arange(slab,dtype=jnp.int32),(q.shape[0],slab))
        off,slot,valid=compact_indices(st,count,rank)
        if isinstance(table,tuple):
            ii=jnp.take_along_axis(st,off,axis=1)+slot; idx=order[jnp.minimum(ii,len(order)-1)]
        else: idx=table[jnp.take_along_axis(cid,off,axis=1),jnp.minimum(slot,table.shape[1]-1)]
        idx=jnp.where(valid,idx,-1)
        value,point=M._pick(q,P,w,idx); take=value<best[0]
        return jnp.where(take,value,best[0]),jnp.where(take,point,best[1])
    return jax.lax.fori_loop(0,(jnp.max(jnp.sum(count,axis=1))+slab-1)//slab,body,init)

def reach_query(q,P,table,lo,ncell,cell,rp=M.RP,slab=64,live=None):
    """Boolean cell-by-cell query: stop each slab/cell once all rows have hits."""
    order,start,stop,_=table
    offsets=M._offs(1)
    offsets=offsets[jnp.argsort(jnp.sum(offsets*offsets,axis=1),stable=True)]
    cc=jnp.floor((q-lo)/cell).astype(jnp.int32)[:,None]+offsets[None]
    ok=jnp.all((cc>=0)&(cc<ncell),axis=2)
    # Conservative cell bounds only prune clear misses. The final f32 norm and
    # strict radius test remain exactly those of the original forward scatter.
    lower=lo+cc.astype(q.dtype)*cell;upper=lower+cell
    scale=jnp.max(jnp.maximum(jnp.abs(q[:,None]),jnp.maximum(jnp.abs(lower),jnp.abs(upper))),axis=2)
    margin=2e-3+32*jnp.finfo(q.dtype).eps*scale
    gap=jnp.maximum(jnp.maximum(lower-q[:,None],q[:,None]-upper),0)
    ok=ok&(jnp.linalg.norm(gap,axis=2)<=rp+margin)
    if live is not None:ok=ok&live[:,None]
    cid=jnp.where(ok,(cc[...,0]*ncell+cc[...,1])*ncell+cc[...,2],0)
    st=start[cid];count=jnp.where(ok,stop[cid]-st,0)
    remaining=jnp.cumsum(count[:,::-1],axis=1)[:,::-1]
    def outer_cond(c):
        cellidx,hit=c
        return (cellidx<27)&jnp.any((~hit)&(remaining[:,jnp.minimum(cellidx,26)]>0))
    def outer_body(c):
        cellidx,hit=c;s=st[:,cellidx];total=count[:,cellidx]
        def cond(c):
            k,hit=c;return k<jnp.max(jnp.where(hit,0,total))
        def body(c):
            k,hit=c;slot=k+jnp.arange(slab,dtype=jnp.int32)
            valid=(slot[None]<total[:,None])&(~hit[:,None])
            idx=order[jnp.minimum(s[:,None]+slot[None],len(order)-1)]
            idx=jnp.where(valid,idx,0)
            d=jnp.linalg.norm(q[:,None]-P[idx],axis=2)-jnp.zeros((),q.dtype)
            return k+slab,hit|jnp.any(valid&(d<rp),axis=1)
        return cellidx+1,jax.lax.while_loop(cond,body,(jnp.int32(0),hit))[1]
    return jax.lax.while_loop(outer_cond,outer_body,(jnp.int32(0),jnp.zeros(len(q),bool)))[1]

def reach_need(need,G,p,pvalid,acent,h,n,ax,NS,ncell,rp=M.RP):
    B=need.shape[0]; cell=M.PCELL; lo=acent-(n-1)*h/2-(rp+.5)
    pp=p
    def cells(p,valid,lo):
        c=jnp.floor((p-lo)/cell).astype(jnp.int32)
        ok=valid&jnp.all((c>=0)&(c<ncell),axis=1)
        cid=jnp.where(ok,(c[:,0]*ncell+c[:,1])*ncell+c[:,2],ncell**3)
        order=jnp.argsort(cid,stable=True).astype(jnp.int32)
        bounds=jnp.searchsorted(cid[order],jnp.arange(ncell**3+1,dtype=jnp.int32),side='left').astype(jnp.int32)
        return order,bounds[:-1],bounds[1:],jnp.arange(1,dtype=jnp.int32)
    table=jax.vmap(cells)(pp,pvalid,lo)
    count=jnp.sum(need,axis=1,dtype=jnp.int32); sel=jax.vmap(lambda x:jnp.nonzero(x,size=NS,fill_value=0)[0])(need)
    q=jnp.take_along_axis(G,sel[...,None],axis=1)+jnp.zeros(3,G.dtype).at[ax].set(h/2)
    chunk=256; pad=(-NS)%chunk; qp=jnp.pad(q,((0,0),(0,pad),(0,0))); out=jnp.zeros((B,n**3),bool)
    def body(k,out):
        qq=jax.lax.dynamic_slice_in_dim(qp,k*chunk,chunk,axis=1)
        live=k*chunk+jnp.arange(chunk)[None]<count[:,None]
        hit=jax.vmap(lambda q,p,t,l,v:reach_query(q,p,t,l,ncell,cell,rp,live=v))(qq,pp,table,lo,live)
        ids=jax.lax.dynamic_slice_in_dim(jnp.pad(sel,((0,0),(0,pad))),k*chunk,chunk,axis=1)
        live=k*chunk+jnp.arange(chunk)[None]<count[:,None]
        return jax.vmap(lambda o,i,h,v:o.at[jnp.where(v,i,n**3)].set(h,mode='drop'))(out,ids,hit,live)
    return jax.lax.fori_loop(0,(jnp.max(count)+chunk-1)//chunk,body,out)

GEO_NCELL=128

def span_hash(P,valid,lo,cell):
    """Preserve the original 1024-grid sort; add a bounded direct span table.

    States outside the auxiliary 128-grid retain the original hashed search path.
    This avoids changing clipping or silently restricting supported coordinates.
    """
    scid,order=G._hash_cells(P,valid,lo,cell)
    c=G._cell_of(P,lo,cell); fits=jnp.all((~valid)|jnp.all(c<GEO_NCELL,axis=1))
    cid=(c[:,0]*GEO_NCELL+c[:,1])*GEO_NCELL+c[:,2]
    cid=jnp.where(valid&jnp.all(c<GEO_NCELL,axis=1),cid,GEO_NCELL**3)
    counts=jnp.zeros(GEO_NCELL**3+1,jnp.int32).at[cid].add(valid.astype(jnp.int32))
    bounds=jnp.concatenate([jnp.zeros(1,jnp.int32),jnp.cumsum(counts[:-1],dtype=jnp.int32)])
    return (scid,bounds,fits),order

def candidate_spans(pts,lo,cell,hashed,order,K):
    scid,bounds,fits=hashed
    c=G._cell_of(pts,lo,cell)[:,None]+jnp.asarray(G.OFFS27)[None]
    ok=jnp.all((c>=0)&(c<G.NCV),axis=2)
    cid=jnp.where(ok,(c[...,0]*G.NCV+c[...,1])*G.NCV+c[...,2],-1)
    def direct(_):
        inaux=ok&jnp.all(c<GEO_NCELL,axis=2)
        idx=jnp.where(inaux,(c[...,0]*GEO_NCELL+c[...,1])*GEO_NCELL+c[...,2],0).astype(jnp.int32)
        return jnp.where(inaux,bounds[idx],0),jnp.where(inaux,bounds[idx+1],0)
    st,en=jax.lax.cond(fits,direct,lambda _:(jnp.searchsorted(scid,cid,side='left').astype(jnp.int32),jnp.searchsorted(scid,cid,side='right').astype(jnp.int32)),None)
    return st,jnp.where(ok,jnp.minimum(en-st,K),0),jnp.max(jnp.where(ok,en-st,0))

def span_cand(pts,lo,cell,hashed,order,K):
    st,count,occ=candidate_spans(pts,lo,cell,hashed,order,K)
    kk=st[...,None]+jnp.arange(K,dtype=jnp.int32)
    idx=jnp.where(jnp.arange(K)<count[...,None],order[jnp.minimum(kk,len(order)-1)],-1)
    return idx.reshape(len(pts),-1),occ

def compact_burial(pts,own,lo,cell,hashed,order,P,R,bound,K,knn):
    st,count,occ=candidate_spans(pts,lo,cell,hashed,order,K); capacity=128
    def small(_):
        rank=jnp.broadcast_to(jnp.arange(capacity,dtype=jnp.int32),(len(pts),capacity))
        off,slot,valid=compact_indices(st,count,rank); ii=jnp.take_along_axis(st,off,axis=1)+slot
        idx=jnp.where(valid,order[jnp.minimum(ii,len(order)-1)],-1)
        return G._burial(pts,own,idx,P,R,bound,knn)
    def full(_):
        kk=st[...,None]+jnp.arange(K,dtype=jnp.int32)
        cand=jnp.where(jnp.arange(K)<count[...,None],order[jnp.minimum(kk,len(order)-1)],-1).reshape(len(pts),-1)
        return G._burial(pts,own,cand,P,R,bound,knn)
    b,ba=jax.lax.cond(jnp.max(jnp.sum(count,axis=1))<=capacity,small,full,None)
    return b,ba,occ
