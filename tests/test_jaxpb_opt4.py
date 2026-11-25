"""OPT4 exact tests; execute only on Coulson compute nodes."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/name) for name in ("src","jobs")]
import jax
import jax.numpy as jnp
import numpy as np
from jaxpb import maps as M, geometry as G
# Candidate arms live in jobs until independently gated.
from opt4_kernels import compact_query as _compact_query, reach_need as _reach_need
compact_query=getattr(M,"_compact_query",_compact_query)
reach_need=getattr(M,"_reach_need",_reach_need)

def _compact_queries(K,csr,weighted=False,reach=1):
 rng=np.random.default_rng(17); p=rng.uniform(0,12,(8192,3)).astype(np.float32); p[:4096]=1e6; p[4099:,0]+=4
 p[4096]=[1.5,.5,.5]; p[4097]=[.5,.5,.5]; p[4098]=[.5,.5,.5]
 q=rng.uniform(-2,14,(269,3)).astype(np.float32); q[:3]=[[1.,.5,.5],[50.,50.,50.],[0.,0.,0.]]
 w=rng.uniform(.5,2.,len(p)).astype(np.float32) if weighted else np.zeros(len(p),np.float32)
 w[4096:4099]=1. if weighted else 0.
 lo=jnp.zeros(3,jnp.float32); pp=jnp.asarray(p); ww=jnp.asarray(w)
 cells,_=M.build_cells(pp,ww,lo,13,1.,K,batch_size=1<<30)
 order,start,stop,slots=cells; kk=start[:,None]+slots
 dense=jnp.where(kk<stop[:,None],order[jnp.minimum(kk,len(p)-1)],-1)
 table=cells if csr else dense
 def compare(q):
  c=jnp.floor(q).astype(jnp.int32)[:,None]+M._offs(reach)[None]; ok=jnp.all((c>=0)&(c<13),axis=2)
  ids=jnp.where(ok,(c[...,0]*13+c[...,1])*13+c[...,2],0)
  idx=jnp.where(ok[...,None],dense[ids],-1).reshape(len(q),-1)
  return M._pick(q,pp,ww,idx),compact_query(q,pp,ww,table,lo,13,1.,M._offs(reach))
 old,new=jax.jit(compare)(jnp.asarray(q))
 for a,b in zip(old,new): assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
 assert int(new[1][0])==4097
 assert int(new[1][1])==-1


def _compact_dense_slabs(K,csr):
 # Put the central cell first in atom order, but a spatially earlier cell must
 # win the equal-distance tie even though it has a larger atom index.
 cells=[13]+[i for i in range(27) if i!=13]
 p=np.full((8192,3),1e6,np.float32)
 for order,cid in enumerate(cells):
  c=np.array([cid//9,(cid//3)%3,cid%3],np.float32)+.5
  p[order*K:(order+1)*K]=c
 q=jnp.asarray([[1.,1.5,1.5],[50.,50.,50.],[0.,0.,0.],[2.9,1.5,1.5]],jnp.float32)
 pp=jnp.asarray(p);w=jnp.ones(len(p),jnp.float32);lo=jnp.zeros(3,jnp.float32)
 table,_=M.build_cells(pp,w,lo,13,1.,K,batch_size=1<<30)
 order,start,stop,slots=table;kk=start[:,None]+slots
 dense=jnp.where(kk<stop[:,None],order[jnp.minimum(kk,len(p)-1)],-1)
 def compare(q):
  c=jnp.floor(q).astype(jnp.int32)[:,None]+M._offs(1)[None];ok=jnp.all((c>=0)&(c<13),axis=2)
  ids=jnp.where(ok,(c[...,0]*13+c[...,1])*13+c[...,2],0)
  idx=jnp.where(ok[...,None],dense[ids],-1).reshape(len(q),-1)
  return M._pick(q,pp,w,idx),compact_query(q,pp,w,table if csr else dense,lo,13,1.,M._offs(1))
 old,new=jax.jit(compare)(q)
 for x,y in zip(old,new):assert np.asarray(x).tobytes()==np.asarray(y).tobytes()
 assert int(new[1][0])==5*K
 assert int(new[1][1])==-1

def _reverse_reach(h,n,PB,NS):
 rng=np.random.default_rng(19); B=4; ac=rng.uniform(-3,3,(B,3)).astype(np.float32)
 p=rng.uniform(-(n-1)*h/2-2,(n-1)*h/2+2,(B,PB,3)).astype(np.float32)+ac[:,None]
 # Exact grid edge, half-grid offsets, out-of-box probes and no valid probes.
 p[:,0]=ac-(n-1)*h/2; p[:,1]=ac+(n-1)*h/2+np.float32(1.4); valid=np.ones((B,PB),bool); valid[1]=False; valid[2,PB//2:]=False
 need=rng.random((B,n**3))<.03; need[0,0]=need[0,-1]=True; need[3]=False
 grid=jax.vmap(lambda a:jnp.stack(jnp.meshgrid(*M.grid_coords(a,h,n),indexing='ij'),axis=-1).reshape(-1,3))(jnp.asarray(ac))
 for ax in range(3):
  f=jax.jit(lambda p,v,need: (M.reach_mask_b(p,v,jnp.sum(v,axis=1),jnp.asarray(ac),h,n,ax,h,PB),reach_need(need,grid,p,v,jnp.asarray(ac),h,n,ax,NS,M.ncells(h,n,'p'))))
  old,new=f(jnp.asarray(p),jnp.asarray(valid),jnp.asarray(need))
  assert np.asarray(old & need).tobytes()==np.asarray(new).tobytes()


def _reach_strict_radius():
 n=9;h=1.;B=4;PB=16;ac=jnp.zeros((B,3),jnp.float32)
 grid=jax.vmap(lambda a:jnp.stack(jnp.meshgrid(*M.grid_coords(a,h,n),indexing='ij'),axis=-1).reshape(-1,3))(ac)
 center=(n//2*n+n//2)*n+n//2;need=jnp.zeros((B,n**3),bool).at[:,center].set(True)
 for ax in range(3):
  p=np.full((B,PB,3),1e6,np.float32);p[:,0]=0.;p[:,0,ax]=h/2
  rp=np.float32(M.RP);perp=(ax+1)%3
  p[:,0,perp]=[rp,np.nextafter(rp,np.float32(0)),np.nextafter(rp,np.float32(np.inf)),rp]
  valid=np.zeros((B,PB),bool);valid[:3,0]=True
  def compare(p,v):
   return M.reach_mask_b(p,v,jnp.sum(v,axis=1),ac,h,n,ax,h,PB)&need,reach_need(need,grid,p,v,ac,h,n,ax,1,M.ncells(h,n,'p'))
  old,new=jax.jit(compare)(jnp.asarray(p),jnp.asarray(valid))
  assert np.asarray(old).tobytes()==np.asarray(new).tobytes()
  assert np.asarray(new)[:,center].tolist()==[False,True,False,False]

def _geometry_direct_spans_empty_ties_and_fallback(outside,nvalid):
 from opt4_kernels import span_hash as sh,span_cand as sc,compact_burial as cb
 span_hash=getattr(G,"_span_hash",sh);span_cand=getattr(G,"_span_cand",sc);compact_burial=getattr(G,"_compact_burial",cb)
 rng=np.random.default_rng(23)
 with G.x64():
  p=rng.uniform(0,12,(8192,3)); valid=np.arange(len(p))<nvalid; p[~valid]=1e6; p[:3]=[1.,1.,1.]
  if outside:p[3]=[600.,0.,0.]
  P=jnp.asarray(p); V=jnp.asarray(valid); lo=jnp.zeros(3); cell=jnp.float64(4.); R=jnp.full(len(p),2.9)
  pts=jnp.asarray(np.concatenate([rng.uniform(-1,13,(256,3)),[[0,0,0],[1,1,1],[600,0,0],[4096,0,0]]]))
  own=jnp.zeros((len(pts),2),jnp.int64)
  def compare(P,V,pts):
   sc,order=G._hash_cells(P,V,lo,cell); aux,order2=span_hash(P,V,lo,cell)
   old,oc=G._cand(pts,lo,cell,sc,order,16); new,nc=span_cand(pts,lo,cell,aux,order2,16)
   return (old,oc,G._burial(pts,own,old,P,R,3.001,24)),(new,nc,compact_burial(pts,own,lo,cell,aux,order2,P,R,3.001,16,24)[:2])
  a,b=jax.jit(compare)(P,V,pts)
  for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):assert np.asarray(x).tobytes()==np.asarray(y).tobytes()


def test_compact_queries():
 with G.x32():
  for K in (24,48):
   for csr in (False,True):
    for weighted in (False,True): _compact_queries(K,csr,weighted)
    _compact_dense_slabs(K,csr)
    _compact_queries(K,csr,True,reach=2)

def test_reverse_reach():
 with G.x32():
  for h,n,PB,NS in ((1.,9,64,729),(.25,17,256,4913),(.25,81,4096,32768)): _reverse_reach(h,n,PB,NS)
  _reach_strict_radius()

def test_geometry_direct_spans_empty_ties_and_fallback():
 for outside in (False,True):
  for nvalid in (24,250): _geometry_direct_spans_empty_ties_and_fallback(outside,nvalid)


def test_model_burial_tiles_real_shape_and_empty():
 from opt4_arms import site_burial as sb
 site_burial=getattr(G,"_site_burial",sb)
 rng=np.random.default_rng(29)
 with G.x64():
  for N,empty in ((128,True),(405120,False)):
   p=jnp.asarray(rng.normal(0,2,(32,3)));R=jnp.full(32,2.9)
   pts=rng.normal(0,4,(N,3));pts[0]=np.nan;pts[1]=np.asarray(p[0])+np.array([2.9-1e-6-1e-12,0,0]);pts[2]=np.asarray(p[0])+np.array([2.9-1e-6+1e-12,0,0])
   pts=jnp.asarray(pts); own=jnp.broadcast_to(jnp.array([1,2]),(N,2));cand=jnp.full(32,-1,jnp.int32) if empty else jnp.arange(32,dtype=jnp.int32)
   def compare(pts):
    return G._burial(pts,own,jnp.broadcast_to(cand,(N,32)),p,R,3.001,24),site_burial(pts,own,cand,p,R,3.001,24)
   old,new=jax.jit(compare)(pts)
   for x,y in zip(old,new):assert np.asarray(x).tobytes()==np.asarray(y).tobytes()


def test_boundary_laplacian_real_shape_and_padding():
 from jaxpb import pb as P
 from opt4_arms import boundary_laplacian as old_arm
 fn=getattr(P,"_boundary_laplacian",old_arm)
 rng=np.random.default_rng(31);n=81
 with G.x32():
  phi=jnp.asarray(rng.normal(0,10,(n,n,n)).astype(np.float32))
  ids=jnp.asarray(np.concatenate([np.array([-1,0,n-1,n*n,n**3-1]),rng.integers(0,n**3,16000)]).astype(np.int32))
  def compare(phi):
   lap=(jnp.roll(phi,-1,0)+jnp.roll(phi,1,0)+jnp.roll(phi,-1,1)+jnp.roll(phi,1,1)+jnp.roll(phi,-1,2)+jnp.roll(phi,1,2)-6*phi)
   return lap.reshape(-1)[jnp.maximum(ids,0)],fn(phi,ids)
  a,b=jax.jit(compare)(phi)
  assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
