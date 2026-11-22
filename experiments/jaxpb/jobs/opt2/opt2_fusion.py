"""Localise representation vs decoding; numerical work only inside Slurm."""
import os,sys
assert os.environ.get('SLURM_JOB_ID')
sys.path.insert(0,'tests')
from test_opt2 import *

def packed_coeff(phi,rhs,ex,ey,ez,dv,om,it,mc,done,*,n,check):
 masks=jnp.asarray(SOR.colour_layout(n)[1])
 def half(p,c):
  v=p[1-c];ld=lambda s:SOR._neighbour(v,c,s,n)
  nb=(ex[c]*ld(n*n)+SOR._neighbour(ex[1-c],c,-n*n,n)*ld(-n*n)
      +ey[c]*ld(n)+SOR._neighbour(ey[1-c],c,-n,n)*ld(-n)
      +ez[c]*ld(1)+SOR._neighbour(ez[1-c],c,-1,n)*ld(-1))
  gs=(nb+rhs[c])*dv[c];old=p[c]
  return p.at[c].set(jnp.where(masks[c],old+om*(gs-old),old))
 def body(k,s):
  p,mc=s;new=half(half(p,0),1)
  return new,jnp.where(k==check-1,jnp.max(jnp.abs(new-p)),mc)
 new,mc2=jax.lax.fori_loop(0,check,body,(phi,mc))
 return new,mc2

for n in (5,7,8,9):
 rng=np.random.default_rng(n)
 with PB.GEO.x32():
  m={k:jnp.asarray(rng.random((n,n,n))<.45) for k in ('ex','ey','ez','ion')};kap=jnp.float32(.017)
  ex,ey,ez,d=PB.operator_coeffs(m,PB.EPSIN,PB.EPSOUT,kap);dv=jnp.where(PB.face_mask(n),0.,1./d)
  phi=jnp.asarray(rng.normal(size=(n,n,n)),jnp.float32);rhs=jnp.asarray(rng.normal(size=(n,n,n)),jnp.float32);om=jnp.float32(1.43)
  ca,cb=map(jnp.asarray,SOR.colour_masks(n));state=(jnp.int32(0),jnp.float32(1e9),jnp.bool_(False))
  for check in (1,10):
   a=jax.jit(partial(reference_chunk,check=check))(phi,rhs,ex,ey,ez,dv,ca,cb,om,*state)
   b=jax.jit(partial(packed_coeff,n=n,check=check))(*[SOR.pack(v,n) for v in (phi,rhs,ex,ey,ez,dv)],om,*state)
   av=np.asarray(a[0]);bv=np.asarray(SOR.unpack(b[0],n));print(n,check,'packed coefficients',int((av!=bv).sum()),np.max(abs(av-bv)),flush=True)
