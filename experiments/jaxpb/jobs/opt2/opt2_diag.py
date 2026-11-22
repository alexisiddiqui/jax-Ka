import os
assert os.environ.get('SLURM_JOB_ID')
from functools import partial
import sys
sys.path.insert(0,'tests')
from test_opt2 import *
for n in (5,7,8,9):
 rng=np.random.default_rng(n)
 with PB.GEO.x32():
  m={k:jnp.asarray(rng.random((n,n,n))<.45) for k in ('ex','ey','ez','ion')}
  kap=jnp.float32(.017)
  ex,ey,ez,d=PB.operator_coeffs(m,PB.EPSIN,PB.EPSOUT,kap)
  dv=jnp.where(PB.face_mask(n),0.,1./d)
  cd=SOR.pack(SOR.op_code(m).reshape(n,n,n),n);lut=SOR.op_lut(kap)
  ca,cb=map(jnp.asarray,SOR.colour_masks(n))
  phi=jnp.asarray(rng.normal(size=(n,n,n)),jnp.float32);rhs=jnp.asarray(rng.normal(size=(n,n,n)),jnp.float32);om=jnp.float32(1.43)
  old=jax.jit(partial(reference_chunk,check=1))
  new=jax.jit(partial(getattr(SOR,"sor_chunk_packed",SOR.sor_chunk),n=n,check=1))
  for step in range(3):
   a=old(phi,rhs,ex,ey,ez,dv,ca,cb,om,jnp.int32(0),jnp.float32(1e9),jnp.bool_(False))
   b=new(SOR.pack(phi,n),SOR.pack(rhs,n),cd,lut,om,jnp.int32(0),jnp.float32(1e9),jnp.bool_(False))
   av=np.asarray(a[0]);bv=np.asarray(SOR.unpack(b[0],n));delta=abs(av-bv);bad=delta!=0
   print(n,step,'bad',int(bad.sum()),'max',delta.max(),'by_colour',int((bad&np.asarray(ca)).sum()),int((bad&np.asarray(cb)).sum()),'mc',a[2],b[2],flush=True)
   if bad.any():
    ix=np.unravel_index(np.argmax(delta),delta.shape);print('worst',ix,av[ix],bv[ix],flush=True)
   phi=a[0]
  # Compare decoded reciprocal with the original separately, without stencil fusion.
  raw=np.asarray(SOR.op_code(m)).reshape(n,n,n).astype(np.int32)
  idx=(raw&1)+(np.roll(raw,1,0)&1)+((raw>>1)&1)+((np.roll(raw,1,1)>>1)&1)+((raw>>2)&1)+((np.roll(raw,1,2)>>2)&1)+8*((raw>>3)&1)
  dd=np.asarray(lut)[idx];print('dv',np.max(abs(dd[~SOR.face_mask_np(n)]-np.asarray(dv)[~SOR.face_mask_np(n)])),flush=True)
