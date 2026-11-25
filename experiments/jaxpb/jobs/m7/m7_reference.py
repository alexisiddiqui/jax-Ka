"""Independent f64 reference: JAX SciPy CG on a sliced 79^3 interior.

No production PCG stencil, recurrence, stopping rule or preconditioner is used.
The interior operator is assembled from six sliced face coefficients. Relative
L2 tolerance 1e-13, absolute 1e-12; true D-scaled infinity residual checked <=1e-10.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxpb import geometry as G
WIDTH=4


def one(bc,rhs,ex,ey,ez,d):
    bc,rhs,ex,ey,ez,d=(x.astype(jnp.float64) for x in (bc,rhs,ex,ey,ez,d))
    mid=lambda x:x[1:-1,1:-1,1:-1]
    ep=mid(ex);em=ex[:-2,1:-1,1:-1];fp=mid(ey);fm=ey[1:-1,:-2,1:-1];gp=mid(ez);gm=ez[1:-1,1:-1,:-2];dd=mid(d)
    def nb(x):
        return (ep*x[2:,1:-1,1:-1]+em*x[:-2,1:-1,1:-1]+fp*x[1:-1,2:,1:-1]
                +fm*x[1:-1,:-2,1:-1]+gp*x[1:-1,1:-1,2:]+gm*x[1:-1,1:-1,:-2])
    def mv(x):return dd*x-nb(jnp.pad(x,1))
    b=mid(rhs)+nb(bc)
    x,_=cg(mv,b,tol=1e-13,atol=1e-12,maxiter=4000)
    residual=jnp.max(jnp.abs((b-mv(x))/dd))
    return bc.at[1:-1,1:-1,1:-1].set(x),residual

reference=jax.jit(jax.vmap(one))


def pool(phi0,rhs0,om0,op0,valid,coefs,stats=None,compare=None):
    ids=np.flatnonzero(valid);ops=np.asarray(op0);store=jnp.zeros_like(phi0);res=np.zeros(len(valid),np.float32)
    maxima={k:0. for k in ('coarse','focused','model')};rmax=0.
    with G.x64():
        for start in range(0,len(ids),WIDTH):
            group=ids[start:start+WIDTH];ix=np.pad(group,(0,WIDTH-len(group)))
            phi,r=reference(phi0[ix],rhs0[ix],*(x[ops[ix]] for x in coefs));rh=np.asarray(r)[:len(group)]
            assert np.all(np.isfinite(rh)) and np.all(rh<=1e-10),rh
            rmax=max(rmax,float(rh.max(initial=0)));res[group]=rh
            if compare is not None:
                delta=np.asarray(jnp.max(jnp.abs(phi[:len(group)]-compare[0][group].astype(jnp.float64)),axis=(1,2,3)))
                for i,v in zip(group,delta):
                    name='focused' if compare[1]==2 else ('coarse' if i<len(valid)//2 else 'model')
                    maxima[name]=max(maxima[name],float(v))
            store=store.at[group].set(phi[:len(group)].astype(jnp.float32))
    if stats is not None:stats.append(dict(blocks=0,lane_blocks=0,needed=0,residual=rmax,stored_residual=rmax))
    return store,np.zeros(len(valid),np.int32),res,maxima,rmax
