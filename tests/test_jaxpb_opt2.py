"""Exact SOR arithmetic and stop-rule checks; execute in a Coulson allocation."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpb import pb as PB, sor as SOR, records as RC


from jaxpb_opt2_reference import sor_chunk as reference_chunk


def exact(a, b):
    assert np.asarray(a).dtype == np.asarray(b).dtype
    assert np.asarray(a).shape == np.asarray(b).shape
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.parametrize('n,check', [(5, 1), (7, 10), (8, 10), (9, 3)])
def test_packed_chunk_bitwise(n, check):
    if jax.default_backend() == "cpu":
        pytest.skip("CPU packed stencil reverted: shape-dependent FMA breaks exact parity")
    rng = np.random.default_rng(n)
    with PB.GEO.x32():
        m = {k: jnp.asarray(rng.random((n,n,n)) < .45) for k in ('ex','ey','ez','ion')}
        kap = jnp.float32(.017)
        ex,ey,ez,d = PB.operator_coeffs(m, PB.EPSIN, PB.EPSOUT, kap)
        dv = jnp.where(PB.face_mask(n), 0., 1./d)
        code = SOR.pack(SOR.op_code(m).reshape(n,n,n), n)
        lut = SOR.op_lut(kap)
        phi = jnp.asarray(rng.normal(size=(n,n,n)), jnp.float32)
        rhs = jnp.asarray(rng.normal(size=(n,n,n)), jnp.float32)
        om = jnp.float32(1.43)
        ca,cb = map(jnp.asarray, SOR.colour_masks(n))
        old = jax.jit(partial(reference_chunk,check=check))
        new = jax.jit(partial(SOR.sor_chunk_packed,n=n,check=check))
        for done in (False, True):
            state = (jnp.int32(20), jnp.float32(.123), jnp.bool_(done))
            a = old(phi,rhs,ex,ey,ez,dv,ca,cb,om,*state)
            b = new(SOR.pack(phi,n),SOR.pack(rhs,n),code,lut,om,*state)
            exact(a[0],SOR.unpack(b[0],n))
            for x,y in zip(a[1:],b[1:]): exact(x,y)


@pytest.mark.parametrize('n,check', [(5,1), (7,10), (8,10), (9,3)])
def test_cpu_chunk_bitwise(n, check):
    if jax.default_backend() != 'cpu':
        pytest.skip('CPU reference path')
    rng=np.random.default_rng(n)
    with PB.GEO.x32():
        m={k:jnp.asarray(rng.random((n,n,n))<.45) for k in ('ex','ey','ez','ion')}
        ex,ey,ez,d=PB.operator_coeffs(m,PB.EPSIN,PB.EPSOUT,jnp.float32(.017))
        dv=jnp.where(PB.face_mask(n),0.,1./d)
        phi=jnp.asarray(rng.normal(size=(n,n,n)),jnp.float32)
        rhs=jnp.asarray(rng.normal(size=(n,n,n)),jnp.float32)
        ca,cb=map(jnp.asarray,SOR.colour_masks(n))
        old=jax.jit(partial(reference_chunk,check=check))
        new=jax.jit(partial(SOR.sor_chunk,check=check))
        for it,done in [(0,False),(RC.NLIT-check,False),(20,True)]:
            args=(phi,rhs,ex,ey,ez,dv,ca,cb,jnp.float32(1.43),jnp.int32(it),jnp.float32(.123),jnp.bool_(done))
            for x,y in zip(old(*args),new(*args)): exact(x,y)


def test_pool_stop_freeze_and_compaction(monkeypatch):
    # Different amplitudes/omegas force lanes to finish at different check blocks;
    # padding, no-live input, and the cap retain the same reference semantics.
    monkeypatch.setattr(RC, 'LANE_WIDTHS', (1, 2, 4, 8))
    monkeypatch.setattr(RC, 'CPU_LANE_WIDTHS', (1, 2, 4, 8))
    n=17; w=8
    with PB.GEO.x32():
        m={k:jnp.zeros((n,n,n),bool) for k in ('ex','ey','ez','ion')}
        ex,ey,ez,d=PB.operator_coeffs(m,PB.EPSIN,PB.EPSOUT,jnp.float32(0.))
        dv=jnp.where(PB.face_mask(n),0.,1./d)
        phi=jnp.zeros((w,n,n,n),jnp.float32)
        rhs=np.zeros((w,n,n,n),np.float32)
        rhs[:,8,8,8]=[0.,1.,1e3,1e6,1e8,20.,1e5,2e3]
        rhs=jnp.asarray(rhs); om=jnp.asarray([1.,1.,1.2,1.4,1.8,1.1,1.5,1.6],jnp.float32)
        valid=np.array([1,1,1,1,1,0,1,0],bool)
        ca,cb=map(jnp.asarray,SOR.colour_masks(n))
        from jaxpb_opt2_reference import run_pool as reference_pool
        old_coefs=(ex[None],ey[None],ez[None],dv[None])
        p,i,c=reference_pool(phi,rhs,om,jnp.zeros(w,jnp.int32),valid,old_coefs)
        codes=((ex[None],ey[None],ez[None],dv[None]) if jax.default_backend() == "cpu" else
               (SOR.op_code(m)[None],SOR.op_lut(jnp.float32(0.))[None]))
        st=[]
        a,ai,ac=SOR.run_pool(phi,rhs,om,jnp.zeros(w,jnp.int32),valid,codes,stats=st)
        exact(p,a);exact(np.where(valid,np.asarray(i),0).astype(np.int32),ai)
        exact(np.where(valid,np.asarray(c),0).astype(np.float32),ac)
        assert len(set(ai[valid])) >= 3
        assert st[0]['lane_blocks'] < st[0]['blocks']*w
        z,zi,zc=SOR.run_pool(phi,rhs,om,jnp.zeros(w,jnp.int32),np.zeros(w,bool),codes)
        assert not np.asarray(z).any() and not zi.any() and not zc.any()

