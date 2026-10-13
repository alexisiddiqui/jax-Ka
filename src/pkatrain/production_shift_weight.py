"""Training-only inverse-square-root binding-shift weights (2026-10-13)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from pkabench.runtime import atomic_json, digest
from .production_graphs import ProductionStore

CUTS = (0.1, 0.25, 0.5, 1., 2., 4.)

def fit(manifest, fraction, destination, cap=5.):
    """Mean-one under equal-complex reduction; normalization never uses validation."""
    from .production_loading import select
    records = select(manifest, 'train', fraction)
    identity = {'version':1, 'fraction':fraction, 'cuts':list(CUTS), 'cap':cap,
                'store_sha256':manifest['store_verification_sha256'], 'pool_sha256':manifest['pool_sha256'],
                'definition':'abs(pKa_AB - pKa_partner_removed)', 'normalization':'equal complex, all eligible sites'}
    destination=Path(destination)
    if destination.exists():
        import json
        saved=json.loads(destination.read_text())
        assert saved['identity']==identity, 'weight-table provenance changed'
        return saved
    store=ProductionStore(manifest['store'])
    def count(r):
        a=store.raw(r['id']); d=np.abs(a['targets'][0]-a['targets'][1])
        assert len(d) and np.isfinite(d).all()
        c=np.bincount(np.searchsorted(np.asarray(CUTS,d.dtype),d,side='right'),minlength=7)
        return c,c/len(d)
    counts=np.zeros(7,np.int64); mass=np.zeros(7)
    with ThreadPoolExecutor(16) as pool:
        for c,m in pool.map(count,records): counts+=c; mass+=m
    store.close()
    assert np.all(counts>0), 'merge empty bins before fitting weights'
    raw=1/np.sqrt(counts); lo,hi=0.,1e9
    for _ in range(100):
        mid=(lo+hi)/2
        if np.dot(mass,np.minimum(cap,mid*raw))/mass.sum()<1:lo=mid
        else:hi=mid
    weights=np.minimum(cap,hi*raw)
    assert abs(np.dot(mass,weights)/mass.sum()-1)<1e-10
    result={'identity':identity,'complexes':len(records),'counts':counts.tolist(),
            'complex_mass':(mass/mass.sum()).tolist(),'weights':weights.tolist(),
            'source_sha256':digest(__file__),'test_data_included':False}
    atomic_json(destination,result)
    return result

def site_weights(targets, table):
    import jax.numpy as jnp
    # Match the float32 training targets and census at bin boundaries.
    delta=jnp.abs(targets[:,0]-targets[:,1])
    index=jnp.searchsorted(jnp.asarray(CUTS,delta.dtype),delta,side='right')
    return jnp.asarray(table['weights'],delta.dtype)[index]

def validation_bins(rows):
    """Unweighted diagnostic metrics; never used to construct training weights."""
    d=np.array([r['teacher_ab']-r['teacher_free'] for r in rows]);p=np.array([r['predicted_ab']-r['predicted_free'] for r in rows]);ii=np.array([r['interface'] for r in rows],bool)
    result=[];edges=[0,*CUTS,float('inf')]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=ii&(np.abs(d)>=lo)&(np.abs(d)<hi)
        if not m.any():continue
        e=p[m]-d[m]
        result.append({'lower':lo,'upper':hi if np.isfinite(hi) else None,'sites':int(m.sum()),
                       'complexes':len({r['complex_id'] for r,keep in zip(rows,m) if keep}),
                       'mae':float(np.mean(abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),
                       'signed_bias':float(np.mean(e)),'magnitude_bias':float(np.mean(abs(p[m])-abs(d[m])))})
    return result
