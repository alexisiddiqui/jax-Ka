"""Production JAX 1024 worker, v2: native compact cache, host arrays released.

Identical readout to production_worker.py (v1). Only execution changes:
the structural cache is identity-restricted to the native sequence and stored
compactly, cache arrays are jit arguments, and host copies are released after
device transfer. Equations, 1024-step solver, grid and validity rules are
unchanged; equivalence to the accepted v1 smoke is checked before release.
"""
import gc
import os
import resource
import sys
import time
from pathlib import Path
from dataclasses import asdict
from .runtime import require_compute, atomic_json, digest, config_hash

VERSION = 'production-1024-v2'


def settings(config):
    return {'model':'jaxka','grid':[-2,16,.25],'gap_policy':'cap','solver':asdict(config),
            'readout':'existing _grid_pka_result applied to shared curves',
            'cache':'native identity-restricted compact','backend':'packed'}


def run(campaign,cid,state,out):
    steps=1024
    require_compute()
    import numpy as np
    from jaxpropka import TitrationModel
    from jaxpropka.parameters import ModelConfig
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache, native_identities
    from jaxpropka.model import _grid_pka_result
    from .prep import read_cif
    from .schema import PH,GROUPS,KEY,read_table,write_table
    campaign=Path(campaign); out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    config=ModelConfig(steps=steps); started=time.monotonic()
    cif=campaign/'structures'/cid/f'{state}.cif'
    topology=load_topology(read_cif(cif),gap_policy='cap',freeze_disulfides=True)
    candidates=build_candidates(topology,missing_sidechain='error')
    cache=build_cache(topology,candidates,identities=native_identities(topology))
    fingerprint=cache.fingerprint(); cache_seconds=time.monotonic()-started
    model=TitrationModel(cache,config=config,backend='packed')
    model.release_host_arrays(); del cache,candidates,topology; gc.collect()
    cache=model.cache  # keys, frozen and masks only; large tensors are released
    curves=model.curves(PH)(model.native_probabilities)
    mid=_grid_pka_result(model.arrays,curves,curves.ph,config)
    y=np.asarray(curves.protonated); active=np.asarray(curves.probability)>0
    valid=np.asarray(mid.valid); bracket=np.asarray(mid.bracketed); vals=np.asarray(mid.value)
    residual=np.asarray(curves.residual); converged=bool(np.asarray(curves.converged).all())
    monotone=np.asarray(mid.sampled_monotone); slopes=np.asarray(mid.slope)
    lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(cache.keys)}
    config_settings=settings(config)
    rows=[]
    for s in read_table(campaign/'structures'/cid/'sites.parquet'):
        if state!='AB' and s['partner']!=state: continue
        r={k:s[k] for k in KEY}; r.update(state=state,method='jaxka',method_version=VERSION,
            config_sha256=config_hash(config_settings),status='not_reported',pka=None,curve=None,curve_source=None,intrinsic_pka=None)
        i=lookup.get((s['chain'],s['resnum'],s['icode'])); g=GROUPS.index(s['group'])
        if i is not None:
            if not active[i,g]:
                if s['group']=='CYS' and bool(cache.frozen[i]):
                    r.update(status='not_titrating',curve=[0.]*73,curve_source='native')
            else:
                status='ok' if valid[i,g] else 'out_of_range' if not bracket[i,g] else 'failed'
                if not converged: status='failed'
                r.update(status=status,pka=float(vals[i,g]) if valid[i,g] else None,
                    curve=y[:,i,g].tolist() if status!='failed' else None,curve_source='native',
                    intrinsic_pka=float(curves.intrinsic_pka[i,g]))
        rows.append(r)
    write_table(out/'predictions.parquet','predictions',rows)
    np.savez_compressed(out/'curves.npz',protonated=y,active=active,residual=residual,midpoint=vals,valid=valid)
    atomic_json(out/'receipt.json',{'state':state,'steps':steps,'version':VERSION,'config':config_settings,
        'input_sha256':digest(cif),'cache_fingerprint':fingerprint,
        'predictions_sha256':digest(out/'predictions.parquet'),'job':os.environ['SLURM_JOB_ID'],
        'max_residual':float(residual.max()),'grid_converged':converged,'wall_seconds':time.monotonic()-started,
        'cache_seconds':cache_seconds,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'nonmonotone_bracketed':int((active&bracket&~monotone).sum()),
        'slope_failed_bracketed':int((active&bracket&(slopes>=-config.slope_min)).sum())})


if __name__=='__main__':
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4]))
