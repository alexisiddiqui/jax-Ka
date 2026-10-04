"""Production JAX 1024 worker copied from the verified isolated readout."""
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict
from .runtime import require_compute, atomic_json, digest, config_hash


def run(campaign,cid,state,out):
    steps=1024
    require_compute()
    import numpy as np
    from jaxpropka import prepare,TitrationModel
    from jaxpropka.parameters import ModelConfig
    from jaxpropka.model import _grid_pka_result
    from .prep import read_cif
    from .schema import PH,GROUPS,KEY,read_table,write_table
    campaign=Path(campaign); out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    config=ModelConfig(steps=steps); started=time.monotonic()
    cif=campaign/'structures'/cid/f'{state}.cif'
    cache=prepare(read_cif(cif),topology_options={'gap_policy':'cap','freeze_disulfides':True},geometry_options={'missing_sidechain':'error'})
    model=TitrationModel(cache,config=config,backend='packed')
    curves=model.curves(PH)(model.native_probabilities)
    # Apply the original readout to the already-computed grid. Avoid compiling
    # a second full-structure solve for midpoint extraction.
    mid=_grid_pka_result(model._d,curves,curves.ph,config)
    y=np.asarray(curves.protonated); active=np.asarray(curves.probability)>0
    valid=np.asarray(mid.valid); bracket=np.asarray(mid.bracketed); vals=np.asarray(mid.value)
    residual=np.asarray(curves.residual); converged=bool(np.asarray(curves.converged).all())
    monotone=np.asarray(mid.sampled_monotone); slopes=np.asarray(mid.slope)
    lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(cache.keys)}
    settings={'model':'jaxka','grid':[-2,16,.25],'gap_policy':'cap','solver':asdict(config),'readout':'existing _grid_pka_result applied to shared curves'}
    rows=[]
    for s in read_table(campaign/'structures'/cid/'sites.parquet'):
        if state!='AB' and s['partner']!=state: continue
        r={k:s[k] for k in KEY}; r.update(state=state,method='jaxka',method_version='production-1024-v1',
            config_sha256=config_hash(settings),status='not_reported',pka=None,curve=None,curve_source=None,intrinsic_pka=None)
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
    atomic_json(out/'receipt.json',{'state':state,'steps':steps,'config':settings,'input_sha256':digest(cif),
        'predictions_sha256':digest(out/'predictions.parquet'),'job':os.environ['SLURM_JOB_ID'],
        'max_residual':float(residual.max()),'grid_converged':converged,'wall_seconds':time.monotonic()-started,
        'nonmonotone_bracketed':int((active&bracket&~monotone).sum()),
        'slope_failed_bracketed':int((active&bracket&(slopes>=-config.slope_min)).sum())})


if __name__=='__main__':
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4]))
