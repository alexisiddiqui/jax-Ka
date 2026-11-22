"""Certify identical map StableHLO across frozen versions; run on an allocated node."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np
from jaxpb import records as RC

p=argparse.ArgumentParser(); p.add_argument('--tags', nargs=2, required=True)
p.add_argument('--platform', choices=['cpu','gpu'], required=True); p.add_argument('--out', required=True)
a=p.parse_args()
jax.config.update('jax_enable_x64', False)  # same precision context as pb.solve_state
assert os.environ.get('SLURM_JOB_ID') and jax.default_backend()==a.platform
T=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline')
roots=[T/f'opt1-{tag}-20261121/src/jaxpb' for tag in a.tags]
mods=[]
for root in roots:
    spec=importlib.util.spec_from_file_location('jaxpb.maps', root/'maps.py')
    m=importlib.util.module_from_spec(spec); sys.modules['jaxpb.maps']=m; spec.loader.exec_module(m); mods.append(m)
rows=[]
for c in range(len(RC.NA_C)):
    for xl in (False,True):
        hashes=[]
        for m in mods:
            sz=m.Sizes.xl() if xl else m.Sizes()
            arr=lambda *shape: jax.ShapeDtypeStruct(shape,np.float32)
            args=(arr(RC.NA_C[c],3),arr(RC.NA_C[c]),arr(RC.NP_C[c],3),arr(16,3),
                  arr(16,RC.MA,3),arr(16,RC.MA),arr(16,sz.PBs,3))
            ir=m.maps_batch.lower(*args,sz=sz).compiler_ir('stablehlo')
            txt=ir.operation.get_asm(enable_debug_info=False)
            hashes.append(hashlib.sha256(txt.encode()).hexdigest())
        row=dict(size_class=c,xl=xl,hashes=hashes,identical=hashes[0]==hashes[1]); rows.append(row);print(row,flush=True)
rest={f.name: f.read_bytes()==(roots[1]/f.name).read_bytes() for f in roots[0].glob('*.py') if f.name!='maps.py'}
# B and C share the map source; certify the corresponding pre-setup-change source pair too.
extra={}
if a.tags==['C8','C9']:
    broots=[T/f'opt1-{tag}-20261121/src/jaxpb' for tag in ('B8','B9')]
    extra={f.name:f.read_bytes()==(broots[1]/f.name).read_bytes() for f in broots[0].glob('*.py') if f.name!='maps.py'}
    extra['shared_old_map']=(broots[0]/'maps.py').read_bytes()==(roots[0]/'maps.py').read_bytes()
    extra['shared_new_map']=(broots[1]/'maps.py').read_bytes()==(roots[1]/'maps.py').read_bytes()
res=dict(passed=all(r['identical'] for r in rows) and all(rest.values()) and all(extra.values()),
         platform=a.platform,jax_enable_x64=bool(jax.config.jax_enable_x64),tags=a.tags,variants=rows,other_solver_sources_identical=rest,
         B_variant_sources_identical=extra,job=os.environ['SLURM_JOB_ID'])
json.dump(res,open(T/'opt1'/a.out,'w'),indent=2)
assert res['passed']
