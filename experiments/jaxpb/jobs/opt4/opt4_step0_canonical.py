"""Add exact CPU-1 fingerprints to the unmodified physics used by step-0.
The profiling script's outputs remain diagnostic; only production outputs enter
this baseline and its PB label shards.
"""
import os,sys,json,hashlib
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
if '--pin' in sys.argv:
 pin=int(sys.argv[sys.argv.index('--pin')+1]);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:pin]);os.environ['OMP_NUM_THREADS']=str(pin)
T=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline');base=T/'opt4-control1';sys.path[:0]=[str(base/'src'),str(base/'jobs')]
import jax,numpy as np
import bench_jaxpb as BJ
from opt2 import digest,FIELDS
from jaxpb import api as A,maps as M,geometry as G,records as R
O=T/'opt4';tag=sys.argv[sys.argv.index('--tag')+1];caps={};rows={};program_globals={'__name__':'__main__','__file__':str(T/'opt4-step0v3/jobs/opt4_step0.py')}
def capture(mod,name,key,select=lambda out:out):
 f=getattr(mod,name)
 def w(*args,**kw):
  out=f(*args,**kw);caps.setdefault(key,[]).append(jax.tree_util.tree_map(digest,select(out)));return out
 setattr(mod,name,w)
capture(M,'maps_batch','maps');capture(G,'geometry_state','geometry',lambda out:(out.probes[:R.NP_C[out.c]],out.flags));capture(G,'geometry_sites','geometry_sites')
solve=A.solve_pb
def solve_capture(*args,**kw):
 out=solve(*args,**kw);caps['pb']={k:digest(getattr(out[0],k)) for k in FIELDS};return out
A.solve_pb=solve_capture;energies=A.energies
result=dict(tag='canonical-'+tag,job=os.environ['SLURM_JOB_ID'],source_base=str(base),passes=[dict(rows=rows)],reference_kind='unchanged bdffc91 physics; CPU-1 outputs during profiling')
def energies_capture(*args,**kw):
 caps.clear();out=energies(*args,**kw);rec=args[0][0];e=out[0];caps['phi']=[entry['hashes'] for entry in program_globals['current']['sor']];caps['energy']={k:digest(getattr(e,k)) for k in ('g','W','occ','start','nst','pairs')}
 path=O/('canonical-'+tag+'-labels.npz');fields=A._state_arrays(rec.n_sites,e,None);A.write_shard(str(path),[dict(state_id=rec.state_id,alias=rec.state_id,input='OPT4',status='ok',arrays=fields,diag={})],dict(task='OPT4'),{})
 with np.load(path) as z:caps['labels']={k:digest(z[k]) for k in z.files if k not in ('provenance_json','timing_json')}
 rows[rec.state_id]=dict(hashes=json.loads(json.dumps(caps)));BJ.jdump(result,str(O/('canonical-'+tag+'.json')));return out
A.energies=energies_capture
script=T/'opt4-step0v3/jobs/opt4_step0.py';exec(compile(script.read_text(),str(script),'exec'),program_globals)
result['completed']=True;result['passed']=True;BJ.jdump(result,str(O/('canonical-'+tag+'.json')))
