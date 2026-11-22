"""Memory-analysis measurements on a separate Slurm process; never in compile gates."""
import os,sys
assert os.environ.get('SLURM_JOB_ID')
import jax
from jaxpb import api as API,sor as SOR
import bench_jaxpb as BJ
import prof_geomaps as PG
import prof2
cpu=jax.default_backend()=='cpu'
ct=BJ.CompileTime();inst=prof2.Inst(ct,cpu)
inst.targets.append((SOR,'device_solve','device_solve',False))
inst.install(memana_on=True)
backend='xla' if cpu else sys.argv[1]
ids=['4066edfcdb3ea8e3_AB']
rows=[]
for r in PG.load(BJ.INPUTS,ids):
 inst.begin(r);API.energies([r],API.Config(backend=backend));rows.append(inst.end())
BJ.jdump(dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),backend=backend,cpu=cpu,
              programs=list(inst.mem.values()),states=ids,measurements=rows),BJ.TP+'/opt2/memory-'+('cpu' if cpu else backend)+'.json')
