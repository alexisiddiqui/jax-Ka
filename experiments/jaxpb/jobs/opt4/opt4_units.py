import os,sys,runpy,json,time,argparse
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import bench_jaxpb as BJ
p=argparse.ArgumentParser();p.add_argument("--tag",default="units-"+os.environ["SLURM_JOB_ID"]);a=p.parse_args()
ns=runpy.run_path('tests/test_opt4.py'); result=dict(job=os.environ['SLURM_JOB_ID'],tests={})
for name,fn in ns.items():
 if name.startswith('test_'):
  t=time.perf_counter(); fn(); result['tests'][name]=dict(passed=True,seconds=time.perf_counter()-t); print(name,'PASS',flush=True)
result['completed']=True; result['passed']=True; result['host_hwm_kb']=BJ.status('VmHWM'); BJ.jdump(result,BJ.TP+'/opt4/'+a.tag+'.json')
