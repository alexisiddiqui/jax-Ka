"""Benchmark real SOR inputs at fixed CPU lane tiles, inside one allocation."""
import json,os,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np
import jax
from jaxpb import sor as SOR,records as RC,api as API
import bench_jaxpb as BJ
import prof_geomaps as PG
from opt2 import digest
orig=SOR.run_pool
rows=[]
class Finished(Exception):pass

def tune(*args,**kwargs):
 global rows
 refs=None
 for widths in [(16,64,128,256),(4,8),(4,8,16),(8,16,32),(16,32,64)]:
  RC.CPU_LANE_WIDTHS=widths
  times=[];stats=[]
  for rep in range(2):
   st=[];t=time.perf_counter();out=orig(*args[:6],stats=st);jax.block_until_ready(out);dt=time.perf_counter()-t
   times.append(dt);stats.append(st)
  hs=[digest(a) for a in out]
  if refs is None:refs=hs
  row=dict(widths=widths,seconds=times,stats=stats,bitwise=hs==refs,host_hwm_kb=BJ.status('VmHWM'))
  rows.append(row);print(row,flush=True)
  BJ.jdump(dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),rows=rows),BJ.TP+'/opt2/tune-cpu.json')
 raise Finished
SOR.run_pool=tune
r=PG.load(BJ.INPUTS,['4066edfcdb3ea8e3_AB'])[0]
try:API.energies([r],API.Config(backend='xla'))
except Finished:pass
