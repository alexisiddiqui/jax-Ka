"""Quantify decomposition rounding at 12 CPUs on one complete real state."""
import os,runpy,sys
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
T=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline');base=T/'opt4-control1'
sys.path[:0]=[str(base/'src'),str(base/'jobs')]
import opt2
opt2.CPU_IDS=['4066edfcdb3ea8e3_AB']
runpy.run_path(str(T/'opt4-step0v3/jobs/opt4_step0.py'),run_name='__main__')
