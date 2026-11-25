"""Resume our baseline and stop at a durable row checkpoint, before its last state."""
import argparse,ctypes,json,os,select,signal,time,datetime
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
p=argparse.ArgumentParser();p.add_argument('--rows',type=int,default=19);a=p.parse_args()
O=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/opt4');control=json.loads((O/'aux-control.json').read_text());assert control['job']==os.environ['SLURM_JOB_ID'];pid=control['baseline_pid'];path=O/'base12.json'
lib=ctypes.CDLL(None);fd=lib.inotify_init1(os.O_NONBLOCK);assert fd>=0;wd=lib.inotify_add_watch(fd,os.fsencode(O),0x00000008|0x00000080);assert wd>=0
started=datetime.datetime.now(datetime.timezone.utc).isoformat();os.kill(pid,signal.SIGCONT);last=None
while True:
 st=path.stat().st_mtime_ns
 if st!=last:
  last=st;doc=json.loads(path.read_text());rows=doc.get('current',{}).get('rows',{})
  if len(rows)>=a.rows:
   os.kill(pid,signal.SIGSTOP)
   out=dict(resumed_utc=started,job=os.environ['SLURM_JOB_ID'],pid=pid,rows=list(rows),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),action='STOP after durable checkpoint')
   (O/'hold-checkpoint.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True);break
 if select.select([fd],[],[],2)[0]:
  try:os.read(fd,65536)
  except BlockingIOError:pass
os.close(fd)
