"""Reuse this task's allocation, with the baseline stopped during isolated arms.
Only the exact OPT4 baseline process is controlled. Frozen sources are untouched.
"""
import argparse,datetime,json,os,signal,subprocess,threading,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--target-root',required=True);p.add_argument('--target-tag',required=True);a=p.parse_args()
ROOT=Path.cwd();O=ROOT.parent/'opt4';target=Path(a.target_root);me=os.getpid();matches=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  argv=(d/'cmdline').read_bytes().split(b'\0')
  if b'jobs/opt4_gate.py' in argv and a.target_tag.encode() in argv and (d/'cwd').resolve()==target:
   env=(d/'environ').read_bytes().split(b'\0');assert ('SLURM_JOB_ID='+os.environ['SLURM_JOB_ID']).encode() in env;matches.append(int(d.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(matches)==1,matches
original=matches[0]; status=O/'aux-control.json'; result=dict(job=os.environ['SLURM_JOB_ID'],baseline_pid=original,baseline_root=str(target),queue=a.queue,actions=[],memory={})
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rss(pid):
 try:
  for line in Path(f'/proc/{pid}/status').read_text().splitlines():
   if line.startswith('VmRSS:'):return int(line.split()[1])
 except (FileNotFoundError,ProcessLookupError):pass
 return 0

def descendants(pid):
 ids=[pid]
 try:
  for c in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():ids+=descendants(int(c))
 except (FileNotFoundError,ProcessLookupError):pass
 return ids

def save():
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,status)
def action(sig,label):
 os.kill(original,sig);result['actions'].append(dict(action=label,utc=now()));save()
action(signal.SIGSTOP,'STOP baseline for isolated measurements')
for _ in range(100):
 if '\nState:\tT' in '\n'+Path(f'/proc/{original}/status').read_text():break
 time.sleep(.05)
else:action(signal.SIGCONT,'CONT after failed stop verification');raise RuntimeError('baseline did not stop')
print('STOPPED baseline',original,'in job',result['job'],flush=True)
stop=threading.Event(); pipeline=None

def sample():
 os.sched_setaffinity(0,[max(os.sched_getaffinity(0))]);lastsave=0
 while not stop.wait(.2):
  try:
   doc=json.loads((O/(a.queue+'.status.json')).read_text());key=doc.get('current',{}).get('id','idle')
  except (FileNotFoundError,json.JSONDecodeError):key='starting'
  ids=descendants(pipeline.pid) if pipeline else []
  arm=sum(rss(pid) for pid in ids);base=rss(original);row=result['memory'].setdefault(key,dict(peak_arm_rss_kb=0,peak_aggregate_rss_kb=0,peak_baseline_rss_kb=0))
  row['peak_arm_rss_kb']=max(row['peak_arm_rss_kb'],arm);row['peak_baseline_rss_kb']=max(row['peak_baseline_rss_kb'],base);row['peak_aggregate_rss_kb']=max(row['peak_aggregate_rss_kb'],arm+base)
  if time.monotonic()-lastsave>30:save();lastsave=time.monotonic()
try:
 pipeline=subprocess.Popen(['bash',str(ROOT/'jobs/opt4_cpu.sbatch'),str(ROOT),'jobs/opt4_pipeline.py','--queue',a.queue],cwd=ROOT)
 thread=threading.Thread(target=sample,daemon=True);thread.start();rc=pipeline.wait();stop.set();thread.join();result['returncode']=rc
 q=json.loads((O/a.queue).read_text())
 # An explicit final stop leaves the baseline held for the caller's Slurm cleanup.
 if not (rc==0 and q.get('stop')):action(signal.SIGCONT,'CONT baseline after coordinator exit')
 result['completed']=True;save()
finally:
 stop.set()
 if not result.get('completed'):
  try:action(signal.SIGCONT,'CONT baseline after coordinator error')
  except ProcessLookupError:pass
