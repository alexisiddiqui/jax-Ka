"""Run frozen OPT4 arms sequentially in one strictly bounded amd96 allocation.

Only the output command queue changes; source snapshots remain immutable. Each
arm gets a fresh Python process and its own cwd, environment, log and evidence.
"""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
T=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'); O=T/'opt4'
p=argparse.ArgumentParser();p.add_argument('--queue',required=True);a=p.parse_args();queue=O/a.queue
status=O/(a.queue+'.status.json'); result=dict(job=os.environ['SLURM_JOB_ID'],cpus=int(os.environ['SLURM_CPUS_PER_TASK']),runs=[]); assert result['cpus']<=12
executed=set();last=time.monotonic()
def save():
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,status)
while True:
 q=json.loads(queue.read_text()) if queue.exists() else {'commands':[]}
 pending=[c for c in q['commands'] if c['id'] not in executed]
 if not pending:
  result.pop('current',None);save()
  if q.get('stop') or time.monotonic()-last>3600:break
  time.sleep(30);continue
 c=pending[0];root=Path(c['root']);assert root.parent==T and root.name.startswith('opt4-');args=c['args'];assert args[0].startswith('jobs/opt4') and args[0].endswith('.py')
 env=dict(os.environ);env['PYTHONPATH']=str(root/'src')+':'+str(root/'jobs');env['OMP_NUM_THREADS']=str(result['cpus']);env['PYTHONDONTWRITEBYTECODE']='1'
 log=O/(c['id']+'-'+result['job']+'.out');row=dict(id=c['id'],frozen=str(root),args=args,log=str(log),started=datetime.datetime.now(datetime.timezone.utc).isoformat());result['current']=row;save()
 with log.open('w') as f: rc=subprocess.call([os.environ.get('OPT4_PYTHON', '/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python')]+args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 row.update(returncode=rc,ended=datetime.datetime.now(datetime.timezone.utc).isoformat());result['runs'].append(row);executed.add(c['id']);result.pop('current',None);save();last=time.monotonic()
result['completed']=True;save()
