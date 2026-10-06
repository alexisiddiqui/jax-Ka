"""Ready-only GPU baseline/training launch after preflight and profile gates."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from pkabench.runtime import require_compute,atomic_json
from .records import read
from .dispatch import queue

def run(out):
    require_compute();m=read(out/'manifest.json');assert read(out/'release.json')['passed']
    root=Path(os.environ['PKABENCH_RUNTIME']);source=Path(os.environ['PKABENCH_SOURCE'])
    dest=out/'gpu-dispatch';dest.mkdir(exist_ok=True)
    lock=(dest/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    submit=source.parent/'_HPC/submission/jax-Ka/pkabench/submit-training-gpu.sh'
    journal=dest/'jobs.json';jobs=read(journal) if journal.exists() else []
    began=time.monotonic();missing={};last=None
    def folder(action,arg):return out/('baseline' if action=='baseline' else f'seed-{arg}')
    while True:
        if (out/'TRAINING_HOLD.json').exists():
            atomic_json(dest/'status.json',{'complete':False,'held':True,'reason':read(out/'TRAINING_HOLD.json')})
            return
        live=queue();active=set()
        for job in jobs:
            if job.get('retired'):continue
            task=(job['action'],job['argument']);target=folder(*task)
            if (target/'verification.json').exists():
                assert read(target/'verification.json')['passed'];continue
            active.add(task)
            if job['id'] in live:missing.pop(job['id'],None);continue
            missing.setdefault(job['id'],time.monotonic())
            if time.monotonic()-missing[job['id']]<120:continue
            if task[0]=='train' and (target/'resume_required.json').exists():
                job['retired']=True;active.remove(task);atomic_json(journal,jobs)
            else:raise RuntimeError(f'GPU job {job} ended without a valid completion artifact')
        if (out/'dispatch/error.json').exists():raise RuntimeError(read(out/'dispatch/error.json'))
        wanted=[]
        if not (out/'baseline/verification.json').exists():wanted.append(('baseline','0'))
        profiles=out/'profiles/verification.json'
        if profiles.exists():
            assert read(profiles)['passed']
            wanted.extend(('train',str(s)) for s in m['config']['seeds'] if not (out/f'seed-{s}/verification.json').exists())
        for action,arg in wanted:
            if (action,arg) in active or len(active)>=3:continue
            argv=['bash',str(submit),action,str(out)] + ([arg] if action=='train' else [])
            result=subprocess.run(argv,text=True,capture_output=True)
            if result.returncode==3:break
            result.check_returncode();jid=result.stdout.strip().splitlines()[-1]
            jobs.append(dict(id=jid,action=action,argument=arg));atomic_json(journal,jobs);active.add((action,arg))
        if profiles.exists() and not wanted:
            subprocess.run([str(root/'envs/train-shared-v1/bin/python'),'-m','pkatrain.run','report',str(out)],check=True)
            atomic_json(dest/'status.json',{'complete':True});return
        status=dict(complete=False,profiles_ready=profiles.exists(),active_tasks=[list(t) for t in sorted(active)],jobs=jobs)
        atomic_json(dest/'status.json',status)
        if status!=last:print(json.dumps(status),flush=True);last=status
        if time.monotonic()-began>36000:
            renew=submit.with_name('submit-shared-training.sh')
            result=subprocess.run(['bash',str(renew),'gpu-dispatch',str(out)],env=dict(os.environ,PKATRAIN_AFTEROK=os.environ['SLURM_JOB_ID']),text=True,capture_output=True)
            if result.returncode==0:return
            if result.returncode!=3:result.check_returncode()
        time.sleep(30)

if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    try:run(out)
    except Exception as error:
        import traceback
        atomic_json(out/'gpu-dispatch/error.json',{'error':repr(error),'traceback':traceback.format_exc()});raise
