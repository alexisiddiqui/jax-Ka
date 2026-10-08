"""Ready-only Slurm orchestration with a user-wide 400-core admission cap."""
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pkabench.runtime import require_compute,atomic_json,digest
from .records import read


def command(args):return subprocess.check_output(args,text=True).strip()

def queue():
    rows={}
    for line in command(['squeue','-r','-h','-u',getpass.getuser(),'-o','%i|%T|%C']).splitlines():
        jid,state,cpu=line.split('|');rows[jid]={'state':state,'cores':int(cpu)}
    return rows


def artifact(out,task):
    action,cid=task
    return {'prepare':out/'prepared'/cid/'receipt.json','gate':out/'gates'/f'{cid}.json',
        'union-gradient':out/'union_gradients'/f'{cid}.json',
        'profile':out/'profiles/complexes'/f'{cid}.json','smoke':out/'smoke/verification.json',
        'train':out/f'seed-{cid}/verification.json','baseline':out/'baseline/verification.json'}[action]


def dispatch(out,prepare_only=False,profiles_only=False):
    require_compute();m=read(out/'manifest.json'); root=Path(os.environ['PKABENCH_RUNTIME']); source=Path(os.environ['PKABENCH_SOURCE'])
    dest=out/('dispatch-prepare' if prepare_only else 'dispatch-profiles' if profiles_only else 'dispatch');dest.mkdir(exist_ok=True)
    lock=(dest/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=source.parent/'_HPC/submission/jax-Ka/pkabench/shared-training.sbatch'; py=root/'envs/train-shared-v1/bin/python'
    batches=[read(p) for p in sorted((dest/'batches').glob('*.json'))] if (dest/'batches').exists() else []
    missing={}; last=None; began=time.monotonic()
    while True:
        live=queue();inflight=set()
        # Renew the lightweight controller before its own walltime expires.
        # Existing workers continue; the successor reloads their journal.
        if time.monotonic()-began>36000 and sum(r['cores'] for r in live.values())<=392:
            env=dict(os.environ,PKATRAIN_AFTEROK=os.environ['SLURM_JOB_ID'])
            submit=script.with_name('submit-shared-training.sh')
            action='prepare-all' if prepare_only else 'profile-all' if profiles_only else 'dispatch'
            successor=subprocess.check_output(['bash',str(submit),action,str(out)],env=env,text=True).strip()
            atomic_json(dest/'handoff.json',{'successor_job_id':successor,'previous_job_id':os.environ['SLURM_JOB_ID']})
            return
        for batch in batches:
            for i,task in enumerate(batch['tasks']):
                if i in batch.get('resumed_indices',[]):continue
                if artifact(out,task).exists():continue
                jid=f"{batch['job_id']}_{i}"
                if jid in live:inflight.add(tuple(task));missing.pop(jid,None)
                else:
                    missing.setdefault(jid,time.monotonic());inflight.add(tuple(task))
                    if time.monotonic()-missing[jid]>120:
                        action,cid=task
                        run_folder=out/('smoke' if action=='smoke' else f'seed-{cid}')
                        if action in ('train','smoke') and (run_folder/'resume_required.json').exists():
                            # Mark this departed allocation superseded in the journal,
                            # and allow the exact checkpointed task to be resubmitted.
                            batch.setdefault('resumed_indices',[]).append(i)
                            atomic_json(dest/'batches'/f"{batch['tag']}.json",batch)
                            inflight.discard(tuple(task));continue
                        state=command(['sacct','-X','-n','-j',jid,'--format=State,ExitCode'])
                        raise RuntimeError(f'Task failed without completion artifact: {task} job={jid} {state}')
        if prepare_only:
            wanted=[('prepare',cid) for cid in m['train']+m['val'] if not artifact(out,('prepare',cid)).exists()]
            if not wanted:
                atomic_json(dest/'status.json',{'complete':True,'prepared':len(m['train'])+len(m['val'])});return
            phase='preparation only'
        elif profiles_only:
            wanted=[('profile',cid) for cid in m['train'] if not artifact(out,('profile',cid)).exists()]
            if not wanted:
                subprocess.run([str(py),'-m','pkatrain.run','profiles',str(out)],check=True)
                atomic_json(dest/'status.json',{'complete':True,'profiles':len(m['train'])});return
            phase='loss profiles only'
        elif not (out/'real_gate.json').exists():
            wanted=[]
            for cid in m['smoke']:
                if not artifact(out,('prepare',cid)).exists():wanted.append(('prepare',cid))
                elif not artifact(out,('gate',cid)).exists():wanted.append(('gate',cid))
                elif not artifact(out,('union-gradient',cid)).exists():wanted.append(('union-gradient',cid))
            if not wanted:
                subprocess.run([str(py),'-m','pkatrain.run','gates',str(out)],check=True);continue
            phase='numerical gates'
        else:
            assert read(out/'real_gate.json')['passed']
            wanted=[]
            if not (out/'smoke/verification.json').exists():wanted.append(('smoke','17'))
            for cid in m['train']+m['val']:
                if not artifact(out,('prepare',cid)).exists():wanted.append(('prepare',cid))
                elif cid in m['train'] and not artifact(out,('profile',cid)).exists():wanted.append(('profile',cid))
            if all(artifact(out,('prepare',cid)).exists() for cid in m['val']) and not (out/'baseline/verification.json').exists():wanted.append(('baseline','0'))
            if all(artifact(out,('profile',cid)).exists() for cid in m['train']) and not (out/'profiles/verification.json').exists():
                subprocess.run([str(py),'-m','pkatrain.run','profiles',str(out)],check=True)
            if (out/'smoke/verification.json').exists() and (out/'profiles/verification.json').exists():
                for seed in m['config']['seeds']:
                    if not (out/f'seed-{seed}/verification.json').exists():wanted.append(('train',str(seed)))
            phase='preparation/profiles' if not (out/'profiles/verification.json').exists() else 'training'
            if not wanted:
                subprocess.run([str(py),'-m','pkatrain.run','report',str(out)],check=True)
                atomic_json(dest/'status.json',{'complete':True,'report':str(out/'report/report.md')});return
        ready=[t for t in wanted if tuple(t) not in inflight]
        if ready:
            with (root/'submission.lock').open('w') as admission:
                fcntl.flock(admission,fcntl.LOCK_EX)
                current=queue();used=sum(r['cores'] for r in current.values()); slots=max(0,(400-used)//8)
                selected=ready[:slots]
                if selected:
                    tag=str(time.time_ns()); path=dest/'batches'/f'{tag}.txt';path.parent.mkdir(exist_ok=True)
                    path.write_text(''.join(' '.join(t)+'\n' for t in selected))
                    excluded={'comp1400'}
                    for line in command(['sinfo','-N','-h','-p','generalaccess,amd96','-o','%N %t']).splitlines():
                        node,state=line.split()
                        if state not in ('idle','mix','alloc'):excluded.add(node)
                    job=command(['sbatch','--parsable','--exclude='+','.join(sorted(excluded)),f'--array=0-{len(selected)-1}',
                        '--output='+str(root/'logs/pkatrain-unit-%A_%a.out'),str(script),'batch',str(out),str(path)]).split(';')[0]
                    batch={'job_id':job,'tasks':selected,'tag':tag,'cores_before':used,'cores_after':used+8*len(selected)}
                    atomic_json(path.with_suffix('.json'),batch);batches.append(batch)
                    with (root/'receipts/submissions.tsv').open('a') as f:f.write(f'{job}\tpkatrain-batch\t{len(selected)} tasks\n')
                    print(json.dumps(batch),flush=True)
        status={'complete':False,'phase':phase,'pending_tasks':len(wanted),'inflight':len(inflight),'ready':len(ready),
            'gates_completed':sum((out/'gates'/f'{cid}.json').exists() for cid in m['smoke']),
            'prepared':sum((out/'prepared'/cid/'receipt.json').exists() for cid in m['train']+m['val'])}
        atomic_json(dest/'status.json',status)
        if status!=last:print(json.dumps(status),flush=True);last=status
        time.sleep(30)

if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    try:dispatch(out,prepare_only='--prepare-only' in sys.argv[2:],profiles_only='--profiles-only' in sys.argv[2:])
    except Exception as e:
        import traceback
        atomic_json(out/'dispatch/error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
