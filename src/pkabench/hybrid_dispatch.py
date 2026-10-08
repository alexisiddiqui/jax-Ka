"""Rolling, user-wide CPU-capped dispatch; no dependency-blocked work arrays."""
import fcntl
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from . import hybrid_mc as science
from . import hybrid_parallel as runner
from .runtime import atomic_json, digest, require_compute


def command(args):
    return subprocess.check_output(args,text=True).strip()


def queue():
    text=command(['squeue','-r','-h','-u',getpass.getuser(),'-o','%i|%T|%C'])
    rows={}
    for line in text.splitlines():
        jid,state,cores=line.split('|'); rows[jid.strip()]={'state':state.strip(),'cores':int(cores)}
    return rows


def taskkey(task): return '/'.join(task)


def ready_tasks(tasks,done,inflight):
    return [t for t in tasks if taskkey(t) not in done and taskkey(t) not in inflight
        and (t[2]=='teacher' or taskkey((t[0],t[1],'teacher')) in done)]


def main(out):
    require_compute(); out=out.resolve(); m=runner.checked(out); root=Path(os.environ['PKABENCH_RUNTIME']); source=Path(os.environ['PKABENCH_SOURCE'])
    dest=out/'parallel-v1'; dest.mkdir(exist_ok=True)
    with (dest/'dispatcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch(out,dest,m,root,source)


def dispatch(out,dest,m,root,source):
    script=source.parent/'_HPC/submission/jax-Ka/pkabench/hybrid-parallel.sbatch'
    pinned=[Path(__file__),Path(runner.__file__),Path(science.__file__),source/'src/pkabench/hybrid_score.py',script,
        source/'tests/test_hybrid_mc.py',source/'tests/test_hybrid_dispatch.py']
    for method in ('teacher','catboost-17'):
        gate=science.read_json(dest/'preflight'/method/'verification.json')
        assert gate['passed'] and gate['exact_rows_match'] and gate['execution_code_sha256']==digest(Path(runner.__file__))
    execution=dest/'execution.json'
    if execution.exists():
        for path,h in science.read_json(execution)['code_sha256'].items(): assert digest(path)==h
    else:
        atomic_json(execution,{'manifest_sha256':digest(out/'manifest.json'),'code_sha256':{str(p):digest(p) for p in pinned},
            'settings':'Same science and per-calculation seed; one process per state/method; two CPUs/4 GB per job.',
            'cpu_cap':400,'excludes':['comp1400'],'dispatcher_job':os.environ['SLURM_JOB_ID']})
    python=root/'envs/finetune-v1/bin/python'
    subprocess.run([str(python),str(source/'tests/test_hybrid_mc.py')],check=True)
    subprocess.run([str(python),str(source/'tests/test_hybrid_dispatch.py')],check=True)
    all_tasks=[(cid,state,method) for cid in m['eligible'] for state in ('AB','A','B') for method in science.METHODS]
    sizes={}
    for cid in m['eligible']:
        request=science.read_json(out/'requests'/f'{cid}.json')
        for state in ('AB','A','B'): sizes[cid,state]=request['states'][state]['sites']
    done=set(); preserved={}
    for t in all_tasks:
        if runner.valid_result(out,*t):
            key=taskkey(t); done.add(key); preserved[key]=digest(out/'complexes'/t[0]/t[1]/t[2]/'result.json')
    if not (dest/'preserved.json').exists(): atomic_json(dest/'preserved.json',preserved)
    batches=[]
    for path in sorted((dest/'batches').glob('*.json')) if (dest/'batches').exists() else []:
        batches.append(science.read_json(path))
    missing_since={}; last_status=None
    while True:
        # Only atomic result.json files mark completion; a directory alone never does.
        for t in all_tasks:
            key=taskkey(t)
            if key not in done and runner.valid_result(out,*t): done.add(key)
        pilot_passed=(out/'pilot/verification.json').exists()
        if pilot_passed:
            gate=science.read_json(out/'pilot/verification.json'); assert gate['passed'] and gate['manifest_sha256']==digest(out/'manifest.json')
        phase='full' if pilot_passed else 'pilot'; cids=m['eligible'] if pilot_passed else m['pilot']
        tasks=[t for t in all_tasks if t[0] in cids]
        if all(taskkey(t) in done for t in tasks):
            # Completed old receipts remain unchanged; new receipts cover exactly 18 results.
            if not (out/phase/'verification.json').exists():
                print(json.dumps({'event':'collect','phase':phase}),flush=True)
                subprocess.run([str(python),'-m','pkabench.hybrid_parallel','collect',str(out),phase],check=True)
            gate=science.read_json(out/phase/'verification.json'); assert gate['passed']
            if phase=='full':
                for key,h in science.read_json(dest/'preserved.json').items():
                    assert digest(out/'complexes'/key/'result.json')==h,'A completed original result changed'
                atomic_json(dest/'status.json',{'complete':True,'phase':'full','completed':len(done),'total':len(all_tasks),
                    'preserved_original_results':len(science.read_json(dest/'preserved.json')),'report':str(out/'full/report.md')})
                return
            continue
        live=queue(); inflight={}
        for batch in batches:
            for i,t in enumerate(batch['tasks']):
                key=taskkey(t); jid=f"{batch['job_id']}_{i}"
                if key in done: continue
                if jid in live:
                    inflight[key]=jid; missing_since.pop(jid,None)
                else:
                    missing_since.setdefault(jid,time.monotonic()); inflight[key]=jid
                    if time.monotonic()-missing_since[jid]>120:
                        accounting=command(['sacct','-X','-n','-j',jid,'--format=JobID,State,ExitCode'])
                        raise RuntimeError(f'Missing result after task left queue: {key}, {jid}, {accounting}')
        ready=ready_tasks(tasks,done,inflight)
        # Unlock missing replay states first, then start the largest ready calculations.
        ready.sort(key=lambda t:(t[2]!='teacher',-sizes[t[0],t[1]],t))
        if ready:
            with (root/'submission.lock').open('w') as admission:
                fcntl.flock(admission,fcntl.LOCK_EX)
                current=queue(); used=sum(r['cores'] for r in current.values()); slots=max(0,(400-used)//2)
                selected=ready[:slots]
                if selected:
                    tag=str(time.time_ns()); batchpath=dest/'batches'/f'{tag}.txt'; batchpath.parent.mkdir(exist_ok=True)
                    batchpath.write_text(''.join(' '.join(t)+'\n' for t in selected))
                    excluded={'comp1400'}
                    for line in command(['sinfo','-N','-h','-p','generalaccess,amd96','-o','%N %t']).splitlines():
                        node,state=line.split()
                        if state not in ('idle','mix','alloc'): excluded.add(node)
                    args=['sbatch','--parsable','--exclude='+','.join(sorted(excluded)),f'--array=0-{len(selected)-1}',
                        '--output='+str(root/'logs/%x-%A_%a.out'),str(script),'batch',str(out),str(batchpath)]
                    job=command(args).split(';')[0]; assert job.isdigit()
                    batch={'job_id':job,'tasks':selected,'task_file_sha256':digest(batchpath),'used_cores_before':used,'requested_cores':2*len(selected),'phase':phase}
                    atomic_json(batchpath.with_suffix('.json'),batch); batches.append(batch)
                    with (root/'receipts/submissions.tsv').open('a') as log: log.write(f'{job}\thybrid-parallel-batch\t{phase} {len(selected)} tasks\n')
                    print(json.dumps({'event':'submitted','job_id':job,'phase':phase,'tasks':len(selected),'total_requested_cores':used+2*len(selected)}),flush=True)
        status={'complete':False,'phase':phase,'completed':sum(taskkey(t) in done for t in tasks),'total':len(tasks),
            'ready_at_last_poll':len(ready),'inflight_before_submission':len(inflight),'user_requested_cores_at_poll':sum(r['cores'] for r in live.values()),
            'preserved_original_results':len(science.read_json(dest/'preserved.json')),'dispatcher_job':os.environ['SLURM_JOB_ID']}
        atomic_json(dest/'status.json',status)
        marker=(phase,status['completed'],status['inflight_before_submission'],len(ready))
        if marker!=last_status: print(json.dumps(status),flush=True); last_status=marker
        time.sleep(15)

if __name__=='__main__':
    out=Path(sys.argv[1])
    try: main(out)
    except Exception as e:
        import traceback
        atomic_json(out/'parallel-v1/dispatcher_error.json',{'error':repr(e),'traceback':traceback.format_exc(),'job_id':os.environ.get('SLURM_JOB_ID')})
        raise
