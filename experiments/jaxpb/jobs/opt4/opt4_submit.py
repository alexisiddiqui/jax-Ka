"""Transport immutable snapshots and submit only allowed Coulson allocations."""
import argparse, ast, hashlib, json, shlex, subprocess, tarfile, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('tag'); p.add_argument('--reuse',action='store_true'); p.add_argument('--gpu',action='store_true'); p.add_argument('--cpus',type=int,choices=(1,5,6,12),default=12); p.add_argument('--hours',type=int,default=24); p.add_argument('--node',choices=('comp0650','comp0651','comp0652')); p.add_argument('--one-thread-per-core',action='store_true'); p.add_argument('--dependency'); p.add_argument('--dependency-type',choices=('afterok','afterany'),default='afterok'); cut=sys.argv.index('--'); a=p.parse_args(sys.argv[1:cut]); args=sys.argv[cut+1:]
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'; root=f'{T}/opt4-{a.tag}'
log=Path('results/opt4-jobs.json'); prior=json.loads(log.read_text()) if log.exists() else []; snapshots=prior+(json.loads(Path('results/opt4-freezes.json').read_text()) if Path('results/opt4-freezes.json').exists() else []); batch_root=root; source_base=None
def ssh(c): return subprocess.check_output(['ssh','Coulson-Chemistry',c],text=True).strip()
if not a.reuse:
    files=list(Path('src/jaxpb').glob('*'))+list(Path('tests').glob('*.py'))+list(Path('jobs').glob('opt4*'))
    files += [Path('jobs')/n for n in ('bench_jaxpb.py','prof_geomaps.py','prof2.py','speed.py','opt2.py')]+list(Path('jobs/a40site').glob('*.py'))
    files=[f for f in files if f.is_file() and f.suffix!='.pyc']
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(),str(f))
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    ssh(f'set -e; mkdir {shlex.quote(root)}; mkdir -p {T}/opt4')
    with tarfile.open('/tmp/opt4-transfer.tar.gz','w:gz') as tf:
        for f in files: tf.add(f)
    with open('/tmp/opt4-transfer.tar.gz','rb') as fi: subprocess.run(['ssh','Coulson-Chemistry',f'tar xzf - -C {root}'],stdin=fi,check=True)
else:
    selected=next(x for x in snapshots if x['frozen']==root); hashes=selected['hashes']; source_base=selected.get('source_base'); checked=selected
    while True:
        got=ssh(f'cd {checked["frozen"]} && sha256sum '+' '.join(shlex.quote(x) for x in checked['hashes']))
        assert {l.split()[1]:l.split()[0] for l in got.splitlines()}==checked['hashes']
        batch_root=checked['frozen']
        if not checked.get('source_base'):break
        checked=next(x for x in snapshots if x['frozen']==checked['source_base'])
cmd=['sbatch','--parsable','-J',f'opt4-{a.tag}','-t',f'{a.hours}:00:00',f'--cpus-per-task={a.cpus}','--mem-per-cpu=2G',f'--output={T}/opt4/%x-%j.out']
if a.node:
    assert not a.gpu, 'A40 node is fixed by the GPU batch script'
    cmd += ['--nodelist='+a.node]
if a.one_thread_per_core: cmd += ['--threads-per-core=1']
if a.dependency: cmd += [f'--dependency={a.dependency_type}:{a.dependency}']
cmd += [f'{batch_root}/jobs/opt4_{"gpu" if a.gpu else "cpu"}.sbatch',root]+args
command=shlex.join(cmd); jid=ssh(command); row=dict(tag=a.tag,job=jid,frozen=root,gpu=a.gpu,cpus=a.cpus,command=command,hashes=hashes,**({'source_base':source_base} if source_base else {})); log.write_text(json.dumps(prior+[row],indent=2)+'\n'); print(json.dumps({k:v for k,v in row.items() if k!='hashes'},indent=2))
