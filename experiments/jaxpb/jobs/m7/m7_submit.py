"""Transport immutable snapshots and submit only allowed Coulson allocations."""
import argparse, ast, hashlib, json, shlex, subprocess, tarfile, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('tag'); p.add_argument('--reuse',action='store_true'); p.add_argument('--baseline',action='store_true'); p.add_argument('--gpu',action='store_true'); p.add_argument('--cpus',type=int,choices=(1,5,6,12),default=12); p.add_argument('--hours',type=int,default=24); p.add_argument('--node',choices=('comp0650','comp0651','comp0652')); p.add_argument('--one-thread-per-core',action='store_true'); p.add_argument('--dependency'); p.add_argument('--dependency-type',choices=('afterok','afterany'),default='afterok'); cut=sys.argv.index('--'); a=p.parse_args(sys.argv[1:cut]); args=sys.argv[cut+1:]
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'; root=f'{T}/m7-{a.tag}'
log=Path('results/m7-jobs.json'); prior=json.loads(log.read_text()) if log.exists() else []; snapshots=prior+(json.loads(Path('results/m7-freezes.json').read_text()) if Path('results/m7-freezes.json').exists() else []); batch_root=root; source_base=None
def ssh(c): return subprocess.check_output(['ssh','Coulson-Chemistry',c],text=True).strip()
if not a.reuse:
    files=list(Path('src/jaxpb').glob('*'))+list(Path('tests').glob('*.py'))+list(Path('jobs').glob('m7*'))
    files += [Path('jobs')/n for n in ('bench_jaxpb.py','prof_geomaps.py','prof2.py','speed.py','opt2.py','opt4_kernels.py','opt4_arms.py')]+list(Path('jobs/a40site').glob('*.py'))
    files=[f for f in files if f.is_file() and f.suffix!='.pyc']
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(),str(f))
    data={str(f):f.read_bytes() for f in files}
    if a.baseline:
        sources=subprocess.check_output(['git','ls-tree','-r','--name-only','31b30bc','src/jaxpb'],text=True).splitlines()
        data={k:v for k,v in data.items() if not k.startswith('src/jaxpb/')}
        for f in sources:data[f]=subprocess.check_output(['git','show','31b30bc:'+f])
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in data.items()}
    ssh(f'set -e; mkdir {shlex.quote(root)}; mkdir -p {T}/m7')
    with tarfile.open('/tmp/m7-transfer.tar.gz','w:gz') as tf:
        for name,blob in data.items():
            import io
            ti=tarfile.TarInfo(name);ti.size=len(blob);tf.addfile(ti,io.BytesIO(blob))
    with open('/tmp/m7-transfer.tar.gz','rb') as fi: subprocess.run(['ssh','Coulson-Chemistry',f'tar xzf - -C {root}'],stdin=fi,check=True)
else:
    selected=next(x for x in snapshots if x['frozen']==root); hashes=selected['hashes']; source_base=selected.get('source_base'); checked=selected
    while True:
        got=ssh(f'cd {checked["frozen"]} && sha256sum '+' '.join(shlex.quote(x) for x in checked['hashes']))
        assert {l.split()[1]:l.split()[0] for l in got.splitlines()}==checked['hashes']
        batch_root=checked['frozen']
        if not checked.get('source_base'):break
        checked=next(x for x in snapshots if x['frozen']==checked['source_base'])
cmd=['sbatch','--parsable','-J',f'm7-{a.tag}','-t',f'{a.hours}:00:00',f'--cpus-per-task={a.cpus}','--mem-per-cpu=2G',f'--output={T}/m7/%x-%j.out']
if a.node:
    assert not a.gpu, 'A40 node is fixed by the GPU batch script'
    cmd += ['--nodelist='+a.node]
if a.one_thread_per_core: cmd += ['--threads-per-core=1']
if a.dependency: cmd += [f'--dependency={a.dependency_type}:{a.dependency}']
cmd += [f'{batch_root}/jobs/m7_{"gpu" if a.gpu else "cpu"}.sbatch',root]+args
command=shlex.join(cmd); jid=ssh(command); row=dict(tag=a.tag,job=jid,frozen=root,gpu=a.gpu,cpus=a.cpus,command=command,hashes=hashes,**({'source_base':source_base} if source_base else {})); log.write_text(json.dumps(prior+[row],indent=2)+'\n'); print(json.dumps({k:v for k,v in row.items() if k!='hashes'},indent=2))
