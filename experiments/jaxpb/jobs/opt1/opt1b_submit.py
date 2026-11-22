"""Local transport/submission only; every numerical operation is a Slurm job."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

T = '/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--reuse', action='store_true')
p.add_argument('--dependency')
p.add_argument('--checks', action='store_true')
p.add_argument('--maps', action='store_true')
p.add_argument('--tune', action='store_true')
p.add_argument('--arms', nargs='+', default=['pallas', 'xla', 'cpu'])
a = p.parse_args()
root = f'{T}/opt1b-{a.tag}-20261122'
# Frozen directories are immutable. Reuse permits new job processes on exactly the same source.
if a.reuse:
    log = json.loads(Path('results/opt1b-jobs.json').read_text())
    prior = next(r for r in log if r['tag'] == a.tag and 'source_sha256' in r)
    current = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src/jaxpb').glob('*.py')}
    assert current == prior['source_sha256'], 'local source differs from this frozen variant'
    subprocess.run(['ssh', 'Coulson-Chemistry', f'test -d {root}'], check=True)
else:
    subprocess.run(['ssh', 'Coulson-Chemistry', f'set -e; mkdir {root}; test -d {T}/opt1b || mkdir {T}/opt1b'], check=True)
    archive = Path('/tmp/opt1b-transfer.tar.gz')
    with tarfile.open(archive, 'w:gz') as tf:
        for directory in ('src', 'jobs', 'tests'):
            for f in Path(directory).rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc':
                    tf.add(f)
    with archive.open('rb') as f:
        subprocess.run(['ssh', 'Coulson-Chemistry', f'tar xzf - -C {root}'], stdin=f, check=True)
rows = []
for arm in a.arms:
    backend = 'xla' if arm == 'cpu' else arm
    selection = {'pallas': 'bench', 'xla': 'xla12', 'cpu': 'cpu', 'all': 'all', 'fresh': 'all'}[arm]
    if arm in ('all', 'fresh'):
        backend = 'pallas'
    ref = 'cpu' if arm == 'cpu' else 'xla' if arm == 'xla' else 'pallas'
    passes = 1 if arm in ('cpu', 'fresh') else 2
    runner = 'opt1b_cpu.sbatch' if arm == 'cpu' else 'a40.sbatch'
    fresh = f' --fresh-reference {T}/opt1b/{a.tag}-all.json' if arm == 'fresh' else ''
    cmd = (f'sbatch --parsable -J opt1b-{a.tag}-{arm} -t 12:00:00 '
           f'--output={T}/opt1b/%x-%j.out {root}/jobs/{runner} {root} jobs/opt1b.sh '
           f'--tag {a.tag}-{arm} --backend {backend} --selection {selection} --passes {passes} '
           f'--reference {T}/opt1/baseline-{ref}.json{fresh}')
    if a.tune:
        cmd = f'sbatch --parsable -J opt1b-tune-{arm} -t 06:00:00 --output={T}/opt1b/%x-%j.out {root}/jobs/{runner} {root} jobs/opt1b_tune.sh'
    if a.checks or a.maps:
        script = 'maps' if a.maps else 'checks'
        cmd = f'sbatch --parsable -J opt1b-{script}-{arm} -t 06:00:00 --output={T}/opt1b/%x-%j.out {root}/jobs/{runner} {root} jobs/opt1b_{script}.sh'
    if a.dependency:
        cmd = cmd.replace('sbatch --parsable', f'sbatch --parsable --dependency=afterok:{a.dependency}')
    jid = subprocess.check_output(['ssh', 'Coulson-Chemistry', cmd], text=True).strip()
    rows.append(dict(source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src/jaxpb').glob('*.py')}, tag=a.tag, arm=arm, job=jid, frozen=root, command=cmd))
log = Path('results/opt1b-jobs.json')
prior = json.loads(log.read_text()) if log.exists() else []
log.write_text(json.dumps(prior + rows, indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('tag','arm','job','frozen')} for r in rows], indent=2))
