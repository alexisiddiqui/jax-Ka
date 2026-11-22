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
p.add_argument('--fresh-baseline',action='store_true')
p.add_argument('--checks', action='store_true')
p.add_argument('--maps', action='store_true')
p.add_argument('--tune', action='store_true')
p.add_argument('--script')
p.add_argument('--passes', type=int)
p.add_argument('--arms', nargs='+', default=['pallas', 'xla', 'cpu'])
a = p.parse_args()
root = f'{T}/opt2-{a.tag}-20261122'
# Frozen directories are immutable. Reuse permits new job processes on exactly the same source.
if a.reuse:
    log = json.loads(Path('results/opt2-jobs.json').read_text())
    prior = next(r for r in log if r['tag'] == a.tag and 'source_sha256' in r)
    source_hashes = prior['source_sha256']
    check = subprocess.check_output(['ssh', 'Coulson-Chemistry', f'cd {root} && sha256sum src/jaxpb/*.py'], text=True)
    assert {line.split()[1]:line.split()[0] for line in check.splitlines()} == source_hashes, 'frozen source was modified'
    subprocess.run(['ssh', 'Coulson-Chemistry', f'test -d {root}'], check=True)
else:
    subprocess.run(['ssh', 'Coulson-Chemistry', f'set -e; mkdir {root}; test -d {T}/opt2 || mkdir {T}/opt2'], check=True)
    archive = Path('/tmp/opt2-transfer.tar.gz')
    with tarfile.open(archive, 'w:gz') as tf:
        for directory in ('src', 'jobs', 'tests'):
            for f in Path(directory).rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc':
                    tf.add(f)
    with archive.open('rb') as f:
        subprocess.run(['ssh', 'Coulson-Chemistry', f'tar xzf - -C {root}'], stdin=f, check=True)
rows = []
for arm in a.arms:
    backend = 'xla' if arm in ('cpu','xla-fresh') else arm
    selection = {'pallas': 'bench', 'xla': 'xla12', 'cpu': 'cpu', 'all': 'all', 'fresh': 'all', 'xla-fresh':'xla12'}[arm]
    if arm in ('all', 'fresh'):
        backend = 'pallas'
    ref = 'cpu' if arm == 'cpu' else 'xla' if arm in ('xla','xla-fresh') else 'all' if arm in ('all', 'fresh') else 'pallas'
    passes = a.passes or (1 if arm in ('cpu', 'fresh','xla-fresh') else 2)
    runner = 'opt2_cpu.sbatch' if arm == 'cpu' else 'a40.sbatch'
    fresh = f' --fresh-reference {T}/opt2/{a.tag}-all.json' if arm == 'fresh' else ''
    if a.fresh_baseline: fresh = f' --fresh-reference {T}/opt1b/s5-all.json'
    if arm == 'xla-fresh': fresh = f' --fresh-reference {T}/opt2/{a.tag}-xla.json'
    cmd = (f'sbatch --parsable -J opt2-{a.tag}-{arm} -t 12:00:00 '
           f'--output={T}/opt2/%x-%j.out {root}/jobs/{runner} {root} jobs/opt2.sh '
           f'--tag {a.tag}-{arm} --backend {backend} --selection {selection} --passes {passes} '
           f'--reference {T}/opt1b/s5-{ref}.json{fresh}')
    if a.tune:
        cmd = f'sbatch --parsable -J opt2-tune-{arm} -t 06:00:00 --output={T}/opt2/%x-%j.out {root}/jobs/{runner} {root} jobs/opt2_tune.sh'
    if a.checks or a.maps:
        script = 'maps' if a.maps else 'checks'
        cmd = f'sbatch --parsable -J opt2-{script}-{arm} -t 06:00:00 --output={T}/opt2/%x-%j.out {root}/jobs/{runner} {root} jobs/opt2_{script}.sh'
    if a.script:
        cmd = f'sbatch --parsable -J opt2-{a.tag}-diag-{arm} -t 02:00:00 --output={T}/opt2/%x-%j.out {root}/jobs/{runner} {root} {a.script}'
    if a.dependency:
        cmd = cmd.replace('sbatch --parsable', f'sbatch --parsable --dependency=afterok:{a.dependency}')
    jid = subprocess.check_output(['ssh', 'Coulson-Chemistry', cmd], text=True).strip()
    rows.append(dict(source_sha256=source_hashes if a.reuse else {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src/jaxpb').glob('*.py')}, tag=a.tag, arm=arm, job=jid, frozen=root, command=cmd))
log = Path('results/opt2-jobs.json')
prior = json.loads(log.read_text()) if log.exists() else []
log.write_text(json.dumps(prior + rows, indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('tag','arm','job','frozen')} for r in rows], indent=2))
