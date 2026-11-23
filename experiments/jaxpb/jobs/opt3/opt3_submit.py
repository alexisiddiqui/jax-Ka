"""Local transport only. Every calculation runs inside an allowed Coulson allocation."""
import argparse
import ast
import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--reuse', action='store_true')
p.add_argument('--gpu', action='store_true')
p.add_argument('--cpus', type=int, choices=(1, 12), default=12)
p.add_argument('--hours', type=int, default=12)
p.add_argument('--dependency')
p.add_argument('args', nargs=argparse.REMAINDER)
a = p.parse_args()
T = '/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
root = f'{T}/opt3-{a.tag}'
log = Path('results/opt3-jobs.json')
prior = json.loads(log.read_text()) if log.exists() else []
def ssh(command):
    return subprocess.check_output(['ssh', 'Coulson-Chemistry', command], text=True).strip()
if not a.reuse:
    files = list(Path('src/jaxpb').glob('*.py')) + [Path('tests')/n for n in ('test_titrate.py','test_opt3.py','test_api.py')]
    files += list(Path('jobs').glob('opt3*')) + [Path('jobs/bench_jaxpb.py')]
    files += [f for f in Path('jobs/a40site').rglob('*') if f.is_file() and f.suffix=='.py']
    files = [f for f in files if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc']
    for f in files:
        if f.suffix == '.py': ast.parse(f.read_text(), filename=str(f))
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    ssh(f'set -e; mkdir {shlex.quote(root)}; mkdir -p {T}/opt3')
    with tarfile.open('/tmp/opt3-transfer.tar.gz', 'w:gz') as tf:
        for f in files: tf.add(f)
    with open('/tmp/opt3-transfer.tar.gz', 'rb') as fi:
        subprocess.run(['ssh', 'Coulson-Chemistry', f'tar xzf - -C {root}'], stdin=fi, check=True)
else:
    hashes = next(r['hashes'] for r in prior if r['frozen'] == root)
    got = ssh(f'cd {root} && sha256sum ' + ' '.join(shlex.quote(x) for x in hashes))
    assert {l.split()[1]: l.split()[0] for l in got.splitlines()} == hashes
args = a.args[1:] if a.args and a.args[0] == '--' else a.args
assert args
cmd = ['sbatch', '--parsable', '-J', f'opt3-{a.tag}', '-t', f'{a.hours:02}:00:00',
       f'--cpus-per-task={a.cpus}', '--mem-per-cpu=2G', f'--output={T}/opt3/%x-%j.out']
if a.dependency: cmd += [f'--dependency=afterok:{a.dependency}']
cmd += [f'{root}/jobs/opt3_{"gpu" if a.gpu else "cpu"}.sbatch', root] + args
command = shlex.join(cmd)
jid = ssh(command)
row = dict(tag=a.tag, job=jid, frozen=root, gpu=a.gpu, cpus=a.cpus, command=command, hashes=hashes)
log.write_text(json.dumps(prior + [row], indent=2) + '\n')
print(json.dumps({k: v for k, v in row.items() if k != 'hashes'}, indent=2))
