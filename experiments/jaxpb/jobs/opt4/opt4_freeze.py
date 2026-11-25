"""Freeze a new immutable snapshot without requesting another allocation."""
import ast,hashlib,json,shlex,subprocess,sys,tarfile
from pathlib import Path
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline';tag=sys.argv[1];root=T+'/opt4-'+tag
files=list(Path('src/jaxpb').glob('*'))+list(Path('tests').glob('*.py'))+list(Path('jobs').glob('opt4*'))
files += [Path('jobs')/n for n in ('bench_jaxpb.py','prof_geomaps.py','prof2.py','speed.py','opt2.py')]+list(Path('jobs/a40site').glob('*.py'));files=[f for f in files if f.is_file() and f.suffix!='.pyc']
for f in files:
 if f.suffix=='.py':ast.parse(f.read_text(),str(f))
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
subprocess.run(['ssh','Coulson-Chemistry','mkdir '+shlex.quote(root)],check=True)
with tarfile.open('/tmp/opt4-freeze.tar.gz','w:gz') as tf:
 for f in files:tf.add(f)
with open('/tmp/opt4-freeze.tar.gz','rb') as f:subprocess.run(['ssh','Coulson-Chemistry','tar xzf - -C '+shlex.quote(root)],stdin=f,check=True)
p=Path('results/opt4-freezes.json');prior=json.loads(p.read_text()) if p.exists() else [];prior.append(dict(tag=tag,frozen=root,hashes=hashes));p.write_text(json.dumps(prior,indent=2)+'\n');print(root)
