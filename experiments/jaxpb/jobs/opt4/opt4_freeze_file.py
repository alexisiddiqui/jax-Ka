"""Freeze a small script override with an explicit immutable source dependency."""
import argparse,ast,hashlib,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('file');p.add_argument('--base',required=True);a=p.parse_args()
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
assert a.base.startswith(T+'/opt4-') and a.file.startswith('jobs/opt4')
root=T+'/opt4-'+a.tag
source='import sys\nsys.path[:0]='+repr([a.base+'/src',a.base+'/jobs'])+'\n'+Path(a.file).read_text();ast.parse(source)
subprocess.run(['ssh','Coulson-Chemistry','mkdir '+shlex.quote(root)+' && mkdir '+shlex.quote(root+'/jobs')],check=True)
subprocess.run(['ssh','Coulson-Chemistry','cat > '+shlex.quote(root+'/'+a.file)],input=source.encode(),check=True)
p=Path('results/opt4-freezes.json');rows=json.loads(p.read_text());rows.append(dict(tag=a.tag,frozen=root,source_base=a.base,hashes={a.file:hashlib.sha256(source.encode()).hexdigest()}));p.write_text(json.dumps(rows,indent=2)+'\n');print(root)
