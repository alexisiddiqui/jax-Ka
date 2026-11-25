"""Local transport: append a frozen arm or terminate the allocated pipeline."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stop',action='store_true');p.add_argument('--queue',default='cpu-queue.json');p.add_argument('--root');p.add_argument('--id');split=sys.argv.index('--') if '--' in sys.argv else len(sys.argv);a=p.parse_args(sys.argv[1:split]);args=sys.argv[split+1:]
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/opt4';local=Path('results/opt4-'+a.queue);q=json.loads(local.read_text()) if local.exists() else dict(commands=[])
if a.stop:q['stop']=True
else:
 assert a.root and a.id and args;assert a.id not in [c['id'] for c in q['commands']];q['commands'].append(dict(id=a.id,root=a.root,args=args))
local.write_text(json.dumps(q,indent=2)+'\n')
with local.open('rb') as f:subprocess.run(['ssh','Coulson-Chemistry',f'cat > {shlex.quote(T+"/"+a.queue+".tmp")} && mv {shlex.quote(T+"/"+a.queue+".tmp")} {shlex.quote(T+"/"+a.queue)}'],stdin=f,check=True)
print(json.dumps(q,indent=2))
