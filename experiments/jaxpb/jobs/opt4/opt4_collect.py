"""Local read-only collection of phase-4 evidence."""
import json,subprocess,tarfile
from pathlib import Path
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
cmd=f'cd {T}/opt4 && tar czf - --ignore-failed-read *.json *.out'
p=Path('results/opt4-evidence.tar.gz')
with p.open('wb') as f:subprocess.run(['ssh','Coulson-Chemistry',cmd],stdout=f,check=True)
with tarfile.open(p) as tf:
 for m in tf.getmembers():
  if m.isfile() and '/' not in m.name and m.name.endswith('.json'):
   data=json.load(tf.extractfile(m)); print(m.name,'complete',data.get('completed'),'rows',len(data.get('rows',{})),'passes',len(data.get('passes',[])),'current',list(data.get('current',{}).get('rows',{})))
rows=json.loads(Path('results/opt4-jobs.json').read_text()); ids=','.join(r['job'] for r in rows)
out=subprocess.check_output(['ssh','Coulson-Chemistry',f'sacct -n -P -j {ids} --format=JobID,JobName,Partition,AllocCPUS,ReqMem,State,Elapsed,MaxRSS,ExitCode'],text=True)
Path('results/opt4-accounting.txt').write_text(out); print(out)
