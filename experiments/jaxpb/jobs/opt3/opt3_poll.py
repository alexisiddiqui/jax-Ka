"""Read only our OPT3 jobs; append the squeue poll and show short log tails."""
import datetime
import json
import shlex
import subprocess
from pathlib import Path
rows=json.load(open('results/opt3-jobs.json'))
ids=','.join(r['job'] for r in rows)
cmd=f'squeue -j {ids} -o "%.18i %.9P %.26j %.8T %.10M %.6D %R"'
result=subprocess.check_output(['ssh','Coulson-Chemistry',cmd],text=True)
with open('results/opt3-squeue.jsonl','a') as f:
    f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,output=result))+'\n')
print(result)
active={line.split()[0] for line in result.splitlines()[1:]}
paths=[f'/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/opt3/opt3-{r["tag"]}-{r["job"]}.out' for r in rows if r['job'] in active]
if paths:
    print(subprocess.check_output(['ssh','Coulson-Chemistry','tail -n 3 '+shlex.join(paths)],text=True))
