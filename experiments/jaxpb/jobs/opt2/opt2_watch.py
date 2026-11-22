"""Local squeue polling, with sleeps; read only the jobs recorded by this phase."""
import json
import subprocess
import time
from pathlib import Path

while True:
    ids = ','.join(str(r['job']) for r in json.loads(Path('results/opt2-jobs.json').read_text()))
    run = subprocess.run(['ssh', 'Coulson-Chemistry', 'squeue -h -j '+ids+' -o "%.12i %.23j %.8T %.9M %.20R"'],
                         text=True, capture_output=True, check=False)
    if run.returncode:
        print('squeue transport failed; retrying after sleeps:', run.stderr.strip(), flush=True)
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), '\n'+run.stdout, flush=True)
    with Path('results/opt2-squeue.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=time.time(), ids=ids, squeue=run.stdout,returncode=run.returncode,stderr=run.stderr))+'\n')
    if run.returncode == 0 and not run.stdout.strip(): break
    for i in range(3):
        time.sleep(60)
        print('Waiting for submitted OPT2 jobs; next squeue poll in', (2-i)*60, 'seconds.', flush=True)
