"""Local transport-only polling, every three minutes, without touching any jobs."""
import datetime,json,subprocess,time
from pathlib import Path
p=Path('results/opt4-squeue.jsonl')
while True:
 rows=json.loads(Path('results/opt4-jobs.json').read_text()); ids=','.join(dict.fromkeys(r['job'] for r in rows))
 try:
  r=subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','Coulson-Chemistry','squeue -h -j '+ids+' -o "%.18i %.32j %.9P %.8T %.10M %.6C"'],text=True,capture_output=True,timeout=40)
 except subprocess.TimeoutExpired:
  r=subprocess.CompletedProcess([],124,'','Coulson SSH timed out after40s')
 row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ids=ids,returncode=r.returncode,output=r.stdout,stderr=r.stderr)
 with p.open('a') as f: f.write(json.dumps(row)+'\n')
 print(row['utc'],r.stdout.strip() or ('no queued jobs' if r.returncode==0 else 'poll failed: '+r.stderr.strip()),flush=True)
 if r.returncode==0 and not r.stdout.strip(): break
 time.sleep(180)
