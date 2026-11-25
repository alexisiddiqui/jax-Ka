"""Transport-only monitoring: squeue every three minutes with sleep.

Stop promptly on a required-job failure; never cancel or alter any cluster job.
After the queue closes, collect evidence and build the reports locally.
"""
import datetime,json,subprocess,time,sys
from pathlib import Path
p=Path('results/m7-squeue.jsonl')
required={'baseline-cpu','baseline-gpu','candidate-cpu-parallel','candidate-gpu','fresh-cpu-parallel','fresh-gpu','converged-v3','fresh-converged4','converged-cpu3','close-gpu3','close-cpu3','labels4','snapshot-firstpass','memory-final','label-details','tests-final3','mode-cli','probe-gpu3','probe-cpu3','analytic-final'}
while True:
 rows=json.loads(Path('results/m7-jobs.json').read_text());ids=','.join(dict.fromkeys(r['job'] for r in rows))
 wanted={r['job']:r['tag'] for r in rows if r['tag'] in required}
 try:
  r=subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','Coulson-Chemistry','squeue -h -j '+ids+' -o "%i|%j|%P|%T|%M|%C|%R"'],text=True,capture_output=True,timeout=40)
 except subprocess.TimeoutExpired:
  r=subprocess.CompletedProcess([],124,'','Coulson SSH timeout')
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();row=dict(utc=stamp,ids=ids,returncode=r.returncode,output=r.stdout,stderr=r.stderr)
 with p.open('a') as f:f.write(json.dumps(row)+'\n')
 print(stamp,r.stdout.strip() or ('no queued jobs' if r.returncode==0 else 'poll failed'),flush=True)
 if r.returncode==0:
  try:
   ac=subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','Coulson-Chemistry','sacct -X -n -P -j '+','.join(wanted)+' --format=JobIDRaw,State,ExitCode'],text=True,capture_output=True,timeout=40)
  except subprocess.TimeoutExpired:
   ac=subprocess.CompletedProcess([],124,'','Coulson accounting timeout')
  if ac.returncode:
   print('Accounting unavailable; retry after sleep',flush=True);time.sleep(180);continue
  failed=[]
  for line in ac.stdout.splitlines():
   fields=line.split('|')
   if len(fields)>=3 and fields[0] in wanted and fields[1].split()[0] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL'):
    failed.append(dict(job=fields[0],tag=wanted[fields[0]],state=fields[1],exitcode=fields[2]))
  if failed:print('REQUIRED JOB FAILURE',json.dumps(failed),flush=True);sys.exit(2)
  if not r.stdout.strip():break
 time.sleep(180)
subprocess.run([sys.executable,'jobs/m7_collect.py'],check=True)
subprocess.run([sys.executable,'jobs/m7_results.py'],check=True)
print('All M7 jobs and report generation completed',flush=True)
