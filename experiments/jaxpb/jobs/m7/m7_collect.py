"""Local transport only: collect Slurm outputs and immutable-source checksums."""
import csv,io,json,subprocess,tarfile,hashlib,re,time
from pathlib import Path
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
jobs=json.loads(Path('results/m7-jobs.json').read_text());ids=','.join(r['job'] for r in jobs)
def ssh(cmd):
 for attempt in range(3):
  try:
   return subprocess.check_output(['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','Coulson-Chemistry',cmd],timeout=300,stderr=subprocess.PIPE)
  except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
   if attempt==2:raise
   time.sleep(5)
blob=ssh(f'cd {T}/m7 && tar czf - -- *.json *.out')
Path('results/m7-evidence.tar.gz').write_bytes(blob)
D=Path('/tmp/m7-evidence');D.mkdir(exist_ok=True)
with tarfile.open('results/m7-evidence.tar.gz') as tf:tf.extractall(D,filter='data')
accounting=ssh(f'sacct -P -j {ids} --format=JobIDRaw,JobName%40,Partition,ReqCPUS,AllocCPUS,ReqMem,State,Elapsed,MaxRSS,NodeList,AllocTRES%100,ExitCode')
Path('results/m7-accounting.txt').write_bytes(accounting)
records={r['JobIDRaw']:r for r in csv.DictReader(io.StringIO(accounting.decode()),delimiter='|')}
allocations={}
def gib(value):
 match=re.fullmatch(r'([\d.]+)([KMGT])(?:[cn])?',value)
 assert match,value
 return float(match[1])*{'K':2**-20,'M':2**-10,'G':1,'T':1024}[match[2]]
for job in jobs:
 r=records[job['job']];allocated=int(r['AllocCPUS']);requested=int(r['ReqCPUS'])
 assert r['Partition']==('oc' if job['gpu'] else 'amd96'),r
 assert requested==job['cpus'] and 0<=allocated<=12,r
 assert gib(r['ReqMem'])==2*requested and '--mem-per-cpu=2G' in job['command'],r
 tres=dict(p.split('=',1) for p in r['AllocTRES'].split(',') if '=' in p)
 if allocated:
  assert gib(tres['mem'])==2*allocated and tres['node']=='1',r
  if job['gpu']:assert r['NodeList']=='comp1400' and tres.get('gres/gpu')=='1',r
 allocations[job['job']]=dict(tag=job['tag'],passed=True,**r)
Path('results/m7-compute-audit.json').write_text(json.dumps(dict(passed=True,allocations=allocations,note='Slurm may round one requested CPU to two allocated hardware threads; both counts and allocated memory are checked. A40 type is fixed in the hashed GPU sbatch script; accounting reports generic gres/gpu.'),indent=2)+'\n')
# Hashing and file transfer are metadata operations, never scientific login-node work.
audit={}
for root in dict.fromkeys(r['frozen'] for r in jobs):
 row=next(r for r in jobs if r['frozen']==root)
 raw=ssh('cd '+root+' && sha256sum '+' '.join(row['hashes']))
 got={line.split()[1]:line.split()[0] for line in raw.decode().splitlines()}
 audit[root]=dict(frozen_unchanged=got==row['hashes'],files=len(got),hashes=got)
 assert audit[root]['frozen_unchanged'],root
Path('results/m7-source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Collected',len(blob),'bytes;',len(audit),'unchanged snapshots')
