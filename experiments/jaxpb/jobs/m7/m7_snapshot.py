"""Snapshot one fully completed pass while the required repeat continues.

Run on comp1400 in an allowed allocation: only reads our parent's cache/output;
never signals or edits the parent process or its frozen source. Array digests
must match every first-pass checkpoint field before the copy is published.
"""
import argparse,datetime,hashlib,json,os,shutil,tarfile,time,zipfile
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
T=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline');O=T/'m7'
p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=int);a=p.parse_args()
r=json.loads((O/'converged-final.json').read_text())
assert r['env']['job']==str(a.parent) and r['cfg']['pb_mode']=='converged'
assert r['passes'] and len(r['passes'][0]['rows'])==104
ids=set(r['states']);rows=r['passes'][0]['rows'];assert set(rows)==ids
assert all(v['flags']['residual_max']<=1e-9 and v['flags']['stored_residual_max']<=1e-4 for v in rows.values())
h=hashlib.sha256()
for f in sorted(Path('src/jaxpb').glob('*.py')):h.update(f.name.encode());h.update(f.read_bytes())
assert h.hexdigest()[:16]==r['code_sha'],'snapshot code must equal parent code'
def digest(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1<<20),b''):h.update(part)
 return h.hexdigest()
source=O/'converged-final-energies.npz';out=O/'converged-firstpass-energies.npz';tmp=out.with_suffix('.tmp-'+os.environ['SLURM_JOB_ID'])
for attempt in range(10):
 try:
  shutil.copyfile(source,tmp)
  with np.load(tmp,allow_pickle=False) as z:
   assert {key.split('/')[0] for key in z.files}==ids
   checked=0
   for key in z.files:
    sid,field=key.split('/');assert digest(z[key])==rows[sid]['hashes'][field],key;checked+=1
   assert checked==sum(len(v['hashes'])-1 for v in rows.values()) # sitpot fingerprint is in checkpoint, not NPZ
  break
 except (OSError,EOFError,ValueError,AssertionError,zipfile.BadZipFile) as exc:
  if attempt==9:raise
  print('copy overlapped parent publication; retry',repr(exc),flush=True);time.sleep(2)
os.replace(tmp,out)
# All parent first-pass executables are already published; copy only regular
# cache entries, from this explicitly identified M7 job's node-local directory.
cache=Path('/tmp')/('jaxpb-cc-'+str(a.parent));files=sorted(f for f in cache.iterdir() if f.is_file()) if cache.exists() else []
assert all(not f.is_symlink() for f in files)
# Some Slurm installations isolate /tmp by allocation. An empty cache is valid:
# the independent fresh process will compile cold, and its counters report it.
archive=O/'firstpass-cache.tar';ctmp=archive.with_suffix('.tmp-'+os.environ['SLURM_JOB_ID'])
with tarfile.open(ctmp,'w') as tf:
 for f in files:tf.add(f,arcname=f.name)
os.replace(ctmp,archive)
r.pop('current',None);r['passes']=r['passes'][:1]
r.update(completed=True,passed=True,snapshot=dict(stage='one complete 104-state pass; parent repeat continues',parent_job=str(a.parent),snapshot_job=os.environ['SLURM_JOB_ID'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),energies=str(out),energy_sha256=sha(out),verified_arrays=checked,cache=str(archive),cache_entries=len(files),parent_cache_visible=bool(files),cache_bytes=archive.stat().st_size))
(O/'converged-firstpass.json').write_text(json.dumps(r,indent=1))
print(json.dumps(r['snapshot'],indent=1),flush=True)
