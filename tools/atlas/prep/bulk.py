"""Resume-safe, separate ATLAS preparation and teacher labelling; one 72-CPU allocation."""
import os,sys,json,time,hashlib,io,gzip,tarfile,shutil,subprocess,concurrent.futures,multiprocessing,collections,fcntl,math
from pathlib import Path
import pilot as P
R=P.R; O=P.O; B=O/'bulk-v1'; B.mkdir(exist_ok=True)
KEEP=['AB.cif','A.cif','B.cif','student_AB.cif','source.cif','input_atom_mask.json','sites.json','labels.json','meta.json','provenance.json','conformers.json']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,data):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def initprep():P.D=Path(os.environ['ATLAS_BATCH_ROOT']);P.initprep()
def initlabel():P.D=Path(os.environ['ATLAS_BATCH_ROOT']);P.initlabel()
def prep(r):return P.prepare(r)
def label(r):return P.label(r)
def pack(r):
 d=Path(os.environ['ATLAS_BATCH_ROOT'])/r['id'];buf=io.BytesIO()
 labels=json.loads((d/'labels.json').read_text())
 for model in ('pkai','pkai_plus'):
  for state in ('AB','A','B'):
   rows=labels[model][state];keys=[tuple(x[:4]) for x in rows]
   assert len(keys)==len(set(keys)) and all(math.isfinite(x[4]) for x in rows)
 with tarfile.open(fileobj=buf,mode='w') as tf:
  for name in KEEP:tf.add(d/name,arcname=name)
 payload=gzip.compress(buf.getvalue(),compresslevel=3)
 return r['id'],payload,hashlib.sha256(payload).hexdigest()
def run_labels():
 tasks=json.loads(Path(sys.argv[2]).read_text());results=[]
 if tasks:
  with concurrent.futures.ProcessPoolExecutor(max_workers=72,initializer=initlabel,mp_context=multiprocessing.get_context('spawn')) as ex:
   results=list(ex.map(label,tasks,chunksize=1))
 atomic(Path(sys.argv[3]),results)
def main():
 assert os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_CPUS_PER_TASK'])==72
 assert int(os.environ.get('SLURM_JOB_NUM_NODES','1'))==1
 lock=(B/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 report=json.loads((O/'benchmark-report.json').read_text())
 assert sha(O/'candidates.json')==report['source_hashes']['candidates.json']
 assert sha(Path(P.__file__))==report['code_hashes']['pilot.py']
 rows=json.loads((O/'candidates.json').read_text());assert len(rows)==157773
 protocol=dict(candidates_sha256=sha(O/'candidates.json'),pilot_sha256=sha(Path(P.__file__)),bulk_sha256=sha(Path(__file__)),batch_size=288,count=len(rows),training_enabled=False,source_confidence_retained=True,mask_status='provisional; final site weights, masks, confidence policy and graphs required before training')
 if (B/'protocol.json').exists():assert json.loads((B/'protocol.json').read_text())==protocol,'Protocol changed; use a new version'
 else:atomic(B/'protocol.json',protocol)
 (B/'shards').mkdir(exist_ok=True);(B/'receipts').mkdir(exist_ok=True)
 local=Path(os.environ['TMPDIR'])/'atlas-batch';local.mkdir(exist_ok=True);os.environ['ATLAS_BATCH_ROOT']=str(local)
 started=time.time();totals=collections.Counter();batches=(len(rows)+287)//288
 def status(stage,batch,**extra):
  elapsed=time.time()-started
  value=dict(stage=stage,job=os.environ['SLURM_JOB_ID'],node=os.environ.get('SLURMD_NODENAME'),batch=batch,batches=batches,total_candidates=len(rows),counts=dict(totals),elapsed_seconds=elapsed,updated_unix=time.time(),training_enabled=False,**extra)
  atomic(B/'progress.json',value);print(json.dumps(value),flush=True)
 status('starting',0)
 with concurrent.futures.ProcessPoolExecutor(max_workers=72,initializer=initprep,mp_context=multiprocessing.get_context('spawn')) as ex:
  for batch,start in enumerate(range(0,len(rows),288)):
   receipt=B/'receipts'/f'{batch:04d}.json';archive=B/'shards'/f'{batch:04d}.tar';part=archive.with_suffix('.tar.partial')
   tasks=rows[start:start+288]
   if receipt.exists():
    r=json.loads(receipt.read_text());assert r['ids']==[x['id'] for x in tasks] and archive.exists() and sha(archive)==r['archive_sha256']
    totals.update(r['counts']);continue
   tick=time.time();status('preparing',batch)
   prepared=list(ex.map(prep,tasks,chunksize=1));candidates=[r for r in prepared if r['status']=='prepared']
   req=local/'label-requests.json';res=local/'label-results.json';atomic(req,candidates)
   status('labelling',batch,prepared_in_batch=len(candidates))
   subprocess.run(['/scratch/u6xq/hussain.u6xq/_install/jax-Ka-venvs/pkai/bin/python',str(Path(__file__).resolve()),'label',str(req),str(res)],check=True)
   labelled=json.loads(res.read_text());good=[r for r in labelled if r['status']=='labelled'];members=[]
   status('packing',batch,labelled_in_batch=len(good))
   with tarfile.open(part,'w') as tf,concurrent.futures.ThreadPoolExecutor(max_workers=72) as packpool:
    for cid,data,digest in packpool.map(pack,good):
     ti=tarfile.TarInfo(cid+'.tar.gz');ti.size=len(data);offset=tf.offset+512;tf.addfile(ti,io.BytesIO(data));members.append(dict(id=cid,member=ti.name,offset=offset,bytes=len(data),sha256=digest))
   # Re-open the completed archive before committing its receipt.
   with tarfile.open(part) as tf:assert [m.name for m in tf.getmembers()]==[r['member'] for r in members]
   part.replace(archive)
   errors=[r for r in prepared if r['status']=='error']+[r for r in labelled if r['status']=='error']
   counts=dict(attempted=len(tasks),labelled=len(good),rejected=sum(r['status']=='rejected' for r in prepared),errors=len(errors))
   assert sum(counts[k] for k in ('labelled','rejected','errors'))==len(tasks)
   rec=dict(batch=batch,ids=[r['id'] for r in tasks],counts=counts,seconds=time.time()-tick,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=members,preparation=prepared,labelling=labelled)
   atomic(receipt,rec);totals.update(counts)
   # Only disposable node-local intermediates are removed, after archive and receipt commit.
   for r in tasks:
    folder=local/r['id']
    if folder.exists():shutil.rmtree(folder)
   status('batch_complete',batch+1,batch_seconds=rec['seconds'])
   if len(errors)>max(3,len(tasks)*.05):raise RuntimeError('Unexpected error rate; durable outputs retained for inspection')
 status('complete',batches)
 atomic(B/'completion.json',dict(counts=dict(totals),candidates=len(rows),elapsed_seconds=time.time()-started,training_enabled=False,protocol_sha256=sha(B/'protocol.json'),job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='label':run_labels()
 else:main()
