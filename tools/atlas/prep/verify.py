import os,json,math,hashlib,time,io,contextlib,concurrent.futures,gzip,tarfile
from pathlib import Path
R=Path(os.environ['PKABENCH_RUNTIME']);O=R/'pretraining/atlas-prepared-v1';D=O/'pilot';A=R/'audits/atlas-prep-v1'
assert os.environ.get('SLURM_JOB_ID')
prep=json.loads((O/'prep-288.json').read_text());lab=json.loads((O/'label-288.json').read_text());stage=json.loads((O/'stage.json').read_text())
assert len(lab['results'])==285 and all(r['status']=='labelled' for r in lab['results'])
count=0
for r in lab['results']:
 d=D/r['id'];l=json.loads((d/'labels.json').read_text());assert not l['errors'] and l['termini']
 for m in ('pkai','pkai_plus'):
  for state in ('AB','A','B'):
   rows=l[m][state];keys=[tuple(v[:4]) for v in rows];assert len(keys)==len(set(keys))
   assert all(math.isfinite(v[4]) for v in rows);count+=len(rows)
# Compare batched outputs with the installed public predictor on two complete complexes/all states/models.
import sys
sys.path.insert(0,str(R/'envs/pkai/lib/python3.11/site-packages/pkai'))
from pKAI import pKAI
checks=0;maxdiff=0
for r in lab['results'][:2]:
 d=D/r['id'];l=json.loads((d/'labels.json').read_text())
 for m,name in [('pkai','pKAI'),('pkai_plus','pKAI+')]:
  for state in ('AB','A','B'):
   with contextlib.redirect_stdout(io.StringIO()):native=list(pKAI(str(d/f'{state}-termini.pdb'),model_name=name,device='cpu',threads=1))
   actual=l[m][state];assert len(native)==len(actual)
   for ref,pred in zip(native,actual):
    diff=abs(float(ref[3])-float(pred[4]));assert diff<=.00501,(ref,pred)
    maxdiff=max(maxdiff,diff);checks+=1
# Benchmark safe archival: each output entry becomes one tar member containing a gzipped tar.
def pack(r):
 d=D/r['id']; buf=io.BytesIO()
 names=['AB.cif','A.cif','B.cif','student_AB.cif','input_atom_mask.json','sites.json','labels.json','meta.json','provenance.json','conformers.json']
 with tarfile.open(fileobj=buf,mode='w') as tf:
  for name in names:tf.add(d/name,arcname=name)
 return r['id'],gzip.compress(buf.getvalue(),compresslevel=3)
t=time.monotonic();packed=O/'pilot-entries.tar';total=0
with tarfile.open(packed,'w') as tf,concurrent.futures.ThreadPoolExecutor(max_workers=72) as ex:
 for cid,b in ex.map(pack,lab['results']):
  ti=tarfile.TarInfo(cid+'.tar.gz');ti.size=len(b);tf.addfile(ti,io.BytesIO(b));total+=len(b)
packseconds=time.monotonic()-t
with tarfile.open(packed) as tf:assert len(tf.getmembers())==len(lab['results'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
n=stage['candidates'];scale=n/288
report=dict(stage=stage,pilot_attempted=288,prepared=285,labelled=285,rejected=3,rejection_reason='interpartner_disulfide',label_errors=0,finite_unique_label_rows=count,native_predictor_checked_values=checks,native_predictor_max_difference_before_rounding=maxdiff,
 workers=72,allocated_nodes=1,allocated_gpus=1,execution='72 single-thread CPU workers; persistent models; feature reuse for both teachers; prediction batches of 256',
 timing_seconds=dict(preparation=prep['wall_seconds'],labelling=lab['wall_seconds'],packing=packseconds),
 extrapolated_hours=dict(preparation=prep['wall_seconds']*scale/3600,labelling=lab['wall_seconds']*scale/3600,packing=packseconds*scale/3600,total=(prep['wall_seconds']+lab['wall_seconds']+packseconds)*scale/3600),
 packed_pilot_bytes=total,projected_packed_bytes=round(total*scale),training_enabled=False,bulk_labelling_started=False,
 limitations=['288-complex pilot; not a guarantee of sustained filesystem throughput','Queue time excluded','9 antibody-like entries remain quarantined; final CDR adjudication pending','Final training graph generation and additional confidence-site policy are outside this preparation/label ETA'],
 source_hashes={name:sha(O/name) for name in ['candidates.json','decisions.json','heldout.fasta','antibody-domains.fasta','benchmark-sample.json']},code_hashes={p.name:sha(p) for p in A.glob('*.py')},model_hashes={p.name:sha(p) for p in (R/'envs/pkai/lib/python3.11/site-packages/pkai/models').glob('*.pt')})
(O/'benchmark-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
