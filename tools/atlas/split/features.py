import os,json,hashlib,collections,concurrent.futures,multiprocessing,io,gzip,tarfile,time
from pathlib import Path
import numpy as np
from Bio.Align import substitution_matrices
from biotite.sequence import ProteinSequence
R=Path(os.environ['PKABENCH_RUNTIME']);A=R/'audits/atlas-v1';O=R/'audits/atlas-split-v1';S=R/'pretraining/atlas-prepared-v1'
AA='ARNDCQEGHILKMFPSTWYV';AI={c:i for i,c in enumerate(AA)};MAT=substitution_matrices.load('BLOSUM62');B=np.array([[MAT[a,b] for b in AA] for a in AA],dtype=float)
def read(p):return json.loads(p.read_text())
def fa(p):
 d={};k=None
 for l in p.read_text().splitlines():
  if l.startswith('>'):k=l[1:];d[k]=''
  elif k:d[k]+=l.strip()
 return d
def source(rel):
 p=R/rel
 return p if p.exists() else R/'overlay'/rel
def seqfeature(seq,mask=None):
 codes=np.array([AI.get(x,-1) for x in seq]);valid=codes>=0
 if mask is not None:valid&=np.asarray(mask)
 if not valid.any():return None
 f=np.bincount(codes[valid],minlength=20).astype(float);f/=f.sum();v=f@B;v/=max(np.linalg.norm(v),1e-12)
 pairs=valid[:-1]&valid[1:];c=np.zeros((20,20))
 if pairs.any():np.add.at(c,(codes[:-1][pairs],codes[1:][pairs]),1);c/=c.sum()
 d=(B.T@c@B).ravel();d/=max(np.linalg.norm(d),1e-12)
 return np.r_[v,d]
def combine(features):
 features=[f for f in features if f is not None]
 if not features:return None
 a=np.array(features);v=np.r_[a.mean(axis=0),a.std(axis=0)];return (v/max(np.linalg.norm(v),1e-12)).astype(np.float32)
def task(r):
 from biotite.structure.io import pdbx
 import biotite.structure as st
 from scipy.spatial import cKDTree
 whole=combine([seqfeature(s) for s in r['sequences']]);assert whole is not None
 meta={k:v for k,v in r.items() if k not in ('sequences','archive','partners','chain_names')};meta['residues']=sum(map(len,r['sequences']));meta['chains']=len(r['sequences']);meta['interface_residues']=0
 try:
  if r['dataset']=='ATLAS':
   ix=r['archive']
   with Path(ix['shard']).open('rb') as f:f.seek(ix['offset']);raw=f.read(ix['bytes'])
   assert hashlib.sha256(raw).hexdigest()==ix['sha256']
   with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as tf:text=tf.extractfile('student_AB.cif').read().decode()
  elif r['dataset']=='PINDER':
   p=source(f'pretraining/pinder-pkai-v1/entries/{r["id"]}/student_AB.cif.gz')
   if not p.exists():p=source(f'pretraining/pinder-pkai-v1/entries/{r["id"]}/AB.cif.gz')
   text=gzip.decompress(p.read_bytes()).decode()
  elif r['dataset']=='pKPDB':
   if len(r['sequences'])<2:raise ValueError('monomer: no interchain interface')
   p=source(f'pretraining/pkpdb-v1/structures/{r["id"][1:3]}/{r["id"]}.cif.gz');text=gzip.decompress(p.read_bytes()).decode()
   defects=read(source(f'pretraining/pkpdb-full-v1/entries/{r["id"]}/defects.json'));r['chain_names']=[x['chain'] for x in defects['sequences']]
  else:
   p=R/'training/shared-v4-float32/records'/f'{r["id"]}.json'
   if p.exists():
    record=read(p);rel=record['structures']['AB'].split('/pkabench/',1)[-1];p=source(rel)
   else:
    candidates=[source(f'campaigns/{campaign}/structures/{r["id"]}/AB.cif') for campaign in ('production-nojax-v1','production-1024-v1','frozen-smoke-v1')]
    p=next((q for q in candidates if q.exists()),None)
    if p is None:raise ValueError('benchmark coordinates unavailable in local prepared campaigns')
   r['chain_names']=[c for side in r['partners'] for c in side];meta['coordinate_source']=str(p.relative_to(R));text=p.read_text()
  cf=pdbx.CIFFile.read(io.StringIO(text));atoms=pdbx.get_structure(cf,model=1,altloc='occupancy',use_author_fields=False)
  keep=~np.isin(np.char.upper(atoms.element),['H','D']);names=set(ProteinSequence.convert_letter_1to3(c) for c in AA);keep&=np.isin(atoms.res_name,list(names))
  if r.get('chain_names'):keep&=np.isin(atoms.chain_id,r['chain_names'])
  atoms=atoms[keep];chains=list(dict.fromkeys(map(str,atoms.chain_id)));assert len(chains)>1,'no two protein chains'
  features=[];counts=[]
  for c in chains:
   sub=atoms[atoms.chain_id==c];other=atoms[atoms.chain_id!=c]
   if r.get('partners'):
    side=next((ss for ss in r['partners'] if c in ss),None)
    if side:other=atoms[~np.isin(atoms.chain_id,side)]
   if not len(other):continue
   dist=cKDTree(other.coord).query(sub.coord)[0];starts=st.get_residue_starts(sub,add_exclusive_stop=True)
   seq=''.join(ProteinSequence.convert_letter_3to1(str(sub.res_name[a])) for a in starts[:-1]);mask=[bool(dist[a:b].min()<=10.) for a,b in zip(starts[:-1],starts[1:])]
   # Discontinuous interface residues never become artificial neighbours: retain positional masking.
   # Also break adjacency across missing author residue positions.
   splitseq=[];splitmask=[]
   for j,(a,b) in enumerate(zip(starts[:-1],starts[1:])):
    if j and int(sub.res_id[a])-int(sub.res_id[starts[j-1]])!=1:splitseq.append('X');splitmask.append(False)
    splitseq.append(seq[j]);splitmask.append(mask[j])
   features.append(seqfeature(''.join(splitseq),splitmask));counts.append(sum(mask))
  inter=combine(features)
  if inter is None:raise ValueError('no interface residues within 10 A')
  meta['interface_residues']=sum(counts);meta['interface_status']='available';return meta,whole,inter
 except Exception as e:meta['interface_status']=str(e);return meta,whole,None
if __name__=='__main__':
 assert os.environ.get('SLURM_JOB_ID')
 seq={**fa(A/'atlas.fasta'),**fa(A/'references.fasta')};refs=read(A/'reference_membership.json');manifest=read(S/'split-v1/manifest.json');bm={r['complex_id']:r for r in read(A/'references/benchmark-index.json')['candidates']};tasks=[];counts={}
 def choose(group,rows,cap):
  rows=sorted(rows,key=lambda r:hashlib.sha256(('phate-20261011|'+r['id']).encode()).hexdigest());counts[group]=dict(population=len(rows),sample=min(cap,len(rows)));tasks.extend(rows[:cap])
 for sp,cap in [('train',1800),('val',2000)]:
  rr=[dict(id=r['id'],dataset='ATLAS',split=sp,group='ATLAS '+sp,sequences=[seq[k] for k in r['chains']],archive=r['archive'],source_class=r['class'],pLDDT=float(r['pLDDT']),component=r['sequence_component']) for r in manifest if r['split']==sp];choose('ATLAS '+sp,rr,cap)
 for dataset,sp,key,cap in [('PINDER','train','PINDER_train',1000),('PINDER','val','PINDER_val',800),('pKPDB','train','pKPDB_train',1200),('pKPDB','val','pKPDB_val',800),('Benchmark','train','benchmark_train',600),('Benchmark','val','benchmark_val',284),('Benchmark','test','benchmark_test',600)]:
  rr=[]
  for cid,keys in refs[key].items():
   row=dict(id=cid,dataset=dataset,split=sp,group=dataset+' '+sp,sequences=[seq[k] for k in keys])
   if dataset=='Benchmark':row['partners']=[bm[cid]['partner_A_chains'],bm[cid]['partner_B_chains']];row['active_validation']=cid in refs['benchmark_active_val142']
   rr.append(row)
  choose(dataset+' '+sp,rr,cap)
 # Include every active benchmark validation record even if frozen membership representation changes.
 assert set(refs['benchmark_active_val142']) <= {r['id'] for r in tasks if r['dataset']=='Benchmark' and r['split']=='val'}
 (O/'sampling.json').write_text(json.dumps(counts,indent=2));print('FEATURE TASKS',len(tasks),flush=True)
 metas=[];whole=[];inter=[];indices=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=72,mp_context=multiprocessing.get_context('spawn')) as ex:
  for i,(meta,w,f) in enumerate(ex.map(task,tasks,chunksize=4)):
   metas.append(meta);whole.append(w)
   if f is not None:indices.append(i);inter.append(f)
   if i%500==0:print('features',i,flush=True)
 np.savez_compressed(O/'features.npz',whole=np.array(whole),interface=np.array(inter),interface_indices=np.array(indices))
 (O/'points.json').write_text(json.dumps(metas));(O/'feature-summary.json').write_text(json.dumps(dict(whole=len(whole),interface=len(inter),interface_status=dict(collections.Counter(m['interface_status'] for m in metas)),by_group={g:dict(whole=sum(m['group']==g for m in metas),interface=sum(m['group']==g and m['interface_status']=='available' for m in metas)) for g in counts},method='BLOSUM62-weighted amino-acid frequencies plus adjacent-pair frequencies; each block L2 normalized; complex = mean and SD across chains, L2 normalized',interface_definition='observed canonical heavy-atom distance <=10 A across partners; any interchain contact for pKPDB; no dipeptide links across excluded residues or numbering gaps'),indent=2))
