import os,json,time,hashlib,collections,multiprocessing,concurrent.futures,functools
from pathlib import Path
import numpy as np
import parasail
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.optimize import linear_sum_assignment
R=Path(os.environ['PKABENCH_RUNTIME']);O=R/'audits/atlas-alignment-v1'
NWORKERS=72
AA=set('ARNDCQEGHILKMFPSTWYVBZX*')
def save(name,data):
 p=O/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def score(a,b):
 result=parasail.sw_scan_16(a,b,11,1,parasail.blosum62)
 if result.saturated:result=parasail.sw_scan_32(a,b,11,1,parasail.blosum62)
 assert not result.saturated
 return result.score

def chain_rows(rows):
 for i in rows:
  marker=O/f'{VIEW}-rows' / str(i)
  if marker.exists():continue
  profile=parasail.profile_create_16(SEQS[i],parasail.blosum62)
  row=np.zeros(len(SEQS)-i,dtype=np.int32);row[0]=SELF[i]
  for offset,j in enumerate(range(i+1,len(SEQS)),1):
   result=parasail.sw_scan_profile_16(profile,SEQS[j],11,1)
   if result.saturated:result=parasail.sw_scan_32(SEQS[i],SEQS[j],11,1,parasail.blosum62)
   assert not result.saturated
   row[offset]=result.score
  SCORES[i,i:]=row;SCORES[i:,i]=row
  # Flush data before checkpoint; only disjoint upper-triangle pairs are written.
  SCORES.flush();marker.touch()
 return len(rows)

def complex_rows(rows):
 for i in rows:
  aa=CHAINS[i];out=np.empty(len(CHAINS)-i,dtype=np.float32)
  for count,j in enumerate(range(i,len(CHAINS))):
   bb=CHAINS[j]
   if len(aa)==2 and len(bb)==2:
    s=max(int(SCORES[aa[0],bb[0]])+int(SCORES[aa[1],bb[1]]),int(SCORES[aa[0],bb[1]])+int(SCORES[aa[1],bb[0]]))
   elif len(aa)==1:s=int(SCORES[aa[0],bb].max())
   elif len(bb)==1:s=int(SCORES[aa,bb[0]].max())
   else:
    scores=SCORES[np.ix_(aa,bb)];x,y=linear_sum_assignment(scores,maximize=True);s=int(scores[x,y].sum())
   d=1-2*s/(DENOM[i]+DENOM[j]);assert -1e-7<=d<=1+1e-7,(i,j,d)
   out[count]=max(d,0)
  DIST[i,i:]=out;DIST[i:,i]=out
 return len(rows)

def pilot(batch):
 st=time.time();cells=0
 for i,j in batch:score(SEQS[i],SEQS[j]);cells+=len(SEQS[i])*len(SEQS[j])
 return len(batch),cells,time.time()-st

def main():
 global VIEW,SEQS,SELF,SCORES,CHAINS,DENOM,DIST
 assert os.environ.get('SLURM_JOB_ID')
 points=json.loads((O/'sequences.json').read_text());old=json.loads((R/'audits/atlas-split-v1/points.json').read_text())
 assert [(p['id'],p['group']) for p in points]==[(p['id'],p['group']) for p in old]
 assert [p['interface_residues'] for p in points]==[p['interface_residues'] for p in old]
 # Independent reference scores verify the actual SIMD backend, gap semantics, symmetry and saturation handling.
 ref=PairwiseAligner(mode='local',substitution_matrix=substitution_matrices.load('BLOSUM62'),open_gap_score=-11,extend_gap_score=-1)
 tests=[('AAAAWWWW','AAAAWWWW'),('ACDEFGHIKLMNPQRSTVWY','ACDEYGHIKLMNPQRSTVWY'),('AAAAWWWW','AAAAQQQQWWWW'),('WWWW','AAAA'),('ACDEFGHIK','ACDEFGHIK'*30)]
 rng=np.random.default_rng(20261011)
 for _ in range(30):
  a=''.join(rng.choice(list('ARNDCQEGHILKMFPSTWYV'),size=int(rng.integers(10,200))))
  b=''.join(rng.choice(list('ARNDCQEGHILKMFPSTWYV'),size=int(rng.integers(10,200))))
  tests.append((a,b))
 for a,b in tests:assert score(a,b)==score(b,a)==int(ref.score(a,b))
 save('alignment-tests.json',dict(passed=True,pairs=len(tests),reference='Bio.Align.PairwiseAligner local BLOSUM62 gap open -11 extend -1'))
 summary={}
 for VIEW in ('whole','interface'):
  beg=time.time();indices=[i for i,p in enumerate(points) if p[VIEW+'_sequences']]
  seqsets=[[ ''.join(c if c in AA else 'X' for c in s.upper()) for s in points[i][VIEW+'_sequences']] for i in indices]
  unique=sorted(set(s for ss in seqsets for s in ss));selfscores={s:score(s,s) for s in unique}
  uninformative=[dict(point_index=i,id=points[i]['id'],group=points[i]['group'],chain_index=c,length=len(s),alphabet=''.join(sorted(set(s)))) for i,ss in zip(indices,seqsets) for c,s in enumerate(ss) if selfscores[s]<=0]
  save(f'{VIEW}-uninformative-chains.json',uninformative)
  seqsets=[[s for s in ss if selfscores[s]>0] for ss in seqsets]
  assert all(seqsets),'A complex has no informative chains; review before plotting'
  SEQS=sorted(set(s for ss in seqsets for s in ss));lookup={s:i for i,s in enumerate(SEQS)}
  CHAINS=[np.array([lookup[s] for s in ss],dtype=int) for ss in seqsets];SELF=np.array([selfscores[s] for s in SEQS],dtype=np.int64)
  print(VIEW,'uninformative chains excluded',len(uninformative),flush=True)
  DENOM=np.array([SELF[ix].sum() for ix in CHAINS],dtype=np.int64)
  save(f'{VIEW}-index.json',dict(point_indices=indices,unique_sequences=len(SEQS),sequence_sha256=hashlib.sha256(json.dumps(SEQS).encode()).hexdigest(),chain_count_distribution=dict(collections.Counter(map(len,CHAINS)))))
  n=len(SEQS);npairs=n*(n-1)//2
  scorepath=O/f'{VIEW}-chain-scores.npy'
  SCORES=np.lib.format.open_memmap(scorepath,mode='r+' if scorepath.exists() else 'w+',dtype=np.int32,shape=(n,n))
  (O/f'{VIEW}-rows').mkdir(exist_ok=True)
  print(VIEW,'complexes',len(indices),'unique chains',n,'pairwise alignments',npairs,flush=True)
  rng=np.random.default_rng(20261011);pairs=rng.integers(0,n,size=(7200,2)).tolist()
  with concurrent.futures.ProcessPoolExecutor(max_workers=NWORKERS,mp_context=multiprocessing.get_context('fork')) as ex:
   st=time.time();samples=list(ex.map(pilot,[pairs[i:i+100] for i in range(0,len(pairs),100)]));wall=time.time()-st
   lengths=np.array(list(map(len,SEQS)),dtype=np.float64);totalcells=(lengths.sum()**2-(lengths**2).sum())/2
   rate=sum(r[1] for r in samples)/wall;eta=totalcells/rate
   save(f'{VIEW}-pilot.json',dict(pairs=len(pairs),wall_seconds=wall,estimated_alignment_seconds=eta,estimated_total_cells=totalcells,cells_per_second=rate,workers=NWORKERS))
   print(VIEW,'PILOT alignment ETA seconds',round(eta),flush=True)
   todo=[i for i in range(n) if not (O/f'{VIEW}-rows'/str(i)).exists()];done=n-len(todo);last=time.time()
   futures=[ex.submit(chain_rows,todo[i:i+4]) for i in range(0,len(todo),4)]
   for fut in concurrent.futures.as_completed(futures):
    done+=fut.result()
    if time.time()-last>=30 or done==n:
     save('progress.json',dict(stage='chain alignments',view=VIEW,rows_done=done,rows_total=n,elapsed_seconds=time.time()-beg,job_id=os.environ['SLURM_JOB_ID']))
     print(VIEW,'chain rows',done,'/',n,'seconds',round(time.time()-beg),flush=True);last=time.time()
  assert np.array_equal(np.diag(SCORES),SELF)
  # Exhaustive block-wise symmetry and upper-bound tests, without copying the full matrix.
  for a in range(0,n,256):
   b=min(a+256,n);block=SCORES[a:b,:]
   assert np.array_equal(block,SCORES[:,a:b].T)
   assert (block>=0).all() and (2*block<=SELF[a:b,None]+SELF[None,:]).all()
  nc=len(indices);DIST=np.lib.format.open_memmap(O/f'{VIEW}-distances.npy',mode='w+',dtype=np.float32,shape=(nc,nc))
  print(VIEW,'assembling complex distances',flush=True)
  with concurrent.futures.ProcessPoolExecutor(max_workers=NWORKERS,mp_context=multiprocessing.get_context('fork')) as ex:
   done=0
   for count in ex.map(complex_rows,[list(range(i,min(i+8,nc))) for i in range(0,nc,8)]):done+=count
  DIST.flush();assert np.allclose(np.diag(DIST),0)
  for a in range(0,nc,256):assert np.array_equal(DIST[a:a+256,:],DIST[:,a:a+256].T)
  assert np.isfinite(DIST).all() and DIST.min()>=0 and DIST.max()<=1
  summary[VIEW]=dict(points=nc,unique_chains=n,chain_pairs=npairs,seconds=time.time()-beg,point_indices=indices,distance_quantiles=np.quantile(DIST[np.triu_indices(nc,1)],[0,.01,.1,.5,.9,.99,1]).tolist())
  save('alignment-summary.json',summary);print(VIEW,'DISTANCES COMPLETE',summary[VIEW]['seconds'],flush=True)
 save('progress.json',dict(stage='alignment distances complete',job_id=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
