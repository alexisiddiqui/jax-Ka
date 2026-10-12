import os,json,time,hashlib,csv,concurrent.futures,multiprocessing
from pathlib import Path
import numpy as np
import phate
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix,issparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from Bio.Align import PairwiseAligner,substitution_matrices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(os.environ['PKABENCH_RUNTIME']);A=R/'audits/atlas-alignment-v1';O=R/'audits/atlas-phate-check-v1';F=R/'audits/atlas-split-v1'
PARAMS=dict(knn=15,decay=40,n_landmark=500,t='auto',gamma=1,mds='metric',mds_solver='smacof',n_jobs=72,random_state=20261011,verbose=0)
COLORS={'ATLAS':'#0072B2','PINDER':'#009E73','pKPDB':'#D55E00','Benchmark':'#882E72'}
def write(name,x): (O/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def quant(x):return dict(zip(['min','p01','p10','median','p90','p99','max'],map(float,np.quantile(x,[0,.01,.1,.5,.9,.99,1]))))
def corr(a,b):
 r=spearmanr(a,b).statistic;return float(r) if np.isfinite(r) else None

def inspect_pair(task):
 view,q,t,kind,dist=task
 mat=substitution_matrices.load('BLOSUM62');aligner=PairwiseAligner(mode='local',substitution_matrix=mat,open_gap_score=-11,extend_gap_score=-1)
 clean=lambda s:''.join(c if c in 'ARNDCQEGHILKMFPSTWYVBZX*' else 'X' for c in s.upper())
 aa=[clean(s) for s in POINTS[q][view+'_sequences']];bb=[clean(s) for s in POINTS[t][view+'_sequences']]
 scores=np.array([[aligner.score(a,b) for b in bb] for a in aa]);rows,cols=linear_sum_assignment(scores,maximize=True)
 cq=ct=ident=columns=0;details=[]
 for a,b in zip(rows,cols):
  if scores[a,b]<=0:continue
  al=aligner.align(aa[a],bb[b])[0];coords=np.asarray(al.coordinates);match=width=0
  for k in range(coords.shape[1]-1):
   x0,y0=coords[:,k];x1,y1=coords[:,k+1];dx,dy=x1-x0,y1-y0;width+=max(dx,dy)
   if dx and dy:
    assert dx==dy
    match+=sum(x==y for x,y in zip(aa[a][x0:x1],bb[b][y0:y1]))
  qcovered=int(coords[0,-1]-coords[0,0]);tcovered=int(coords[1,-1]-coords[1,0]);cq+=qcovered;ct+=tcovered;ident+=match;columns+=width
  details.append(dict(query_chain=int(a),target_chain=int(b),score=float(scores[a,b]),identity=match/width,query_coverage=qcovered/len(aa[a]),target_coverage=tcovered/len(bb[b])))
 return dict(view=view,kind=kind,query_id=POINTS[q]['id'],query_group=POINTS[q]['group'],target_id=POINTS[t]['id'],target_group=POINTS[t]['group'],distance=float(dist),identity=ident/columns if columns else 0,query_coverage=cq/sum(map(len,aa)),target_coverage=ct/sum(map(len,bb)),query_chains=len(aa),target_chains=len(bb),matched_chain_count=len(details),chains=details)

def main():
 global POINTS
 assert os.environ.get('SLURM_JOB_ID');O.mkdir(exist_ok=True)
 POINTS=json.loads((A/'sequences.json').read_text());feat=np.load(F/'features.npz');oldmeta=json.loads((A/'alignment-provenance.json').read_text())
 public=[{k:p[k] for k in ('id','dataset','group','split','residues','chains','interface_residues')} for p in POINTS]
 results={'scope':'Diagnostics of the plotted sample only; nearest-training matches are not an exhaustive full-training search or a new leakage audit.','parameters':PARAMS,'pca_note':'PCA50/no-PCA is a controlled ablation of the original composition features with cosine distance. PCA50 on alignment-distance rows is a separate distance-profile representation, with Euclidean distance after PCA; it is not precomputed alignment geometry.','views':{}};tasks=[]
 for view in ('whole','interface'):
  start=time.time();m=json.loads((A/f'{view}-duplicate-map.json').read_text());indices=np.array(m['point_indices']);reps=np.array(m['representative_local_indices']);labels=np.array(m['group_per_local_point']);fullD=np.load(A/f'{view}-distances.npy',mmap_mode='r');D=np.asarray(fullD[np.ix_(reps,reps)]);pp=[POINTS[i] for i in indices[reps]];n=len(pp)
  assert len(indices)==len(feat[view]);X=feat[view][reps];rng=np.random.default_rng(20261011)
  sample=rng.choice(n,size=min(1024,n),replace=False);reference=np.array(D[sample],copy=True);reference[np.arange(len(sample)),sample]=np.inf
  near=np.argpartition(reference,14,axis=1)[:,:15]
  lens=np.array([sum(map(len,p[view+'_sequences'])) for p in pp]);chains=np.array([len(p[view+'_sequences']) for p in pp]);ii=rng.integers(n,size=100000);jj=rng.integers(n,size=100000);keep=ii!=jj;ii,jj=ii[keep],jj[keep];pairdist=D[ii,jj]
  vd={'unique_profiles':n,'points':len(indices),'random_pair_distance':quant(pairdist),'distance_spearman_log_length_ratio':corr(pairdist,np.abs(np.log(lens[ii]/lens[jj]))),'distance_spearman_chain_count_difference':corr(pairdist,np.abs(chains[ii]-chains[jj])),'fits':{}}
  # Inspect one neighbour in another dataset and nearest available plotted training entry.
  for group in sorted(set(p['group'] for p in pp)):
   candidates=np.array([i for i,p in enumerate(pp) if p['group']==group]);chosen=rng.choice(candidates,size=min(80,len(candidates)),replace=False)
   for q in chosen:
    for kind in ('cross_dataset','training'):
     mask=np.array([p['dataset']!=pp[q]['dataset'] if kind=='cross_dataset' else p['split']=='train' for p in pp]);mask[q]=False
     if not mask.any():continue
     d=np.where(mask,D[q],np.inf);target=int(np.argmin(d));tasks.append((view,int(indices[reps[q]]),int(indices[reps[target]]),kind,float(d[target])))
  for name in ('alignment','composition_pca50','composition_no_pca','profile_pca50'):
   beg=time.time();settings=PARAMS.copy()
   if name=='alignment':data=D;settings.update(n_pca=None,knn_dist='precomputed_distance')
   elif name=='profile_pca50':
    with threadpool_limits(limits=8):
     pca=PCA(n_components=50,svd_solver='randomized',iterated_power=3,random_state=20261011);data=pca.fit_transform(D)
    vd['profile_pca50_explained_variance']=float(pca.explained_variance_ratio_.sum());settings.update(n_pca=None,knn_dist='euclidean')
   else:data=X;settings.update(n_pca=50 if name.endswith('pca50') else None,knn_dist='cosine')
   model=phate.PHATE(**settings);xy_unique=model.fit_transform(data);assert np.isfinite(xy_unique).all()
   if name=='alignment':
    oldxy=np.load(A/f'{view}-k15-deduplicated-smacof.npy');assert np.allclose(xy_unique[labels],oldxy,atol=1e-6);xy=oldxy
   else:xy=xy_unique[labels]
   np.save(O/f'{view}-{name}.npy',xy)
   for i,coord in zip(indices,xy):public[i][view+'-'+name]=coord.round(7).tolist()
   ed=cdist(xy_unique[sample],xy_unique);ed[np.arange(len(sample)),sample]=np.inf;kn=np.argpartition(ed,14,axis=1)[:,:15];recall=np.array([len(set(a)&set(b))/15 for a,b in zip(near,kn)])
   # Compare a map with its own input neighbourhoods as well as the alignment baseline.
   if name=='alignment':own=reference
   else:
    reduced=model.graph.data_nu if hasattr(model.graph,'data_nu') else data
    own=cdist(np.asarray(reduced)[sample],np.asarray(reduced),metric=settings['knn_dist']);own[np.arange(len(sample)),sample]=np.inf
   ownkn=np.argpartition(own,14,axis=1)[:,:15];ownrecall=np.array([len(set(a)&set(b))/15 for a,b in zip(ownkn,kn)])
   K=model.graph.kernel;K=K.toarray() if issparse(K) else np.asarray(K);K=np.array(K,copy=True);np.fill_diagonal(K,0);components=connected_components(csr_matrix(K>0),directed=False)[0];weight=K.sum(axis=1);effective=weight**2/np.maximum((K*K).sum(axis=1),1e-300)
   radial=np.linalg.norm(xy_unique-np.median(xy_unique,axis=0),axis=1)
   vd['fits'][name]={'t':int(model.optimal_t),'mean_alignment_neighbor_recall15':float(recall.mean()),'mean_own_input_neighbor_recall15':float(ownrecall.mean()),'graph_components':int(components),'effective_neighbors':quant(effective),'median_radius_over_max':float(np.median(radial)/max(radial.max(),1e-12)),'radius':quant(radial),'axis_length_spearman':[corr(xy_unique[:,c],lens) for c in (0,1)],'axis_chain_count_spearman':[corr(xy_unique[:,c],chains) for c in (0,1)],'seconds':time.time()-beg}
   del model,K
   print(view,name,vd['fits'][name],flush=True)
  vd['seconds']=time.time()-start;results['views'][view]=vd;write('diagnostics.json',results)
 print('NEIGHBOUR ALIGNMENTS',len(tasks),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=72,mp_context=multiprocessing.get_context('fork')) as ex:matches=list(ex.map(inspect_pair,tasks,chunksize=8))
 write('nearest-matches.json',matches)
 with (O/'nearest-matches.csv').open('w') as f:
  keys=[k for k in matches[0] if k!='chains'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in matches)
 results['nearest_match_summary']={}
 for view in ('whole','interface'):
  for kind in ('cross_dataset','training'):
   rr=[r for r in matches if r['view']==view and r['kind']==kind];results['nearest_match_summary'][view+'-'+kind]={'pairs':len(rr),'median_identity':float(np.median([r['identity'] for r in rr])),'median_query_coverage':float(np.median([r['query_coverage'] for r in rr])),'median_target_coverage':float(np.median([r['target_coverage'] for r in rr])),'fraction_both_coverage_ge80':float(np.mean([min(r['query_coverage'],r['target_coverage'])>=.8 for r in rr]))}
 results['validation_by_group']={}
 for view in ('whole','interface'):
  for group in sorted(set(r['query_group'] for r in matches if r['query_group'].endswith(' val'))):
   rr=[r for r in matches if r['view']==view and r['kind']=='training' and r['query_group']==group]
   if rr:results['validation_by_group'][view+' / '+group]={'n':len(rr),'median_identity':float(np.median([r['identity'] for r in rr])),'median_min_coverage':float(np.median([min(r['query_coverage'],r['target_coverage']) for r in rr])),'median_distance':float(np.median([r['distance'] for r in rr]))}
 results['input_hashes']={name:hashlib.sha256((A/name).read_bytes()).hexdigest() for name in ('alignment-provenance.json','whole-index.json','interface-index.json')};results['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 write('diagnostics.json',results);write('comparison-points.json',public)
 fig,axes=plt.subplots(2,2,figsize=(12,10))
 for row,view in enumerate(('whole','interface')):
  for ds,color in COLORS.items():
   rr=[r for r in matches if r['view']==view and r['kind']=='training' and r['query_group'].endswith(' val') and r['query_group'].startswith(ds+' ')]
   if rr:
    axes[row,0].scatter([r['identity'] for r in rr],[min(r['query_coverage'],r['target_coverage']) for r in rr],s=18,alpha=.5,color=color,label=ds)
    vals=sorted(r['distance'] for r in rr);axes[row,1].plot(vals,np.arange(1,len(vals)+1)/len(vals),color=color,label=ds)
  axes[row,0].set(title=view+' · validation vs nearest plotted training',xlabel='Alignment identity (matches / alignment columns)',ylabel='Minimum query / target coverage',xlim=(0,1),ylim=(0,1));axes[row,1].set(title=view+' · nearest-training distance',xlabel='Normalized alignment distance',ylabel='Cumulative fraction',xlim=(0,1),ylim=(0,1));axes[row,0].legend();axes[row,1].legend()
 fig.suptitle('Neighbour audit of the plotted sample — not the full training pool',fontsize=14);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(O/'nearest-training-audit.png',dpi=180);fig.savefig(O/'nearest-training-audit.pdf');plt.close(fig)
 print('ALL CHECKS COMPLETE',flush=True)
if __name__=='__main__':main()
