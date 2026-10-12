import os,json,time,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import phate
import parasail
from Bio.Align import PairwiseAligner, substitution_matrices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
R=Path(os.environ['PKABENCH_RUNTIME']);O=R/'audits/atlas-alignment-v1'
assert os.environ.get('SLURM_JOB_ID')
points=json.loads((O/'sequences.json').read_text())
public=[{k:v for k,v in p.items() if not k.endswith('_sequences')} for p in points]
colors={'ATLAS':'#0072B2','PINDER':'#009E73','pKPDB':'#D55E00','Benchmark':'#882E72'}
params=dict(decay=40,n_landmark=500,t='auto',gamma=1,n_pca=None,knn_dist='precomputed_distance',mds='metric',mds_solver='smacof',n_jobs=72,random_state=20261011)
meta={'parameters':params,'fits':{},'method':'Smith-Waterman BLOSUM62 gap-open 11, extension 1; maximum-score one-to-one chain assignment; d=1-2*matched_score/(sum_self_scores_A+sum_self_scores_B). Unmatched chains contribute no score but remain in denominator.','interface':'Contact residues within 10 A heavy-atom distance, in original chain order; non-contact positions omitted. This is an ordered contact-subsequence alignment, not a full-chain alignment masked after traceback.','versions':{p:importlib.metadata.version(p) for p in ('phate','parasail','numpy','scipy')},'sources':['https://github.com/jeffdaily/parasail','https://phate.readthedocs.io/en/stable/'],'code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*.py')}}
meta['uninformative_chains']={view:json.loads((O/f'{view}-uninformative-chains.json').read_text()) for view in ('whole','interface')}
plots=[];fig,axes=plt.subplots(2,4,figsize=(18,9))
for row,view in enumerate(('whole','interface')):
 index=json.loads((O/f'{view}-index.json').read_text())['point_indices'];D=np.load(O/f'{view}-distances.npy');assert np.isfinite(D).all()
 alphabet=set('ARNDCQEGHILKMFPSTWYVBZX*')
 unique=sorted(set(''.join(c if c in alphabet else 'X' for c in seq.upper()) for i in index for seq in points[i][view+'_sequences']))
 reference=PairwiseAligner(mode='local',substitution_matrix=substitution_matrices.load('BLOSUM62'),open_gap_score=-11,extend_gap_score=-1)
 unique=[seq for seq in unique if reference.score(seq,seq)>0]
 scores=np.load(O/f'{view}-chain-scores.npy',mmap_mode='r');assert len(unique)==scores.shape[0]
 rng=np.random.default_rng(20261011)
 for a,b in rng.integers(0,len(unique),size=(128,2)):
  assert int(scores[a,b])==int(reference.score(unique[a],unique[b])),'SIMD profile score differs from independent reference'
 meta.setdefault('profile_validation',{})[view]={'passed':True,'reference_pairs':128}
 print(view,'128 stored profile scores independently validated',flush=True)
 n_unique,labels=connected_components(csr_matrix(D==0),directed=False)
 representatives=np.array([np.flatnonzero(labels==i)[0] for i in range(n_unique)])
 for group,representative in enumerate(representatives):
  members=np.flatnonzero(labels==group)
  assert np.allclose(D[members],D[representative],atol=1e-7,rtol=0),'Zero-distance group has unequal profiles; cannot collapse safely'
 fitD=D[np.ix_(representatives,representatives)]
 assert np.all(fitD[np.triu_indices(n_unique,1)]>0)
 meta.setdefault('duplicate_handling',{})[view]={'input_points':len(D),'unique_distance_profiles':int(n_unique),'collapsed_points':len(D)-int(n_unique),'largest_group':int(np.bincount(labels).max()),'method':'Collapse zero-distance groups only after verifying identical distance profiles; map all entries back to representative coordinates.'}
 (O/f'{view}-duplicate-map.json').write_text(json.dumps({'point_indices':index,'representative_local_indices':representatives.tolist(),'group_per_local_point':labels.tolist()}))
 print(view,'unique distance profiles',n_unique,'from',len(D),'points',flush=True)
 pp=[public[i] for i in index]
 for k in (15,5,30):
  name=f'{view}-k{k}';path=O/f'{name}-deduplicated-smacof.npy';start=time.time()
  if path.exists():xy=np.load(path);optimal=None
  else:
   model=phate.PHATE(knn=k,**params);xy_unique=model.fit_transform(fitD);xy=xy_unique[labels];np.save(path,xy);optimal=int(model.optimal_t)
  assert np.isfinite(xy).all() and xy.shape==(len(pp),2)
  meta['fits'][name]={'points':len(pp),'optimal_t':optimal,'elapsed_seconds':time.time()-start}
  for p,pos in zip(pp,xy):p[name]=pos.round(6).tolist()
  plot=make_subplots(rows=2,cols=2,subplot_titles=list(colors))
  lo=xy.min(axis=0);hi=xy.max(axis=0);pad=(hi-lo)*.06
  for c,(ds,color) in enumerate(colors.items()):
   rr,cc=c//2+1,c%2+1
   plot.add_trace(go.Scattergl(x=xy[:,0],y=xy[:,1],mode='markers',marker=dict(color='#9eabb9',size=3,opacity=.12),hoverinfo='skip',showlegend=False),row=rr,col=cc)
   for sp,sym in [('train','circle'),('val','diamond'),('test','x')]:
    ix=[i for i,p in enumerate(pp) if p['dataset']==ds and p['split']==sp]
    if not ix:continue
    plot.add_trace(go.Scattergl(x=xy[ix,0],y=xy[ix,1],mode='markers',name=ds+' '+sp,legendgroup=sp,showlegend=True,text=[pp[i]['id'] for i in ix],marker=dict(color=color,size=5 if sp=='train' else 7,symbol=sym,opacity=.65),hovertemplate='%{text}<br>'+ds+' '+sp+'<extra></extra>'),row=rr,col=cc)
   plot.update_xaxes(range=[lo[0]-pad[0],hi[0]+pad[0]],title_text='PHATE 1',row=rr,col=cc)
   plot.update_yaxes(range=[lo[1]-pad[1],hi[1]+pad[1]],title_text='PHATE 2',row=rr,col=cc)
   if k==15:
    ax=axes[row,c];ax.scatter(xy[:,0],xy[:,1],c='#aab4be',s=2,alpha=.1,rasterized=True)
    for sp,m in [('train','o'),('val','D'),('test','x')]:
     ix=[i for i,p in enumerate(pp) if p['dataset']==ds and p['split']==sp]
     if ix:ax.scatter(xy[ix,0],xy[ix,1],c=color,s=4 if sp=='train' else 8,marker=m,alpha=.65,linewidths=.4 if sp=='test' else 0,rasterized=True)
    ax.set(title=f'{ds} · {view}',xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),xlabel='PHATE 1',ylabel='PHATE 2');ax.spines[['top','right']].set_visible(False)
  plot.update_layout(height=950,template='plotly_white',title=f'{view.title()} alignment distances · k={k} · automatic t={optimal}',legend=dict(orientation='h'),margin=dict(l=50,r=20,t=80,b=100))
  plots.append('<section id="'+name+'"><h2>'+name+'</h2>'+plot.to_html(full_html=False,include_plotlyjs=len(plots)==0,config={'displaylogo':False})+'</section>')
  print('EMBED COMPLETE',name,meta['fits'][name],flush=True)
fig.suptitle('PHATE from BLOSUM62 alignment distances · k=15\nSame coordinates across datasets within each row; circles train, diamonds validation, crosses test',fontsize=15);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(O/'atlas-alignment-phate.png',dpi=200);fig.savefig(O/'atlas-alignment-phate.pdf');plt.close(fig)
(O/'alignment-phate-points.json').write_text(json.dumps(public,separators=(',',':')))
(O/'alignment-provenance.json').write_text(json.dumps(meta,indent=2))
intro='''<h1>Dataset coverage from sequence-alignment distances</h1><p>All 9,084 original whole-sequence points and 6,848 available interface points are retained, including all 2,000 ATLAS validation entries. The frozen split is unchanged. Each dataset is highlighted over the same grey reference points, with shared axes within a fit.</p><p>Smith–Waterman local alignment uses BLOSUM62, gap-open 11 and extension 1. Chains are matched one-to-one to maximize total alignment score. Distance = 1 − 2 × matched score / (total self-score of both complexes). Unmatched chains remain in the denominator. This is a normalized dissimilarity, not percent identity or a guaranteed metric.</p><p>Interface sequences contain contact residues in chain order (≤10 Å heavy-atom distance). Non-contact residues are omitted, so gaps between contact patches are collapsed. The interface view measures similarity of ordered contact subsequences, not full-chain homology.</p><p>PHATE uses precomputed distances, decay 40, 500 landmarks, automatic diffusion time, gamma 1, metric MDS with the SMACOF solver, and seed 20261011. Neighbourhood sizes 5, 15 and 30 are shown for sensitivity; their axes are independent. No PCA is applied to the distance matrix.</p><p>Dot density reflects the plotted sample, not the full dataset population. PHATE separation does not establish train/validation independence; use the original alignment leakage audit. Benchmark interface coordinates are available for 142 active validation entries only; monomers have no interface.</p><p><a href="atlas-alignment-phate.pdf">PDF figure (k=15)</a> · <a href="atlas-alignment-phate.png">PNG figure</a> · <a href="alignment-provenance.json">Methods and provenance</a> · <a href="atlas-split-report.html">Original split report and composition maps</a></p>'''
intro += '<p>Duplicate handling: zero-distance groups with identical distance profiles are represented once during PHATE fitting, then all entries are mapped back to those coordinates. '+ '; '.join(v+': '+str(x['unique_distance_profiles'])+' distinct profiles from '+str(x['input_points'])+' points' for v,x in meta['duplicate_handling'].items())+'. This prevents duplicate abundance from collapsing the adaptive neighbourhood bandwidth.</p>'
intro += '<p>Jump to: '+ ' · '.join('<a href="#'+v+'-k'+str(k)+'">'+v+' k='+str(k)+'</a>' for v in ('whole','interface') for k in (15,5,30))+'</p>'
intro += '<p>Zero-information chains excluded from normalization: '+', '.join(f'{v}: {len(rows)}' for v,rows in meta['uninformative_chains'].items())+'. All complexes retain at least one informative chain; affected IDs are recorded in the provenance.</p>'
(O/'atlas-alignment-report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS alignment PHATE</title><style>body{font:16px/1.6 system-ui;color:#203047;background:#f6f8fb;margin:0}main{max-width:1350px;padding:28px;margin:auto}section{background:white;border:1px solid #dde4ed;border-radius:12px;padding:18px;margin:24px 0}a{color:#225b9d}</style><main>'+intro+''.join(plots)+'<p>Method references: <a href="https://github.com/jeffdaily/parasail">Parasail</a> · <a href="https://phate.readthedocs.io/en/stable/">PHATE</a></p></main></html>')
print('ALL OUTPUTS COMPLETE',flush=True)
