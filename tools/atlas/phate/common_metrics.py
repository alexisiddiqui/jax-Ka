import os,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(os.environ['PKABENCH_RUNTIME']);A=R/'audits/atlas-alignment-v1';O=R/'audits/atlas-phate-check-v1';F=R/'audits/atlas-split-v1'
assert os.environ.get('SLURM_JOB_ID')
d=json.loads((O/'diagnostics.json').read_text());points=json.loads((O/'comparison-points.json').read_text());feat=np.load(F/'features.npz');colors={'ATLAS':'#0072B2','PINDER':'#009E73','pKPDB':'#D55E00','Benchmark':'#882E72'}
fig,axes=plt.subplots(2,4,figsize=(18,10))
for row,view in enumerate(('whole','interface')):
 m=json.loads((A/f'{view}-duplicate-map.json').read_text());indices=np.array(m['point_indices']);reps=np.array(m['representative_local_indices']);X=feat[view][reps];n=len(reps);sample=np.random.default_rng(20261011).choice(n,size=min(1024,n),replace=False)
 raw=cdist(X[sample],X,metric='cosine');raw[np.arange(len(sample)),sample]=np.inf;rawkn=np.argpartition(raw,14,axis=1)[:,:15]
 for name,metrics in d['views'][view]['fits'].items():
  xy=np.load(O/f'{view}-{name}.npy')[reps];dist=cdist(xy[sample],xy);dist[np.arange(len(sample)),sample]=np.inf;kn=np.argpartition(dist,14,axis=1)[:,:15];metrics['mean_original_feature_neighbor_recall15']=float(np.mean([len(set(a)&set(b))/15 for a,b in zip(rawkn,kn)]))
 allxy=np.load(O/f'{view}-composition_pca50.npy');lo=allxy.min(axis=0);hi=allxy.max(axis=0);pad=(hi-lo)*.06
 for col,(ds,color) in enumerate(colors.items()):
  ax=axes[row,col];ax.scatter(allxy[:,0],allxy[:,1],s=2,color='#9eabb9',alpha=.12,rasterized=True)
  for split,marker in [('train','o'),('val','D'),('test','x')]:
   mask=np.array([points[i]['dataset']==ds and points[i]['split']==split for i in indices]);ax.scatter(allxy[mask,0],allxy[mask,1],s=5 if split=='train' else 9,color=color,alpha=.6,marker=marker,linewidths=.4 if split=='test' else 0,rasterized=True)
  ax.set(title=f'{ds} · {view}',xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),xlabel='PHATE 1',ylabel='PHATE 2');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Original composition features with PCA 50 → PHATE\nShared coordinates across datasets within each row; circles train, diamonds validation, crosses test',fontsize=15);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(O/'pca50-overview.png',dpi=180);fig.savefig(O/'pca50-overview.pdf');plt.close(fig)
d['common_feature_metric_note']='Original-feature recall uses unreduced 840-dimensional composition features and cosine distance as the common reference for all maps. Same fixed sample of up to 1024 unique alignment profiles.'
(O/'diagnostics.json').write_text(json.dumps(d,indent=2,allow_nan=False));print('COMMON METRICS AND FIGURES COMPLETE')
