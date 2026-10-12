import os,json,time,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import phate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
R=Path(os.environ['PKABENCH_RUNTIME']);O=R/'audits/atlas-split-v1'
assert os.environ.get('SLURM_JOB_ID')
f=np.load(O/'features.npz');points=json.loads((O/'points.json').read_text());sampling=json.loads((O/'sampling.json').read_text());split=json.loads((O/'split-summary.json').read_text());features=json.loads((O/'feature-summary.json').read_text());out={};timing={}
for name in ('whole','interface'):
 p=O/f'phate-{name}.npy';beg=time.time()
 if p.exists():xy=np.load(p)
 else:
  model=phate.PHATE(n_components=2,knn=15,decay=40,n_landmark=500,t='auto',gamma=1,n_pca=50,knn_dist='cosine',mds='metric',n_jobs=72,random_state=20261011,verbose=1)
  xy=model.fit_transform(f[name]);np.save(p,xy);(O/f'phate-{name}-t.txt').write_text(str(model.optimal_t))
 assert np.isfinite(xy).all();out[name]=xy;timing[name]=time.time()-beg;print(name,'PHATE completed',xy.shape,flush=True)
for i,p in enumerate(points):p['whole']=out['whole'][i].round(5).tolist();p['interface']=None
for i,j in enumerate(f['interface_indices']):points[int(j)]['interface']=out['interface'][i].round(5).tolist()
(O/'phate-points.json').write_text(json.dumps(points,separators=(',',':')))
colors={'ATLAS':'#3c78d8','PINDER':'#16a085','pKPDB':'#d18b22','Benchmark':'#9958b0'};symbols={'train':'circle','val':'diamond','test':'x'};markers={'train':'o','val':'D','test':'x'}
fig=make_subplots(rows=1,cols=2,subplot_titles=['Whole-chain sequence features','Interface sequence features'])
for col,name in enumerate(('whole','interface'),1):
 for group in sampling:
  pp=[p for p in points if p['group']==group and p[name] is not None]
  if not pp:continue
  ds=pp[0]['dataset'];sp=pp[0]['split'];data=np.array([p[name] for p in pp]);custom=[[p['id'],p['group'],p['residues'],p['interface_residues'],p.get('source_class',''),p.get('component',''),p.get('active_validation',False)] for p in pp]
  fig.add_trace(go.Scattergl(x=data[:,0],y=data[:,1],mode='markers',name=group,legendgroup=group,showlegend=col==1,customdata=custom,marker=dict(color=colors[ds],symbol=symbols[sp],size=5 if sp=='train' else 7,opacity=.48 if sp=='train' else .8),hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>Full residues: %{customdata[2]}<br>Interface residues: %{customdata[3]}<br>Source: %{customdata[4]}<br>Component: %{customdata[5]}<br>Active benchmark validation: %{customdata[6]}<extra></extra>'),row=1,col=col)
 fig.update_xaxes(title_text='PHATE 1 (arbitrary units)',row=1,col=col);fig.update_yaxes(title_text='PHATE 2 (arbitrary units)',scaleanchor='x' if col==1 else 'x2',scaleratio=1,row=1,col=col)
fig.update_layout(height=670,template='plotly_white',legend=dict(orientation='h',y=-.17,x=0,groupclick='togglegroup'),margin=dict(l=45,r=20,b=160,t=65),title='ATLAS intended split alongside existing cohorts')
plot=fig.to_html(full_html=False,include_plotlyjs=False,config={'displaylogo':False,'toImageButtonOptions':{'format':'svg','filename':'atlas-phate-split'}})
fig2,axes=plt.subplots(1,2,figsize=(14,6.6))
for ax,name,title in zip(axes,('whole','interface'),('Whole-chain sequence features','Interface sequence features')):
 for group in sampling:
  pp=[p for p in points if p['group']==group and p[name] is not None]
  if not pp:continue
  ds,sp=pp[0]['dataset'],pp[0]['split'];xy=np.array([p[name] for p in pp]);ax.scatter(xy[:,0],xy[:,1],s=5 if sp=='train' else 12,alpha=.4 if sp=='train' else .7,c=colors[ds],marker=markers[sp],label=group,linewidths=.5 if sp=='test' else 0,rasterized=True)
 ax.set(title=title,xlabel='PHATE 1',ylabel='PHATE 2');ax.spines[['top','right']].set_visible(False);ax.set_aspect('equal',adjustable='datalim')
handles,labels=axes[0].get_legend_handles_labels();fig2.legend(handles,labels,loc='lower center',ncol=5,frameon=False);fig2.tight_layout(rect=(0,.14,1,.95));fig2.suptitle('Joint PHATE fits of BLOSUM62-weighted sequence features',fontsize=14)
fig2.savefig(O/'atlas-phate-split.png',dpi=220);fig2.savefig(O/'atlas-phate-split.pdf');plt.close(fig2)
# Static membership bars use full cohort counts, independently of the plotted sample sizes.
bar=go.Figure()
for sp in ('train','val','test'):
 vals=[sum(v['population'] for g,v in sampling.items() if g==ds+' '+sp) for ds in colors]
 bar.add_bar(name=sp,x=list(colors),y=vals,text=[f'{v:,}' if v else '' for v in vals],textposition='auto',marker_color={'train':'#7898bb','val':'#edaa40','test':'#a29cb4'}[sp])
bar.update_layout(barmode='stack',template='plotly_white',height=350,yaxis_title='Complexes / entries (full cohort)',title='Intended ATLAS membership and existing cohort membership')
barhtml=bar.to_html(full_html=False,include_plotlyjs=True,config={'displaylogo':False})
rows=''.join(f'<tr><td>{g}</td><td>{s["population"]:,}</td><td>{s["sample"]:,}</td><td>{features["by_group"][g]["interface"]:,}</td></tr>' for g,s in sampling.items())
meta=dict(phate_version=importlib.metadata.version('phate'),blosum='BLOSUM62',feature_method=features['method'],interface_definition=features['interface_definition'],sampling=sampling,features=features,code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*.py')},parameters=dict(knn=15,decay=40,n_landmark=500,n_pca=50,knn_dist='cosine',t='auto',seed=20261011),timing_seconds=timing,input_hashes={p:hashlib.sha256((O/p).read_bytes()).hexdigest() for p in ('features.npz','points.json','split-summary.json')},sources=['https://phate.readthedocs.io/en/stable/','https://biopython.org/docs/latest/Tutorial/chapter_pairwise.html'])
(O/'visualisation-provenance.json').write_text(json.dumps(meta,indent=2))
html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS intended split and PHATE coverage</title><style>body{{font:16px/1.6 system-ui;margin:0;background:#f6f8fb;color:#203047}}main{{max-width:1350px;margin:auto;padding:30px}}h1{{font-size:32px;line-height:1.25}}section{{background:white;border:1px solid #dde4ed;border-radius:12px;padding:22px;margin:22px 0}}.numbers{{display:flex;gap:35px;flex-wrap:wrap}}.number{{font-size:30px;font-weight:700}}.muted{{color:#536579}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #e2e7ee;padding:9px}}.note{{border-left:4px solid #ddaa44;padding-left:16px}}a{{color:#225b9d}}code{{overflow-wrap:anywhere}}</style><main><p class="muted">11 October 2026 · Frozen ATLAS membership · Separate staging</p><h1>Where ATLAS training and validation will sit</h1><div class="numbers"><div><div class="number">{split['train']:,}</div>ATLAS planned train</div><div><div class="number">{split['val']:,}</div>ATLAS planned validation</div><div><div class="number">0</div>detected validation → training hits after clearance</div></div>
<p>ATLAS flow: 181,671 high-confidence source complexes → 157,773 candidates after screening → 155,677 successfully labelled complexes. Preparation rejected 1,953 and source-sequence checks quarantined 143.</p><p class="note">Split membership is frozen; training admission remains gated on final site masks, weights and graph QC. Existing training and validation manifests are unchanged. The 143 source-sequence mismatches remain quarantined. ATLAS labels are pKAI/pKAI+ predictions, so this set measures teacher agreement on held-out predicted structures; retain independent experimental benchmarks for experimental pKa accuracy.</p>
<section>{barhtml}<p class="muted">Full cohort counts. pKPDB includes monomers. Benchmark counts refer to the frozen structural train/val/test assignments; 142 of its 284 validation entries are in the active labelled benchmark.</p></section>
<section>{plot}<p>Colour identifies the dataset; circles are training, diamonds validation, crosses test. Click legend entries to hide/show groups, drag to zoom, and hover for entry details. Both datasets and splits share each fit; the two panels are independent embeddings and their axis positions are not comparable.</p><p class="note">PHATE proximity is descriptive, not a leakage test. These are BLOSUM62-weighted composition and adjacent-pair features, not pairwise alignment distances or protein language-model embeddings. Interface residues are selected by ≤10 Å heavy-atom distance, with no artificial adjacency across omitted residues.</p></section>
<section><h2>Sampling and structural coverage</h2><table><tr><th>Dataset / split</th><th>Full membership</th><th>Whole-chain points</th><th>Interface points</th></tr>{rows}</table><p>Dot density reflects the displayed sample, not full cohort prevalence. All ATLAS validation candidates and current PINDER/pKPDB validation entries are shown; large training and benchmark-test groups use deterministic samples. Monomers and unavailable benchmark coordinates remain in the whole-chain panel and are omitted from the interface panel. Available interface coverage is reported explicitly.</p></section>
<section><h2>How membership was chosen</h2><p>{split["components"]:,} connected components; largest component: {split["largest_components"][0]:,} complexes. Eligible training-independent validation population: {split["eligible_validation_population"]:,} complexes. Validation spans {split["validation_components"]:,} components, with at most {split["largest_validation_component"]:,} entries from any one component (cap: 100).</p><p>Either matching partner joins ATLAS complexes into a connected component at ≥70% identity with ≥80% coverage of both chains, or ≥90% identity with ≥80% coverage of either chain. Exact shared chains are joined directly. Whole components stay together. Validation is drawn from components without detected matches to PINDER, pKPDB or frozen benchmark training sequences, stratified by source group, homo/heterodimer type, length and pLDDT.</p><p>The selected validation sequences were searched against the remaining ATLAS and existing training sequences in both directions. Zero hits remained at the declared thresholds. MMseqs2 is a sensitive search, not proof of absence of remote homology. Stratification targets the eligible, training-independent population; it cannot guarantee a representative sample of already-familiar families.</p><p>Features use normalized amino-acid and adjacent-pair frequencies transformed with BLOSUM62, then chain-wise mean and standard deviation to make complexes invariant to chain order. PHATE uses cosine neighbours, k=15, 500 landmarks, 50 PCA dimensions, automatic diffusion time and seed 20261011.</p><p><a href="https://phate.readthedocs.io/en/stable/">PHATE method documentation</a> · <a href="https://biopython.org/docs/latest/Tutorial/chapter_pairwise.html">BLOSUM62 implementation documentation</a></p></section>
<p><a href="atlas-phate-split.png">PNG figure</a> · <a href="atlas-phate-split.pdf">PDF figure</a> · <a href="atlas-split-summary.json">Split audit</a> · <a href="atlas-visualisation-provenance.json">Representation and sampling provenance</a> · <a href="atlas-validation.tsv">Validation membership</a> · <a href="atlas-train.tsv">Training membership</a></p></main></html>'''
(O/'atlas-split-report.html').write_text(html);print('ALL OUTPUTS COMPLETE',flush=True)
