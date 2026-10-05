"""Shared geometry-to-native-tautomer intrinsic baseline; no scalar pair reduction."""
import argparse,csv,json,os
from collections import Counter,defaultdict
from pathlib import Path
from .runtime import require_compute,atomic_json,digest

def read(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()

KEY=['complex_id','chain','resnum','icode','group']
RADII=(3,6,10,15)
POINTS=('center','atom0','atom1')
NUMERIC=[f'{p}_{e}_{r}' for p in POINTS for r in RADII for e in ('heavy','N','O','S')]+[f'{p}_{q}' for p in POINTS for q in ('formal_field','positive_6','negative_6','positive_10','negative_10')]+['atom0_present','atom1_present']
FEATURES=['group','tautomer']+NUMERIC

def init(out):
    root=Path(os.environ['PKABENCH_RUNTIME']); source=root/'tierB/native-v2'
    gate=json.loads((source/'readiness.json').read_text()); assert gate['passed'] and gate['native_order_mc_replay_passed'] and not gate['test_data_included']
    v=json.loads((source/'verification.json').read_text()); assert digest(source/'native_state_index.json')==v['index_sha256']
    manifest=json.loads((source/'manifest.json').read_text()); orig=Path(manifest['source']); assignments=read(source/'assignments.parquet')
    assert Counter(r['split'] for r in assignments)=={'train':778,'val':151}
    assert not {r['component_id'] for r in assignments if r['split']=='train'}&{r['component_id'] for r in assignments if r['split']=='val'}
    index=json.loads((source/'native_state_index.json').read_text()); out.mkdir(parents=True,exist_ok=False); (out/'shards').mkdir()
    config={'source':str(source),'structure_source':str(orig),'readiness_sha256':digest(source/'readiness.json'),'index_sha256':v['index_sha256'],
      'code_sha256':digest(Path(__file__)),'seed_list':[17,29,43],'features':FEATURES,'iterations':500,'depth':6,'learning_rate':.03,
      'loss':'Huber delta 1 on pKint minus force-field tautomer pKmod','selection':'fixed hyperparameters and iteration count; no validation checkpoint selection',
      'weights':'equal sequence groups, equal complexes per group, equal site-state records per complex, equal tautomers per site-state; normalize mean to one',
      'support':'all native supervision-eligible site-states, including finite intrinsics whose coupled midpoint is out of range',
      'test_data_included':False,'environment_lock_sha256':digest(root/'finetune/diagnostic-v1/environment.lock'),
      'input_sha256':{n:digest(source/n) for n in ('assignments.parquet','structures.parquet','site_masks.parquet')}}
    atomic_json(out/'manifest.json',config); print(json.dumps({'states':len(index),'assignments':len(assignments),'features':len(FEATURES)}),flush=True)

def check(out):
    m=json.loads((out/'manifest.json').read_text()); assert digest(Path(__file__))==m['code_sha256']; source=Path(m['source'])
    assert digest(source/'readiness.json')==m['readiness_sha256'] and digest(source/'native_state_index.json')==m['index_sha256']
    for n,h in m['input_sha256'].items(): assert digest(source/n)==h
    return m

def features(out,shard,shards):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree
    from .prep import read_cif
    from .annotate import SITE_ATOMS
    m=check(out); source=Path(m['source']); original=Path(m['structure_source']); assignment={r['complex_id']:r for r in read(source/'assignments.parquet')}
    states=json.loads((source/'native_state_index.json').read_text()); states=sorted(states,key=lambda s:(s['complex_id'],s['state']))[shard::shards]
    root=Path(os.environ['PKABENCH_RUNTIME']); ff=root/'envs/pypka/lib/python3.10/site-packages/pypka/G54A7/sts'
    rows=[]; missing=[]; constants={}; inputs={}; counts=Counter()
    for state in states:
        cid=state['complex_id']; st=state['state']; a=assignment[cid]; base=Path(state['export'])
        assert a['split']==state['split'] and a['split'] in ('train','val')
        assert digest(base/'sites.json')==state['sites_sha256']; inputs[str(base/'sites.json')]=state['sites_sha256']
        sites=json.loads((base/'sites.json').read_text()); path=original/'structures'/cid/f'{st}.cif'
        # State hash is checked against the canonical source receipts, including reused states.
        side=json.loads((original/'jobs/pypka'/f'{cid}.json').read_text())
        # The non-JAX receipt retains the original state hashes when available; resolve original receipt otherwise.
        production=Path(json.loads((source/'manifest.json').read_text())['original'])
        original_side=json.loads((production/'jobs/pypka'/f'{cid}.json').read_text())
        assert digest(path)==original_side['input_state_sha256'][st]; inputs[str(path)]=digest(path)
        atoms=read_cif(path); coords=np.asarray(atoms.coord,dtype=float); tree=cKDTree(coords)
        residue_indices=defaultdict(list)
        for i in range(len(atoms)): residue_indices[str(atoms.chain_id[i]),int(atoms.res_id[i]),str(atoms.ins_code[i])].append(i)
        charged=[]; q=[]; charged_keys=[]
        for rk,ii in residue_indices.items():
            aa=str(atoms.res_name[ii[0]]); charge={'ASP':-1.,'GLU':-1.,'LYS':1.,'ARG':1.}.get(aa,0.)
            if charge:
                jj=[i for i in ii if str(atoms.atom_name[i]) in SITE_ATOMS[aa]]
                if jj: charged.append(coords[jj].mean(axis=0)); q.append(charge); charged_keys.append(rk)
        charged=np.asarray(charged,dtype=float).reshape(-1,3); q=np.asarray(q)
        for s in sites:
            counts['native_sites']+=1
            if not s['supervision_eligible']: counts['masked_sites']+=1; continue
            rk=(s['chain'],s['resnum'],s['icode']); own=residue_indices[rk]; byname={str(atoms.atom_name[i]):i for i in own}; names=SITE_ATOMS[s['group']]; functional=[byname[n] for n in names if n in byname]
            if not functional: missing.append({'complex_id':cid,'state':st,'site':list(rk),'group':s['group'],'reason':'no observed functional atom'}); continue
            points={'center':coords[functional].mean(axis=0)}
            f={}
            for slot in (0,1):
                exists=slot<len(names) and names[slot] in byname; f[f'atom{slot}_present']=int(exists); points[f'atom{slot}']=coords[byname[names[slot]]] if exists else None
            ownset=set(own); qmask=np.asarray([x!=rk for x in charged_keys],dtype=bool)
            for pname,center in points.items():
                if center is None:
                    for n in NUMERIC:
                        if n.startswith(pname+'_'): f[n]=None
                    f[pname+'_present']=0
                    continue
                indices=np.asarray([j for j in tree.query_ball_point(center,15) if j not in ownset],dtype=int)
                dist=np.linalg.norm(coords[indices]-center,axis=1); elements=atoms.element[indices]
                for r in RADII:
                    sel=dist<=r; f[f'{pname}_heavy_{r}']=int(sel.sum())
                    for e in ('N','O','S'): f[f'{pname}_{e}_{r}']=int(np.sum(sel&(elements==e)))
                d=np.linalg.norm(charged[qmask]-center,axis=1); charges=q[qmask]
                f[f'{pname}_formal_field']=float(np.sum(charges*np.exp(-d/10)/(d+.5)))
                for r in (6,10):
                    f[f'{pname}_positive_{r}']=int(np.sum((d<=r)&(charges>0))); f[f'{pname}_negative_{r}']=int(np.sum((d<=r)&(charges<0)))
            assert set(f)==set(NUMERIC),(set(f)-set(NUMERIC),set(NUMERIC)-set(f))
            assert all(x is None or np.isfinite(x) for x in f.values())
            for name,target in zip(s['tautomers'][:-1],s['intrinsic_pka_tautomers']):
                ffgroup={'NTERM':'NTR','CTERM':'CTR'}.get(s['group'],s['group']); constantpath=ff/f'{ffgroup}tau{int(name[-1])+1}.st'
                pmod=float(constantpath.read_text().splitlines()[0]); constants[str(constantpath)]=digest(constantpath)
                rows.append({n:s[n] for n in KEY}|{'state':st,'split':a['split'],'component_id':a['component_id'],'role':a['role'],'interface':s['interface'],
                    'tautomer':name,'intrinsic_pka':target,'model_pka':pmod,**f})
    dest=out/'shards'/f'{shard:02d}.parquet'; pq.write_table(pa.Table.from_pylist(rows),dest)
    atomic_json(out/'shards'/f'{shard:02d}.json',{'rows':len(rows),'state_count':len(states),'counts':dict(counts),'missing_geometry':missing,'inputs_sha256':inputs,
        'constants_sha256':constants,'output_sha256':digest(dest),'manifest_sha256':digest(out/'manifest.json')})
    print(json.dumps({'shard':shard,'rows':len(rows),'missing_geometry':len(missing)}),flush=True)

def assemble(out,shards):
    import pyarrow as pa
    import pyarrow.parquet as pq
    m=check(out); rows=[]; missing=[]; input_hashes={}
    for shard in range(shards):
        p=out/'shards'/f'{shard:02d}.parquet'; r=json.loads((p.with_suffix('.json')).read_text()); assert r['manifest_sha256']==digest(out/'manifest.json') and digest(p)==r['output_sha256']
        rows.extend(read(p)); missing.extend(r['missing_geometry']); input_hashes.update(r['inputs_sha256']); input_hashes.update(r['constants_sha256'])
    keys=[tuple(r[n] for n in KEY+['state','tautomer']) for r in rows]; assert len(keys)==len(set(keys))
    assert set(r['split'] for r in rows)=={'train','val'}
    assert not {r['component_id'] for r in rows if r['split']=='train'}&{r['component_id'] for r in rows if r['split']=='val'}
    assert not (set(FEATURES)&{'intrinsic_pka','model_pka','complex_id','chain','resnum','split','interface','state'})
    pq.write_table(pa.Table.from_pylist(rows),out/'features.parquet')
    atomic_json(out/'feature_gate.json',{'passed':True,'counts':dict(Counter(r['split'] for r in rows)),'missing_geometry':missing,'inputs_sha256':input_hashes,'features_sha256':digest(out/'features.parquet'),'test_data_included':False})
    print(json.dumps({'counts':dict(Counter(r['split'] for r in rows)),'missing_geometry':len(missing)}),flush=True)

def frame(out):
    import pandas as pd
    m=check(out); gate=json.loads((out/'feature_gate.json').read_text()); assert gate['passed'] and digest(out/'features.parquet')==gate['features_sha256']
    d=pd.read_parquet(out/'features.parquet'); assert set(d.split)=={'train','val'}
    return m,d

def fit(out,seed):
    import numpy as np
    from catboost import CatBoostRegressor
    m,d=frame(out); dest=out/f'seed-{seed}'; dest.mkdir(exist_ok=False); tr=d[d.split=='train'].copy()
    nt=tr.groupby(KEY+['state']).tautomer.transform('count'); ns=tr[KEY+['state']].drop_duplicates().groupby('complex_id').size(); nc=tr[['complex_id','component_id']].drop_duplicates().groupby('component_id').size()
    weight=1/(nt*tr.complex_id.map(ns)*tr.component_id.map(nc)); weight/=weight.mean()
    x=d[FEATURES].copy(); x[NUMERIC]=x[NUMERIC].astype(float)
    target=tr.intrinsic_pka-tr.model_pka
    model=CatBoostRegressor(iterations=m['iterations'],depth=m['depth'],learning_rate=m['learning_rate'],loss_function='Huber:delta=1.0',random_seed=seed,thread_count=2,allow_writing_files=False,verbose=100)
    model.fit(x.loc[tr.index],target,sample_weight=weight,cat_features=['group','tautomer'])
    pred=model.predict(x)+d.model_pka.values; assert np.isfinite(pred).all()
    result=d[KEY+['state','tautomer','split','component_id','role','interface','intrinsic_pka','model_pka']].copy(); result['prediction']=pred
    # Train-only per-tautomer weighted constant is an additional absolute baseline.
    means={g:float(np.average(tr.loc[idx].intrinsic_pka,weights=weight.loc[idx])) for g,idx in tr.groupby(['group','tautomer']).groups.items()}
    result['train_constant']=[means.get((r.group,r.tautomer),r.model_pka) for r in result.itertuples()]
    result.to_parquet(dest/'predictions.parquet',index=False); model.save_model(str(dest/'intrinsic.cbm'))
    import pandas as pd
    pd.DataFrame({'feature':FEATURES,'importance':model.feature_importances_}).to_csv(dest/'feature_importance.csv',index=False)
    atomic_json(dest/'receipt.json',{'complete':True,'seed':seed,'prediction_sha256':digest(dest/'predictions.parquet'),'model_sha256':digest(dest/'intrinsic.cbm'),
      'manifest_sha256':digest(out/'manifest.json'),'feature_gate_sha256':digest(out/'feature_gate.json'),'train_rows':len(tr),'validation_rows':len(d)-len(tr),'validation_used_for_fit':False,'test_data_included':False})

def group_scores(d):
    # A site contributes equally regardless of how many tautomers it has.
    d=d.copy(); d['ae']=abs(d.prediction-d.target); d['se']=(d.prediction-d.target)**2
    sites=d.groupby(['component_id']+KEY+(['state'] if 'state' in d.columns else []))[['ae','se']].mean()
    complexes=sites.groupby(['component_id','complex_id']).mean(); complexes['rmse']=complexes.se**.5
    return complexes.groupby('component_id')[['ae','rmse']].mean().rename(columns={'ae':'mae'})

def collect(out):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m,d=frame(out); scores=[]; contrasts=[]; support={}; input_hashes={}
    for seed in m['seed_list']:
        dest=out/f'seed-{seed}'; receipt=json.loads((dest/'receipt.json').read_text()); path=dest/'predictions.parquet'
        assert receipt['complete'] and digest(path)==receipt['prediction_sha256'] and receipt['manifest_sha256']==digest(out/'manifest.json')
        r=pd.read_parquet(path); input_hashes[str(path)]=digest(path)
        ids=KEY+['tautomer']; ab=r[r.state=='AB']; free=r[r.state!='AB']; assert not free.duplicated(ids).any()
        pairs=ab.merge(free,on=ids,suffixes=('_ab','_free'),validate='one_to_one'); assert (pairs.split_ab==pairs.split_free).all() and (pairs.component_id_ab==pairs.component_id_free).all()
        for field in ('split','component_id','role','interface'): pairs[field]=pairs[field+'_ab']
        for field in ('intrinsic_pka','prediction','model_pka','train_constant'): pairs[field]=pairs[field+'_ab']-pairs[field+'_free']
        assert np.all(pairs.model_pka==0) and np.all(pairs.train_constant==0)
        support={'absolute_rows':len(r),'paired_tautomer_rows':len(pairs),'paired_sites':len(pairs[KEY].drop_duplicates()),'validation_paired_interface_sites':len(pairs[(pairs.split=='val')&pairs.interface][KEY].drop_duplicates())}
        for task,data in [('absolute',r),('paired_delta',pairs)]:
            for split in ('train','val'):
                for subset in ('all','interface'):
                    for role in ('all','antibody_antigen','general'):
                        sub=data[data.split==split].copy()
                        if subset=='interface': sub=sub[sub.interface]
                        if role!='all': sub=sub[sub.role==role]
                        if len(sub)==0: continue
                        groupvalues={}
                        for method in ('catboost','model_compound','train_constant'):
                            df=sub.copy(); df['target']=df.intrinsic_pka; df['prediction']=df[{'catboost':'prediction','model_compound':'model_pka','train_constant':'train_constant'}[method]]
                            g=group_scores(df); groupvalues[method]=g; rng=np.random.default_rng(20261005); vals=g.mae.values
                            ci=np.quantile(rng.choice(vals,(2000,len(vals))).mean(axis=1),[.025,.975]) if len(vals)>=5 else [np.nan,np.nan]
                            scores.append({'task':task,'split':split,'subset':subset,'role':role,'seed':seed,'method':method,'tautomer_rows':len(df),'sites':len(df[KEY].drop_duplicates()),'groups':len(g),'mae':g.mae.mean(),'rmse':g.rmse.mean(),'ci_low':ci[0],'ci_high':ci[1]})
                        if split=='val' and subset=='interface' and role=='all':
                            for baseline in ('model_compound','train_constant'):
                                delta=groupvalues['catboost'].mae-groupvalues[baseline].mae; rng=np.random.default_rng(20261005); ci=np.quantile(rng.choice(delta.values,(2000,len(delta))).mean(axis=1),[.025,.975])
                                contrasts.append({'task':task,'seed':seed,'baseline':baseline,'mae_change':float(delta.mean()),'ci95':ci.tolist()})
    frame_scores=pd.DataFrame(scores); frame_scores.to_csv(out/'scores.csv',index=False); atomic_json(out/'comparisons.json',contrasts)
    selected=frame_scores[(frame_scores.split=='val')&(frame_scores.subset=='interface')&(frame_scores.role=='all')]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    methods=('model_compound','train_constant','catboost')
    for ax,task in zip(axes,('absolute','paired_delta')):
        for i,method in enumerate(methods):
            rr=selected[(selected.task==task)&(selected.method==method)]; ax.scatter([i]*len(rr),rr.mae)
        ax.set(xticks=range(3),xticklabels=['Model compound','Train constant','CatBoost'],ylabel='Group-macro intrinsic pKa MAE',title=task); ax.tick_params(axis='x',rotation=20); ax.set_ylim(bottom=0)
    fig.suptitle('Native tautomer targets; validation interface; three seeds'); fig.savefig(out/'validation.png',dpi=180); plt.close(fig)
    imp=pd.concat([pd.read_csv(out/f'seed-{s}/feature_importance.csv') for s in m['seed_list']]).groupby('feature').importance.mean().nlargest(15).sort_values()
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained'); imp.plot.barh(ax=ax); ax.set_title('Mean CatBoost feature importance'); fig.savefig(out/'feature_importance.png',dpi=180); plt.close(fig)
    lines=['# Native-state intrinsic baseline','','These targets are native tautomer intrinsic pKas, not coupled midpoints. No validation labels were used for fitting or checkpoint selection. Test data was excluded.','','| Task | Method | Seed | Validation interface MAE |','|---|---|---:|---:|']
    for r in selected.itertuples(): lines.append(f'| {r.task} | {r.method} | {r.seed} | {r.mae:.4f} |')
    lines+=['','All finite eligible native sites are included, even when the coupled midpoint is outside the pH grid. Tautomers average within sites before complex and sequence-group averaging. Paired predictions subtract the shared model outputs. Fixed constants predict zero intrinsic shift.','',f'Support: {support}.','',
      'Geometry features use observed heavy atoms, separate functional-atom neighborhoods and fixed residue formal-charge proxies. They do not use teacher energies, charge occupancies, PDB identity or state flags. Missing functional-atom slots remain missing. This is a coarse structural baseline, not a learned pair matrix or a validated coupled-pKa model.','',
      'Paired bootstrap changes and antibody/general strata are retained. Do not compare these MAEs numerically with the experiment 02 midpoint MAEs.','',f'![Validation]({out}/validation.png)',f'![Importance]({out}/feature_importance.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',{'passed':True,'seeds':m['seed_list'],'support':support,'test_data_included':False,'validation_used_for_fit':False,'prediction_sha256':input_hashes,
      'artifacts_sha256':{n:digest(out/n) for n in ('scores.csv','comparisons.json','validation.png','feature_importance.png','report.md')}})
    print((out/'report.md').read_text(),flush=True)

if __name__=='__main__':
    affinity=os.sched_getaffinity(0); require_compute(); os.sched_setaffinity(0,affinity)
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['init','features','assemble','fit','collect']); p.add_argument('--out',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=32); p.add_argument('--seed',type=int,default=17); a=p.parse_args()
    if a.stage=='init': init(a.out)
    elif a.stage=='features': features(a.out,a.shard,a.shards)
    elif a.stage=='assemble': assemble(a.out,a.shards)
    elif a.stage=='fit': fit(a.out,a.seed)
    else: collect(a.out)
