"""Versioned paired-learning diagnostic. No test data is read by this module."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter, defaultdict
from .runtime import require_compute, digest, atomic_json

ARMS=('frozen','last','all','scratch','catboost')
SEEDS=(17,29,43)
FEATURES=['delta_sasa','functional_delta_sasa','min_partner_distance','distance_interface_centroid',
 'delta_heavy_count_6A','delta_heavy_count_10A','potential_partner_donor_atoms_4A',
 'potential_partner_acceptor_atoms_4A','delta_formal_charge_6A','delta_formal_charge_10A',
 'salt_bridge_proxy_count_4A','nearest_opposite_charge_distance']

def table(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()

def key(r): return tuple(r[k] for k in ('complex_id','chain','resnum','icode','group'))

def setup(out,handoff):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    out.mkdir(parents=True,exist_ok=False)
    v=json.loads((handoff/'verification.json').read_text())
    assert v['passed'] and not v['test_data_included'] and v['frozen_pkai_state_predictions_checked']>0
    assert not (handoff/'INVALID.json').exists()
    for name,sha in v['artifacts_sha256'].items(): assert digest(handoff/name)==sha
    rows=table(handoff/'pkai_pairs.parquet'); cats=table(handoff/'catboost_pairs.parquet')
    assert set(r['split'] for r in rows+cats)=={'train','val'}
    groups={s:{r['component_id'] for r in rows+cats if r['split']==s} for s in ('train','val')}
    assert not groups['train']&groups['val']
    # Preserve all representable rows for shell evaluation; fit interface rows only.
    xab=np.empty((len(rows),4008),dtype=np.float32); xfree=np.empty_like(xab)
    uses=defaultdict(list)
    for i,r in enumerate(rows):
        uses[r['ab_feature_file']].append((i,r['ab_feature_row'],True))
        uses[r['free_feature_file']].append((i,r['free_feature_row'],False))
    for path,indices in uses.items():
        assert digest(Path(path))==v['feature_files_sha256'][path]
        with np.load(path) as d:
            for i,j,ab in indices: (xab if ab else xfree)[i]=d['x'][j]
    assert np.isfinite(xab).all() and np.isfinite(xfree).all()
    np.save(out/'ab.npy',xab); np.save(out/'free.npy',xfree)
    pq.write_table(pa.Table.from_pylist(rows),out/'rows.parquet')
    pq.write_table(pa.Table.from_pylist(cats),out/'catboost.parquet')
    runtime=Path(os.environ['PKABENCH_RUNTIME']); env=runtime/'envs/finetune-v1'
    uv=Path('/home/coulson/oc/lina4225/_runtime/BioFeaturisers/cuda-86/toolchain/bin/uv')
    assert not env.exists(),'Use a new version for an existing training environment'
    subprocess.run([str(uv),'venv','--python',str(runtime/'envs/runner/bin/python'),str(env)],check=True)
    subprocess.run([str(uv),'pip','install','--python',str(env/'bin/python'),'numpy==1.26.4','pyarrow==23.0.1','scipy==1.15.3','catboost==1.2.8','matplotlib==3.10.8'],check=True)
    lock=subprocess.check_output([str(uv),'pip','freeze','--python',str(env/'bin/python')],text=True)
    (out/'environment.lock').write_text(lock)
    model=runtime/'envs/pkai/lib/python3.11/site-packages/pkai/models/pKAI_model.pt'
    manifest={'handoff':str(handoff),'handoff_verification_sha256':digest(handoff/'verification.json'),
      'model':str(model),'model_sha256':digest(model),'code_sha256':digest(Path(__file__)),
      'torch_environment_lock_sha256':digest(runtime/'manifests/pkai.requirements.lock'),
      'environment':str(env),'environment_lock_sha256':digest(out/'environment.lock'),
      'input_sha256':{n:digest(out/n) for n in ('rows.parquet','catboost.parquet','ab.npy','free.npy')},
      'arms':list(ARMS),'seeds':list(SEEDS),'epochs':40,'patience':8,'batch_size':128,
      'learning_rates':{'last':.001,'all':.0001,'scratch':.001},'weight_decay':.0001,
      'fit_support':'training interface only','selection':'validation interface group-macro MAE; epoch zero is eligible',
      'normalization':'per residue training-interface target SD, floored at 0.25',
      'weighting':'equal groups, equal complexes per group, equal sites per complex; multiply by 2 for abs(target)>0.5; normalize mean weight to 1',
      'dropout':'disabled in both paired branches, including fitting; weights shared',
      'test_data_included':False,'counts':dict(Counter(r['split']+(':interface' if r['interface'] else ':other') for r in rows))}
    atomic_json(out/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)

def read_inputs(out):
    m=json.loads((out/'manifest.json').read_text())
    assert digest(Path(__file__))==m['code_sha256']
    assert digest(Path(m['model']))==m['model_sha256']
    for n,sha in m['input_sha256'].items(): assert digest(out/n)==sha
    assert digest(out/'environment.lock')==m['environment_lock_sha256']
    return m

def macro(rows,pred,indices):
    import numpy as np
    complexes=defaultdict(list)
    for i in indices: complexes[rows[i]['complex_id']].append(i)
    groups=defaultdict(list)
    for ii in complexes.values():
        err=np.asarray([pred[i]-rows[i]['target_delta_pka'] for i in ii]); y=np.asarray([rows[i]['target_delta_pka'] for i in ii])
        denom=float(np.mean(y*y))
        groups[rows[ii[0]]['component_id']].append({'mae':float(np.mean(abs(err))), 'rmse':float(np.sqrt(np.mean(err*err))),
          'skill':float(1-np.mean(err*err)/denom) if denom>0 else None})
    gm={g:{k:float(np.mean([a[k] for a in rr if a[k] is not None])) if any(a[k] is not None for a in rr) else None for k in ('mae','rmse','skill')} for g,rr in groups.items()}
    return {k:float(np.mean([r[k] for r in gm.values() if r[k] is not None])) if any(r[k] is not None for r in gm.values()) else None for k in ('mae','rmse','skill')},gm

def weights(rows,indices):
    import numpy as np
    sitecounts=Counter(rows[i]['complex_id'] for i in indices)
    gc=defaultdict(set)
    for i in indices: gc[rows[i]['component_id']].add(rows[i]['complex_id'])
    w=np.asarray([(2 if abs(rows[i]['target_delta_pka'])>.5 else 1)/(sitecounts[rows[i]['complex_id']]*len(gc[rows[i]['component_id']])) for i in indices],dtype=np.float32)
    return w/w.mean()

def fit(out,arm,seed):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    m=read_inputs(out); dest=out/f'{arm}-{seed}'; dest.mkdir(exist_ok=False)
    rows=table(out/('catboost.parquet' if arm=='catboost' else 'rows.parquet'))
    train=np.asarray([i for i,r in enumerate(rows) if r['split']=='train' and r['interface']]); val=np.asarray([i for i,r in enumerate(rows) if r['split']=='val' and r['interface']])
    assert len(train)>0 and len(val)>0
    target=np.asarray([r['target_delta_pka'] for r in rows],dtype=np.float32)
    history=[]; extras={}
    if arm=='catboost':
        from catboost import CatBoostRegressor
        names=FEATURES+sorted(k for k in rows[0] if k.startswith('residue_') and k!='residue_delta_sasa')
        # Explicit allowlist prevents labels/baseline pKas from entering features.
        x=np.asarray([[float(r[n]) if r[n] is not None else np.nan for n in names] for r in rows])
        residual=np.asarray([r['catboost_target'] for r in rows]); prediction=np.asarray([r['propka_delta'] for r in rows])
        importance=np.zeros(len(names)); classes={g:('acid' if g in ('ASP','GLU','CYS','TYR','CTERM') else 'base') for g in {r['group'] for r in rows}}
        for cls in ('acid','base'):
            tr=np.asarray([i for i in train if classes[rows[i]['group']]==cls]); va=np.asarray([i for i in val if classes[rows[i]['group']]==cls]); allidx=np.asarray([i for i,r in enumerate(rows) if classes[r['group']]==cls])
            assert len(tr) and len(va)
            model=CatBoostRegressor(iterations=500,depth=5,learning_rate=.03,loss_function='Huber:delta=1.0',random_seed=seed,thread_count=2,allow_writing_files=False,verbose=False)
            model.fit(x[tr],residual[tr],sample_weight=weights(rows,tr))
            prediction[allidx]+=model.predict(x[allidx]); model.save_model(str(dest/f'{cls}.cbm'))
            importance+=model.feature_importances_/2
        with (dest/'feature_importance.csv').open('w') as f:
            w=csv.writer(f); w.writerow(['feature','importance']); w.writerows(zip(names,importance.tolist()))
        extras={'features':names,'iterations':500,'validation_used_for_fit':False}
    else:
        runtime=Path(os.environ['PKABENCH_RUNTIME']); sys.path.append(str(runtime/'envs/pkai/lib/python3.11/site-packages'))
        import torch
        torch.set_num_threads(2); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
        model=torch.jit.load(m['model'],map_location='cpu'); model.eval()
        if arm=='scratch':
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name.endswith('weight'): torch.nn.init.kaiming_uniform_(p,a=math.sqrt(5))
                    else:
                        weight=dict(model.named_parameters())[name[:-4]+'weight']; bound=1/math.sqrt(weight.shape[1]); torch.nn.init.uniform_(p,-bound,bound)
        for name,p in model.named_parameters(): p.requires_grad_(arm in ('all','scratch') or (arm=='last' and name.startswith('layers.3.')))
        # mmap avoids loading unused far-field feature arrays into the training heap.
        ab=np.load(out/'ab.npy',mmap_mode='r'); free=np.load(out/'free.npy',mmap_mode='r')
        def predict():
            result=np.empty(len(rows))
            with torch.no_grad():
                for start in range(0,len(rows),256):
                    a=torch.tensor(np.array(ab[start:start+256])); b=torch.tensor(np.array(free[start:start+256]))
                    result[start:start+256]=(model(a)-model(b)).reshape(-1).numpy()
            assert np.isfinite(result).all()
            return result
        prediction=predict(); initial=prediction.copy()
        # Native paired output must match both the handoff and identical-input cancellation.
        if arm!='scratch':
            ii=[i for i,r in enumerate(rows) if r['frozen_pkai_ab'] is not None]
            assert max(abs(prediction[i]-(rows[i]['frozen_pkai_ab']-rows[i]['frozen_pkai_free'])) for i in ii)<=.0105
        with torch.no_grad():
            a=torch.tensor(np.array(ab[:8])); assert torch.equal(model(a)-model(a),torch.zeros((len(a),1)))
        best=macro(rows,prediction,val)[0]['mae']; bestepoch=0
        model.save(str(dest/'best.pt'))
        if arm!='frozen':
            scales={g:max(.25,float(np.std([target[i] for i in train if rows[i]['group']==g]))) for g in {rows[i]['group'] for i in train}}
            scale=torch.tensor([scales[rows[i]['group']] for i in train]); tw=torch.tensor(weights(rows,train)); y=torch.tensor(target[train])
            a=torch.tensor(np.array(ab[train])); b=torch.tensor(np.array(free[train])); before={n:p.detach().clone() for n,p in model.named_parameters()}
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=m['learning_rates'][arm],weight_decay=m['weight_decay'])
            stall=0
            for epoch in range(1,m['epochs']+1):
                order=torch.randperm(len(train)); losses=[]
                for ix in order.split(m['batch_size']):
                    opt.zero_grad(); delta=(model(a[ix])-model(b[ix])).reshape(-1)
                    loss=(torch.nn.functional.huber_loss(delta/scale[ix],y[ix]/scale[ix],reduction='none')*tw[ix]).mean()
                    assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); losses.append(float(loss.detach()))
                prediction=predict(); score=macro(rows,prediction,val)[0]['mae']; history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_group_mae':score})
                print(json.dumps({'arm':arm,'seed':seed,**history[-1]}),flush=True)
                if score<best-1e-6: best=score; bestepoch=epoch; stall=0; model.save(str(dest/'best.pt'))
                else: stall+=1
                if stall>=m['patience']: break
            assert any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters()),'No parameters changed'
            extras['training_scales']=scales
        model=torch.jit.load(str(dest/'best.pt')); model.eval(); prediction=predict()
        extras.update(best_epoch=bestepoch,best_validation_group_mae=best,initial_validation_group_mae=macro(rows,initial,val)[0]['mae'],device='cpu')
    assert np.isfinite(prediction).all()
    pq.write_table(pa.Table.from_pylist([r|{'prediction':float(p),'arm':arm,'seed':seed} for r,p in zip(rows,prediction)]),dest/'predictions.parquet')
    atomic_json(dest/'history.json',history)
    atomic_json(dest/'receipt.json',{'complete':True,'arm':arm,'seed':seed,'manifest_sha256':digest(out/'manifest.json'),'prediction_sha256':digest(dest/'predictions.parquet'),'train_sites':len(train),'val_sites':len(val),'test_data_included':False,**extras})
    print(json.dumps({'complete':True,'arm':arm,'seed':seed,**extras}),flush=True)

def collect(out):
    import numpy as np
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=read_inputs(out); runs={}
    for arm in ARMS:
        for seed in ((17,) if arm=='frozen' else SEEDS):
            dest=out/f'{arm}-{seed}'; rec=json.loads((dest/'receipt.json').read_text())
            assert rec['complete'] and rec['manifest_sha256']==digest(out/'manifest.json') and digest(dest/'predictions.parquet')==rec['prediction_sha256']
            runs[arm,seed]=table(dest/'predictions.parquet')
    common=set.intersection(*({key(r) for r in rr} for rr in runs.values()))
    summaries=[]; paired={}
    for (arm,seed),allrows in runs.items():
        rows=[r for r in allrows if key(r) in common]; pred=np.asarray([r['prediction'] for r in rows])
        for split in ('train','val'):
            for subset in ('interface','shell_0_20'):
                for role in ('all','antibody_antigen','general'):
                    idx=[i for i,r in enumerate(rows) if r['split']==split and r[subset] and (role=='all' or r['role']==role)]
                    if not idx: continue
                    score,gm=macro(rows,pred,idx); rng=np.random.default_rng(20261005); values=np.asarray([g['mae'] for g in gm.values()])
                    ci=np.quantile(rng.choice(values,(2000,len(values))).mean(axis=1),[.025,.975]).tolist() if len(values)>=5 else [None,None]
                    y=np.asarray([rows[i]['target_delta_pka'] for i in idx]); p=pred[idx]; sig=abs(y)>.5
                    rho=float(spearmanr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else None
                    summaries.append({'arm':arm,'seed':seed,'split':split,'subset':subset,'role':role,'sites':len(idx),'groups':len(gm),**score,'mae_ci_low':ci[0],'mae_ci_high':ci[1],'pooled_spearman':rho,'pooled_sign_accuracy_abs_ref_gt_0_5':float(np.mean(np.sign(y[sig])==np.sign(p[sig]))) if sig.any() else None})
                    if split=='val' and subset=='interface' and role=='all': paired[arm,seed]=gm
    with (out/'arms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    base=paired['frozen',17]; comparisons=[]
    for (arm,seed),gm in paired.items():
        if arm=='frozen': continue
        gs=sorted(base.keys()&gm.keys()); delta=np.asarray([gm[g]['mae']-base[g]['mae'] for g in gs]); rng=np.random.default_rng(20261005)
        ci=np.quantile(rng.choice(delta,(2000,len(delta))).mean(axis=1),[.025,.975]).tolist()
        comparisons.append({'arm':arm,'seed':seed,'mae_change_vs_frozen':float(delta.mean()),'ci95':ci})
    atomic_json(out/'comparisons.json',comparisons)
    rows=[r for r in summaries if r['split']=='val' and r['subset']=='interface' and r['role']=='all']
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for j,arm in enumerate(ARMS):
        rr=[r for r in rows if r['arm']==arm]; ax.scatter([j]*len(rr),[r['mae'] for r in rr])
    ax.set(xticks=range(len(ARMS)),xticklabels=ARMS,ylabel='Validation group-macro ΔpKa MAE',title='Common interface support; dots show training seeds')
    fig.savefig(out/'arms.png',dpi=180); plt.close(fig)
    imp=defaultdict(list)
    for seed in SEEDS:
        with (out/f'catboost-{seed}/feature_importance.csv').open() as f:
            for r in csv.DictReader(f): imp[r['feature']].append(float(r['importance']))
    names=sorted(imp,key=lambda n:np.mean(imp[n]))[-15:]; fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    ax.barh(names,[np.mean(imp[n]) for n in names]); ax.set_title('CatBoost feature importance, mean across seeds'); fig.savefig(out/'feature_importance.png',dpi=180); plt.close(fig)
    lines=['# Experiment 02 diagnostic results','','Validation was used for neural checkpoint selection; these are development results, not a new held-out test claim. All arms below use identical common interface support. Training used only training-interface sites.','','| Arm | Seed | Validation MAE | 95% group bootstrap interval | Skill |','|---|---:|---:|---|---:|']
    for r in rows: lines.append(f"| {r['arm']} | {r['seed']} | {r['mae']:.4f} | {r['mae_ci_low']:.4f}–{r['mae_ci_high']:.4f} | {r['skill']:.4f} |")
    lines+=['','Paired changes versus frozen pKAI are in comparisons.json; negative favors the trained arm. Seed dispersion and bootstrap intervals describe different sources of variability. Antibody/general strata and shell results are in arms.csv. Spearman and sign accuracy are explicitly pooled secondary diagnostics.','',
     'No result alone proves an architecture limitation or establishes experimental accuracy. CatBoost and pKAI have different representable supports, so comparisons use their intersection; receipts retain method-specific counts. Experimental Set 2 and a prospectively fixed final test evaluation remain separate.','',f'![Validation arms]({out}/arms.png)',f'![Feature importance]({out}/feature_importance.png)']
    (out/'decision.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',{'passed':True,'runs':len(runs),'common_sites':len(common),'test_data_included':False,'manifest_sha256':digest(out/'manifest.json'),'artifacts_sha256':{n:digest(out/n) for n in ('arms.csv','comparisons.json','arms.png','feature_importance.png','decision.md')}})
    print((out/'decision.md').read_text(),flush=True)

def main():
    # This isolated diagnostic can use its allocated CPUs; leave shared guards unchanged.
    affinity=os.sched_getaffinity(0); require_compute(); os.sched_setaffinity(0,affinity)
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['setup','fit','collect']); p.add_argument('--out',type=Path,required=True); p.add_argument('--handoff',type=Path); p.add_argument('--arm',choices=ARMS); p.add_argument('--seed',type=int,default=17); a=p.parse_args()
    if a.stage=='setup': setup(a.out,a.handoff)
    elif a.stage=='fit': fit(a.out,a.arm,a.seed)
    else: collect(a.out)
if __name__=='__main__': main()
