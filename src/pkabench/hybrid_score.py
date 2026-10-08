"""Group-weighted validation report and implementation gate for hybrid MC."""
import json
import sys
from pathlib import Path
from collections import Counter
from .runtime import require_compute, atomic_json, digest
from .hybrid_mc import KEY, METHODS, read_json, sitekey, changed_energies

def group_errors(frame):
    """Equal states per site, sites per complex, complexes per frozen group."""
    sites=frame.groupby(['component_id']+KEY)[['ae','se']].mean()
    complexes=sites.groupby(['component_id','complex_id']).mean()
    complexes['rmse']=complexes.se**.5
    return complexes.groupby('component_id')[['ae','rmse']].mean().rename(columns={'ae':'mae'})

def summarize(frame,metadata):
    import numpy as np
    groups=group_errors(frame)
    result=dict(metadata,observations=len(frame),sites=len(frame[KEY].drop_duplicates()),complexes=frame.complex_id.nunique(),groups=len(groups),
        mae=float(groups.mae.mean()),rmse=float(groups.rmse.mean()),ci_low=None,ci_high=None)
    if len(groups)>=5:
        rng=np.random.default_rng(20261005); v=groups.mae.to_numpy()
        boot=v[rng.integers(len(v),size=(2000,len(v)))].mean(axis=1)
        result['ci_low'],result['ci_high']=map(float,np.quantile(boot,[.025,.975]))
    return result,groups

def collect(out,phase):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import hybrid_mc
    manifest=read_json(out/'manifest.json'); mh=digest(out/'manifest.json')
    assert digest(Path(hybrid_mc.__file__))==manifest['code_sha256']
    assert not manifest['test_data_included']
    cids=manifest['eligible'] if phase=='full' else manifest['pilot']
    if phase=='probe':
        cids=[c for c in cids if (out/'complexes'/c/'receipt.json').exists()]
        assert cids, 'No completed pilot complex for scoring preflight'
    if phase=='full':
        gate=read_json(out/'pilot/verification.json'); assert gate['passed'] and gate['manifest_sha256']==mh
    records=[]; coverage=[]; inputs={}; replays=[]; pkai=[]
    prod=Path(read_json(Path(manifest['native'])/'manifest.json')['source'])
    for cid in cids:
        request=read_json(out/'requests'/f'{cid}.json'); assert request['split']=='val'
        assert digest(out/'requests'/f'{cid}.json')==manifest['request_sha256'][cid]
        receipt_path=out/'complexes'/cid/'receipt.json'; receipt=read_json(receipt_path)
        assert receipt['passed'] and receipt['manifest_sha256']==mh and len(receipt['outputs_sha256'])==18
        inputs[str(receipt_path)]=digest(receipt_path)
        for path,h in receipt['outputs_sha256'].items(): assert digest(path)==h
        for state in ('AB','A','B'):
            s=request['states'][state]; raw=Path(s['source']); original=read_json(raw/'mc-energies.json')
            assert all(digest(raw/name)==h for name,h in s['source_hashes'].items())
            sites=read_json(Path(s['export'])/'sites.json'); assert digest(Path(s['export'])/'sites.json')==s['sites_sha256']
            teacher=read_json(out/'complexes'/cid/state/'teacher/result.json')
            assert teacher['passed'] and teacher['max_replay_curve_error']<=1e-12 and teacher['max_replay_pka_error']<=1e-10
            replays.append({k:teacher[k] for k in ('complex_id','state','max_replay_curve_error','max_replay_pka_error')})
            refs={sitekey(r):r for r in teacher['rows']}
            coverage.append(dict(complex_id=cid,state=state,role=request['role'],n_residues=request['n_residues'],
                native_sites=len(sites),replaced_sites=len(s['replacements']),masked_sites=sum(not r['supervision_eligible'] for r in sites),
                eligible_unpredicted_sites=sum(r['supervision_eligible'] for r in sites)-len(s['replacements'])))
            for method in METHODS[1:]:
                folder=out/'complexes'/cid/state/method; result=read_json(folder/'result.json')
                assert result['passed'] and result['manifest_sha256']==mh and result['code_sha256']==manifest['code_sha256']
                assert digest(folder/'energies.json')==result['energy_sha256']
                # Independently recheck immutable physical context and exact replacement support.
                assert read_json(folder/'energies.json')==changed_energies(original,sites,s['replacements'],method)
                assert {sitekey(r) for r in result['rows']}==refs.keys()
                assert len(result['rows'])==len(refs)
                for r in result['rows']:
                    ref=refs[sitekey(r)]
                    assert r['replaceable']==ref['replaceable'] and r['supervision_eligible']==ref['supervision_eligible']
                    curve=np.asarray(r['curve']); target=np.asarray(ref['curve'])
                    assert curve.shape==(73,) and np.isfinite(curve).all() and ((curve>=0)&(curve<=1)).all()
                    if r['replaceable']:
                        records.append({k:r[k] for k in KEY}|dict(state=state,method=method,component_id=request['component_id'],role=request['role'],
                            interface=r['interface'],prediction=r['pka'],target=ref['pka'],curve_ae=float(abs(curve-target).mean()),curve_se=float(((curve-target)**2).mean())))
        # Read only selected validation entries, never aggregate test tables.
        pp=prod/'jobs/pkai'/f'{cid}.parquet'; rp=pp.with_suffix('.json')
        if pp.exists() and rp.exists():
            side=read_json(rp); assert digest(pp)==side['output_sha256']; inputs[str(pp)]=digest(pp)
            for r in pq.read_table(pp).to_pylist():
                assert r['complex_id']==cid and r['method']=='pkai'
                if r['status']=='ok' and r['pka'] is not None:
                    pkai.append({k:r[k] for k in KEY}|dict(state=r['state'],pkai=float(r['pka'])))
    d=pd.DataFrame(records); assert not d.duplicated(KEY+['state','method']).any()
    p=pd.DataFrame(pkai,columns=KEY+['state','pkai']); assert not p.duplicated(KEY+['state']).any()
    # Pair without using midpoint availability: all curves count, including out-of-range pKas.
    ab=d[d.state=='AB'].drop(columns='state'); free=d[d.state!='AB'].drop(columns='state')
    assert not free.duplicated(KEY+['method']).any()
    paired=ab.merge(free,on=KEY+['method'],suffixes=('_ab','_free'),validate='one_to_one')
    for col in ('component_id','role','interface'):
        if col!='interface': assert (paired[col+'_ab']==paired[col+'_free']).all()
        paired[col]=paired[col+'_ab']
    paired['prediction']=paired.prediction_ab-paired.prediction_free; paired['target']=paired.target_ab-paired.target_free
    paired['curve_ae']=(paired.curve_ae_ab+paired.curve_ae_free)/2
    paired['curve_se']=(paired.curve_se_ab+paired.curve_se_free)/2
    pab=p[p.state=='AB'].drop(columns='state'); pf=p[p.state!='AB'].drop(columns='state')
    assert not pf.duplicated(KEY).any()
    pdelt=pab.merge(pf,on=KEY,suffixes=('_ab','_free'),validate='one_to_one')
    pdelt['pkai']=pdelt.pkai_ab-pdelt.pkai_free
    joined=d.merge(p,on=KEY+['state'],how='left',validate='many_to_one')
    joined_pair=paired.merge(pdelt[KEY+['pkai']],on=KEY,how='left',validate='many_to_one')
    metrics=[]; contrasts=[]; midpoint_coverage=[]; group_frames=[]
    for task,frame,idcols in [('absolute_pka',joined,KEY+['state']),('paired_delta',joined_pair,KEY)]:
        for subset in ('all','interface'):
            for role in ('all','antibody_antigen','general'):
                sub=frame.copy()
                if subset=='interface': sub=sub[sub.interface]
                if role!='all': sub=sub[sub.role==role]
                if sub.empty: continue
                meta=dict(task=task,subset=subset,role=role)
                # Absolute primary support additionally requires the matching free/bound site.
                if task=='absolute_pka':
                    keys=paired[KEY].drop_duplicates(); sub=sub.merge(keys,on=KEY,validate='many_to_one')
                if sub.empty: continue
                valid=np.isfinite(sub.prediction)&np.isfinite(sub.target)
                for method in METHODS[1:]:
                    sel=sub.method==method
                    midpoint_coverage.append(dict(meta,method=method,eligible=len(sub[sel]),teacher_midpoints=int(sub.loc[sel,'target'].notna().sum()),
                        predicted_midpoints=int(sub.loc[sel,'prediction'].notna().sum()),matched_midpoints=int((sel&valid).sum())))
                # Own support is descriptive; comparisons use complete common support below.
                for method,g in sub[valid].groupby('method'):
                    g=g.copy(); g['ae']=abs(g.prediction-g.target); g['se']=(g.prediction-g.target)**2
                    score,_=summarize(g,dict(meta,method=method,support='method_available')); metrics.append(score)
                # Common hybrid support; no pKAI restriction for within-hybrid comparisons.
                counts=sub.assign(valid=valid).groupby(idcols).valid.agg(['sum','count'])
                common=counts[(counts['sum']==len(METHODS)-1)&(counts['count']==len(METHODS)-1)].reset_index()[idcols]
                matched=sub.merge(common,on=idcols,validate='many_to_one')
                cached={}
                for method,g in matched.groupby('method'):
                    g=g.copy(); g['ae']=abs(g.prediction-g.target); g['se']=(g.prediction-g.target)**2
                    score,groups=summarize(g,dict(meta,method=method,support='common_hybrid')); metrics.append(score); cached[method]=groups
                    group_frames.append(groups.reset_index().assign(**meta,method=method,support='common_hybrid'))
                for seed in (17,29,43):
                    name=f'catboost-{seed}'
                    for control in ('model_compound','train_constant'):
                        if name not in cached or control not in cached: continue
                        delta=cached[name].mae-cached[control].mae
                        ci=[None,None]
                        if len(delta)>=5:
                            rng=np.random.default_rng(20261005); values=delta.to_numpy(); boots=values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
                            ci=list(map(float,np.quantile(boots,[.025,.975])))
                        contrasts.append(dict(meta,method=name,control=control,groups=len(delta),mae_change=float(delta.mean()),ci_low=ci[0],ci_high=ci[1]))
                # Frozen pKAI on exactly the same midpoint/shift support as all hybrid methods.
                matched=matched[matched.pkai.notna()]
                for method,g in matched.groupby('method'):
                    g=g.copy(); g['ae']=abs(g.prediction-g.target); g['se']=(g.prediction-g.target)**2
                    score,_=summarize(g,dict(meta,method=method,support='common_with_pkai')); metrics.append(score)
                g=matched.drop_duplicates(idcols).copy()
                if len(g):
                    g['ae']=abs(g.pkai-g.target); g['se']=(g.pkai-g.target)**2
                    score,_=summarize(g,dict(meta,method='frozen_pkai',support='common_with_pkai')); metrics.append(score)
                # pKAI's own coverage on the entire eligible teacher support is explicit too.
                g=sub.drop_duplicates(idcols).copy(); ok=g.pkai.notna()&g.target.notna()
                midpoint_coverage.append(dict(meta,method='frozen_pkai',eligible=len(g),teacher_midpoints=int(g.target.notna().sum()),predicted_midpoints=int(g.pkai.notna().sum()),matched_midpoints=int(ok.sum())))
                if ok.any():
                    g=g[ok].copy(); g['ae']=abs(g.pkai-g.target); g['se']=(g.pkai-g.target)**2
                    score,_=summarize(g,dict(meta,method='frozen_pkai',support='method_available')); metrics.append(score)
    for subset in ('all','interface'):
        for role in ('all','antibody_antigen','general'):
            sub=paired.copy()
            if subset=='interface': sub=sub[sub.interface]
            if role!='all': sub=sub[sub.role==role]
            for method,g in sub.groupby('method'):
                g=g.copy(); g['ae']=g.curve_ae; g['se']=g.curve_se
                score,_=summarize(g,dict(task='paired_curves',subset=subset,role=role,method=method,support='all_paired_curves')); metrics.append(score)
    dest=out/phase; dest.mkdir(exist_ok=True)
    scores=pd.DataFrame(metrics); scores.to_csv(dest/'scores.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'replacement_coverage.csv',index=False)
    pd.DataFrame(midpoint_coverage).to_csv(dest/'midpoint_coverage.csv',index=False)
    pd.concat(group_frames,ignore_index=True).to_csv(dest/'group_scores.csv',index=False)
    d.to_parquet(dest/'site_errors.parquet',index=False); paired.to_parquet(dest/'paired_errors.parquet',index=False)
    atomic_json(dest/'contrasts.json',contrasts)
    totals={field:sum(r[field] for r in coverage) for field in ('native_sites','replaced_sites','masked_sites','eligible_unpredicted_sites')}
    summary={'complexes':len(cids),'roles':dict(Counter(read_json(out/'requests'/f'{c}.json')['role'] for c in cids)),
        'paired_sites':len(paired[KEY].drop_duplicates()),'paired_interface_sites':len(paired[paired.interface][KEY].drop_duplicates()),
        **totals,'teacher_intrinsic_fraction':1-totals['replaced_sites']/totals['native_sites'],
        'max_replay_curve_error':max(r['max_replay_curve_error'] for r in replays),'max_replay_pka_error':max(r['max_replay_pka_error'] for r in replays)}
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,task,support,title in zip(axes,['paired_curves','absolute_pka','paired_delta'],['all_paired_curves','common_with_pkai','common_with_pkai'],['Paired curves (occupancy MAE)','Coupled midpoint (pKa MAE)','Bound − free (pKa MAE)']):
        sub=scores[(scores.task==task)&(scores.subset=='interface')&(scores.role=='all')&(scores.support==support)].set_index('method')
        names=[m for m in METHODS[1:]+['frozen_pkai'] if m in sub.index]
        vals=[sub.loc[m,'mae'] for m in names]
        ax.bar(range(len(names)),vals,color=['#3974a5' if m.startswith('catboost') else '#999999' for m in names])
        for i,m in enumerate(names):
            lo=sub.loc[m,'ci_low']; hi=sub.loc[m,'ci_high']
            if pd.notna(lo) and pd.notna(hi): ax.plot([i,i],[lo,hi],color='black',linewidth=1)
        ax.set_xticks(range(len(names)),[m.replace('model_compound','model compound').replace('train_constant','train constant').replace('frozen_pkai','frozen pKAI') for m in names],rotation=45,ha='right'); ax.set_title(title); ax.set_ylim(bottom=0)
    fig.suptitle(f'Hybrid MC: {phase} validation, eligible interface sites\nTeacher interactions retained; group bootstrap 95% intervals')
    fig.tight_layout(); fig.savefig(dest/'validation.png',dpi=170); plt.close(fig)
    primary=scores[(scores.subset=='interface')&(scores.role=='all')&(((scores.task=='paired_curves')&(scores.support=='all_paired_curves'))|((scores.task!='paired_curves')&(scores.support=='common_with_pkai')))]
    lines=[f'# Hybrid MC {phase} validation', '', 'Implementation checks passed. This is agreement with the current PypKa teacher, not experimental accuracy or a standalone model.', '',
        f"{len(cids)} complexes; {summary['paired_interface_sites']} eligible paired interface sites; {summary['paired_sites']} eligible paired sites overall.",
        f"All {len(replays)} original-state replays passed: maximum curve error {summary['max_replay_curve_error']:.3g}, maximum midpoint error {summary['max_replay_pka_error']:.3g}.",
        f"Replaced {totals['replaced_sites']} of {totals['native_sites']} native site-states ({100*(1-summary['teacher_intrinsic_fraction']):.1f}%). Retained {totals['masked_sites']} masked and {totals['eligible_unpredicted_sites']} eligible unpredicted site-states with teacher intrinsics. All interactions remain teacher-derived.", '',
        '| Task | Method | MAE | 95% group CI | Sites | Groups |','|---|---|---:|---|---:|---:|']
    for r in primary.to_dict('records'):
        ci=f"{r['ci_low']:.4f}–{r['ci_high']:.4f}" if pd.notna(r['ci_low']) else 'not estimated (<5 groups)'
        lines.append(f"| {r['task']} | {r['method']} | {r['mae']:.4f} | {ci} | {r['sites']} | {r['groups']} |")
    lines += ['', 'Midpoint and shift rows above use common support with frozen pKAI; curves use all paired eligible sites, including sites without a midpoint in the sampled pH range. See midpoint_coverage.csv and scores.csv for per-method coverage and common-hybrid scores. Curve error averages the 73 pH points and both states. Each site, complex and frozen group receives equal weight at its respective aggregation stage. Intervals resample whole groups, 2,000 times.', '',
        'Pilot selection deliberately spans size and role and is not a representative prevalence estimate.' if phase=='pilot' else 'Full analysis includes every validation complex with all three native states and at least one eligible paired interface site. Nine validation complexes are excluded for documented data/support reasons, not prediction errors.', '',
        'Masked sites remain part of the physical system but are not scored as model predictions. Missing coupled midpoints are not imputed. No test data, new PB solve, or model fitting was used. MC settings match the saved teacher; hybrid midpoint shifts need not be zero even for constant intrinsics because teacher interactions differ between AB and free states.', '',
        'Original auxiliary states_ddG is preserved and is not a valid decomposition of the substituted intrinsic energies. Final scoring uses sampled occupancies and midpoints only.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    artifacts={p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='verification.json'}
    atomic_json(dest/'verification.json',{'passed':True,'phase':phase,'manifest_sha256':mh,'score_code_sha256':digest(Path(__file__)),
        'inputs_sha256':inputs,'artifacts_sha256':artifacts,'summary':summary,'test_data_included':False,'new_pb_solves':0,'hybrid':True})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    require_compute(); assert sys.argv[2] in ('probe','pilot','full'); collect(Path(sys.argv[1]).resolve(),sys.argv[2])
