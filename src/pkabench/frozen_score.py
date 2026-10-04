"""Frozen-mask PB agreement, per-complex/group macro scores and group bootstrap."""
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from .runtime import require_compute, atomic_json, digest
from .schema import key, read_table, PH
from .score import correlation
from .linkage import integrate

METRICS=('mae','rmse','skill','spearman','sign_accuracy','error_cancellation')


def measures(ref,pred,ab_errors=None,free_errors=None):
    a=np.asarray(ref,float); b=np.asarray(pred,float)
    if not len(a): return {m:None for m in METRICS}
    mse=float(np.mean((a-b)**2)); denom=float(np.mean(a*a)); signs=np.abs(a)>=.5
    return {'mae':float(np.mean(np.abs(a-b))),'rmse':float(np.sqrt(mse)),
        'skill':1-mse/denom if denom>0 else None,'spearman':correlation(a,b,True),
        'sign_accuracy':float(np.mean(np.sign(a[signs])==np.sign(b[signs]))) if signs.any() else None,
        'error_cancellation':correlation(ab_errors,free_errors) if ab_errors is not None else None}


def aggregate(complexes,replicates=2000,seed=20261004):
    """Equal complex weight within group; equal group weight across groups."""
    bygroup=defaultdict(list)
    for r in complexes: bygroup[r['component_id']].append(r)
    groups=[]
    for gid,rr in sorted(bygroup.items()):
        row={'component_id':gid,'complexes':len(rr)}
        for m in METRICS:
            v=[r[m] for r in rr if r[m] is not None and np.isfinite(r[m])]
            row[m]=float(np.mean(v)) if v else None
        groups.append(row)
    result={'complexes':len(complexes),'groups':len(groups),'sites':sum(r['n'] for r in complexes)}
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(groups),(replicates,len(groups))) if groups else None
    for m in METRICS:
        values=np.array([r[m] if r[m] is not None else np.nan for r in groups]); valid=np.isfinite(values)
        result[m]=float(values[valid].mean()) if valid.any() else None
        result[m+'_groups']=int(valid.sum()); ci=None
        if valid.sum()>=5:
            sampled=values[draws]; counts=np.isfinite(sampled).sum(axis=1)
            boot=np.nansum(sampled,axis=1)[counts>0]/counts[counts>0]
            ci=np.quantile(boot,[.025,.975]).tolist()
        result[m+'_ci95']=ci
    return result,groups


def pairs(rows,sites,masks,method):
    index={}; output={}
    for r in rows:
        if r['method']!=method: continue
        k=(key(r),r['state'])
        if k in index: raise ValueError('Duplicate prediction identity')
        index[k]=r
    for s in sites:
        k=key(s); m=masks[k]
        if not (m['training_eligible'] or m['evaluation_eligible']): continue
        a=index.get((k,'AB')); f=index.get((k,s['partner']))
        if a and f and a['status']==f['status']=='ok' and all(r['pka'] is not None and np.isfinite(r['pka']) for r in (a,f)):
            output[k]=(a['pka']-f['pka'],a['pka'],f['pka'])
    return output


def complete_linkage(rows,sites,masks):
    if not sites: return {'status':'no_sites','delta_q':None,'delta_g':None}
    if any(not (masks[key(s)]['training_eligible'] or masks[key(s)]['evaluation_eligible']) for s in sites):
        return {'status':'masked_uncertain_charge_coverage','delta_q':None,'delta_g':None}
    indexed={(key(r),r['state']):r for r in rows}; delta=np.zeros(73); sources=set()
    for s in sites:
        rr=[indexed.get((key(s),state)) for state in ('AB',s['partner'])]
        if any(r is None or r['status'] in ('failed','not_reported') or r['curve'] is None for r in rr):
            return {'status':'incomplete_charge_coverage','delta_q':None,'delta_g':None}
        a,b=[np.asarray(r['curve'],float) for r in rr]
        if any(x.shape!=(73,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)) for x in (a,b)): raise ValueError('Invalid native/HH curve')
        delta+=a-b; sources.update(r['curve_source'] for r in rr)
    return {'status':'ok','curve_sources':sorted(sources),'delta_q':delta.tolist(),'delta_g':integrate(delta).tolist()}


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def score(campaign):
    require_compute(); campaign=Path(campaign)
    manifest=json.loads((campaign/'manifest.json').read_text()); methods=[m for m in manifest['methods'] if m!='pypka']
    assert digest(campaign/'site_masks.parquet')==manifest['site_masks_sha256']
    rows=read_table(campaign/'predictions.parquet'); sites=read_table(campaign/'sites.parquet'); mm=read_table(campaign/'site_masks.parquet')
    annotations={key(s):s for s in sites}; masks={key(m):m for m in mm}; assert len(annotations)==len(sites)==len(masks)
    assignments={r['complex_id']:r for r in read_table(campaign/'assignments.parquet')}
    eligible={k for k,m in masks.items() if m['training_eligible'] or m['evaluation_eligible']}
    ref=pairs(rows,sites,masks,'pypka'); models={m:pairs(rows,sites,masks,m) for m in methods}
    common=set(ref)
    for values in models.values(): common &= set(values)
    summary=[]; percomplex=[]; pergroup=[]; coverage=[]; represent=[]
    def subsets(k):
        s=annotations[k]; a=assignments[k[0]]; delta=abs(ref[k][0]) if k in ref else 0; sasa=s['residue_delta_sasa']; d=s['min_partner_distance']
        names=[]
        if masks[k]['interface']: names.append('interface')
        if d<=20: names.append('shell_0_20')
        if masks[k]['interface']:
            names.extend(['interface:residue:'+s['group'],'interface:role:'+a['role'],'interface:shift:'+('<0.1' if delta<.1 else '0.1-0.5' if delta<.5 else '>=0.5'),'interface:dSASA:'+('10-30' if sasa<30 else '30-60' if sasa<60 else '>=60')])
        for lo,hi in ((0,5),(5,10),(10,15),(15,20)):
            if lo<=d<hi or hi==20 and d==20: names.append(f'shell_{lo}_{hi}')
        return names
    for method,model in models.items():
        for split in ('train','val','test'):
            universe={k for k in eligible if assignments[k[0]]['split']==split}
            for label in ('interface','shell_0_20'):
                selected={k for k in universe if masks[k]['interface']} if label=='interface' else {k for k in universe if annotations[k]['min_partner_distance']<=20}
                coverage.append({'method':method,'split':split,'subset':label,'eligible_sites':len(selected),'teacher_sites':len(selected&ref.keys()),'method_sites':len(selected&model.keys()),'matched_sites':len(selected&ref.keys()&model.keys()),'all_method_common_sites':len(selected&common),'teacher_coverage':len(selected&ref.keys())/len(selected) if selected else None,'method_coverage':len(selected&model.keys())/len(selected) if selected else None})
            for scope,keys in (('pairwise',universe&ref.keys()&model.keys()),('all_method_common',universe&common)):
                buckets=defaultdict(list)
                for k in sorted(keys):
                    for subset in subsets(k): buckets[subset].append(k)
                for subset,kk in sorted(buckets.items()):
                    grouped=defaultdict(list)
                    for k in kk: grouped[k[0]].append(k)
                    cr=[]
                    tags={'method':method,'split':split,'scope':scope,'subset':subset}
                    for cid,ks in grouped.items():
                        record=tags|{'complex_id':cid,'component_id':assignments[cid]['component_id'],'n':len(ks)}|measures([ref[k][0] for k in ks],[model[k][0] for k in ks],[model[k][1]-ref[k][1] for k in ks],[model[k][2]-ref[k][2] for k in ks])
                        cr.append(record)
                    agg,gr=aggregate(cr,replicates=2000 if subset in ('interface','shell_0_20') else 400)
                    summary.append(tags|agg); percomplex.extend(cr); pergroup.extend(tags|g for g in gr)
            for lo,hi in ((0,5),(5,10),(10,15),(15,20)):
                kk=[k for k in universe&ref.keys()&model.keys() if abs(ref[k][0])>=.1 and (lo<=annotations[k]['min_partner_distance']<hi or hi==20 and annotations[k]['min_partner_distance']==20)]
                bycid=defaultdict(list)
                for k in kk: bycid[k[0]].append(abs(model[k][0])<.01)
                bygroup=defaultdict(list)
                for cid,values in bycid.items(): bygroup[assignments[cid]['component_id']].append(float(np.mean(values)))
                represent.append({'method':method,'split':split,'shell':f'{lo}-{hi}','reference_threshold':.1,'zero_threshold':.01,'matched_sites':len(kk),'complexes':len(bycid),'groups':len(bygroup),'group_macro_structural_zero_fraction':float(np.mean([np.mean(v) for v in bygroup.values()])) if bygroup else None})
    curves=[]
    bysites=defaultdict(list); bypred=defaultdict(list)
    for s in sites: bysites[s['complex_id']].append(s)
    for r in rows: bypred[r['complex_id'],r['method']].append(r)
    for cid,ss in bysites.items():
        for method in manifest['methods']:
            result=complete_linkage(bypred[cid,method],ss,masks)
            side=json.loads((campaign/'jobs'/method/f'{cid}.json').read_text())
            unsupported=sorted({g for v in side.get('extra',{}).values() for g in v.get('unsupported_groups',[])})
            if unsupported: result={'status':'incomplete_charge_coverage','unsupported_groups':unsupported,'delta_q':None,'delta_g':None}
            curves.append({'complex_id':cid,'method':method,**result})
    atomic_json(campaign/'scores_set1.json',summary); write_csv(campaign/'scores_set1.csv',summary); write_csv(campaign/'scores_per_complex.csv',percomplex); write_csv(campaign/'scores_per_group.csv',pergroup)
    write_csv(campaign/'coverage.csv',coverage); write_csv(campaign/'representability.csv',represent); atomic_json(campaign/'linkage.json',curves)
    atomic_json(campaign/'scoring_report.json',{'methods':methods,'expected_eligible_sites':len(eligible),'teacher_paired_sites':len(ref),'common_paired_sites':len(common),'prediction_status_counts':dict(Counter(f"{r['method']}:{r['status']}" for r in rows)),
        'summary_rows':len(summary),'linkage_statuses':dict(Counter(r['status'] for r in curves)),
        'aggregation':'Per-complex site metrics, equal complex means within sequence group, equal group means across groups. Percentile bootstrap resamples whole groups; CIs require >=5 valid groups.',
        'coverage':'Pairwise support plus common support across all scored methods, reported by frozen split. Failed/out-of-range/not-reported midpoints excluded explicitly, never imputed.',
        'interpretation':'PB agreement on a selected engineering smoke sample, not independent experimental accuracy or production benchmark estimates. pKAI/pKAI+ share teacher lineage.',
        'manifest_sha256':digest(campaign/'manifest.json'),'scorer_sha256':digest(Path(__file__))})
    print(json.dumps({'eligible':len(eligible),'teacher_paired':len(ref),'all_method_common':len(common),'summary_rows':len(summary)}),flush=True)
