"""Aggregate OPT3B evidence in a permitted Coulson CPU allocation (no PB)."""
import json
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np
import bench_jaxpb as BJ
O=Path(BJ.TP)/'opt3b'

def load(tag): return json.load(open(O/(tag+'.json')))
def run(r,label): return next(x for x in r['runs'] if x['label']==label)
def parity(a,b):
    return {k:a[k]==b[k] for k in ('outputs','raw_counts','shard_fields')}
def diff(z,aa,bb,ids,other=None):
    other=z if other is None else other
    dc=dr=total=rawtotal=states=rawstates=mm=0; p=[]; locations=[]; rawlocations=[]
    for sid in ids:
        a,b=z[f'{aa}/{sid}/prot'],other[f'{bb}/{sid}/prot']
        if np.any(a!=b):
            ph,site=np.argwhere(a!=b)[0]; locations.append(dict(state=sid,ph_index=int(ph),site=int(site),a=int(a[ph,site]),b=int(b[ph,site])))
        c,d=z[f'{aa}/{sid}/counts'],other[f'{bb}/{sid}/counts']
        rawstates+=int(np.any(c!=d))
        if np.any(c!=d):
            ph,microstate=np.argwhere(c!=d)[0]
            rawlocations.append(dict(state=sid,ph_index=int(ph),microstate=int(microstate),a=int(c[ph,microstate]),b=int(d[ph,microstate])))
        dc+=int(np.count_nonzero(a!=b));total+=a.size
        dr+=int(np.count_nonzero(c!=d));rawtotal+=c.size
        states+=int(np.any(a!=b))
        a,b=z[f'{aa}/{sid}/pka'],other[f'{bb}/{sid}/pka']
        finite=np.isfinite(a)&np.isfinite(b);p.extend(np.abs(a[finite]-b[finite]).tolist())
        mm+=int(np.count_nonzero(np.isfinite(a)!=np.isfinite(b)))
    return dict(differing_curve_counts=dc,curve_count_entries=total,differing_raw_counts=dr,raw_entries=rawtotal,
                differing_states=states,raw_differing_states=rawstates,states=len(ids),pka_mae=float(np.mean(p)),max_abs_delta_pka=max(p),midpoint_mismatches=mm,first_curve_difference_per_state=locations,first_raw_difference_per_state=rawlocations)

def main():
    cpu,ref,fresh,gpu=(load(k) for k in ('cpu-final2','ref-final2','fresh-final2','guard2'))
    api_check=load('api-check'); assert api_check['passed']
    unit=load('units-final2'); t1,t12=(load(k) for k in ('time1-final2','time12-final2'))
    assert all(r.get('completed') for r in (cpu,ref,fresh,gpu,t1,t12)), 'Incomplete experiment'
    zcpu,zgpu=(np.load(O/(tag+'-arrays.npz')) for tag in ('ref-final2','guard2'))
    teach=np.load(O/'teacher.npz')
    zfast=np.load(O/'cpu-final2-arrays.npz')
    out=dict(milestone='OPT3B',date='2026-11-23',baseline='a5d5c1d',committed=False,
             aggregation_job=os.environ['SLURM_JOB_ID'],gates={},rebaseline={},cross_backend={},timings={})
    g=out['gates']
    g['cpu_bitwise']={lab:parity(run(cpu,lab),run(ref,'new-'+lab)) for lab in ('50k','200k')}
    g['determinism']=dict(two_runs=cpu['repeat_exact'],fresh={lab:parity(run(cpu,lab),run(fresh,lab)) for lab in ('50k','200k')},
                          public_api=api_check['checks'],public_api_validation=api_check,superseded_curve_axis_checker=fresh['public_api'],cpu1_vs_cpu12_20x200k=parity(run(t1,'200k-all'),run(t12,'200k-all')))
    g['flat_compiles']=dict(total=cpu['compiles']['n_compiled'],warm=sum(x['compiles'] for x in run(cpu,'warm')['buckets']),
                           after_warm=sum(x['compiles'] for row in cpu['runs'][1:] for x in row['buckets']))
    g['unit_tests']=dict(passed=unit['passed'],new=unit['new'],existing=unit['existing'],api=unit['api'])
    for lab in ('50k','200k'):
        ids=run(ref,'new-'+lab)['states']
        out['rebaseline']['cpu-'+lab]=dict(change=diff(zcpu,'new-'+lab,'old-'+lab,ids),
                                         seed_noise=diff(zcpu,'noise-'+lab,'old-'+lab,ids))
    for lab in ('bench200k','l3'):
        ids=run(gpu,'new-'+lab)['states']
        out['rebaseline']['pallas-'+lab]=dict(change=diff(zgpu,'new-'+lab,'old-'+lab,ids),
                                            seed_noise=diff(zgpu,'noise-'+lab,'old-'+lab,ids))
    def metric(label):
        e=[];mm=0
        for sid in teach.files:
            a=zgpu[f'{label}/{sid}/pka'];b=teach[sid];f=np.isfinite(a)&np.isfinite(b)
            e.extend(np.abs(a[f]-b[f]).tolist());mm+=int(np.count_nonzero(np.isfinite(a)!=np.isfinite(b)))
        return dict(n=len(e),mae=float(np.mean(e)),max=float(np.max(e)),midpoint_mismatches=mm)
    out['l3_teacher']={k:metric(k+'-l3') for k in ('old','new','noise')}
    def cohort_teacher(z,label,ids):
        e=[];mm=0
        for sid in ids:
            a=z[f'{label}/{sid}/pka'];b=teach[sid if sid.startswith('L3:') else 'L3:'+sid]
            assert a.shape==b.shape,(sid,a.shape,b.shape)
            finite=np.isfinite(a)&np.isfinite(b)
            e.extend(np.abs(a[finite]-b[finite]).tolist());mm+=int(np.count_nonzero(np.isfinite(a)!=np.isfinite(b)))
        return dict(n=len(e),mae=float(np.mean(e)),max=float(np.max(e)),midpoint_mismatches=mm)
    out['teacher_metrics']={'pallas-l3':out['l3_teacher']}
    out['teacher_metrics']['pallas-bench200k']={k:cohort_teacher(zgpu,k+'-bench200k',run(gpu,'new-bench200k')['states']) for k in ('old','new','noise')}
    for lab in ('50k','200k'):
        out['teacher_metrics']['cpu-'+lab]={k:cohort_teacher(zcpu,k+'-'+lab,run(ref,'new-'+lab)['states']) for k in ('old','new','noise')}

    for lab in ('50k','200k'):
        a,b=run(gpu,'pallas-'+lab),run(gpu,'ref-'+lab)
        c=run(cpu,lab)
        out['cross_backend'][lab]=dict(pallas_vs_gpu_ref=parity(a,b),pallas_vs_cpu=parity(a,c),gpu_ref_vs_cpu=parity(b,c),
            differences=dict(pallas_vs_gpu_ref=diff(zgpu,'pallas-'+lab,'ref-'+lab,c['states']),
                             pallas_vs_cpu=diff(zgpu,'pallas-'+lab,lab,c['states'],other=zfast),
                             gpu_ref_vs_cpu=diff(zgpu,'ref-'+lab,lab,c['states'],other=zfast)))
    for tag,r in [('cpu1',t1),('cpu12',t12)]:
        row=run(r,'200k-all');out['timings'][tag]=dict(s_per_state=row['s_per_state'],buckets=row['buckets'],env=r['env'],job=r['job'],host_hwm_kb=r['host_hwm_kb'])
    ztime=np.load(O/'time12-final2-arrays.npz')
    out['cross_backend']['bench200k']=dict(pallas_vs_cpu=parity(run(gpu,'new-bench200k'),run(t12,'200k-all')),
        differences=diff(zgpu,'new-bench200k','200k-all',run(t12,'200k-all')['states'],other=ztime))
    out['timings']['phase3_cpu_s_per_state']={'cpu1':238.5,'cpu12':41.7}
    out['timings']['pallas']={lab:{name:run(gpu,name+'-'+lab)['s_per_state'] for name in ('old','new','noise')} for lab in ('bench200k','l3')}
    # Predeclare "well inside noise" quantitatively: change < 10% of seed noise
    # for curve-count disagreement, pKa MAE and max; teacher MAE drift < noise drift.
    checks={}
    for name,x in out['rebaseline'].items():
        checks[name]={k:x['change'][k]<=.1*x['seed_noise'][k] for k in ('differing_curve_counts','pka_mae','max_abs_delta_pka')}
    l=out['l3_teacher']
    drift=abs(l['new']['mae']-l['old']['mae']);noise=abs(l['noise']['mae']-l['old']['mae'])
    g['rebaseline']=dict(checks=checks,teacher_mae_drift=drift,teacher_seed_mae_drift=noise,teacher_within_noise=drift<=noise,
                         criterion='change <=10% seed noise for count disagreement, pKa MAE and maximum; teacher MAE drift <= seed drift')
    out['timings']['cpu_speedup_vs_phase3']={k:out['timings']['phase3_cpu_s_per_state'][k]/out['timings'][k]['s_per_state'] for k in ('cpu1','cpu12')}
    g['speed']={lab:v['new']<=v['old'] for lab,v in out['timings']['pallas'].items()}
    out['passed']=all(all(v.values()) for v in g['cpu_bitwise'].values()) and g['determinism']['two_runs'] and all(all(v.values()) for v in g['determinism']['fresh'].values()) and all(g['determinism']['public_api'].values()) and all(g['determinism']['cpu1_vs_cpu12_20x200k'].values()) and g['unit_tests']['passed'] and g['flat_compiles']['after_warm']==0 and all(all(v.values()) for v in checks.values()) and drift<=noise and all(g['speed'].values())
    BJ.jdump(out,str(O/'summary.json')); print(json.dumps(out,indent=2),flush=True)
    assert out['passed'], 'OPT3B gate failure; inspect summary.json'

if __name__=='__main__':main()
