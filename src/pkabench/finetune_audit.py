"""Independent pandas aggregation check and same-support physics/null baselines."""
import sys,json,csv,os
from pathlib import Path
from .runtime import require_compute,digest,atomic_json

def main():
    require_compute()
    import numpy as np
    import pandas as pd
    out=Path(sys.argv[1]); verification=json.loads((out/'verification.json').read_text()); assert verification['passed']
    for n,h in verification['artifacts_sha256'].items(): assert digest(out/n)==h
    manifest=json.loads((out/'manifest.json').read_text()); assert digest(out/'manifest.json')==verification['manifest_sha256']
    assert digest(Path(__file__).with_name('finetune_experiment.py'))==manifest['code_sha256']
    keys=['complex_id','chain','resnum','icode','group']; runs={}
    for arm in manifest['arms']:
        for seed in ([17] if arm=='frozen' else manifest['seeds']):
            folder=out/f'{arm}-{seed}'; rec=json.loads((folder/'receipt.json').read_text()); assert digest(folder/'predictions.parquet')==rec['prediction_sha256']
            df=pd.read_parquet(folder/'predictions.parquet'); assert not df.duplicated(keys).any(); assert set(df.split)=={'train','val'}
            assert set(df.loc[df.split=='train','component_id']).isdisjoint(df.loc[df.split=='val','component_id'])
            runs[arm,seed]=df.set_index(keys)
    common=set.intersection(*(set(d.index) for d in runs.values())); assert len(common)==verification['common_sites']; common=sorted(common)
    def calc(d):
        d=d.copy(); d['ae']=(d.prediction-d.target_delta_pka).abs(); d['se']=(d.prediction-d.target_delta_pka)**2; d['ref2']=d.target_delta_pka**2
        c=d.groupby(['component_id','complex_id'])[['ae','se','ref2']].mean(); c['mae']=c.ae; c['rmse']=np.sqrt(c.se); c['skill']=1-c.se/c.ref2.replace(0,np.nan)
        return c.groupby('component_id')[['mae','rmse','skill']].mean()
    frozen=runs['frozen',17].loc[common]
    for d in runs.values():
        assert np.array_equal(d.loc[common].target_delta_pka.values,frozen.target_delta_pka.values)
        assert np.array_equal(d.loc[common].split.values,frozen.split.values)
    scores=pd.read_csv(out/'arms.csv'); checked=0
    for _,s in scores.iterrows():
        d=runs[s.arm,int(s.seed)].loc[common].reset_index(); d=d[(d.split==s.split)&d[s.subset]]
        if s.role!='all': d=d[d.role==s.role]
        g=calc(d); assert len(d)==s.sites and len(g)==s.groups
        for name in ['mae','rmse','skill']:
            v=g[name].mean(); assert np.isclose(v,s[name],rtol=1e-10,atol=1e-10,equal_nan=True),(s.arm,name,v,s[name]); checked+=1
    baseline=[]; comparisons=[]
    for method in ['propka','zero']:
        df=frozen.reset_index(); df['prediction']=df.propka_delta if method=='propka' else 0.
        assert df.prediction.notna().all()
        for split in ['train','val']:
            for role in ['all','antibody_antigen','general']:
                d=df[(df.split==split)&df.interface]
                if role!='all': d=d[d.role==role]
                g=calc(d); vals=g.mae.to_numpy(); rng=np.random.default_rng(20261005)
                ci=np.quantile(rng.choice(vals,(2000,len(vals))).mean(axis=1),[.025,.975])
                baseline.append({'method':method,'split':split,'role':role,'sites':len(d),'groups':len(g),**{n:float(v) for n,v in g.mean().items()},'ci95':ci.tolist()})
                if split=='val' and role=='all' and method=='propka':
                    for seed in manifest['seeds']:
                        c=runs['catboost',seed].loc[common].reset_index(); c=c[(c.split=='val')&c.interface]; cg=calc(c)
                        delta=(cg.mae-g.mae).to_numpy(); rng=np.random.default_rng(20261005); ci=np.quantile(rng.choice(delta,(2000,len(delta))).mean(axis=1),[.025,.975])
                        comparisons.append({'seed':seed,'catboost_mae_minus_propka':float(delta.mean()),'ci95':ci.tolist()})
    report={'passed':True,'independent_group_metrics_checked':checked,'identical_teacher_labels_and_splits':True,'no_test_rows':True,'baselines':baseline,'catboost_vs_propka':comparisons,'code_sha256':digest(Path(__file__)),'verification_sha256':digest(out/'verification.json')}
    atomic_json(out/'independent_audit.json',report); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
