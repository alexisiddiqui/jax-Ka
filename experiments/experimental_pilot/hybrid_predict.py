import json, os
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from pkabench.runtime import require_compute, atomic_json, digest
from pkabench.native_intrinsic import FEATURES, NUMERIC, KEY
require_compute()
runtime=Path(os.environ['PKABENCH_RUNTIME']); out=runtime/'experimental/hybrid-v1'
baseline=runtime/'tierB/intrinsic-baseline-v1'
assert json.loads((out/'feature_regression.json').read_text())['passed']
data=pd.read_parquet(out/'features.parquet'); x=data[FEATURES].copy(); x[NUMERIC]=x[NUMERIC].astype(float)
hashes={}
for seed in (17,29,43):
    path=baseline/f'seed-{seed}/intrinsic.cbm'
    receipt=json.loads(path.with_name('receipt.json').read_text())
    assert digest(path)==receipt['model_sha256']
    model=CatBoostRegressor(); model.load_model(str(path))
    assert model.feature_names_==FEATURES
    data[f'catboost-{seed}']=model.predict(x,thread_count=2)+data.model_pka.values
    assert np.isfinite(data[f'catboost-{seed}']).all()
    hashes[str(path)]=digest(path)
for pdb in ('1BNI','1IGD','1PGB'):
    sites=json.loads((out/pdb/'sites.json').read_text()); subset=data[data.complex_id==pdb]
    lookup={tuple(row[k] for k in KEY)+(row['tautomer'],):row for row in subset.to_dict('records')}
    replacements=[]
    for s in sites:
        keys=[tuple(s[k] for k in KEY)+(name,) for name in s['tautomers'][:-1]]
        if s['supervision_eligible'] and all(k in lookup for k in keys):
            replacements.append(dict(site=s,values={f'catboost-{seed}':[lookup[k][f'catboost-{seed}'] for k in keys] for seed in (17,29,43)}))
    atomic_json(out/pdb/'replacements.json',replacements)
data.to_parquet(out/'intrinsic_predictions.parquet',index=False)
atomic_json(out/'model_receipt.json',dict(fit_performed=False,models_sha256=hashes,
    feature_sha256=digest(out/'features.parquet'),prediction_sha256=digest(out/'intrinsic_predictions.parquet')))
print('Frozen inference complete',len(data),flush=True)
