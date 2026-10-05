import json,math
from pathlib import Path
import numpy as np

def test_native_contract():
    from pkabench.runtime import require_compute
    require_compute()
    p=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-1024-v1/jobs/pypka/00313ae18e743f16/attempt-730983/AB')
    d=json.loads((p/'mc-energies.json').read_text()); rr=json.loads((p/'result.json').read_text())['rows']; result={(r['chain'],r['resnum'],r['group']):r for r in rr}
    counts=np.array(d['npossible_states']); owner=np.repeat(np.arange(len(counts)),counts); w=np.array(d['interactions']); allowed=owner[:,None]!=owner[None,:]
    assert w.shape==(counts.sum(),counts.sum()),w.shape
    assert np.isfinite(w[allowed]).all()
    assert not np.any(w[allowed]==-999999),'cross-site sentinel'
    assert np.max(abs(w-w.T))<1e-8,'asymmetry'
    for i,name in enumerate(d['all_sites']):
        c,g,n=name.rsplit('_',2); n=int(n)-(5000 if g in ('NTR','CTR') else 0); g={'NTR':'NTERM','CTR':'CTERM'}.get(g,g)
        r=result[c,n,g]; num=counts[i]; energy=np.array(d['possible_states_g'][i][:num]); occ=np.array(d['possible_states_occ'][i][:num]); names=sorted(r['intrinsic_tautomers']); expected=np.array([r['intrinsic_tautomers'][k] for k in names])*math.log(10)*(1-2*occ[:-1])
        assert np.allclose(energy[:-1],expected,rtol=1e-10,atol=1e-8),(name,energy[:-1],expected)
    from pkabench.schema import read_table
    mapping={(r['chain'],r['resnum']):tuple(r['original']) for r in json.loads((p/'mapping.json').read_text())}
    rows=read_table(p.parent.parent.parent/'00313ae18e743f16.parquet')
    normalized={(r['chain'],r['resnum'],r['icode'],r['group']):r for r in rows if r['state']=='AB'}
    for r in rr:
        k=(*mapping[r['chain'],r['resnum']],r['group'])
        assert np.array_equal(np.asarray(r['curve'],dtype=np.float32),np.asarray(normalized[k]['curve'],dtype=np.float32))
