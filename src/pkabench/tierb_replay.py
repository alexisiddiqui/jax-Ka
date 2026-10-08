"""MC-only replay gate using an already exported training state; no PB solve."""
import json,os,sys
from pathlib import Path
from .runtime import require_compute,atomic_json,digest

def main():
    require_compute()
    from pypka import Titration
    out=Path(sys.argv[1]); profile=sys.argv[2] if len(sys.argv)>2 else 'small'; choices=[]
    failed=json.loads((out/'replay-complex/result.json').read_text()) if profile=='aligned' else None
    for path in (out/'complexes').glob('*/receipt.json'):
        r=json.loads(path.read_text())
        if r['split']!='train': continue
        for s in r['states']:
            if s['status']!='exported': continue
            if failed and (r['complex_id'],s['state'])!=(failed['complex_id'],failed['state']): continue
            if profile=='complex':
                if s['state']!='AB' or not 10<=s['sites']<=60: continue
                sites=json.loads((path.parent/s['state']/'sites.json').read_text())
                if not any(x['group']=='HIS' for x in sites): continue
            choices.append((s['sites'],r['complex_id'],s['state'],s))
    n,cid,state,s=min(choices,key=lambda r:r[:3]); raw=Path(s['source']); dest=out/('replay' if profile=='small' else 'replay-'+profile); dest.mkdir(exist_ok=False)
    assert all(digest(raw/name)==h for name,h in s['source_hashes'].items())
    request=json.loads((raw/'request.json').read_text()); expected=json.loads((raw/'result.json').read_text())['rows']
    params=dict(request['config']); params.update(structure=request['pdb'],load_mc_energies=str(raw/'mc-energies.json'),ncpus=1,pH='-2,16',pHstep=.25,mcsteps=200000,eqsteps=1000,seed=1234567)
    atomic_json(dest/'request.json',params); os.chdir(dest)
    order=json.loads((raw/'mc-energies.json').read_text())['all_sites']; rank={token:i for i,token in enumerate(order)}
    class NativeOrderTitration(Titration):
        def get_all_sites(self,get_list=False):
            value=super().get_all_sites(get_list=get_list)
            if get_list:
                tokens=[f'{site.molecule.chain}_{site.res_name}_{site.res_number}' for site in value]
                assert set(tokens)==set(order) and len(tokens)==len(order)
                return sorted(value,key=lambda site:rank[f'{site.molecule.chain}_{site.res_name}_{site.res_number}'])
            return value
    model=(NativeOrderTitration if profile=='aligned' else Titration)(params); actual={}
    for site in model:
        g={'NTR':'NTERM','CTR':'CTERM'}.get(site.res_name,site.res_name)
        curve=site.getTitrationCurve(); actual[site.molecule.chain,site.getResNumber(),g]={'pka':site.getpK(),'curve':[float(curve[round(-2+i*.25,2)]) for i in range(73)]}
    assert len(actual)==len(expected)
    max_curve=max_pka=0.; rows=[]
    for r in expected:
        a=actual[r['chain'],r['resnum'],r['group']]; err=max(abs(x-y) for x,y in zip(a['curve'],r['curve'])); max_curve=max(max_curve,err)
        assert (a['pka'] is None)==(r['pka'] is None)
        if a['pka'] is not None: max_pka=max(max_pka,abs(a['pka']-r['pka']))
        rows.append({'chain':r['chain'],'resnum':r['resnum'],'group':r['group'],**a})
    atomic_json(dest/'result.json',{'passed':max_curve<=1e-12 and max_pka<=1e-10,'complex_id':cid,'state':state,'sites':n,'max_curve_error':max_curve,'max_pka_error':max_pka,
        'source':str(raw),'source_hashes':s['source_hashes'],'code_sha256':digest(Path(__file__)),'mode':'saved-energy Monte Carlo only; no PB solve','native_site_order_restored':profile=='aligned','rows':rows})
    assert max_curve<=1e-12 and max_pka<=1e-10,(max_curve,max_pka)
    print(json.dumps({'passed':True,'complex_id':cid,'state':state,'sites':n,'max_curve_error':max_curve,'max_pka_error':max_pka}),flush=True)
if __name__=='__main__': main()
