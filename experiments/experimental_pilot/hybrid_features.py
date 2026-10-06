"""Frozen native-intrinsic features for experimental inference; never fit a model."""
import json, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from pkabench.runtime import require_compute, atomic_json, digest
from pkabench.native_intrinsic import FEATURES, NUMERIC, RADII, KEY
from pkabench.prep import read_cif
from pkabench.annotate import SITE_ATOMS

def geometry(path, sites, ff):
    # Identical calculations to native_intrinsic.features(), isolated from its
    # train/validation campaign bookkeeping. Verified below against saved rows.
    atoms=read_cif(path); coords=np.asarray(atoms.coord,dtype=float); tree=cKDTree(coords)
    residues=defaultdict(list)
    for i in range(len(atoms)):
        residues[str(atoms.chain_id[i]),int(atoms.res_id[i]),str(atoms.ins_code[i])].append(i)
    charged=[]; q=[]; charged_keys=[]
    for rk,ii in residues.items():
        aa=str(atoms.res_name[ii[0]]); charge={'ASP':-1.,'GLU':-1.,'LYS':1.,'ARG':1.}.get(aa,0.)
        if charge:
            jj=[i for i in ii if str(atoms.atom_name[i]) in SITE_ATOMS[aa]]
            if jj: charged.append(coords[jj].mean(axis=0)); q.append(charge); charged_keys.append(rk)
    charged=np.asarray(charged,dtype=float).reshape(-1,3); q=np.asarray(q); rows=[]
    for s in sites:
        if not s['supervision_eligible']: continue
        rk=(s['chain'],s['resnum'],s['icode']); own=residues[rk]
        byname={str(atoms.atom_name[i]):i for i in own}; names=SITE_ATOMS[s['group']]
        functional=[byname[n] for n in names if n in byname]
        if not functional: continue
        points={'center':coords[functional].mean(axis=0)}; f={}
        for slot in (0,1):
            exists=slot<len(names) and names[slot] in byname
            f[f'atom{slot}_present']=int(exists)
            points[f'atom{slot}']=coords[byname[names[slot]]] if exists else None
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
                f[f'{pname}_positive_{r}']=int(np.sum((d<=r)&(charges>0)))
                f[f'{pname}_negative_{r}']=int(np.sum((d<=r)&(charges<0)))
        assert set(f)==set(NUMERIC)
        for name in s['tautomers'][:-1]:
            group={'NTERM':'NTR','CTERM':'CTR'}.get(s['group'],s['group'])
            pmod=float((ff/f'{group}tau{int(name[-1])+1}.st').read_text().splitlines()[0])
            rows.append({k:s[k] for k in KEY}|dict(tautomer=name,model_pka=pmod,**f))
    return rows

def main():
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    runtime=Path(os.environ['PKABENCH_RUNTIME']); out=runtime/'experimental/hybrid-v1'
    baseline=runtime/'tierB/intrinsic-baseline-v1'
    manifest=json.loads((baseline/'manifest.json').read_text())
    from pkabench import native_intrinsic
    assert digest(native_intrinsic.__file__)==manifest['code_sha256']
    ff=runtime/'envs/pypka/lib/python3.10/site-packages/pypka/G54A7/sts'
    # A real saved state is an independent regression fixture for every feature.
    native=Path(manifest['source']); index=json.loads((native/'native_state_index.json').read_text())
    fixture=next(r for r in index if r['state']=='AB')
    fs=json.loads((Path(fixture['export'])/'sites.json').read_text())
    actual=geometry(Path(manifest['structure_source'])/'structures'/fixture['complex_id']/'AB.cif',fs,ff)
    expected=pq.read_table(baseline/'features.parquet',filters=[('complex_id','=',fixture['complex_id']),('state','=','AB')]).to_pylist()
    key=lambda r:tuple(r[k] for k in KEY+['tautomer'])
    aa={key(r):r for r in actual}; ee={key(r):r for r in expected}
    assert aa.keys()==ee.keys()
    for k,r in aa.items():
        for name in FEATURES+['model_pka']:
            a,b=r[name],ee[k][name]
            if a is None: assert b is None or np.isnan(b)
            elif isinstance(a,str): assert a==b
            else: assert np.isclose(a,b,rtol=0,atol=1e-12),(k,name,a,b)
    atomic_json(out/'feature_regression.json',dict(passed=True,fixture=fixture['complex_id'],rows=len(actual),source_code_sha256=manifest['code_sha256']))
    gap={tuple(r[k] for k in KEY):r for r in pq.read_table(runtime/'experimental/pilot-v2/natural-gap-tiers-final/site_tiers.parquet').to_pylist()}
    rows=[]
    for pdb in ('1BNI','1IGD','1PGB'):
        raw=out/pdb/'pypka/AB'; energy=json.loads((raw/'mc-energies.json').read_text())
        result=json.loads((raw/'result.json').read_text())
        mapping={(r['chain'],r['resnum']):r['original'] for r in json.loads((raw/'mapping.json').read_text())}
        results={(r['chain'],r['resnum'],r['group']):r for r in result['rows']}
        components={tuple(r[k] for k in KEY):r for r in json.loads((runtime/'experimental/pilot-v2/structures'/pdb/'component_masks.json').read_text())}
        evidence=json.loads((runtime/'experimental/pilot-v2/structures'/pdb/'input_atom_mask.json').read_text())
        sites=[]
        for i,token in enumerate(energy['all_sites']):
            chain,group,num=token.rsplit('_',2)
            number=int(num)-(5000 if group in ('NTR','CTR') else 0)
            group={'NTR':'NTERM','CTR':'CTERM'}.get(group,group)
            orig=mapping[chain,number]; k=(pdb,*orig,group); res=results[chain,number,group]
            names=sorted(res['intrinsic_tautomers']); nr=energy['npossible_states'][i]
            assert nr==len(names)+1
            pki=[res['intrinsic_tautomers'][n] for n in names]
            # Occupancy arrays may include padding after npossible_states.
            expected=np.array(pki)*np.log(10)*(1-2*np.array(energy['possible_states_occ'][i][:nr-1]))
            assert np.allclose(energy['possible_states_g'][i][:nr-1],expected,atol=1e-8,rtol=1e-10)
            artificial=group in ('NTERM','CTERM') and orig in evidence['artificial_terminal_keys']
            eligible=bool(k in gap and gap[k]['tier'] in ('clean','uncertain') and components[k]['component_eval_mask'] and not artificial)
            sites.append(dict(zip(KEY,k))|dict(tautomers=names+['reference'],intrinsic_pka_tautomers=pki,supervision_eligible=eligible,interface=False))
        atomic_json(out/pdb/'sites.json',sites)
        rr=geometry(runtime/'experimental/pilot-v2/structures'/pdb/'AB.cif',sites,ff)
        rows.extend(rr)
    pq.write_table(pa.Table.from_pylist(rows),out/'features.parquet')
    atomic_json(out/'features_receipt.json',dict(rows=len(rows),sha256=digest(out/'features.parquet'),code_sha256=digest(Path(__file__)),fit_performed=False))
    print('Feature regression passed;',len(rows),'inference rows',flush=True)

if __name__=='__main__': main()
