"""Validation-only native-state hybrid MC diagnostic; immutable v1 requests."""
import argparse
import copy
import json
import math
import os
import time
from collections import defaultdict, Counter
from pathlib import Path
from .runtime import require_compute, atomic_json, digest

KEY = ['complex_id', 'chain', 'resnum', 'icode', 'group']
METHODS = ['teacher', 'model_compound', 'train_constant', 'catboost-17', 'catboost-29', 'catboost-43']
MC = dict(ncpus=1, pH='-2,16', pHstep=.25, mcsteps=200000, eqsteps=1000, seed=1234567)

def read_json(path):
    return json.loads(Path(path).read_text())

def sitekey(row):
    return tuple(row[k] for k in KEY)

def init(out):
    import pyarrow.parquet as pq
    root = Path(os.environ['PKABENCH_RUNTIME'])
    native = root/'tierB/native-v2'; baseline = root/'tierB/intrinsic-baseline-v1'
    ready = read_json(native/'readiness.json'); verified = read_json(baseline/'verification.json')
    assert ready['passed'] and ready['native_order_mc_replay_passed'] and verified['passed']
    assert not verified['test_data_included'] and not verified['validation_used_for_fit']
    index = read_json(native/'native_state_index.json')
    assert digest(native/'native_state_index.json') == read_json(native/'verification.json')['index_sha256']
    assignments = {r['complex_id']: r for r in pq.read_table(native/'assignments.parquet', filters=[('split','=','val')]).to_pylist()}
    structures = {r['complex_id']: r for r in pq.read_table(native/'structures.parquet').to_pylist() if r['complex_id'] in assignments}
    predictions = {}; hashes = {}
    for seed in (17,29,43):
        path = baseline/f'seed-{seed}/predictions.parquet'; receipt = read_json(path.with_name('receipt.json'))
        assert digest(path) == receipt['prediction_sha256'] == verified['prediction_sha256'][str(path)]
        hashes[str(path)] = digest(path)
        rows = pq.read_table(path, filters=[('split','=','val')]).to_pylist()
        predictions[seed] = {(sitekey(r),r['state'],r['tautomer']):r for r in rows}
        assert len(predictions[seed]) == len(rows)
    states = defaultdict(dict)
    for row in index:
        if row['split'] == 'val': states[row['complex_id']][row['state']] = row
    requests = {}; rejected = []
    for cid, assignment in sorted(assignments.items()):
        if set(states[cid]) != {'AB','A','B'}:
            rejected.append({'complex_id':cid,'reason':'missing native state','available':sorted(states[cid])}); continue
        request = dict(assignment, n_residues=structures[cid]['n_residues'], states={})
        support = {}
        for state in ('AB','A','B'):
            s = states[cid][state]; path = Path(s['export'])/'sites.json'
            assert digest(path) == s['sites_sha256']
            sites = read_json(path); replacements = []
            for site in sites:
                values = {}
                names = site['tautomers'][:-1]
                if site['supervision_eligible'] and all((sitekey(site),state,t) in predictions[seed] for seed in predictions for t in names):
                    for seed in predictions:
                        values[f'catboost-{seed}'] = [predictions[seed][sitekey(site),state,t]['prediction'] for t in names]
                    for method,field in [('model_compound','model_pka'),('train_constant','train_constant')]:
                        values[method] = [predictions[17][sitekey(site),state,t][field] for t in names]
                        assert all(values[method] == [predictions[seed][sitekey(site),state,t][field] for t in names] for seed in predictions)
                    assert all(math.isfinite(v) for vv in values.values() for v in vv)
                    replacements.append({'site':site,'values':values})
            support[state] = {sitekey(r['site']):r['site'] for r in replacements}
            request['states'][state] = dict(s, replacements=replacements, request_sha256=digest(Path(s['source'])/'request.json'))
        free = support['A'] | support['B']; assert not support['A'].keys() & support['B'].keys()
        paired = support['AB'].keys() & free.keys()
        request['paired_sites'] = len(paired)
        request['paired_interface_sites'] = sum(support['AB'][k]['interface'] for k in paired)
        if not request['paired_interface_sites']:
            rejected.append({'complex_id':cid,'reason':'no eligible paired interface sites'}); continue
        requests[cid] = request
    # Deterministic equal role allocation, five size strata per role, two per stratum.
    # No prediction errors enter this selection. Quantile positions include size extremes.
    selected = []
    roles = sorted({r['role'] for r in requests.values()})
    quota = {role:20//len(roles)+(i<20%len(roles)) for i,role in enumerate(roles)}
    for role in roles:
        candidates = sorted((r for r in requests.values() if r['role']==role),key=lambda r:(r['n_residues'],r['complex_id']))
        n = min(quota[role],len(candidates))
        positions = [round(i*(len(candidates)-1)/max(n-1,1)) for i in range(n)]
        selected.extend(candidates[i]['complex_id'] for i in positions)
    for cid in sorted(requests, key=lambda c:(requests[c]['n_residues'],c)):
        if len(selected)>=min(20,len(requests)): break
        if cid not in selected: selected.append(cid)
    assert len(set(selected)) == len(selected)
    out.mkdir(parents=True,exist_ok=False)
    for cid,r in requests.items(): atomic_json(out/'requests'/f'{cid}.json',r)
    manifest = {'code_sha256':digest(Path(__file__)), 'native':str(native),'baseline':str(baseline),
        'prediction_sha256':hashes,'readiness_sha256':digest(native/'readiness.json'),
        'intrinsic_verification_sha256':digest(baseline/'verification.json'),
        'index_sha256':digest(native/'native_state_index.json'),
        'request_sha256':{cid:digest(out/'requests'/f'{cid}.json') for cid in requests},
        'methods':METHODS,'mc':MC,'pilot':selected,'eligible':sorted(requests),'excluded':rejected,
        'selection':'Up to 20 validation complexes; equal role quotas; evenly spaced residue-count ranks including extremes; no outcome-based selection.',
        'test_data_included':False,'hybrid':True,'masked_sites':'Retain original teacher intrinsics and all interactions.'}
    atomic_json(out/'manifest.json',manifest)
    for phase,cids in [('pilot',selected),('remaining',[c for c in sorted(requests) if c not in selected])]:
        (out/f'{phase}.txt').write_text(''.join(c+'\n' for c in cids))
    summary = {'eligible':len(requests),'excluded':rejected,'pilot':[{'complex_id':c,'role':requests[c]['role'],'n_residues':requests[c]['n_residues'],'paired_interface_sites':requests[c]['paired_interface_sites']} for c in selected]}
    atomic_json(out/'selection.json',summary); print(json.dumps(summary),flush=True)

def changed_energies(original, sites, replacements, method):
    """Change whole eligible sites only; preserve reference, padding and pair energies."""
    data = copy.deepcopy(original)
    bykey = {sitekey(r['site']):r for r in replacements}
    assert len(sites) == len(data['all_sites'])
    for i,site in enumerate(sites):
        if method == 'teacher' or sitekey(site) not in bykey: continue
        replacement = bykey[sitekey(site)]
        assert site['supervision_eligible'] and replacement['site'] == site
        values = replacement['values'][method]
        assert len(values) == data['npossible_states'][i]-1 == len(site['tautomers'])-1
        for j,pka in enumerate(values):
            data['possible_states_g'][i][j] = math.log(10)*pka*(1-2*data['possible_states_occ'][i][j])
    for field in original:
        if field != 'possible_states_g': assert data[field] == original[field],field
    for i,site in enumerate(sites):
        count = data['npossible_states'][i]
        assert data['possible_states_g'][i][count-1:] == original['possible_states_g'][i][count-1:]
        if method == 'teacher' or sitekey(site) not in bykey:
            assert data['possible_states_g'][i] == original['possible_states_g'][i]
    return data

def run_model(params, order):
    from pypka import Titration
    rank = {token:i for i,token in enumerate(order)}
    assert len(rank) == len(order)
    class NativeOrderTitration(Titration):
        def get_all_sites(self,get_list=False):
            value = super().get_all_sites(get_list=get_list)
            if get_list:
                tokens = [f'{s.molecule.chain}_{s.res_name}_{s.res_number}' for s in value]
                assert set(tokens) == set(order) and len(tokens)==len(order)
                return sorted(value,key=lambda s:rank[f'{s.molecule.chain}_{s.res_name}_{s.res_number}'])
            return value
    model = NativeOrderTitration(params); rows = {}
    for site in model:
        group = {'NTR':'NTERM','CTR':'CTERM'}.get(site.res_name,site.res_name)
        curve = site.getTitrationCurve()
        values = [float(curve[round(-2+i*.25,2)]) for i in range(73)]
        assert all(math.isfinite(v) and 0<=v<=1 for v in values)
        pka = site.getpK(); assert pka is None or math.isfinite(pka)
        rows[site.molecule.chain,site.getResNumber(),group] = {'pka':pka,'curve':values}
    assert len(rows) == len(order)
    return rows

def work(out,cid):
    m = read_json(out/'manifest.json'); assert digest(Path(__file__)) == m['code_sha256']
    assert cid in m['eligible'] and digest(out/'requests'/f'{cid}.json') == m['request_sha256'][cid]
    if cid not in m['pilot']:
        gate = read_json(out/'pilot/verification.json'); assert gate['passed'] and gate['manifest_sha256']==digest(out/'manifest.json')
    request = read_json(out/'requests'/f'{cid}.json'); assert request['split']=='val'
    dest = out/'complexes'/cid; dest.mkdir(parents=True,exist_ok=True)
    start = time.monotonic(); completed = []
    for state in ('AB','A','B'):
        s = request['states'][state]; raw = Path(s['source']); original = read_json(raw/'mc-energies.json')
        assert all(digest(raw/name)==h for name,h in s['source_hashes'].items())
        assert digest(raw/'request.json')==s['request_sha256']
        assert digest(Path(s['export'])/'sites.json') == s['sites_sha256']
        sites = read_json(Path(s['export'])/'sites.json'); expected = read_json(raw/'result.json')['rows']
        original_request = read_json(raw/'request.json')
        mapping = {(r['chain'],r['resnum']):r['original'] for r in read_json(raw/'mapping.json')}
        for method in METHODS:
            folder = dest/state/method; path = folder/'result.json'
            if path.exists():
                saved=read_json(path); assert saved['manifest_sha256']==digest(out/'manifest.json') and saved['passed']
                completed.append(str(path)); continue
            folder.mkdir(parents=True,exist_ok=True)
            energies=changed_energies(original,sites,s['replacements'],method)
            atomic_json(folder/'energies.json',energies)
            params=dict(original_request['config']); params.update(MC,structure=original_request['pdb'],load_mc_energies=str(folder/'energies.json'))
            atomic_json(folder/'request.json',params)
            previous=Path.cwd(); began=time.monotonic()
            try:
                os.chdir(folder); actual=run_model(params,original['all_sites'])
            finally: os.chdir(previous)
            max_curve=max_pka=0.; rows=[]
            replaceable={sitekey(r['site']) for r in s['replacements']}
            sites_bykey={sitekey(r):r for r in sites}
            assert len(actual)==len(expected)
            for ref in expected:
                a=actual[ref['chain'],ref['resnum'],ref['group']]
                orig=mapping[ref['chain'],ref['resnum']]
                key=(cid,*orig,ref['group']); site=sites_bykey[key]
                if method=='teacher':
                    max_curve=max(max_curve,max(abs(x-y) for x,y in zip(a['curve'],ref['curve'])))
                    assert (a['pka'] is None)==(ref['pka'] is None)
                    if a['pka'] is not None: max_pka=max(max_pka,abs(a['pka']-ref['pka']))
                rows.append({k:site[k] for k in KEY}|{'state':state,'interface':site['interface'],'replaced':key in replaceable and method!='teacher',
                    'replaceable':key in replaceable,'supervision_eligible':site['supervision_eligible'],**a})
            passed=method!='teacher' or (max_curve<=1e-12 and max_pka<=1e-10)
            result={'passed':passed,'complex_id':cid,'state':state,'method':method,'rows':rows,'seconds':time.monotonic()-began,
                'max_replay_curve_error':max_curve if method=='teacher' else None,'max_replay_pka_error':max_pka if method=='teacher' else None,
                'native_order_restored':True,'unchanged_interactions':True,'unchanged_masked_intrinsics':True,
                'source_hashes':s['source_hashes'],'energy_sha256':digest(folder/'energies.json'),
                'manifest_sha256':digest(out/'manifest.json'),'code_sha256':digest(Path(__file__))}
            atomic_json(path,result); assert passed,(cid,state,max_curve,max_pka)
            completed.append(str(path)); print(json.dumps({'complex_id':cid,'state':state,'method':method,'seconds':result['seconds'],'passed':passed}),flush=True)
            # Energies remain available as provenance; no PB recomputation occurs.
    atomic_json(dest/'receipt.json',{'passed':True,'seconds':time.monotonic()-start,'manifest_sha256':digest(out/'manifest.json'),
        'outputs_sha256':{p:digest(p) for p in completed},'job_id':os.environ['SLURM_JOB_ID']})

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['init','work']); p.add_argument('out',type=Path); p.add_argument('cid',nargs='?'); a=p.parse_args()
    require_compute()
    if a.command=='init': init(a.out.resolve())
    else: work(a.out.resolve(),a.cid)
if __name__=='__main__': main()
