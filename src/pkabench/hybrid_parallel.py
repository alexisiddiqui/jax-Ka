"""Independent state/method execution using the frozen hybrid_mc scientific code."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from . import hybrid_mc as science
from .runtime import require_compute, atomic_json, digest


def checked(out):
    manifest=science.read_json(out/'manifest.json')
    assert digest(Path(science.__file__))==manifest['code_sha256']
    execution=out/'parallel-v1/execution.json'
    if execution.exists():
        for path,h in science.read_json(execution)['code_sha256'].items(): assert digest(path)==h,path
    return manifest


def valid_result(out,cid,state,method):
    path=out/'complexes'/cid/state/method/'result.json'
    if not path.exists(): return False
    r=science.read_json(path)
    assert r['passed'] and r['manifest_sha256']==digest(out/'manifest.json')
    assert r['code_sha256']==science.read_json(out/'manifest.json')['code_sha256']
    assert (r['complex_id'],r['state'],r['method'])==(cid,state,method)
    assert digest(path.parent/'energies.json')==r['energy_sha256']
    if 'execution_code_sha256' in r: assert r['execution_code_sha256']==digest(Path(__file__))
    if method=='teacher': assert r['max_replay_curve_error']<=1e-12 and r['max_replay_pka_error']<=1e-10
    return True


def calculate(out,cid,state,method,folder):
    m=checked(out); assert cid in m['eligible'] and state in ('AB','A','B') and method in science.METHODS
    assert digest(out/'requests'/f'{cid}.json')==m['request_sha256'][cid]
    request=science.read_json(out/'requests'/f'{cid}.json'); assert request['split']=='val'
    if cid not in m['pilot']:
        gate=science.read_json(out/'pilot/verification.json'); assert gate['passed'] and gate['manifest_sha256']==digest(out/'manifest.json')
    if method!='teacher': assert valid_result(out,cid,state,'teacher'),'Teacher replay must pass first'
    s=request['states'][state]; raw=Path(s['source'])
    assert all(digest(raw/name)==h for name,h in s['source_hashes'].items())
    assert digest(raw/'request.json')==s['request_sha256']
    assert digest(Path(s['export'])/'sites.json')==s['sites_sha256']
    original=science.read_json(raw/'mc-energies.json'); sites=science.read_json(Path(s['export'])/'sites.json')
    expected=science.read_json(raw/'result.json')['rows']; original_request=science.read_json(raw/'request.json')
    mapping={(r['chain'],r['resnum']):r['original'] for r in science.read_json(raw/'mapping.json')}
    energies=science.changed_energies(original,sites,s['replacements'],method)
    folder.mkdir(parents=True,exist_ok=False); atomic_json(folder/'energies.json',energies)
    params=dict(original_request['config']); params.update(science.MC,structure=original_request['pdb'],load_mc_energies=str(folder/'energies.json'))
    atomic_json(folder/'request.json',params)
    previous=Path.cwd(); began=time.monotonic()
    try:
        os.chdir(folder); actual=science.run_model(params,original['all_sites'])
    finally: os.chdir(previous)
    max_curve=max_pka=0.; rows=[]
    replaceable={science.sitekey(r['site']) for r in s['replacements']}
    sites_bykey={science.sitekey(r):r for r in sites}; assert len(actual)==len(expected)
    for ref in expected:
        a=actual[ref['chain'],ref['resnum'],ref['group']]; orig=mapping[ref['chain'],ref['resnum']]
        key=(cid,*orig,ref['group']); site=sites_bykey[key]
        if method=='teacher':
            max_curve=max(max_curve,max(abs(x-y) for x,y in zip(a['curve'],ref['curve'])))
            assert (a['pka'] is None)==(ref['pka'] is None)
            if a['pka'] is not None: max_pka=max(max_pka,abs(a['pka']-ref['pka']))
        rows.append({k:site[k] for k in science.KEY}|{'state':state,'interface':site['interface'],
            'replaced':key in replaceable and method!='teacher','replaceable':key in replaceable,
            'supervision_eligible':site['supervision_eligible'],**a})
    passed=method!='teacher' or (max_curve<=1e-12 and max_pka<=1e-10)
    result={'passed':passed,'complex_id':cid,'state':state,'method':method,'rows':rows,'seconds':time.monotonic()-began,
        'max_replay_curve_error':max_curve if method=='teacher' else None,'max_replay_pka_error':max_pka if method=='teacher' else None,
        'native_order_restored':True,'unchanged_interactions':True,'unchanged_masked_intrinsics':True,
        'source_hashes':s['source_hashes'],'energy_sha256':digest(folder/'energies.json'),
        'manifest_sha256':digest(out/'manifest.json'),
        # Shared frozen code implements energy conversion, site ordering and MC.
        'code_sha256':m['code_sha256'],'execution_code_sha256':digest(Path(__file__)),
        'job_id':os.environ['SLURM_JOB_ID']}
    atomic_json(folder/'result.json',result); assert passed,(cid,state,max_curve,max_pka)
    print(json.dumps({k:result[k] for k in ('complex_id','state','method','seconds','passed')}),flush=True)
    return result


def work(out,cid,state,method):
    lock=out/'parallel-v1/locks'/f'{cid}-{state}-{method}.lock'; lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if valid_result(out,cid,state,method): return
        folder=out/'complexes'/cid/state/method
        if folder.exists():
            archive=out/'parallel-v1/interrupted'/f'{cid}-{state}-{method}-{time.time_ns()}'
            archive.parent.mkdir(parents=True,exist_ok=True); folder.rename(archive)
        calculate(out,cid,state,method,folder)


def preflight(out,method):
    # Each invocation is a fresh process, as each dispatched calculation will be.
    cid='4066edfcdb3ea8e3'; state='AB'; assert valid_result(out,cid,state,method)
    dest=out/'parallel-v1/preflight'/method
    got=calculate(out,cid,state,method,dest)
    old=science.read_json(out/'complexes'/cid/state/method/'result.json')
    assert got['rows']==old['rows'],'Independent-process execution differs from original serial execution'
    atomic_json(dest/'verification.json',{'passed':True,'exact_rows_match':True,'original_result_sha256':digest(out/'complexes'/cid/state/method/'result.json'),
        'new_result_sha256':digest(dest/'result.json'),'execution_code_sha256':digest(Path(__file__))})


def collect(out,phase):
    m=checked(out); cids=m['pilot'] if phase=='pilot' else m['eligible']
    for cid in cids:
        paths=[]
        for state in ('AB','A','B'):
            for method in science.METHODS:
                assert valid_result(out,cid,state,method); paths.append(out/'complexes'/cid/state/method/'result.json')
        receipt=out/'complexes'/cid/'receipt.json'
        if receipt.exists():
            r=science.read_json(receipt); assert r['passed'] and r['manifest_sha256']==digest(out/'manifest.json')
            assert len(r['outputs_sha256'])==18 and all(digest(p)==h for p,h in r['outputs_sha256'].items())
        else:
            atomic_json(receipt,{'passed':True,'manifest_sha256':digest(out/'manifest.json'),
                'outputs_sha256':{str(p):digest(p) for p in paths},'job_id':os.environ['SLURM_JOB_ID'],
                'execution_code_sha256':digest(Path(__file__)),'scheduling':'independent state/method tasks'})
    from .hybrid_score import collect as score
    score(out,phase)


def main():
    require_compute(); p=argparse.ArgumentParser(); p.add_argument('command',choices=['work','preflight','collect']); p.add_argument('out',type=Path); p.add_argument('args',nargs='*'); a=p.parse_args()
    out=a.out.resolve()
    if a.command=='work': work(out,*a.args)
    elif a.command=='preflight': preflight(out,*a.args)
    else: collect(out,*a.args)
if __name__=='__main__': main()
