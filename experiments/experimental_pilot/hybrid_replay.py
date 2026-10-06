import json, os, sys, time
from pathlib import Path
from pkabench.runtime import require_compute, atomic_json, digest
from pkabench.hybrid_mc import changed_energies, run_model, MC
require_compute()
pdb,method=sys.argv[1:3]
root=Path(os.environ['PKABENCH_RUNTIME'])/'experimental/hybrid-v1'
raw=root/pdb/'pypka/AB'; dest=root/pdb/method
dest.mkdir(parents=True,exist_ok=False)
original=json.loads((raw/'mc-energies.json').read_text())
sites=json.loads((root/pdb/'sites.json').read_text())
replacements=json.loads((root/pdb/'replacements.json').read_text())
if method!='teacher':
    assert json.loads((root/pdb/'teacher/result.json').read_text())['exact_replay']
energies=changed_energies(original,sites,replacements,method)
atomic_json(dest/'energies.json',energies)
request=json.loads((raw/'request.json').read_text())
params=dict(request['config']); params.update(MC,structure=request['pdb'],load_mc_energies=str(dest/'energies.json'))
atomic_json(dest/'request.json',params)
os.chdir(dest); start=time.monotonic(); actual=run_model(params,original['all_sites'])
mapping={(r['chain'],r['resnum']):r['original'] for r in json.loads((raw/'mapping.json').read_text())}
expected=json.loads((raw/'result.json').read_text())['rows']
assert len(actual)==len(expected)
error_curve=error_pka=0.; rows=[]
for ref in expected:
    got=actual[ref['chain'],ref['resnum'],ref['group']]
    if method=='teacher':
        error_curve=max(error_curve,max(abs(a-b) for a,b in zip(got['curve'],ref['curve'])))
        assert (got['pka'] is None)==(ref['pka'] is None)
        if got['pka'] is not None: error_pka=max(error_pka,abs(got['pka']-ref['pka']))
    chain,num,icode=mapping[ref['chain'],ref['resnum']]
    rows.append(dict(complex_id=pdb,chain=chain,resnum=num,icode=icode,group=ref['group'],**got))
if method=='teacher': assert error_curve<=1e-12 and error_pka<=1e-10
atomic_json(dest/'result.json',dict(method=method,rows=rows,exact_replay=method=='teacher' and error_curve==0 and error_pka==0,
    max_curve_error=error_curve if method=='teacher' else None,max_pka_error=error_pka if method=='teacher' else None,
    replaced_sites=0 if method=='teacher' else len(replacements),total_sites=len(sites),
    retained_teacher_sites=len(sites)-len(replacements),interactions_unchanged=True,
    seconds=time.monotonic()-start,energy_sha256=digest(dest/'energies.json'),source_sha256=digest(raw/'mc-energies.json'),
    replacement_sha256=digest(root/pdb/'replacements.json'),job=os.environ['SLURM_JOB_ID']))
print(pdb,method,'complete',flush=True)
