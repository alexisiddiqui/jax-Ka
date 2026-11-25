"""Run all self-contained existing tests plus M7 gates inside an allocation."""
import os,subprocess,json,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
os.environ['PYTHONPATH']=os.environ.get('PYTHONPATH','')+':/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pkgtest/pytestshim'
O=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/m7')
# Data-driven milestone scripts are exercised by m7_gate instead.
files=['test_records.py','test_assemble.py','test_titrate.py','test_sor_pallas.py','test_api.py','test_opt1.py','test_opt2.py','test_opt3.py','test_opt4.py','test_m7.py']
tag='tests-'+os.environ['SLURM_JOB_ID'];r={'job':os.environ['SLURM_JOB_ID'],'frozen':os.getcwd(),'tests':{}}
for name in files:
    t=time.time(); log=O/(tag+'-'+name+'.out')
    with log.open('w') as f: p=subprocess.run(([os.sys.executable,'-m','pytest','-q','tests/'+name] if name.startswith('test_opt') else [os.sys.executable,'tests/'+name]+(['local'] if name in ('test_assemble.py','test_titrate.py') else ([str(O.parent/'m0/shards'/('m0_goldens_0'+str(k)+'.npz')) for k in range(3)] if name=='test_records.py' else []))),stdout=f,stderr=subprocess.STDOUT)
    r['tests'][name]={'returncode':p.returncode,'seconds':time.time()-t,'log':str(log)}
    (O/(tag+'.json')).write_text(json.dumps(r,indent=2));print(name,p.returncode,flush=True)
r['completed']=True;r['passed']=all(x['returncode']==0 for x in r['tests'].values());(O/(tag+'.json')).write_text(json.dumps(r,indent=2))
