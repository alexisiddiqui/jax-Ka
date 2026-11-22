"""Verify final frozen hashes, phase scope and reuse of the 10a3ddc references."""
import ast,hashlib,json,subprocess
from pathlib import Path
jobs=json.loads(Path('results/opt2-jobs.json').read_text())
last=next(r for r in jobs if r['tag']=='finalr')
local={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src/jaxpb').glob('*.py')}
remote=subprocess.check_output(['ssh','Coulson-Chemistry',f"cd {last['frozen']} && sha256sum src/jaxpb/*.py tests/test_opt2.py tests/opt2_reference.py tests/test_sor_pallas.py tests/test_opt1.py"],text=True)
frozen={line.split()[1]:line.split()[0] for line in remote.splitlines()}
match={p:frozen[p]==h for p,h in local.items()}
for p in ('tests/test_opt2.py','tests/opt2_reference.py','tests/test_sor_pallas.py','tests/test_opt1.py'):
 match[p]=frozen[p]==hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline={p:hashlib.sha256(subprocess.check_output(['git','show','10a3ddc:'+p])).hexdigest() for p in local}
prior=json.loads(Path('results/opt1b-source-audit.json').read_text())['final_sha256']
reuse={p:baseline[p]==prior[p] for p in baseline}
unchanged={p:local[p]==h for p,h in baseline.items() if p not in ('src/jaxpb/pb.py','src/jaxpb/sor.py','src/jaxpb/records.py')}
changed=subprocess.check_output(['git','diff','--name-only','10a3ddc'],text=True).splitlines()
scope=set(changed)<=set(['src/jaxpb/pb.py','src/jaxpb/sor.py','src/jaxpb/records.py','tests/test_sor_pallas.py'])
old_sor=subprocess.check_output(['git','show','10a3ddc:src/jaxpb/sor.py'],text=True)
new_sor=Path('src/jaxpb/sor.py').read_text()
def functions(text):
 return {node.name:ast.get_source_segment(text,node) for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)}
of,nf=functions(old_sor),functions(new_sor)
rf=functions(Path('tests/opt2_reference.py').read_text())
fixture_functions={name:of[name]==text for name,text in rf.items()}
pallas_functions={name:of[name]==nf[name] for name in ('op_code','op_lut','_half_kernel','_half','sor_solve','run_pool_pallas')}
res=dict(passed=all(fixture_functions.values()) and all(pallas_functions.values()) and all(match.values()) and all(reuse.values()) and all(unchanged.values()) and scope,
 reference_fixture_unchanged_functions=fixture_functions,pallas_unchanged_functions=pallas_functions,frozen=last['frozen'],local_matches_frozen=match,unchanged_modules=unchanged,scope_passed=scope,
 reference_reuse_matches_10a3ddc=reuse,baseline_sha256=baseline,final_sha256=frozen,tracked_changes=changed)
Path('results/opt2-source-audit.json').write_text(json.dumps(res,indent=2)+'\n')
print('Source audit:',res['passed'])
