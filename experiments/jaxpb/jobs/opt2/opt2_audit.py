"""Read Slurm accounting for this phase's submitted IDs; no job mutation or solver work."""
import json
import subprocess
from pathlib import Path

jobs=json.loads(Path('results/opt2-jobs.json').read_text())
ids=','.join(str(row['job']) for row in jobs)
fields=['JobIDRaw','JobName%50','Partition','AllocCPUS','ReqMem','ReqTRES%160','NodeList%40','State','ExitCode','Elapsed']
cmd='sacct -X -P -n -j '+ids+' --format='+','.join(fields)
text=subprocess.check_output(['ssh','Coulson-Chemistry',cmd],text=True)
rows=[]
for line in text.splitlines():
    parts=line.split('|')
    if len(parts)<10: continue
    row=dict(zip([f.split('%')[0] for f in fields],parts))
    req=dict(part.split('=',1) for part in row['ReqTRES'].split(',') if '=' in part)
    gpu=row['Partition']=='oc'
    row['compliant']=(row['JobName'].startswith('opt2-') and int(req.get('cpu',0))<=12
        and req.get('mem')=='24G' and row['ReqMem'] in ('2Gc','24G')
        and ((gpu and row['NodeList'] in ('comp1400','None assigned','') and req.get('gres/gpu')=='1')
             or (not gpu and row['Partition']=='amd96' and not any('gpu' in k for k in req))))
    rows.append(row)
a40_script=Path('jobs/a40.sbatch').read_text()
gpu_script_resources=all(line in a40_script for line in ('#SBATCH --gres=gpu:a40:1','#SBATCH --cpus-per-task=12','#SBATCH --mem-per-cpu=2G','#SBATCH --partition=oc','#SBATCH --nodelist=comp1400'))
result=dict(gpu_script_resources=gpu_script_resources, passed=gpu_script_resources and len(rows)==len(jobs) and all(row['compliant'] for row in rows),jobs=rows,
            expected_jobs=len(jobs), actual_jobs=len(rows),
            note='Pending jobs may not yet have allocations; requested resources are audited. This Slurm normalizes ReqMem to 24G and GPU TRES to gres/gpu=1; the submitted frozen a40.sbatch requests a40:1 and 2G per CPU.')
Path('results/opt2-compute-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Resource audit:',result['passed'],len(rows),'jobs')
for row in rows:
    if not row['compliant']: print(row)
