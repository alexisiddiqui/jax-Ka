"""Recover non-JAX tasks missing receipts after worker retirement."""
import json
import os
from pathlib import Path
from .runtime import require_compute,atomic_json,digest
from .schema import read_table,write_table,key


def run(campaign,cid,method):
    require_compute()
    from .production import check,calculate
    campaign=Path(campaign).resolve(); manifest=check(campaign)
    assert method in ('pypka','propka','pkai','pkai_plus','null')
    base=campaign/'jobs'/method; path=base/f'{cid}.json'
    if path.exists():
        receipt=json.loads(path.read_text()); assert digest(path.with_suffix('.parquet'))==receipt['output_sha256']
        print('Existing receipt verified; no overwrite',flush=True); return
    inp=json.loads((campaign/'input-files.json').read_text())[cid]
    for state,sha in inp['state_sha256'].items(): assert digest(campaign/'structures'/cid/f'{state}.cif')==sha
    work=base/cid/f"recovery-{os.environ['SLURM_JOB_ID']}"; work.mkdir(parents=True,exist_ok=False)
    rows,errors,timings,extras,elapsed=calculate(campaign,cid,method,work,manifest['timeout_seconds'][method])
    expected={(key(s),st) for s in read_table(campaign/'structures'/cid/'sites.parquet') for st in ('AB',s['partner'])}
    assert len(rows)==len(expected) and {(key(r),r['state']) for r in rows}==expected
    output=path.with_suffix('.parquet'); write_table(output,'predictions',rows)
    atomic_json(path,{'status':'failed' if errors else 'complete','errors':errors,'state_seconds':timings,'extra':extras,
        'wall_seconds':elapsed,'workdir':str(work),'reused':None,'node':os.environ['SLURMD_NODENAME'],
        'job':os.environ['SLURM_JOB_ID'],'output_sha256':digest(output),'manifest_sha256':digest(campaign/'manifest.json'),
        'task_index':None,'input_state_sha256':inp['state_sha256'],
        'recovery':'Task lacked receipt after worker retirement; previous attempts retained.',
        'recovery_source_sha256':digest(Path(__file__))})
    print(json.dumps({'complex_id':cid,'method':method,'errors':errors,'seconds':elapsed}),flush=True)
