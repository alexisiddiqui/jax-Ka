"""Small native SAbDab teacher pilot; no pool activation or model training."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_units import load_prepared
from pkabench.sabdab_labels import source_evidence,export_states,make_sites,qualify_labels


def one(task):
    unit,output,python,pins=task
    began=time.monotonic();entry=Path(output)/'entries'/unit['unit_id'].split(':',1)[1];entry.mkdir(parents=True)
    times={};stage='source_evidence'
    try:
        t=time.monotonic();topology,meta=load_prepared(unit)
        evidence,real,known,residues=source_evidence(unit,topology,meta)
        times[stage]=time.monotonic()-t;stage='native_environment_and_masks';t=time.monotonic()
        sites=make_sites(topology,meta,evidence,real,known,residues)
        times[stage]=time.monotonic()-t;stage='teacher_export';t=time.monotonic()
        mappings=export_states(topology,meta,real,entry)
        atomic_json(entry/'request.json',dict(mappings=mappings,teacher_input_pins=pins))
        atomic_json(entry/'source-evidence.json',evidence)
        atomic_json(entry/'terminal-evidence.json',dict(real=sorted(real),known=sorted(known),
            policy='deposited entity sequence positions; never canonical renumbering or crop boundaries'))
        times[stage]=time.monotonic()-t;stage='teacher_process';t=time.monotonic()
        with (entry/'teacher.log').open('w') as log:
            subprocess.run([python,'-m','pkabench.pkai_native',str(entry/'request.json'),str(entry/'labels.json')],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        times[stage]=time.monotonic()-t
        labels=json.loads((entry/'labels.json').read_text());counts=qualify_labels(sites,labels)
        atomic_json(entry/'sites.json',sites)
        receipt=dict(unit=unit,status='complete',counts=counts,timings_s=times,teacher_timings_s=labels['timings_s'],
            wall_s=time.monotonic()-began,mask_reasons=dict(Counter(s['gap_rule'] for s in sites)),
            terminal_sites=len(known),true_terminal_sites=len(real),
            gap_defects=[dict(kind=d['kind'],chain=d['chain'],length=d['length']) for d in evidence['defects'] if d['kind'] in ('internal_gap','terminal_gap')],
            files_sha256={p.name:digest(p) for p in entry.iterdir() if p.is_file()},
            native_coordinate_sha256=meta['native_coordinate_sha256'],source_scope_policy=evidence['source_scope'],
            active_release_changed=False)
    except Exception as exc:
        receipt=dict(unit=unit,status='failed',stage=stage,error=repr(exc),traceback=traceback.format_exc(),
            timings_s=times,wall_s=time.monotonic()-began)
    atomic_json(entry/'receipt.json',receipt)
    print(json.dumps(dict(id=unit['unit_id'],status=receipt['status'],counts=receipt.get('counts'),error=receipt.get('error'))),flush=True)
    return receipt


def run(args):
    require_compute(threads=args.workers)
    if args.output.exists():raise FileExistsError(args.output)
    data=json.loads(args.inventory.read_text());reference=json.loads(args.teacher_manifest.read_text())
    package=args.teacher_python.parent.parent/'lib/python3.11/site-packages/pkai'
    names=['pKAI.py','protein.py','residue.py','atom.py','models/pKAI_model.pt','models/pKAI+_model.pt']
    pins={str(package/n):reference['input_pins'][str(package/n)] for n in names}
    for path,expected in pins.items():
        if digest(path)!=expected:raise ValueError('Pinned teacher changed: '+path)
    args.output.mkdir(parents=True);began=time.monotonic()
    atomic_json(args.output/'manifest.json',dict(format='sabdab-native-label-pilot-v1',inventory_sha256=digest(args.inventory),
        teacher_manifest_sha256=digest(args.teacher_manifest),teacher_input_pins=pins,
        workers=args.workers,job=os.environ['SLURM_JOB_ID'],
        protocol='existing fixed-coordinate CPU pKAI/pKAI+ AB/A/B; declared antibody scope and existing gap/component masks',
        code_pins={str(Path(__file__)):digest(__file__),**{str(p):digest(p) for p in Path(__file__).parents[1].joinpath('src/pkabench').glob('*.py')}}))
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(one,[(u,str(args.output),str(args.teacher_python),pins) for u in data['units']]))
    summary=dict(status='complete' if all(r['status']=='complete' for r in rows) else 'partial',
        records=rows,wall_s=time.monotonic()-began,active_release_changed=False,release_ready=False,
        limitations=['Five selected examples do not estimate full-catalogue yield or tail cost.',
            'Published antibody intervals define the unit; deposited antigen gaps retain existing masking rules.'])
    atomic_json(args.output/'summary.json',summary)
    if summary['status']!='complete':raise RuntimeError('Native label pilot has explicit failures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','output','teacher-python','teacher-manifest']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);run(p.parse_args())
