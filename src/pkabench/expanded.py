"""Bounded, stratified missing-residue study; immutable per-reference jobs."""
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from .runtime import atomic_json, digest, require_compute


def plan(campaigns,out,count=30):
    from .prep import read_cif
    import biotite.structure as struc
    require_compute(); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list); eligible=[]
    for campaign in map(Path,campaigns):
        parent=json.loads((campaign/'manifest.json').read_text())
        for path in sorted((campaign/'rows').glob('*.json')):
            row=json.loads(path.read_text())
            if row['status']!='accepted' or not 80<=row['structure']['n_residues']<=600: continue
            work=campaign/'structures'/row['complex_id']
            evidence=json.loads((work/'input_atom_mask.json').read_text())
            defects=evidence['defects']
            if any(s['sequence_source']!='entity_poly_canonical' for s in evidence['sequences']): continue
            missing=sum(d['length'] for d in defects if d['kind']=='terminal_gap')
            incomplete=sum(d['kind']=='missing_atoms' for d in defects)
            if any(d['kind'] in ('internal_gap','removed_additive') for d in defects) or missing>2 or incomplete>2: continue
            atoms=read_cif(work/'AB.cif'); starts=struc.get_residue_starts(atoms)
            charge=sum(str(atoms.res_name[i]) in ('ASP','GLU','HIS','LYS','ARG') for i in starts)/len(starts)
            n=row['structure']['n_residues']; size='80-249' if n<250 else '250-399' if n<400 else '400-600'
            kind='antibody' if row.get('stratum')=='antibody' else 'homomer' if row['structure']['homomeric'] else 'heteromer'
            item={**row,'parent_campaign':str(campaign),'source_root':str(Path(parent['source_audit'])/'sources'),
                'size_stratum':size,'interface_type':kind,'charge_fraction':charge,'charge_stratum':'high' if charge>=.2 else 'low',
                'original_missing_residues':missing,'original_incomplete_residues':incomplete}
            groups[bool(missing or incomplete),kind,size,item['charge_stratum']].append(item); eligible.append(item)
    rng=random.Random(20261003)
    for group in groups.values(): rng.shuffle(group)
    selected=[]; seen=set()
    for near_complete in (False,True):
        keys=sorted(k for k in groups if k[0]==near_complete); rng.shuffle(keys)
        while len(selected)<count:
            changed=False
            for key in keys:
                while groups[key]:
                    item=groups[key].pop(); entry=item['pdb_id'].split('-assembly')[0]
                    if entry in seen: continue
                    seen.add(entry); selected.append(item); changed=True; break
                if len(selected)==count: break
            if not changed: break
    manifest={'references':selected,'requested':count,'eligible':len(eligible),'seed':20261003,
        'selection':'Complete references first, then near-complete (<=2 missing terminal residues, <=2 incomplete residues). Canonical deposited sequences required; no internal gap/additive removal. Within each tier round-robin interface type x size (80-249/250-399/400-600) x charge fraction (<0.2/>=0.2). One reference per PDB entry; sequence independence not guaranteed.',
        'production_allowed':False,'implementation_sha256':digest(Path(__file__))}
    atomic_json(out/'plan.json',manifest)
    print(json.dumps({'selected':len(selected),'eligible':len(eligible),'references':[{k:r[k] for k in ('complex_id','pdb_id','size_stratum','interface_type','charge_fraction')} for r in selected]},indent=2))
    if not selected: raise ValueError('no eligible references')


def prepare(out,shard,shards):
    from biotite.structure.io import pdbx
    from .radial import deletion_plan, deletion_mask
    from .prep import read_cif, Rejection
    from .curation import prepare_revised, POLICY
    from .dataset_audit import filtered_cif
    from .schema import write_table
    require_compute(); out=Path(out); plan=json.loads((out/'plan.json').read_text())
    for row in plan['references'][shard::shards]:
        cid=row['complex_id']; receipt=out/'cases'/f'{cid}.json'
        if receipt.exists(): continue
        campaign=Path(row['parent_campaign']); reference=read_cif(campaign/'structures'/cid/'AB.cif')
        partners={p:row[f'partner_{p}_chains'] for p in ('A','B')}
        variants=deletion_plan(reference,expanded=True,partners=partners)
        expected={'N1','N3','N5','N10','C1','C3','C5','C10','buried1','buried3','buried_charged1','internal_near3','internal_remote3'}
        if plan.get('mode')=='long_tails':
            from .long_tails import deletion_plan as long_deletions
            variants=long_deletions(reference)
            expected={'N20','N30','C20','C30'}
        case={k:row[k] for k in ('complex_id','pdb_id','original_missing_residues','original_incomplete_residues','size_stratum','interface_type','charge_fraction')}
        case.update(n_residues=row['structure']['n_residues'],variants={},unavailable_variants=sorted(expected-set(variants)))
        source=Path(row['source_root'])/f"{row['pdb_id']}.cif"
        if digest(source)!=row['source_sha256']: raise ValueError('reference source hash mismatch')
        for name,spec in [('baseline',None),('repeat',None),*variants.items()]:
            root=out/cid/name; work=root/'structures'/cid; work.mkdir(parents=True,exist_ok=True)
            modified=root/'source.cif'
            if spec:
                original=pdbx.CIFFile.read(source)
                filtered_cif(source,deletion_mask(original,[r['key'] for r in spec['residues']])).write(modified)
            else: modified.write_bytes(source.read_bytes())
            item={k:row[k] for k in ('complex_id','pdb_id','partner_A_chains','partner_B_chains')}; item['source_sha256']=digest(modified)
            try:
                structure,sites,extra=prepare_revised(modified,item,work,diagnostic_perturbation=spec is not None)
                write_table(work/'sites.parquet','sites',sites); write_table(root/'sites.parquet','sites',sites)
                write_table(root/'structures.parquet','structures',[structure])
                atomic_json(root/'manifest.json',{'candidates':[item],'variant':name,'perturbation':spec,
                    'parent_campaign':str(campaign),'policy':POLICY,'production_allowed':False})
                case['variants'][name]={'status':'prepared','perturbation':spec}
            except Rejection as exc: case['variants'][name]={'status':'rejected','code':exc.code,'detail':str(exc),'perturbation':spec}
        atomic_json(receipt,case)


def assemble(out):
    require_compute(); out=Path(out); plan=json.loads((out/'plan.json').read_text())
    cases=[json.loads((out/'cases'/f"{r['complex_id']}.json").read_text()) for r in plan['references']]
    tasks=[]; failures=[]
    for case in cases:
        cid=case['complex_id']
        for name,variant in case['variants'].items():
            if variant['status']=='prepared': tasks.append((str(out/cid/name),cid))
            else: failures.append({'complex_id':cid,'variant':name,'code':variant['code'],'detail':variant['detail']})
    atomic_json(out/'manifest.json',{'cases':cases,'tasks':tasks,'prep_failures':failures,
        'selection':plan['selection'],'requested_cases':plan['requested'],'production_allowed':False,
        'plan_sha256':digest(out/'plan.json')})
    (out/'tasks.tsv').write_text(''.join(f'{root}\t{cid}\n' for root,cid in tasks))
    print(json.dumps({'cases':len(cases),'tasks':len(tasks),'prep_failures':failures},indent=2))


def dispatch(out):
    """Submit one bounded batch; rerunning fills free slots without duplicates."""
    import fcntl
    import subprocess
    from .jobs import queued_cores
    require_compute(); out=Path(out).resolve(); runtime=Path(os.environ['PKABENCH_RUNTIME'])
    manifest=json.loads((out/'manifest.json').read_text())
    if len(manifest['cases'])>30 or manifest['production_allowed'] is not False:
        raise ValueError('dispatcher restricted to <=30 diagnostic references')
    tasks=sorted(manifest['tasks'],key=lambda t:(Path(t[0]).name not in ('baseline','repeat'),Path(t[0]).name,t[1]))
    with (runtime/'submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        submitted=0; unsent=[]
        for position,(root,cid) in enumerate(tasks):
            root=Path(root); ledger=root/'submission.json'; identity=f'{cid}/pypka'
            if ledger.exists() and identity in json.loads(ledger.read_text()): continue
            used=queued_cores(subprocess.check_output(['squeue','-r','-h','-u',os.environ['USER'],'-o','%C'],text=True))
            if used+2>400:
                unsent=[(r,c) for r,c in tasks[position:] if not (Path(r)/'submission.json').exists()]
                break
            state=subprocess.check_output(['sinfo','-N','-h','-p','generalaccess,amd96','-o','%N %t'],text=True)
            excluded={'comp1400'}|{line.split()[0] for line in state.splitlines() if line.split()[1] not in {'idle','mix','alloc'}}
            command=['sbatch','--parsable',f'--exclude={",".join(sorted(excluded))}',f'--output={runtime}/logs/%x-%j.out',
                '/home/coulson/oc/lina4225/_HPC/submission/jax-Ka/pkabench/work.sbatch','run','--campaign',str(root),'--complex-id',cid,'--method','pypka','--timeout','5400']
            job=subprocess.check_output(command,text=True).strip()
            atomic_json(ledger,{identity:{'job':job,'queued_cores_before':used,'requested_cpus':2,'requested_memory_gib':4}}); submitted+=1
        atomic_json(out/'unsent.json',unsent)
    print(json.dumps({'submitted':submitted,'unsent':len(unsent)}),flush=True)
    return len(unsent)


def monitor(out):
    """Compute-node controller; resources count against the same global cap."""
    import time
    import subprocess
    from .jobs import reconcile_scheduler_failure
    from .radial import collect
    require_compute(); out=Path(out); start=time.monotonic()
    while time.monotonic()-start<6800:
        unsent=dispatch(out); manifest=json.loads((out/'manifest.json').read_text()); pending=[]
        active=set(subprocess.check_output(['squeue','-r','-h','-u',os.environ['USER'],'-o','%i'],text=True).split())
        for root,cid in manifest['tasks']:
            root=Path(root); receipt=root/'jobs/pypka'/f'{cid}.json'
            if not receipt.exists() and (root/'submission.json').exists():
                task=json.loads((root/'submission.json').read_text())[f'{cid}/pypka']
                if task['job'].split(';')[0] not in active: reconcile_scheduler_failure(root,cid,'pypka')
            if not receipt.exists(): pending.append((str(root),cid))
        atomic_json(out/'progress.json',{'total':len(manifest['tasks']),'pending':len(pending),'unsent':unsent})
        if not pending:
            collect(out); return
        time.sleep(30)
    raise TimeoutError('Controller time budget reached; resume expanded-radial monitor using existing ledgers')


def status(out):
    from collections import Counter
    from .schema import read_table
    require_compute(); out=Path(out); manifest=json.loads((out/'manifest.json').read_text())
    counts=Counter(); failures=[]; baseline=Counter(); elapsed=[]
    for root,cid in manifest['tasks']:
        root=Path(root); receipt=root/'jobs/pypka'/f'{cid}.json'
        if not receipt.exists(): counts['unfinished']+=1; continue
        side=json.loads(receipt.read_text()); counts[side['status']]+=1; elapsed.append(side['wall_seconds'])
        if side['status']!='complete': failures.append({'complex_id':cid,'variant':root.name,'errors':side['errors']})
        if root.name=='baseline':
            baseline.update(r['status'] for r in read_table(root/'jobs/pypka'/f'{cid}.parquet'))
    result={'teacher_tasks':dict(counts),'failures':failures,'baseline_state_site_statuses':dict(baseline),
        'reference_types':dict(Counter(c['interface_type'] for c in manifest['cases'])),
        'reference_sizes':dict(Counter(c['size_stratum'] for c in manifest['cases'])),
        'complete_references':sum(not(c['original_missing_residues'] or c['original_incomplete_residues']) for c in manifest['cases']),
        'finished_wall_seconds':{'min':min(elapsed,default=0),'max':max(elapsed,default=0)},
        'final_report_present':(out/'report.json').exists()}
    atomic_json(out/'status.json',result); print(json.dumps(result,indent=2))
