"""Prepare matched-cohort GQT/pKAI raw/clean pilot inputs on compute nodes."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from pkabench.runtime import atomic_json, digest, require_compute
from .records import read


def select_queries(data, keep):
    return {k: v[keep] if k in ('query_residue', 'query_group', 'labels') else v
            for k, v in data.items()}


def graph_task(task):
    pilot,out,original=task;cid=original['complex_id'];source=pilot/'entries'/cid
    assert digest(source/'graph.npz')==original['sha256']
    assert digest(source/'sites.json')==original['sites_sha256']
    with np.load(source/'graph.npz') as f:data={k:f[k] for k in f.files}
    sites=read(source/'sites.json');result={}
    for arm in ('raw','clean'):
        keep=np.array([arm=='raw' or s['train_mask'] for s in sites],bool);assert keep.any()
        folder=out/f'gqt-{arm}'/'data'/cid;folder.mkdir(parents=True,exist_ok=True)
        if arm=='raw':
            (folder/'graph.npz').unlink(missing_ok=True)
            (folder/'graph.npz').symlink_to(source/'graph.npz')
        else:np.savez_compressed(folder/'graph.npz',**select_queries(data,keep))
        result[arm]=dict(original,split='train',q=int(keep.sum()),
            keys=[k for k,yes in zip(original['keys'],keep) if yes],sha256=digest(folder/'graph.npz'))
    return result


def prepare_graphs(root, out):
    from .graph_data import bucket
    pilot = root/'pretraining/pkpdb-5k-v2'
    verification = read(pilot/'verification.json')
    assert verification['passed'] and digest(pilot/'pilot.json') == verification['pilot_sha256']
    cohort = read(pilot/'pilot.json')
    validation = Path(cohort['validation_source'])
    assert digest(validation/'manifest.json') == cohort['validation_manifest_sha256']
    vm = read(validation/'manifest.json')
    common = dict(pilot=str(pilot), pilot_sha256=digest(pilot/'pilot.json'),
                  validation_source=str(validation), validation_manifest_sha256=digest(validation/'manifest.json'))
    out.mkdir(parents=True, exist_ok=True)
    tasks=[(pilot,out,r) for r in cohort['records']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        prepared=list(pool.map(graph_task,tasks,chunksize=4))
    for arm in ('raw', 'clean'):
        dest = out/f'gqt-{arm}'; dest.mkdir(exist_ok=True)
        rows = [r[arm] for r in prepared]
        for original in vm['records']:
            if original['split'] != 'val': continue
            cid = original['complex_id']; source = validation/'data'/cid
            assert digest(source/'graph.npz') == original['sha256']
            folder = dest/'data'/cid
            if folder.exists():assert folder.is_symlink() and folder.resolve()==source.resolve(), 'Train/validation ID collision'
            else:folder.symlink_to(source, target_is_directory=True)
            rows.append(original)
        capacities = {}
        for b in sorted({bucket(r) for r in rows}):
            members = [r for r in rows if bucket(r) == b]
            capacities[b] = [int(np.ceil(max(r[k] for r in members)/32)*32) for k in ('n','k','q')]
        manifest = dict(common, records=rows, capacities=capacities,
            train=[r['complex_id'] for r in rows if r['split']=='train'], val=vm['val'],
            config=dict(seed=17, epochs=20, learning_rate=.001, accumulation=8, dtype='float32',
                architecture=dict(width=44, ff=88), parameter_count=49709, strict_backbone=True,
                objective='single-state scalar pKa MSE', radius_A=20, selection='fixed final epoch 20', arm=arm),
            label_source='Historical pKPDB training labels; current PypKa clean validation labels',
            scope='5,000 fixed structures, raw versus cleaned labels; no test fitting or evaluation')
        atomic_json(dest/'manifest.json', manifest)
        atomic_json(dest/'preparation.json', dict(passed=True, complexes=len(rows),
            sites={s:sum(r['q'] for r in rows if r['split']==s) for s in ('train','val')}))
    atomic_json(out/'cohort.json', common)


def feature_task(task):
    """Export observed canonical atoms with reversible numbering, then native features."""
    root, out, row, split = task
    root, out = Path(root), Path(out); cid=row['complex_id']; dest=out/'pkai-data'/split/cid
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'receipt.json').exists(): return read(dest/'receipt.json')
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from biotite.structure.io.pdb import PDBFile
    from pkabench.prep import CANONICAL
    if split == 'train':
        from pkabench.pkpdb_pilot_refs import cif, sequences
        from pkabench.conformers import resolve
        path=root/'pretraining/pkpdb-v1/structures'/cid[1:3]/f'{cid}.cif.gz'
        assert digest(path)==row['source_sha256']
        file=cif(path); chains,_=sequences(file); file,_=resolve(file,[r['chain'] for r in chains])
        atoms=pdbx.get_structure(file, model=1, altloc='occupancy', use_author_fields=True)
        pilot=root/'pretraining/pkpdb-5k-v2/entries'/cid
        assert digest(pilot/'sites.json')==row['sites_sha256']
        sites=read(pilot/'sites.json')
    else:
        vm=read(root/'pretraining/graph-pilot-v1/manifest.json')
        path=Path(vm['source'])/'records'/f'{cid}.json'
        assert digest(path)==row['record_sha256']
        record=read(path); path=Path(record['structures']['AB'])
        assert digest(path)==record['structure_sha256']['AB']
        from pkabench.prep import read_cif
        atoms=read_cif(path)
        gp=root/'pretraining/graph-pilot-v1/data'/cid/'graph.npz'
        assert digest(gp)==row['sha256']
        with np.load(gp) as f: labels=f['labels']
        sites=[dict(zip(('complex_id','chain','resnum','icode','group'),key),pka=float(y),train_mask=True)
               for key,y in zip(row['keys'],labels)]
    atoms=atoms[np.isin(atoms.res_name,list(CANONICAL)) & ~np.isin(np.char.upper(atoms.element),['H','D'])]
    exported=atoms.copy(); mapping={}
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    for number,(s,e) in enumerate(zip(starts[:-1],starts[1:]),1):
        mapping[str(number)]=[str(atoms.chain_id[s]),int(atoms.res_id[s]),str(atoms.ins_code[s]).strip()]
        exported.chain_id[s:e]='A'; exported.res_id[s:e]=number; exported.ins_code[s:e]=''
    exported.hetero[:]=False
    file=PDBFile();file.set_structure(exported);file.write(dest/'input.pdb')
    check=PDBFile.read(dest/'input.pdb').get_structure(model=1)
    assert len(check)==len(atoms) and np.allclose(check.coord,atoms.coord,rtol=0,atol=.00051)
    atomic_json(dest/'request.json',dict(mapping=mapping,sites=sites,complex_id=cid,component_id=row['component_id'],split=split))
    env=os.environ.copy()
    with (dest/'worker.log').open('w') as log:
        subprocess.run([str(root/'envs/finetune-v1/bin/python'),'-m','pkatrain.pkai_scratch','features',str(dest)],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    return read(dest/'receipt.json')


def prepare_pkai(root, out):
    import multiprocessing
    pilot=read(root/'pretraining/pkpdb-5k-v2/pilot.json')
    vm=read(root/'pretraining/graph-pilot-v1/manifest.json')
    tasks=[(str(root),str(out),r,'train') for r in pilot['records']]
    tasks += [(str(root),str(out),r,'val') for r in vm['records'] if r['split']=='val']
    receipts=[]
    cpus=sorted(os.sched_getaffinity(0));workers=min(16,len(cpus)//2)
    assignments=multiprocessing.Queue()
    for i in range(workers):assignments.put(cpus[2*i:2*i+2])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=worker_affinity,initargs=(assignments,)) as pool:
        for receipt in pool.map(feature_task,tasks,chunksize=1):
            receipts.append(receipt)
            if len(receipts)%100==0:
                atomic_json(out/'feature-progress.json',dict(prepared=len(receipts),total=len(tasks)))
                print(json.dumps(dict(prepared=len(receipts),total=len(tasks))),flush=True)
    atomic_json(out/'pkai-features.json',dict(passed=True,records=receipts))


def worker_affinity(assignments):
    # The native worker's single-core guard must choose a different core per process.
    os.sched_setaffinity(0,set(assignments.get()))


if __name__=='__main__':
    import sys
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    root=Path(os.environ['PKABENCH_RUNTIME']);out=root/'pretraining/pkpdb-5k-comparison-v1'
    if sys.argv[1]=='graphs': prepare_graphs(root,out)
    elif sys.argv[1]=='pkai': prepare_pkai(root,out)
    else: raise ValueError(sys.argv[1])
