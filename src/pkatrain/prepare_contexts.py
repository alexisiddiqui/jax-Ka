"""Cache full pKAI atom neighborhoods and verify common GQT residue indexing."""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from .context_augmentation import encode_candidates


def read(path):return json.loads(Path(path).read_text())


def pin(queue):os.sched_setaffinity(0,set(queue.get()))


def prepare_one(task):
    record,layout,dest,total_residues=task;dest=Path(dest)
    path=Path(record['path']);cid=record['complex_id'];folder=dest/'entries'/cid
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():
        receipt=read(folder/'receipt.json')
        if receipt.get('format_version')==2:
            assert digest(folder/'candidates.npz')==receipt['sha256'];return receipt
    from .pkai_scratch import native
    native()
    from protein import Protein
    from residue import AA_ATOMS,ATOM_OHE
    req=read(path/'request.json');rows=read(path/'rows.json')
    assert digest(path/'input.pdb')==record['pdb_sha256']
    assert digest(path/'rows.json')==record['rows_sha256']
    assert digest(path/'features.npz')==record['sha256']
    protein=Protein(path/'input.pdb');residues=list(protein.iter_residues())
    n=layout['stop']-layout['start'];assert len(residues)==len(req['mapping'])
    keypath=dest/'node-keys'/f'{cid}.json';assert digest(keypath)==layout['node_keys_sha256']
    graph_keys=read(keypath)['keys'];assert len(graph_keys)==n
    graph_index={tuple(key):i for i,key in enumerate(graph_keys)};assert len(graph_index)==n
    native_to_graph={r.resnumb:graph_index.get((*req['mapping'][str(r.resnumb)],r.resname)) for r in residues}
    assert len({i for i in native_to_graph.values() if i is not None})==n
    source=Path(layout['graph']);assert digest(source)==layout['graph_sha256']
    with np.load(source) as f:
        query_res=f['query_residue'];nodes=f['nodes']
    three=('ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR')
    assert [three[i] for i in nodes[:,:20].argmax(-1)]==[key[3] for key in graph_keys]
    for i,key in zip(query_res,layout['keys']):assert graph_keys[int(i)][:3]==key[1:4]
    lookup={(*req['mapping'][str(r.resnumb)],r.resname):r for r in residues}
    atoms=list(protein.iter_atoms());coords=np.array([a.coords for a in atoms],np.float64)
    owner=np.array([native_to_graph[a.residue.resnumb]+layout['start'] if native_to_graph[a.residue.resnumb] is not None else total_residues-1 for a in atoms],np.int32)
    native_owner=np.array([a.residue.resnumb for a in atoms])
    selected=[i for i,r in enumerate(rows) if r['train_mask']]
    classes=[];values=[];owners=[];offsets=[0]
    with np.load(path/'features.npz') as f:original=f['x']
    checked=0;clear=np.zeros(total_residues,bool)
    for i in selected:
        row=rows[i];r=lookup[row['chain'],row['resnum'],row['icode'],row['group']]
        assert native_to_graph[r.resnumb] in layout['protected']
        centers=np.array([a.coords for a in r.iter_atoms() if a.aname in AA_ATOMS[r.resname]])
        ids=np.array([],dtype=int);cls=[];distance=np.zeros(len(atoms))
        if len(centers):
            distance=np.sqrt(((coords[:,None]-centers[None,:])**2).sum(-1).min(-1))
            ids=np.flatnonzero((native_owner!=r.resnumb) & (distance<15.))
            assert not np.any(distance[ids]==0)
            r.env_anames=[atoms[j].aname for j in ids];r.env_resnames=[atoms[j].residue.resname for j in ids]
            r.encode_atoms()
            # Native ordering is (distance, class name), not class index.
            ordered=sorted(zip(distance[ids],r.env_oheclasses,ids),key=lambda v:(v[0],v[1]))
            ids=np.array([j for d,c,j in ordered],dtype=int)
            cls=np.array([ATOM_OHE.index(c) for d,c,j in ordered],np.int8)
        cls=np.asarray(cls,np.int8);val=(1/distance[ids]**2).astype(np.float32);own=owner[ids]
        # Every eligible row must reproduce the already-verified native input.
        rebuilt=encode_candidates(cls,val,own,clear,original[i,4000:])
        np.testing.assert_allclose(rebuilt,original[i],rtol=2e-6,atol=1e-8)
        checked+=1;classes.extend(cls);values.extend(val);owners.extend(own);offsets.append(len(classes))
    np.savez_compressed(folder/'candidates.npz',classes=np.asarray(classes,np.int8),values=np.asarray(values,np.float32),
        owners=np.asarray(owners,np.int32),offsets=np.asarray(offsets,np.int64),rows=np.asarray(selected,np.int64)+layout['site_start'])
    receipt=dict(format_version=2,complex_id=cid,rows=len(selected),candidates=len(classes),all_unmasked_rows_verified=checked,
        native_only_residues=[(*req['mapping'][str(r.resnumb)],r.resname) for r in residues if native_to_graph[r.resnumb] is None],
        source_pdb_sha256=record['pdb_sha256'],sha256=digest(folder/'candidates.npz'))
    atomic_json(folder/'receipt.json',receipt);return receipt


def prepare(root,out):
    source=root/'pretraining/pkpdb-5k-comparison-v1';out.mkdir(parents=True,exist_ok=True)
    parent=read(source/'gqt-clean/manifest.json');pkai=read(source/'pkai-features.json');assert pkai['passed']
    keygate=read(out/'node-keys/verification.json');assert keygate['passed']
    records={r['complex_id']:r for r in parent['records'] if r['split']=='train'}
    structures={};residue_start=0;site_start=0;tasks=[]
    for r in pkai['records']:
        if r['split']=='train':
            row=records[r['complex_id']];path=source/'gqt-clean/data'/r['complex_id']/'graph.npz'
            assert digest(path)==row['sha256']
            with np.load(path) as f:protected=np.unique(f['query_residue']).tolist()
            structures[r['complex_id']]=dict(start=residue_start,stop=residue_start+row['n'],protected=protected,
                node_keys_sha256=keygate['hashes'][r['complex_id']],
                site_start=site_start,graph=str(path),graph_sha256=row['sha256'],keys=row['keys'])
            residue_start+=row['n']
        site_start+=r['sites']
    assert len(structures)==5000
    plan=dict(structures=structures,residues=residue_start+1,sites=site_start,probability=.05,
        native_only_sentinel=residue_start,native_only_policy='Retain native-only context identically in both pKAI arms; exclude it from the shared protein-residue mask',
        protection='all clean GQT supervised residue centres, shared with pKAI',
        refresh='one deterministic structure/epoch mask; repeated draws within an epoch share it',
        graph_manifest_sha256=digest(source/'gqt-clean/manifest.json'),pkai_features_sha256=digest(source/'pkai-features.json'))
    atomic_json(out/'plan.json',plan)
    tasks=[(r,structures[r['complex_id']],str(out),residue_start+1) for r in pkai['records'] if r['split']=='train']
    cpus=sorted(os.sched_getaffinity(0));workers=min(16,len(cpus)//2);queue=multiprocessing.Queue()
    for i in range(workers):queue.put(cpus[2*i:2*i+2])
    receipts=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=pin,initargs=(queue,)) as pool:
        for receipt in pool.map(prepare_one,tasks,chunksize=1):
            receipts.append(receipt)
            if len(receipts)%100==0:
                atomic_json(out/'progress.json',dict(prepared=len(receipts),total=len(tasks)))
                print(json.dumps(dict(prepared=len(receipts),total=len(tasks))),flush=True)
    counts=np.zeros(site_start,np.int64);available=np.zeros(site_start,bool)
    total=sum(r['candidates'] for r in receipts)
    arrays={name:np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=dtype,shape=(total,))
            for name,dtype in [('classes',np.int8),('values',np.float32),('owners',np.int32)]}
    offset=0
    for r in receipts:
        path=out/'entries'/r['complex_id']/'candidates.npz';assert digest(path)==r['sha256']
        with np.load(path) as f:
            ids=f['rows'];assert not available[ids].any();available[ids]=True;counts[ids]=np.diff(f['offsets'])
            for name,array in arrays.items():array[offset:offset+r['candidates']]=f[name]
        offset+=r['candidates']
    assert offset==total
    for array in arrays.values():array.flush()
    np.save(out/'offsets.npy',np.concatenate(([0],np.cumsum(counts))))
    np.save(out/'available.npy',available)
    atomic_json(out/'receipts.json',receipts)
    atomic_json(out/'verification.json',dict(passed=True,structures=5000,sites=int(available.sum()),candidates=total,
        files={name:digest(out/name) for name in ('plan.json','offsets.npy','available.npy','classes.npy','values.npy','owners.npy')},
        code_sha256={name:digest(Path(__file__).with_name(name)) for name in ('prepare_contexts.py','context_augmentation.py')}))


if __name__=='__main__':
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    root=Path(os.environ['PKABENCH_RUNTIME'])
    prepare(root,root/'pretraining/augmentation-v1/contexts')
