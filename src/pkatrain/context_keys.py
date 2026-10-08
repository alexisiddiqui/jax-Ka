"""Recover authoritative graph-node identities from the same resolved CIF view."""
import concurrent.futures
import json
import os
from pathlib import Path
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute


def read(path):return json.loads(Path(path).read_text())


def one(task):
    root,out,row=task;root=Path(root);out=Path(out);cid=row['complex_id'];path=out/f'{cid}.json'
    if path.exists():return cid,digest(path)
    from biotite.structure.io import pdbx
    import biotite.structure as struc
    from pkabench.pkpdb_pilot_refs import cif,sequences
    from pkabench.conformers import resolve
    from pkabench.prep import CANONICAL
    source=root/'pretraining/pkpdb-v1/structures'/cid[1:3]/f'{cid}.cif.gz'
    assert digest(source)==row['source_sha256']
    file=cif(source);chains,_=sequences(file);selected=[r['chain'] for r in chains]
    file,_=resolve(file,selected)
    label=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=False)
    author=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True)
    working=label.copy();working.res_id=author.res_id.copy();working.ins_code=author.ins_code.copy()
    keep=np.isin(working.res_name,list(CANONICAL)) & np.isin(working.chain_id,selected) & ~np.isin(np.char.upper(working.element),['H','D'])
    starts=struc.get_residue_starts(working,add_exclusive_stop=True);keys=[]
    for s,e in zip(starts[:-1],starts[1:]):
        if keep[s]:keys.append([str(author.chain_id[s]),int(author.res_id[s]),str(author.ins_code[s]).strip(),str(author.res_name[s])])
    assert len(keys)==row['n']
    atomic_json(path,dict(keys=keys,source_sha256=row['source_sha256'],graph_sha256=row['sha256']))
    return cid,digest(path)


if __name__=='__main__':
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    root=Path(os.environ['PKABENCH_RUNTIME']);out=root/'pretraining/augmentation-v1/contexts/node-keys';out.mkdir(parents=True,exist_ok=True)
    m=read(root/'pretraining/pkpdb-5k-comparison-v1/gqt-clean/manifest.json')
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        hashes=dict(pool.map(one,[(str(root),str(out),r) for r in m['records'] if r['split']=='train'],chunksize=4))
    atomic_json(out/'verification.json',dict(passed=True,hashes=hashes))
