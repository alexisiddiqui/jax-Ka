"""Resumable pKPDB label snapshot and deposited PDB coordinates, not teacher-prepared structures."""
import concurrent.futures
import csv
import gzip
import json
import os
import re
import time
from pathlib import Path
from .download import fetch
from .runtime import atomic_json, digest, require_compute


def run(out):
    require_compute(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    labels=out/'pkas.csv'; receipt=out/'labels-receipt.json'
    if not receipt.exists():
        atomic_json(receipt,fetch('https://bucket.pypka.org/pkas.csv',labels,limit=4*1024**3))
    assert digest(labels)==json.loads(receipt.read_text())['sha256']
    index=out/'index.json'
    if not index.exists():
        ids=set(); other=set(); count=0
        with labels.open(newline='') as stream:
            reader=csv.DictReader(stream,delimiter=';')
            if 'idcode' not in (reader.fieldnames or []):raise ValueError(f'Unexpected pKPDB schema: {reader.fieldnames}')
            for row in reader:
                entry=row['idcode'].strip(); count+=1
                if re.fullmatch(r'[0-9][A-Za-z0-9]{3}',entry):ids.add(entry.lower())
                else:other.add(entry)
        atomic_json(index,dict(pdb_ids=sorted(ids),other_ids=sorted(other),label_rows=count,
            labels_sha256=digest(labels),scope='Deposited asymmetric-unit mmCIF; not historical PypKa-prepared coordinates. Non-PDB IDs are inventoried separately. No training admission implied.'))
    manifest=json.loads(index.read_text()); ids=manifest['pdb_ids']
    def one(entry):
        dest=out/'structures'/entry[1:3]; dest.mkdir(parents=True,exist_ok=True)
        path=dest/f'{entry}.cif.gz'; meta=dest/f'{entry}.json'
        try:
            if meta.exists() and path.exists():
                saved=json.loads(meta.read_text())
                if digest(path)==saved['sha256']:return entry,'cached',path.stat().st_size
            info=fetch(f'https://files.rcsb.org/download/{entry}.cif.gz',path,limit=512*1024**2)
            with gzip.open(path,'rt') as stream:
                if not stream.read(256).lstrip().startswith('data_'):raise ValueError('Not mmCIF')
                while stream.read(1024*1024):pass  # validate gzip checksum too
            atomic_json(meta,dict(info,pdb_id=entry,source_kind='deposited asymmetric unit'))
            return entry,'downloaded',info['bytes']
        except Exception as exc:return entry,'error',repr(exc)
    counts={'cached':0,'downloaded':0,'error':0}; size=0; began=time.time(); errors=[]
    def status(done):
        report=dict(total=len(ids),processed=sum(counts.values()),counts=counts,bytes=size,
            elapsed_seconds=time.time()-began,complete=done,errors=errors,
            other_ids=len(manifest['other_ids']),job=os.environ['SLURM_JOB_ID'])
        atomic_json(out/'status.json',report); print(json.dumps({k:v for k,v in report.items() if k!='errors'}),flush=True)
    status(False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # At most 128 outstanding requests/futures; two simultaneous HTTP requests.
        for start in range(0,len(ids),128):
            for entry,state,value in pool.map(one,ids[start:start+128]):
                counts[state]+=1
                if state=='error':errors.append(dict(pdb_id=entry,error=value))
                else:size+=value
            status(False)
    status(True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
