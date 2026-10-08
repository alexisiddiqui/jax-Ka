"""Deterministic 5k pKPDB pilot construction; no new teacher prediction or fitting."""
import concurrent.futures
import csv
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path
from .runtime import atomic_json,digest,require_compute,config_hash
from .pkpdb_pilot_refs import read,reference_inventory,disallowed_hit
from .pkpdb_pilot_clean import metadata,clean


def labels(root,out):
    path=root/'pretraining/pkpdb-v1/pkas.csv';receipt=read(root/'pretraining/pkpdb-v1/labels-receipt.json')
    assert digest(path)==receipt['sha256']
    dest=out/'labels.sqlite'
    if dest.exists():
        assert read(out/'labels.json')['source_sha256']==receipt['sha256'];return
    temporary=out/'labels.building.sqlite';temporary.unlink(missing_ok=True)
    db=sqlite3.connect(temporary);db.execute('pragma journal_mode=OFF');db.execute('pragma synchronous=OFF')
    db.execute('create table labels(pdb text,chain text,kind text,number text,pka real)')
    batch=[];count=0
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream,delimiter=';'):
            value=float(row['pk'])
            batch.append((row['idcode'].lower(),row['chain'],row['residue_name'],row['residue_number'],value));count+=1
            if len(batch)==50000:db.executemany('insert into labels values(?,?,?,?,?)',batch);db.commit();batch=[]
    if batch:db.executemany('insert into labels values(?,?,?,?,?)',batch)
    db.execute('create index by_pdb on labels(pdb)');db.commit();db.close()
    atomic_json(out/'labels.json',dict(source_sha256=receipt['sha256'],rows=count));temporary.rename(dest)


def search(root,out,rows,references,batch_number,threads):
    if not rows:return {}
    folder=out/'sequence'/f'batch-{batch_number:03d}';folder.mkdir(parents=True,exist_ok=True)
    owners=defaultdict(set);seqs={}
    for r in rows:
        for c in r['chains']:
            key='q'+config_hash(c['sequence'])[:24];seqs[key]=c['sequence'];owners[key].add(r['pdb_id'])
    fasta=folder/'queries.fasta';fasta.write_text(''.join(f'>{k}\n{s}\n' for k,s in sorted(seqs.items())))
    binary=root/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs';hits=folder/'hits.tsv'
    command=[str(binary),'easy-search',str(fasta),str(out/'references.fasta'),str(hits),str(Path(os.environ['TMPDIR'])/f'pkpdb-mmseqs-{batch_number}'),
        '--threads',str(threads),'--min-seq-id','0.3','-c','0','-e','1000000','--alignment-mode','3','--max-seqs','1000000',
        '--exhaustive-search','1','--split-memory-limit','32G','--format-output','query,target,fident,qcov,tcov']
    if not hits.exists() or not (folder/'receipt.json').exists():
        with (folder/'search.log').open('w') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(folder/'receipt.json',dict(command=command,queries_sha256=digest(fasta),references_sha256=digest(out/'references.fasta'),hits_sha256=digest(hits)))
    else:
        receipt=read(folder/'receipt.json')
        assert receipt['queries_sha256']==digest(fasta) and receipt['references_sha256']==digest(out/'references.fasta') and receipt['hits_sha256']==digest(hits)
    exclusions=defaultdict(list)
    with hits.open() as stream:
        for line in stream:
            q,t,i,qc,tc=line.strip().split('\t');i,qc,tc=map(float,(i,qc,tc));kinds=references['references'][t]['kinds']
            if disallowed_hit(i,qc,tc,kinds):
                for pdb in owners[q]:exclusions[pdb].append(dict(query=q,target=t,identity=i,qcov=qc,tcov=tc,kinds=kinds))
    atomic_json(folder/'excluded.json',exclusions)
    return exclusions


def run(root,out,smoke=False):
    threads=int(os.environ['SLURM_CPUS_PER_TASK']);require_compute(threads=threads)
    out.mkdir(parents=True,exist_ok=True);began=time.time()
    if (out/'verification.json').exists() and read(out/'verification.json')['passed']:return
    refs=read(out/'references.json') if (out/'references.json').exists() else reference_inventory(root,out)
    if refs['unresolved']:raise RuntimeError(f'Unresolved experimental reserve sequences: {refs["unresolved"]}')
    labels(root,out)
    index=root/'pretraining/pkpdb-v1/index.json';order=read(index)['pdb_ids'];random.Random(20261006).shuffle(order)
    manifest=dict(target=5000,selection_seed=20261006,label_index_sha256=digest(index),references_sha256=digest(out/'references.json'),
        validation_test_identity_cutoff=.9,validation_test_shorter_coverage=.8,experimental_identity_cutoff=.3,experimental_bidirectional_coverage=.8,
        selection='first 5000 eligible structures in deterministic shuffled order; same cohort for raw and cleaned labels',
        masks='Frozen anchor gap tiers; ligands 15/25 Å, buffers 15/20 Å, glycans 20/25 Å, eligible exposed metals 25/25 Å; buried/coordinated metals rejected',
        scope='Temporary pilot; deposited asymmetric units, 30–1500 declared protein residues, no nonprotein polymer. No pKa recalculation or structural reconstruction.',
        code={p.name:digest(p) for p in list(Path(__file__).parent.glob('pkpdb_pilot*.py'))+
            [Path(__file__).parent/name for name in ('glycan_buffer_policy.py','conformers.py','supervision.py','anchor_tiers.py')]})
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==manifest
    else:atomic_json(out/'protocol.json',manifest)
    accepted=[];audit=[];reserved=set(refs['reserved_pdb_ids']);batch_size=40 if smoke else 2000
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(16,threads//2)) as pool:
        for batch_number,start in enumerate(range(0,len(order),batch_size)):
            ids=order[start:start+batch_size];tasks=[]
            for pdb in ids:
                if pdb in reserved:audit.append(dict(pdb_id=pdb,status='rejected',reason='reserved_pdb_id'));continue
                path=root/'pretraining/pkpdb-v1/structures'/pdb[1:3]/f'{pdb}.cif.gz';tasks.append((pdb,str(path)))
            meta=list(pool.map(metadata,tasks,chunksize=4));audit.extend(r for r in meta if r['status']!='candidate')
            candidates=[r for r in meta if r['status']=='candidate']
            atomic_json(out/'status.json',dict(stage='sequence screening',batch=batch_number,accepted=len(accepted),candidates=len(candidates),target=5000))
            excluded=search(root,out,candidates,refs,batch_number,threads)
            passing=[]
            for r in candidates:
                if r['pdb_id'] in excluded:audit.append(dict(pdb_id=r['pdb_id'],status='rejected',reason='sequence_overlap'))
                else:passing.append(r)
            results=list(pool.map(clean,[(str(root),str(out),r) for r in passing],chunksize=1));audit.extend(results)
            errors=[r for r in results if r['status']=='pipeline_error']
            accepted.extend(r for r in results if r['status']=='accepted')
            report=dict(stage='cleaning',processed=len(audit),accepted=len(accepted),target=5000,
                reasons=dict(Counter(r.get('reason',r['status']) for r in audit)),elapsed_seconds=time.time()-began,pipeline_errors=errors)
            atomic_json(out/'status.json',report);atomic_json(out/'audit.json',audit);print(json.dumps({k:v for k,v in report.items() if k!='pipeline_errors'}),flush=True)
            if errors:raise RuntimeError(f'{len(errors)} unexpected preparation errors; pilot not released')
            if smoke or len(accepted)>=5000:break
    if smoke:
        assert accepted,'Smoke produced no accepted structures'
        atomic_json(out/'smoke.json',dict(passed=True,processed=len(audit),accepted=len(accepted)));return
    if len(accepted)<5000:raise RuntimeError(f'Only {len(accepted)} accepted structures; refusing to relax leakage or cleaning gates')
    selected=accepted[:5000];assert len({r['pdb_id'] for r in selected})==5000
    for r in selected:
        assert r['pdb_id'] not in reserved
        path=out/'entries'/r['pdb_id'];assert digest(path/'graph.npz')==r['sha256'] and digest(path/'sites.json')==r['sites_sha256']
    release=dict(target=5000,records=selected,raw_arm='all unambiguously mapped finite scalar labels',
        clean_arm='the same structures/inputs with train_mask applied from sites.json',
        validation_source=str(root/'pretraining/graph-pilot-v1'),validation_manifest_sha256=digest(root/'pretraining/graph-pilot-v1/manifest.json'),
        protocol_sha256=digest(out/'protocol.json'),reference_sha256=digest(out/'references.json'),
        raw_sites=sum(r['counts']['raw_sites'] for r in selected),clean_sites=sum(r['counts']['clean_sites'] for r in selected),
        training_launched=False,notes=['Historical teacher preparation/version remain unknown; direct author chain/residue/type matches only.',
            'Structures must have at least one clean site in both arms; raw/clean comparison is conditional on this cohort.',
            'Within-training component_id groups exact sequence sets only; 90% exclusion is against frozen held-out chains, not a within-training clustering claim.'])
    atomic_json(out/'pilot.json',release);atomic_json(out/'verification.json',dict(passed=True,structures=5000,raw_sites=release['raw_sites'],clean_sites=release['clean_sites'],pilot_sha256=digest(out/'pilot.json')))
    text=['# Temporary pKPDB 5k pilot','',f'5,000 structures; {release["raw_sites"]:,} raw mapped sites; {release["clean_sites"]:,} clean training sites.',
        'All candidate protein chains were screened against all frozen validation/test chains at ≥90% identity and ≥80% shorter-sequence coverage. Experimental references retain ≥30% identity / ≥80% bidirectional coverage exclusions.',
        'Raw and clean arms share identical inputs and structure membership. No training launched.', '', '| Audit reason | Structures |', '|---|---:|']
    text.extend(f'| {k} | {v} |' for k,v in sorted(Counter(r.get('reason',r['status']) for r in audit).items()))
    text+=['',*release['notes']];(out/'report.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    root=Path(os.environ['PKABENCH_RUNTIME']);out=Path(sys.argv[1]);run(root,out,'--smoke' in sys.argv)
