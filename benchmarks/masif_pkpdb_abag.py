"""Inventory pKPDB antibody/antigen assemblies in the v5 training setup.

All supplied Ab chains share one surface and all supplied Ag chains another,
within the existing training assembly. This includes H+L and preserves copies;
it does not invent a minimal complex or choose chains by alphabetic labels.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import time

from pkabench.runtime import atomic_json,digest,require_compute

UNIT_POLICY='pkpdb-ab-ag-training-assembly-v1'
CONTACT_A=5.
AB_DESCRIPTION=re.compile(r'\b(antibody|immunoglobulin|nanobody|fab|scfv|vhh|igg[1-4]?|ig[akmd])\b',re.I)


def sequence_key(sequence):
    return 's'+hashlib.sha256(sequence.encode()).hexdigest()[:20]


def membership(runtime):
    folder=Path(runtime)/'training/pool-v4'
    manifest=json.loads((folder/'pool-v4.json').read_text());rows=[];seen=set();hashes={}
    for split,name,field in [('train','pkpdb.tsv','pool'),('val','pkpdb-val.tsv','validation')]:
        path=folder/name;hashes[name]=digest(path)
        if hashes[name]!=manifest['datasets']['pkpdb'][field]['sha256']:
            raise ValueError('pKPDB membership differs from pool manifest')
        for row in csv.DictReader(path.open(),delimiter='\t'):
            if row['id'] in seen:raise ValueError('pKPDB membership overlap/duplicate')
            rows.append(dict(row,split=split));seen.add(row['id'])
    return rows,hashes


def prepare(runtime,v5,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows,hashes=membership(runtime)
    v5=Path(v5);manifest=json.loads((v5/'pool/pool-v5/pool-v5.json').read_text())
    if not manifest['base']['pkpdb'].startswith('unchanged:'):
        raise ValueError('v5 no longer inherits the unchanged pKPDB pool')
    # Trusted, locally generated pool-builder snapshot, never an uploaded pickle.
    state_path=v5/'pool/state.pkl'
    with state_path.open('rb') as stream:state=pickle.load(stream)
    candidates={r['id']:r for r in state['candidates']['pkpdb'][0]}
    candidates.update({r['id']:r for r in state['validation']['pkpdb']})
    sequences={};parents={}
    for row in rows:
        seqs=candidates[row['id']]['_sequences']
        if not seqs:raise ValueError('empty pKPDB sequence inventory')
        parents[row['id']]=sorted({sequence_key(s) for s in seqs})
        for sequence in seqs:sequences[sequence_key(sequence)]=sequence
    fasta=output/'sequences.fasta'
    fasta.write_text(''.join('>'+key+'\n'+s+'\n' for key,s in sorted(sequences.items())))
    atomic_json(output/'prepared.json',dict(format='pkpdb-abag-sequences-v1',rows=rows,
        parent_sequence_keys=parents,sequences_sha256=digest(fasta),membership_sha256=hashes,
        pool_state_sha256=digest(state_path),v5_manifest_sha256=digest(v5/'pool/pool-v5/pool-v5.json')))
    print(json.dumps(dict(parents=len(rows),unique_sequences=len(sequences))),flush=True)


def read_fasta(path):
    out={};key=None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):key=line[1:];out[key]=''
        elif key:out[key]+=line.strip()
    return out


def number_chunk(items):
    from anarci import anarci
    nums,alignments,_=anarci(items,scheme='imgt',allowed_species=None,allow={'H','K','L'},assign_germline=False,ncpu=1)
    results={}
    for (key,_),domains,info in zip(items,nums,alignments):
        results[key]=[]
        for (numbering,start,end),ali in zip(domains or [],info or []):
            results[key].append(dict(chain_type=ali['chain_type'],start=int(start),end=int(end),
                bitscore=float(ali['bitscore']),evalue=float(ali['evalue']),
                cdr3=[[int(p[0]),p[1].strip(),a] for p,a in numbering if 105<=p[0]<=117 and a!='-']))
    return results


def number(root,workers):
    import importlib.metadata as md
    root=Path(root);sequences=read_fasta(root/'sequences.fasta');items=sorted(sequences.items())
    parts=[items[i:i+50] for i in range(0,len(items),50)];numbered={};began=time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        for i,res in enumerate(pool.map(number_chunk,parts)):
            numbered.update(res)
            if i%20==0:print(json.dumps(dict(done=len(numbered),of=len(items),wall_s=time.monotonic()-began)),flush=True)
    atomic_json(root/'numbered.json',dict(numbered={k:v for k,v in numbered.items() if v},
        failed=sorted(k for k,v in numbered.items() if not v),sequences=len(sequences),
        sequences_sha256=digest(root/'sequences.fasta'),anarci_version=md.version('anarci'),
        scheme='imgt',allow=['H','K','L'],wall_s=time.monotonic()-began))


def partition_chains(sequences,numbered):
    """Conservative Ig-domain roles; unresolved antibody-labelled chains fail."""
    ab=[];ag=[];evidence=[]
    for i,row in enumerate(sequences):
        domains=numbered.get(sequence_key(row['sequence']),[])
        valid=[d for d in domains if d['chain_type'] in ('H','K','L') and d.get('cdr3')]
        if valid:
            ab.append(i);evidence.append(dict(chain_index=i,chain=row['chain'],
                role='Ab',domain_types=[d['chain_type'] for d in valid],evidence='ANARCI IMGT H/K/L'))
        elif domains or AB_DESCRIPTION.search(row.get('description','')):
            raise ValueError('unresolved antibody-labelled or truncated Ig chain: '+row['chain'])
        else:
            ag.append(i);evidence.append(dict(chain_index=i,chain=row['chain'],role='Ag',
                evidence='non-Ig protein in a contact-connected Ab/Ag assembly'))
    return ab,ag,evidence


def contact_components(atoms,chain,n_chains):
    """Components of heavy-atom chain contacts; source chain indices survive."""
    import numpy as np
    from scipy.spatial import cKDTree
    parent=list(range(n_chains));edges=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    xyz=[];trees=[]
    for i in range(n_chains):
        points=np.asarray([p for j,res in enumerate(atoms) if chain[j]==i
                           for name,p in res.items() if name[0] not in ('H','D')],float).reshape(-1,3)
        xyz.append(points);trees.append(cKDTree(points) if len(points) else None)
    for i in range(n_chains):
        if trees[i] is None:continue
        for j in range(i+1,n_chains):
            if trees[j] is None:continue
            distances,_=trees[i].query(xyz[j],distance_upper_bound=CONTACT_A)
            contacts=int(np.isfinite(distances).sum())
            if contacts>=3:
                edges.append(dict(chain_indices=[i,j],contacting_heavy_atoms=contacts))
                a,b=find(i),find(j);parent[b]=a
    groups={}
    for i in range(n_chains):
        if len(xyz[i]):groups.setdefault(find(i),[]).append(i)
    return sorted(groups.values(),key=lambda x:x[0]),edges


def inventory_one(task):
    runtime,row,expected,numbered=task;runtime=Path(runtime);cid=row['id']
    from pkatrain.production_graphs import source,PKPDB,PKPDB_STRUCTURES
    entry=source(runtime,PKPDB+'/entries/'+cid)
    try:
        defects=json.loads((entry/'defects.json').read_text());sequences=defects['sequences']
        if sorted({sequence_key(r['sequence']) for r in sequences})!=expected:
            raise ValueError('source sequence inventory differs from numbered input')
        if len(sequences)!=int(row['n_chains']):raise ValueError('chain inventory differs from training pool')
        ab,ag,evidence=partition_chains(sequences,numbered)
        if not ag:return dict(id=cid,status='antibody_only',split=row['split'],role_evidence=evidence,units=[])
        from pkatrain.production_sidechains import _pkpdb_atoms
        from pkatrain.production_inputs import _residues
        import numpy as np
        bb,atoms,identity=_pkpdb_atoms(runtime,cid);other,chain,other_identity=_residues('pkpdb',runtime,cid)
        if not np.array_equal(identity,other_identity) or not np.allclose(bb,other,atol=.001,rtol=0):
            raise ValueError('pKPDB canonical atom order differs')
        components,edges=contact_components(atoms,chain,len(sequences))
        structure=source(runtime,PKPDB_STRUCTURES+'/'+cid[1:3]+'/'+cid+'.cif.gz')
        hashes={'defects.json':digest(entry/'defects.json'),'receipt.json':digest(entry/'receipt.json'),
                'structure':digest(structure)}
        units=[];unpaired=[]
        for component,indices in enumerate(components):
            ab_chains=sorted(set(indices)&set(ab));ag_chains=sorted(set(indices)&set(ag))
            if not ab_chains or not ag_chains:unpaired.append(indices);continue
            for role,selected in [('Ab',ab_chains),('Ag',ag_chains)]:
                canonical=np.flatnonzero(np.isin(chain,selected)).astype(int).tolist()
                if not canonical:raise ValueError('empty canonical partner')
                units.append(dict(id='pkpdb-abag/'+cid+'/complex-%03d/'%component+role,
                    dataset='pkpdb',cid=cid,state=role,role=role,split=row['split'],group=row['group'],
                    family='antibody_partner' if role=='Ab' else 'antigen_partner',n=len(canonical),
                    unit_policy=UNIT_POLICY,chain_indices=selected,expected_chains=len(sequences),
                    source_chain_labels=[sequences[i]['chain'] for i in selected],
                    canonical_residue_ids=canonical,source_files_sha256=hashes,
                    component_chain_indices=indices,assembly_contact_edges=[e for e in edges if all(c in indices for c in e['chain_indices'])],
                    antibody_domain_types=[t for e in evidence if e['chain_index'] in ab_chains for t in e.get('domain_types',[])],
                    role_evidence=[e for e in evidence if e['chain_index'] in selected],
                    assembly_scope='contact-connected part of supplied training assembly; no missing chains reconstructed'))
        return dict(id=cid,split=row['split'],status='accepted' if units else 'no_cross_role_contact',
                    units=units,components=components,unpaired_components=unpaired,role_evidence=evidence)
    except Exception as e:return dict(id=cid,split=row['split'],status='quarantined',error=str(e),units=[])


def build_inventory(runtime,root,workers):
    root=Path(root);prepared=json.loads((root/'prepared.json').read_text());data=json.loads((root/'numbered.json').read_text())
    rows,hashes=membership(runtime)
    if hashes!=prepared['membership_sha256'] or rows!=prepared['rows']:raise ValueError('prepared membership changed')
    if digest(root/'sequences.fasta')!=data['sequences_sha256'] or data['sequences_sha256']!=prepared['sequences_sha256']:
        raise ValueError('numbered sequences changed')
    numbered=data['numbered'];tested=set(numbered)|set(data['failed'])
    if tested!=set(read_fasta(root/'sequences.fasta')):raise ValueError('ANARCI input coverage incomplete')
    tasks=[]
    for row in rows:
        keys=prepared['parent_sequence_keys'][row['id']]
        if any(k in numbered for k in keys):tasks.append((str(runtime),row,keys,{k:numbered[k] for k in keys if k in numbered}))
    results=[];began=time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        for i,res in enumerate(pool.map(inventory_one,tasks)):
            results.append(res)
            if i%50==0:print(json.dumps(dict(done=i+1,of=len(tasks),counts=dict(Counter(x['status'] for x in results)))),flush=True)
    units=[u for r in results for u in r['units']]
    metadata=dict(format='masif-pkpdb-abag-inventory-v1',unit_policy=UNIT_POLICY,
        source='native pKPDB antibody/antigen training assemblies inherited by v5',
        membership_sha256=hashes,numbered_sha256=digest(root/'numbered.json'),
        prepared_sha256=digest(root/'prepared.json'),contact_A=CONTACT_A,minimum_contacting_heavy_atoms=3,
        parents_screened=len(rows),Ig_candidate_parents=len(tasks),counts=dict(Counter(r['status'] for r in results)),
        unit_counts=dict(Counter(u['split']+'/'+u['role'] for u in units)),wall_s=time.monotonic()-began)
    content=dict(metadata=metadata,units=units,parents=results)
    content['inventory_sha256']=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    atomic_json(root/'inventory.json',content)
    print(json.dumps(metadata),flush=True)


def load_inventory(path,runtime,split='all'):
    path=Path(path);data=json.loads(path.read_text());expected=data['inventory_sha256']
    content={k:v for k,v in data.items() if k!='inventory_sha256'}
    if hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()!=expected:
        raise ValueError('pKPDB Ab/Ag inventory integrity mismatch')
    _,hashes=membership(runtime)
    if hashes!=data['metadata']['membership_sha256'] or data['metadata']['unit_policy']!=UNIT_POLICY:
        raise ValueError('pKPDB Ab/Ag source membership or unit policy differs')
    units=[u for u in data['units'] if split=='all' or u['split']==split]
    return units,dict(data['metadata'],inventory_sha256=expected,inventory_file_sha256=digest(path))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['prepare','number','inventory'])
    p.add_argument('--runtime',type=Path);p.add_argument('--v5-root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    if a.command=='prepare':prepare(a.runtime,a.v5_root,a.output)
    elif a.command=='number':number(a.output,a.workers)
    else:build_inventory(a.runtime,a.output,a.workers)


if __name__=='__main__':main()
