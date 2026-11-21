"""Source-pinned native recovery units; legacy cache payloads stay immutable."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from masif_pkpdb_abag import membership,contact_components

POLICY='pkpdb-native-targeted-recovery-v1'


def signature(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


def write_inventory(path,units,metadata):
    if not units or len({u['id'] for u in units})!=len(units):raise ValueError('empty or duplicate recovery units')
    data=dict(format=POLICY,metadata=metadata,units=sorted(units,key=lambda u:u['id']))
    data['inventory_sha256']=signature(data);atomic_json(path,data)


def source_hashes(runtime,cid):
    from pkatrain.production_graphs import source,PKPDB,PKPDB_STRUCTURES
    entry=source(runtime,PKPDB+'/entries/'+cid)
    hashes={name:digest(entry/name) for name in ['defects.json','receipt.json']}
    hashes['structure']=digest(source(runtime,PKPDB_STRUCTURES+'/'+cid[1:3]+'/'+cid+'.cif.gz'))
    return hashes


def load_inventory(path,runtime,split='all'):
    data=json.loads(Path(path).read_text());content={k:v for k,v in data.items() if k!='inventory_sha256'}
    if data['format']!=POLICY or signature(content)!=data['inventory_sha256']:raise ValueError('recovery inventory hash or format mismatch')
    rows,hashes=membership(runtime);lookup={r['id']:r for r in rows}
    if data['metadata']['membership_sha256']!=hashes:raise ValueError('recovery source membership changed')
    seen=set()
    for u in data['units']:
        row=lookup.get(u['cid'])
        if u['id'] in seen or u['dataset']!='pkpdb' or u.get('unit_policy')!=POLICY or row is None:
            raise ValueError('invalid recovery identity/policy')
        seen.add(u['id'])
        if u['split']!=row['split'] or u['group']!=row['group'] or u['expected_chains']!=int(row['n_chains']):
            raise ValueError('recovery unit differs from training membership')
        single='chain_index' in u;grouped='chain_indices' in u
        if single==grouped:raise ValueError('choose exactly one canonical chain selection')
        indices=[u['chain_index']] if single else u['chain_indices']
        if not indices or len(indices)!=len(set(indices)) or any(type(i)!=int or not 0<=i<u['expected_chains'] for i in indices):
            raise ValueError('invalid recovery chain selection')
        if single and u['id']!='pkpdb/'+u['cid']+'/chain-%04d'%indices[0]:raise ValueError('chain ID mismatch')
        if set(u.get('source_files_sha256',{}))!={'structure','defects.json','receipt.json'}:raise ValueError('native source hashes missing')
        if grouped and (u.get('role')!='Ab' or len(indices)<2 or not u.get('source_files_sha256')):
            raise ValueError('intact antibody recovery requires pinned multichain Ab source')
    return [u for u in data['units'] if split=='all' or u['split']==split],dict(data['metadata'],inventory_sha256=data['inventory_sha256'],unit_policy=POLICY)


def plan(runtime,audit,output):
    runtime=Path(runtime);audit=Path(audit);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    b=audit.parent;old=b/'masif-pkpdb-train-v1';grouped=b/'masif-pkpdb-abag-v1'
    redo=json.loads((audit/'redo-plan.json').read_text());rows,hashes=membership(runtime);members={r['id']:r for r in rows}
    catalog={r['id']:r for r in json.loads((audit/'full/catalog.json').read_text())['entries']}
    diagnostics=list(csv.DictReader((old/'failure-diagnostics-v1/failure-diagnostics.csv').open()))
    counts=Counter(r['category'] for r in diagnostics);retry=[];held=[];large=[]
    for r in diagnostics:
        category=r['category']
        if category=='PDB2PQR_repair_threshold' and int(r['missing_atoms'])==0:retry.append((r['id'],'terminal_oxygen_retry'))
        elif category in ('collapsed_mesh_faces','severe_mesh_collapse','nonfinite_shape_index','MSMS_segfault'):
            retry.append((r['id'],category))
        elif category=='APBS_resource_outlier':large.append(r)
        else:held.append(r)
    # Exact diagnostic categories are retained; fail if an expected retry class
    # was renamed rather than silently withholding a repairable input.
    atomic_json(output/'failure-triage.json',dict(categories=dict(counts),regular_retries=retry,large_resource=large,held=held,
        diagnostics_sha256=digest(old/'failure-diagnostics-v1/failure-diagnostics.csv')))
    targets={r['id']:'confirmed_cached_geometry' for r in redo['geometry_rebuild'] if not r['already_replaced_by_grouped_cache']}
    targets.update(dict(retry));units=[]
    for rid,reason in sorted(targets.items()):
        _,cid,state=rid.split('/');index=int(state.split('-')[1]);m=members[cid]
        n=(catalog[rid]['summary']['n'] if rid in catalog else next(int(r['residues']) for r in diagnostics if r['id']==rid))
        units.append(dict(id=rid,dataset='pkpdb',cid=cid,state=state,split=m['split'],group=m['group'],
            family='pkpdb_chain',chain_index=index,expected_chains=int(m['n_chains']),n=n,unit_policy=POLICY,recovery_reason=reason,
            source_files_sha256=source_hashes(runtime,cid)))
    write_inventory(output/'chains.json',units,dict(membership_sha256=hashes,source_audit_sha256=digest(audit/'redo-plan.json'),
        source='confirmed legacy geometry and diagnosed native preparation retries',unit_policy=POLICY))
    from pkatrain.production_sidechains import _pkpdb_atoms
    from pkatrain.production_inputs import _residues
    from pkatrain.production_graphs import source,PKPDB,PKPDB_STRUCTURES
    roles={r['id']:r for r in json.loads((grouped/'inventory/inventory.json').read_text())['parents']};antibodies=[]
    for row in redo['additional_antibody_regrouping']:
        cid=row['cid'];m=members[cid];evidence=roles[cid]
        if evidence['status']!='antibody_only':raise ValueError('regrouping is not an antibody-only assembly')
        bb,atoms,identity=_pkpdb_atoms(runtime,cid);other,chain,other_identity=_residues('pkpdb',runtime,cid)
        if not np.array_equal(identity,other_identity) or not np.allclose(bb,other,atol=.001,rtol=0):raise ValueError('canonical source order differs')
        groups,edges=contact_components(atoms,chain,int(m['n_chains']))
        groups=[g for g in groups if len(g)>1]
        if groups!=row['chain_index_groups']:raise ValueError('antibody contact grouping changed')
        entry=source(runtime,PKPDB+'/entries/'+cid);labels=json.loads((entry/'defects.json').read_text())['sequences']
        hashes_source={name:digest(entry/name) for name in ['defects.json','receipt.json']}
        hashes_source['structure']=digest(source(runtime,PKPDB_STRUCTURES+'/'+cid[1:3]+'/'+cid+'.cif.gz'))
        for component,indices in enumerate(groups):
            canonical=np.flatnonzero(np.isin(chain,indices)).astype(int).tolist()
            antibodies.append(dict(id='pkpdb-antibody/'+cid+'/component-%03d/Ab'%component,dataset='pkpdb',cid=cid,state='Ab',role='Ab',
                split=m['split'],group=m['group'],family='antibody_only',n=len(canonical),unit_policy=POLICY,
                chain_indices=indices,expected_chains=int(m['n_chains']),source_chain_labels=[labels[i]['chain'] for i in indices],
                canonical_residue_ids=canonical,source_files_sha256=hashes_source,component_chain_indices=indices,
                assembly_contact_edges=[e for e in edges if all(i in indices for i in e['chain_indices'])],
                role_evidence=[e for e in evidence['role_evidence'] if e['chain_index'] in indices],
                supersedes=['pkpdb/'+cid+'/chain-%04d'%i for i in indices],
                assembly_scope='contact-connected supplied antibody-only assembly; no antigen or absent chain reconstructed'))
    write_inventory(output/'antibodies.json',antibodies,dict(membership_sha256=hashes,source_audit_sha256=digest(audit/'redo-plan.json'),
        source='additional intact antibody-only training assemblies',unit_policy=POLICY))
    atomic_json(output/'summary.json',dict(chain_units=len(units),antibody_units=len(antibodies),antibody_parents=len(redo['additional_antibody_regrouping']),
        chain_reasons=dict(Counter(u['recovery_reason'] for u in units)),held_original_failures=len(held),large_resource_outliers=large))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True);plan(a.runtime,a.audit,a.output)
