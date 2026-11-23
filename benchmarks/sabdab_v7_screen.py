"""Full-pool sequence/component evidence; proposed decisions, never a release."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json, digest, require_compute
from pkabench.source_policy import SourcePolicy
from sabdab_pool_overlap import CDRS, compare, domain, official_h3_links, published_cdr_string
from sabdab_v7_exposure import fasta, read, skey, write_fasta


def domain_cdrs(d):
    if 'numbering' in d:
        return domain(d['numbering'])
    return domain([v for c in CDRS for v in d.get(c,[])])


def normalise_old_cdr(d,c):
    if c in d:
        return domain(d[c])
    lo,hi=CDRS[c]
    return {p:a for p,a in domain_cdrs(d).items() if lo<=p[0]<=hi}


def merge_numbering(root,output=None):
    cached=read(root/'numbering-cached.json');raw=cached['numbered']
    new=read(root/'numbering-new.json');mismatches=[]
    for k,dd in new['numbered'].items():
        previous=raw.get(k,[])
        for d in previous:
            matches=[e for e in dd if e['start']==d['start'] and e['end']==d['end'] and
                     ('L' if e['chain_type'] in {'K','L'} else 'H') ==
                     ('L' if d['chain_type'] in {'K','L'} else 'H')]
            if not matches:
                mismatches.append(dict(sequence_key=k,reason='cached_domain_not_reproduced'))
                continue
            n=domain(matches[0]['numbering'])
            for c,(lo,hi) in CDRS.items():
                # A CDR3-only cache has no prior CDR1/2 assertion to verify.
                if c not in d and 'numbering' not in d:
                    continue
                if normalise_old_cdr(d,c)!={p:a for p,a in n.items() if lo<=p[0]<=hi}:
                    mismatches.append(dict(sequence_key=k,reason='cached_'+c+'_not_reproduced'))
        if not any(m['sequence_key']==k for m in mismatches):
            raw[k]=dd
    attempted=set(cached['attempted'])|set(new['numbered'])
    atomic_json((output or root)/'numbering-verification.json',dict(status='passed' if not mismatches else 'needs_review',
        newly_numbered=len(new['numbered']),anarci_version=new['version'],mismatches=mismatches))
    return raw,attempted,mismatches


def mmsearch(folder,queries,targets,executable,threads,*,cdr=False):
    folder.mkdir(exist_ok=True)
    if not queries or not targets:
        return {}
    write_fasta(folder/'q.fasta',queries);write_fasta(folder/'t.fasta',targets)
    cmd=[str(executable),'easy-search',str(folder/'q.fasta'),str(folder/'t.fasta'),
         str(folder/'hits.tsv'),str(folder/'tmp'),'--min-seq-id','0.7','-c','0.8','--cov-mode','0',
         '--alignment-mode','3','--max-seqs','10000','-s','7.5','--threads',str(threads),
         '--format-output','query,target,fident,qcov,tcov']
    with (folder/'log.txt').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    hits=defaultdict(set)
    for line in (folder/'hits.tsv').read_text().splitlines():
        q,t,ident,qc,tc=line.split('\t')
        if float(ident)>=.7 and min(float(qc),float(tc))>=.8:
            hits[q].add(t)
    return dict(hits)


def run(args):
    require_compute(threads=args.workers,allow_comp1400=True)
    started=time.monotonic();root=args.inventory
    out=args.output
    if out.exists():raise FileExistsError('Choose a new screen output directory')
    out.mkdir(parents=True)
    manifest=read(root/'inventory-manifest.json')
    for name,pin in manifest['input_pins'].items():
        if digest(pin['path'])!=pin['sha256']:raise ValueError('Frozen input changed: '+name)
    policy=SourcePolicy.load(root/'source-policy.json')
    if policy.sha256!=manifest['policy_sha256']:raise ValueError('Policy fingerprint changed')
    units=pq.read_table(root/'units.parquet').to_pylist()
    members=pq.read_table(root/'sequence-membership.parquet').to_pylist()
    sequences=fasta(root/'sequences.fasta')
    raw,attempted,mismatches=merge_numbering(root,out)
    variable_keys=set(read(root/'numbering-cached.json')['official_variable_keys'])
    domains={}; by_sequence=defaultdict(list)
    for k,dd in raw.items():
        if k not in sequences:continue
        for i,d in enumerate(dd):
            cd=domain_cdrs(d)
            if not any(105<=p[0]<=117 for p in cd):continue
            dk=k+'d'+str(i);role='H' if d['chain_type']=='H' else 'L'
            domains[dk]=dict(sequence_key=k,role=role,domain=cd)
            by_sequence[k].append(dk)
    refseq=read(manifest['input_pins']['reference_sequences']['path'])
    scopes={s:{skey(v):v for v in refseq[s]} for s in ['validation','heldout']}
    for u in units:
        if u['source']!='sabdab' and u['original_role']=='validation':
            scopes['validation'].update({k:sequences[k] for k in u['sequence_keys']})
    with Path(manifest['input_pins']['sabdab_csv']['path']).open() as f:
        sab=list(csv.DictReader(f))
    sab_by_id={r['INSTANCE']:r for r in sab}
    official=defaultdict(set);official_by_h3=defaultdict(set);official_domains={};unit_official=defaultdict(list)
    all_test_components={r['ab_ag_cluster'] for r in sab if r['ab_ag_split']=='test'}
    scopes['sabdab_test']={};test_annotation_keys=set();protected_apo=0
    antigen_metadata=pq.read_table(root/'sabdab-antigen-targets.parquet').to_pylist()
    for target in antigen_metadata:
        if target['original_role']=='test':
            k=target['sequence_key'];scopes['sabdab_test'][k]=sequences[k]
    for r in sab:
        if r['ab_ag_split']=='test':
            protected_apo+=not bool(r['agchains'])
        for role in ['H','L']:
            seq=r['V'+role+'_numerable_seq']
            if not seq:continue
            if r['ab_ag_split']=='test':
                for col in [role+'seq',role+'seq_expected','V'+role+'_numerable_seq']:
                    if r[col]:
                        k=skey(r[col]);scopes['sabdab_test'][k]=r[col];test_annotation_keys.add(k)
            if r['type']=='VNAR':continue
            numbered=[(int(p),str(i).strip(),a) for a,(p,i) in json.loads(r['V'+role+'_numbering_list'])]
            cd=domain(numbered)
            dk='o'+hashlib.sha256((role+repr(tuple(cd.items()))).encode()).hexdigest()[:24]
            official_domains[dk]=dict(sequence_key=skey(seq),role=role,domain=cd)
            unit_official['sabdab:'+r['INSTANCE']].append(dk)
            official[(role,published_cdr_string(cd))].add(r['ab_ag_cluster'])
            official[('variable',seq)].add(r['ab_ag_cluster'])
            if role=='H':official_by_h3[published_cdr_string(cd,(105,117))].add(r['ab_ag_cluster'])
            if r['ab_ag_split']=='test':
                domains[dk]=official_domains[dk]
    unit_domains={}
    for u in units:
        if u['source']=='sabdab':
            unit_domains[u['unit_id']]=unit_official[u['unit_id']]
            for k in unit_domains[u['unit_id']]:domains[k]=official_domains[k]
        else:
            unit_domains[u['unit_id']]=sorted({d for k in u['sequence_keys'] for d in by_sequence[k]})
    scope_domains={s:{d for k in ss for d in by_sequence[k]} for s,ss in scopes.items()}
    scope_domains['sabdab_test']|={d for r in sab if r['ab_ag_split']=='test' for d in unit_official['sabdab:'+r['INSTANCE']]}
    hits=defaultdict(dict);nearest=defaultdict(dict)
    for role in ['H','L']:
        qkeys=sorted(k for k,d in domains.items() if d['role']==role)
        qdomains=[domains[k]['domain'] for k in qkeys]
        for scope,rset in scope_domains.items():
            rkeys=sorted(k for k in rset if domains[k]['role']==role)
            if not rkeys:continue
            score=compare(qdomains,[domains[k]['domain'] for k in rkeys])
            for i,q in enumerate(qkeys):
                indexes=np.flatnonzero((score['cdr3'][i]>=.9-1e-7)|(score['cdrs'][i]>=.8-1e-7))
                if len(indexes):hits[scope][q]={domains[rkeys[j]]['sequence_key'] for j in indexes}
                nearest[scope][q]=dict(cdr3=float(score['cdr3'][i].max()),cdrs=float(score['cdrs'][i].max()))
            print('CDR',role,scope,'queries',len(qkeys),'references',len(rkeys),flush=True)
    executable=Path(manifest['runtime'])/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
    for role in ['H','L']:
        # Exact reproduction of the older sorted-position concat convention.
        def concat(d):
            return ''.join(a for lo,hi in CDRS.values() for (p,_),a in sorted(d.items()) if lo<=p<=hi)
        q={k:concat(d['domain']) for k,d in domains.items() if d['role']==role}
        t={k:concat(domains[k]['domain']) for k in scope_domains['heldout'] if domains[k]['role']==role}
        mh=mmsearch(out/('heldout-cdr-'+role),q,t,executable,args.workers,cdr=True)
        for qk,tt in mh.items():
            hits['heldout'].setdefault(qk,set()).update(domains[k]['sequence_key'] for k in tt)
    generic={}
    for scope,ss in scopes.items():
        # Framework similarity is descriptive, not a newly invented Ig gate.
        targets={k:v for k,v in ss.items() if not by_sequence[k] and not (scope=='sabdab_test' and k in test_annotation_keys)}
        generic[scope]=mmsearch(out/('generic-'+scope),sequences,targets,executable,args.workers)
    print('CDR and generic screens complete',flush=True)
    # Link existing-pool Ig domains to the official CDR component graph. The
    # SAbDab units themselves retain their authoritative published components.
    native_domain_keys={d for u in units if u['source']!='sabdab' for d in unit_domains[u['unit_id']]}
    native_domain_keys|=scope_domains['validation']|scope_domains['heldout']
    qh3=sorted({published_cdr_string(domains[k]['domain'],(105,117)) for k in native_domain_keys if domains[k]['role']=='H'})
    refs_h3=sorted(official_by_h3)
    with ProcessPoolExecutor(args.workers) as pool:
        h3links=dict(pool.map(official_h3_links,((q,refs_h3) for q in qh3),chunksize=16))
    domain_components={}
    for dk in native_domain_keys:
        d=domains[dk];cd=d['domain'];components=set(official[(d['role'],published_cdr_string(cd))])
        if d['role']=='H':
            for h in h3links.get(published_cdr_string(cd,(105,117)),[]):components.update(official_by_h3[h])
        seq=''.join(cd.values())
        components.update(official[('variable',seq)])
        domain_components[dk]=components
    validation_components=set().union(*(domain_components.get(k,set()) for k in scope_domains['validation']))
    heldout_components=set().union(*(domain_components.get(k,set()) for k in scope_domains['heldout']))
    prior={r['instance']:r for r in csv.DictReader(open(manifest['input_pins']['prior_overlap']['path']),delimiter='\t')}
    context_cases=read(manifest['input_pins']['prior_context']['path'])['entries']
    context_flags={r['id'] for r in context_cases if r['author_chain_matched_sabdab_instances'] and
                   not any(q['current_antigen_chain_in_sabdab'] for q in r['author_chain_matched_sabdab_instances'])}
    source_groups={s:set().union(*(set(u['source_groups']) for u in units if u['source']=='pinder' and u['original_role']==s)) for s in ['train','validation']}
    test_interface_groups=set().union(*(set(json.loads(r['pinder_clusters'])) for r in prior.values() if r['split']=='test'))
    validation_interface_groups=set().union(*(set(u['source_groups']) for u in units if u['source']=='pinder' and u['original_role']=='validation'))
    bad_numbering={m['sequence_key'] for m in mismatches}
    evidence=[];counts=defaultdict(Counter)
    for u in units:
        uid=u['unit_id'];qds=unit_domains[uid];keys=set(u['sequence_keys'])
        components={u['sabdab_component']} if u['source']=='sabdab' else set().union(*(domain_components.get(d,set()) for d in qds))
        if u['source']=='sabdab':
            p=prior[u['source_native_id']];groups=set(json.loads(p['pinder_clusters']));mapped=bool(groups)
        else:
            groups=set(u['source_groups']) if u['source']=='pinder' else set();mapped=u['source']=='pinder'
        row=dict(unit_id=uid,source=u['source'],source_native_id=u['source_native_id'],original_role=u['original_role'],
                 published_split=u['sabdab_split'],published_component=u['sabdab_component'],
                 protein_partner=u['protein_partner'],official_components=sorted(components),
                 interface_mapped=mapped,interface_groups=sorted(groups),
                 test_interface_cluster_hit=bool(groups&test_interface_groups),
                 validation_interface_cluster_hit=bool(groups&validation_interface_groups),
                 numbering_recheck_failed=bool(keys&bad_numbering),
                 numbered_H_domains=sum(domains[d]['role']=='H' for d in qds),
                 numbered_L_domains=sum(domains[d]['role']=='L' for d in qds))
        reasons=[]
        for scope,ss in scopes.items():
            exact=sorted(keys&ss.keys())
            heavy=sorted({k for d in qds if domains[d]['role']=='H' for k in hits[scope].get(d,set())})
            light=sorted({k for d in qds if domains[d]['role']=='L' for k in hits[scope].get(d,set())})
            generic_keys=sorted(k for k in keys if k in generic[scope])
            comp=(components&all_test_components if scope=='sabdab_test' else
                  components&(validation_components if scope=='validation' else heldout_components))
            row[scope+'_exact_hit']=bool(exact);row[scope+'_heavy_CDR_hit']=bool(heavy)
            row[scope+'_light_CDR_hit']=bool(light);row[scope+'_generic_nonIg_hit']=bool(generic_keys)
            row[scope+'_component_hit']=bool(comp)
            row[scope+'_matched_components']=sorted(comp)
            if u['original_role']=='train':
                for hit,name in [(exact,'exact'),(heavy,'heavy_CDR'),(generic_keys,'nonIg_70_80'),(comp,'published_component')]:
                    if hit:reasons.append(scope+':'+name)
        if u['original_role']=='train':
            if row['test_interface_cluster_hit']:reasons.append('sabdab_test:interface_cluster')
            if row['validation_interface_cluster_hit']:reasons.append('validation:interface_cluster')
            if not u['protein_partner']:reasons.append('nonprotein_antigen_only')
        review=[]
        if u['source']=='pinder' and u['source_native_id'] in context_flags:
            review.append('unresolved_antigen_context_from_prior_audit')
            if u['original_role']=='train':reasons.append('quarantine:unresolved_PINDER_antigen_context')
        if u['source']=='sabdab' and u['source_annotation']=='VNAR':review.append('VNAR_numbering_unsupported')
        if u['source']=='sabdab' and u['source_annotation']=='SD-L':review.append('standalone_light_legacy_gate_requires_policy')
        if row['numbering_recheck_failed']:review.append('cached_CDR_not_reproduced')
        if not mapped:review.append('structural_interface_unassessed')
        review.append('full_antigen_global_homology_pending')
        action='exclude_training' if reasons else ('review_before_acceptance' if u['original_role']=='train' else 'retain_original_role_pending_review')
        if u['original_role']=='validation' and (row['sabdab_test_exact_hit'] or row['sabdab_test_heavy_CDR_hit'] or row['sabdab_test_component_hit'] or row['test_interface_cluster_hit']):
            action='retire_from_v7_validation_test_exposure_review'
        elif u['original_role']=='validation' and u['source']=='pinder' and u['source_native_id'] in context_flags:
            action='quarantine_validation_antigen_context_review'
        row.update(exclusion_reasons=sorted(reasons),review_flags=sorted(review),proposed_action=action,
                   source_priority=policy.rank(u['source']),prepared_identity_established=False)
        evidence.append(row)
        counts[u['source']+'/'+u['original_role']][action]+=1
    pq.write_table(pa.Table.from_pylist(evidence),out/'unit-exposure.parquet',compression='zstd')
    with gzip.open(out/'sequence-CDR-evidence.jsonl.gz','wt') as f:
        for dk,d in domains.items():
            f.write(json.dumps(dict(domain_id=dk,sequence_key=d['sequence_key'],role=d['role'],
                 nearest={s:nearest[s].get(dk) for s in scopes},
                 matched_chains={s:sorted(hits[s].get(dk,set())) for s in scopes}))+'\n')
    atomic_json(out/'component-seeds.json',dict(validation=sorted(validation_components),heldout=sorted(heldout_components),
                                              official_test=sorted(all_test_components)))
    summary=dict(status='sequence_exposure_complete_pending_antigen_and_structure_not_a_release',
                 counts={k:dict(v) for k,v in counts.items()},units=len(units),numbering_mismatches=len(mismatches),
                 domains=len(domains),protected_apo_test_annotations_without_new_labels=protected_apo,
                 policy=policy.to_dict(),policy_sha256=policy.sha256,inventory_manifest_sha256=digest(root/'inventory-manifest.json'),
                 code_sha256=digest(__file__),slurm_job_id=os.environ['SLURM_JOB_ID'],
                 mmseqs=dict(path=str(executable),sha256=digest(executable)),
                 wall_s=time.monotonic()-started,
                 limitations=['Proposed removals are not activated pool memberships.',
                              'Full global antigen homology is pending and generic 70/80 is not its replacement.',
                              'Unmapped interfaces are unassessed, not dissimilar.',
                              'No prepared identity, cache reuse, teacher or feature build is authorised by this inventory.'])
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32)
    run(p.parse_args())
