"""Recheck development roles at actual native scope; publish proposals only."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json,digest,require_compute,config_hash
from pkabench.role_engine import Unit,Evidence,PUBLISHED,assign
from pkabench.source_registry import SourceRegistry
from pkabench.split_authority import SplitAuthority
from pkabench.group_adjudication import make_groups,resolve_groups
from pkatrain.augmentation_store import AugmentationStore
from sabdab_v7_roles import collect,cdr_components,numbering_assessed,general_holdout_guard
from sabdab_v7_antigen import initialise,compare_chunk


def read(path):return json.loads(Path(path).read_text())


def write_rows(path,rows):
    columns=sorted({k for row in rows for k in row})
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in columns} for r in rows]),path,compression='zstd')


def verify_sabdab_antigen(row,meta):
    expected={}
    for author,typ,seq,indices in zip(row['agchains'].split('/'),row['agtypes'].split('/'),
            row['agresolvedseqs'].split('/'),row['ag_index_list'].split('/')):
        if typ not in ('PROTEIN','PEPTIDE'):continue
        expected[author]=(seq,None if indices=='__FULL_CHAIN__' else json.loads(indices))
    chains={c['chain']:c['observed_sequence'] for c in meta['chains']};actual=defaultdict(list)
    for m in meta['residue_map']:
        if m['side']=='B':
            actual[m['source_author_chain']].append((chains[m['chain']][m['number']-1],
                [m['source_author_number'],m['source_author_insertion']]))
    if set(actual)!=set(expected):raise ValueError('Prepared antigen author-chain selection differs from published metadata')
    for author,(sequence,indices) in expected.items():
        observed=actual[author]
        if ''.join(r[0] for r in observed)!=sequence:raise ValueError('Prepared antigen sequence differs from published retained sequence')
        if indices is not None:
            normalised=[(str(a),[int(n),str(i).strip()]) for a,(n,i) in indices]
            if observed!=normalised:raise ValueError('Prepared antigen retained indices differ from published selection')
    return True


def prepare(args):
    require_compute(threads=2,allow_comp1400=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);data=read(args.audit/'inputs.json')
    native=pq.read_table(args.native/'units.parquet').to_pylist()
    sabdab=pq.read_table(args.sabdab/'units.parquet').to_pylist()
    official={r['INSTANCE']:r for r in data['official']}
    units={r['unit_id']:r for r in pq.read_table(args.audit/'units.parquet').to_pylist()}
    original=pq.read_table(args.audit/'scope-membership.parquet').to_pylist()
    original_roles={(m['unit_id'],m['chain_id']):m['annotation_role'] for m in original}
    members=[];sequences=dict(data['sequences']);native_keys=set();native_checked=set()
    for row in native:
        if row['status']!='checked':continue
        uid=row['unit_id'];native_checked.add(uid)
        for c in row['chains']:
            k=c['sequence_key'];seq=c['sequence']
            if k in sequences and sequences[k]!=seq:raise ValueError('Sequence hash collision')
            sequences[k]=seq;native_keys.add(k)
            members.append(dict(unit_id=uid,chain_id=c['chain_id'],sequence_key=k,
                annotation_role=original_roles.get((uid,c['chain_id']),'protein_context_role_unresolved'),
                sequence_kind='native_observed_coordinates'))
    native_sab={r['unit_id']:r for r in sabdab if r['status']!='withheld_preparation'}
    scope_checks=[]
    with AugmentationStore(args.sabdab/'prepared') as store:
        for uid,row in sorted(native_sab.items()):
            record=store.raw(uid);meta=record['metadata']['structure'];source=official[row['source_native_id']]
            try:
                if meta['antibody_crop_metadata']['source_row_sha256']!=config_hash(source):raise ValueError('Published crop source row changed')
                verify_sabdab_antigen(source,meta);status='published_retained_scope_verified';error=None
            except Exception as exc:status='scope_unassessed';error=repr(exc)
            scope_checks.append(dict(unit_id=uid,status=status,error=error))
            for c in row['chains']:
                k=c['sequence_key'];seq=c['sequence']
                if k in sequences and sequences[k]!=seq:raise ValueError('Sequence hash collision')
                sequences[k]=seq
                members.append(dict(unit_id=uid,chain_id=c['chain'],sequence_key=k,
                    annotation_role='antibody' if c['side']=='A' else 'antigen',sequence_kind='native_observed_coordinates'))
    # Unprepared published references still define their authoritative split;
    # they never masquerade as successfully prepared training records.
    for m in original:
        if units[m['unit_id']]['source']=='sabdab' and m['unit_id'] not in native_sab:
            members.append(dict(m,sequence_kind='published_reference_no_prepared_input'))
    inventory=Path(data['inventory']);cached=read(inventory/'numbering-cached.json');raw=cached['numbered']
    fresh=read(inventory/'numbering-new.json');raw.update(fresh['numbered'])
    attempted=set(cached['attempted'])|set(fresh['numbered'])
    extra=Path('/home/coulson/oc/lina4225/_runtime/sabdab-v7-adjudication-v1')
    for path in [extra/'native-scope-v3/numbering-new.json',extra/'native-scope-v4/numbering-new.json']:
        if path.exists():
            more=read(path)['numbered'];raw.update(more);attempted.update(more)
    todo={k:sequences[k] for k in sorted(native_keys) if k not in attempted or any('numbering' not in d for d in raw.get(k,[]))}
    atomic_json(args.output/'numbering-input.json',todo)
    atomic_json(args.output/'numbering-cache.json',dict(numbered=raw,attempted=sorted(attempted)))
    atomic_json(args.output/'sequences.json',sequences)
    write_rows(args.output/'members.parquet',members);write_rows(args.output/'sabdab-scope-checks.parquet',scope_checks)
    summary=dict(status='actual_native_scope_inputs_complete',native_checked=len(native_checked),
        observed_native_unique_sequences=len(native_keys),new_numbering_queries=len(todo),
        sabdab_scope_checks=dict(Counter(r['status'] for r in scope_checks)),
        inputs_sha256=digest(args.audit/'inputs.json'),native_sha256=digest(args.native/'units.parquet'),
        sabdab_sha256=digest(args.sabdab/'units.parquet'),members_sha256=digest(args.output/'members.parquet'))
    atomic_json(args.output/'prepare-summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


def screen(args):
    require_compute(threads=args.workers+1,allow_comp1400=True)
    root=args.output;data=read(args.audit/'inputs.json');sequences=read(root/'sequences.json')
    members=pq.read_table(root/'members.parquet').to_pylist();units={r['unit_id']:r for r in pq.read_table(args.audit/'units.parquet').to_pylist()}
    old_cache=read(root/'numbering-cache.json');numbered=old_cache['numbered'];new=read(root/'numbering-new.json')['numbered'];numbered.update(new)
    attempted=set(old_cache['attempted'])|set(new)
    by_unit=defaultdict(list)
    for m in members:by_unit[m['unit_id']].append(m)
    native={r['unit_id']:r for r in pq.read_table(args.native/'units.parquet').to_pylist()}
    native_keys={m['sequence_key'] for m in members if units[m['unit_id']]['source']!='sabdab'}
    cd=cdr_components(numbered,native_keys,data['official'],args.workers)
    queries={m['sequence_key'] for m in members if units[m['unit_id']]['source']!='sabdab'
        and not m['annotation_role'].startswith('antibody')
        and (not numbered.get(m['sequence_key']) or m['annotation_role'] in ('antigen','antigen_context'))}
    # The completed historical union screen is a superset of the resolved
    # targets. Reuse negative queries as well as hits, replacing component
    # lists with those of the resolved reference scope required by CF2.
    old_hits,receipts=collect(args.audit,'union',digest(args.audit/'inputs.json'))
    if sum(r['queries'] for r in receipts)!=len(data['query_keys']):raise ValueError('Incomplete cached query assessment')
    targets={(t['sequence_key'],t['antigen_type']):t for t in data['targets']}
    union={(t['sequence_key'],t['antigen_type']) for t in data['union_targets']}
    if not set(targets)<=union:raise ValueError('Cached reference scope is not a superset')
    cached_keys=queries & set(data['query_keys']);hits={}
    for key in cached_keys:
        good=[]
        for hit in old_hits.get(key,[]):
            target=targets.get((hit['target'],hit['antigen_type']))
            if target:good.append(dict(hit,components=target['components']))
        if good:hits[key]=good
    library=os.environ.get('SABDAB_EXACT_LCS_LIBRARY');bound_pin=None
    if library:
        validation=read(args.output.parent/'lcs-validation.json')
        if validation['status']!='passed_exact_hit_equality' or digest(library)!=validation['library_sha256']:
            raise ValueError('Exact rejection bound has not passed pinned validation')
        bound_pin=dict(library_sha256=digest(library),validation_sha256=digest(args.output.parent/'lcs-validation.json'))
    new_queries=sorted(queries-cached_keys);start=time.monotonic();processed=0;pairs=0;rejected=0
    with gzip.open(root/'new-antigen-hits.jsonl.gz','wt') as stream:
        with ProcessPoolExecutor(args.workers,initializer=initialise,initargs=(sequences,data['targets'])) as pool:
            chunks=[new_queries[i:i+16] for i in range(0,len(new_queries),16)]
            for result in pool.map(compare_chunk,chunks):
                pairs+=result['pairs'];processed+=result['queries'];rejected+=result.get('lcs_rejected',0)
                for h in result['hits']:
                    hits[h['sequence_key']]=h['hits'];stream.write(json.dumps(h)+'\n')
                if processed%320<16:
                    progress=dict(queries=processed,total=len(new_queries),pairs=pairs,lcs_rejected=rejected,wall_s=time.monotonic()-start)
                    atomic_json(root/'alignment-progress.json',progress);print(json.dumps(progress),flush=True)
    specs=[];evidence=[]
    for uid,u in sorted(units.items()):
        own=u['source']=='sabdab';checked=own or native.get(uid,{}).get('status')=='checked'
        specs.append(Unit(uid,u['source'],u['native_split'],
            'published_sabdab_scope' if own else 'verified_model_observed_coordinates',
            u['native_component'],u['supervised'],u['declared_general']))
        if own:
            for axis in ['antibody_CDR','antigen']:
                evidence.append(Evidence(uid,PUBLISHED,axis,'published_authoritative_component','hit',(u['native_component'],)))
            continue
        ab=set();ag=set();complete=checked
        for m in by_unit[uid]:
            k=m['sequence_key'];ab.update(cd.get(k,[]))
            if k in queries:ag.update(c for h in hits.get(k,[]) for c in h['components'])
            complete &= numbering_assessed(k,m['annotation_role'],numbered,attempted)
        for axis,components in [('antibody_CDR',ab),('antigen',ag)]:
            evidence.append(Evidence(uid,PUBLISHED,axis,'whole_native_observed_scope',
                ('hit' if components else 'no_hit') if checked else 'unassessed',tuple(sorted(components)) if checked else ()))
        if checked and not complete:evidence.append(Evidence(uid,PUBLISHED,'antibody_CDR','incomplete_native_numbering','unassessed'))
    baseline=assign(specs,evidence,data['component_splits'],SplitAuthority.load(args.audit/'authority.json'),SourceRegistry.load(args.audit/'registry.json'))
    groups=make_groups(units,members,baseline,exempt_mapped=True)
    grouped,decisions=resolve_groups(baseline,groups,preferred_removal_source='pinder')
    executable=Path(read(Path(data['inventory'])/'inventory-manifest.json')['runtime'])/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
    final,generic=general_holdout_guard(grouped,units,members,numbered,sequences,executable,root/'generic',args.workers)
    for a in final:
        if a.source=='sabdab' and units[a.unit_id]['supervised'] and a.operational_role!={'train':'train','test':'validation'}[units[a.unit_id]['native_split']]:
            raise ValueError('Observed-scope recheck changed an authoritative SAbDab role')
    write_rows(root/'proposed-assignments.parquet',[dict(asdict(a),release_active=False) for a in final])
    write_rows(root/'component-assignments.parquet',[asdict(a) for a in baseline]);write_rows(root/'published-evidence.parquet',[asdict(e) for e in evidence])
    atomic_json(root/'group-decisions.json',decisions)
    write_rows(root/'generic-evidence.parquet',generic)
    old={r['unit_id']:r for r in pq.read_table(args.assignments).to_pylist()}
    changes=[dict(unit_id=a.unit_id,source=a.source,previous_role=old[a.unit_id]['operational_role'],
        observed_scope_role=a.operational_role,reason_codes=list(a.reason_codes)) for a in final if a.operational_role!=old[a.unit_id]['operational_role']]
    write_rows(root/'role-changes.parquet',changes)
    summary=dict(status='whole_pool_observed_scope_role_recheck_complete',counts={s:dict(Counter(a.operational_role for a in final if a.source==s)) for s in ['pkpdb','pinder','sabdab']},
        role_changes=len(changes),native_antigen_queries=len(queries),cached_antigen_queries=len(cached_keys),
        new_antigen_queries=len(new_queries),new_alignment_pairs=pairs,new_alignment_wall_s=time.monotonic()-start,
        exact_lcs_rejected_pairs=rejected,exact_lcs_bound=bound_pin,
        grouping_policy='CF2 mapped component authority, PINDER-first conflict removal, actual observed exact-sequence groups',
        generic_guard=dict(Counter(r['action'] for r in generic)),published_sabdab_roles_preserved=True,release_active=False,
        assignment_sha256=digest(root/'proposed-assignments.parquet'),members_sha256=digest(root/'members.parquet'))
    atomic_json(root/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='generic_guard'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','screen'])
    for name in ['audit','native','sabdab','assignments','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    {'prepare':prepare,'screen':screen}[args.action](args)
