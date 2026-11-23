"""Published-parameter global antigen homology, with a bounded timing pilot."""
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
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from Bio.Align import PairwiseAligner

from pkabench.runtime import atomic_json, digest, require_compute
from sabdab_v7_exposure import fasta, read, skey


THRESHOLDS={'PROTEIN':.4,'PEPTIDE':.8}
GLOBAL_TARGETS=None
GLOBAL_SEQUENCES=None
GLOBAL_LCS_BOUND=None


def aligner():
    return PairwiseAligner(mode='global',match_score=1,mismatch_score=0,
                          open_gap_score=-2,extend_gap_score=-1)


def identity(q,t,a):
    alignment=a.align(q,t)[0]
    return alignment.counts().identities/max(len(q),len(t))


def initialise(sequences,targets):
    global GLOBAL_SEQUENCES,GLOBAL_TARGETS,GLOBAL_LCS_BOUND
    GLOBAL_SEQUENCES=sequences;GLOBAL_TARGETS=targets
    library=os.environ.get('SABDAB_EXACT_LCS_LIBRARY')
    if library:
        from pkabench.lcs_bound import LCSBound
        GLOBAL_LCS_BOUND=LCSBound(library,sequences,targets)
    else:GLOBAL_LCS_BOUND=None


def compare_chunk(keys):
    a=aligner();out=[];started=time.monotonic();pairs=0;cells=0;rejected=0
    for qkey in keys:
        q=GLOBAL_SEQUENCES[qkey];hits=[]
        bounds=GLOBAL_LCS_BOUND.bounds(q) if GLOBAL_LCS_BOUND is not None else None
        for index,t in enumerate(GLOBAL_TARGETS):
            target=GLOBAL_SEQUENCES[t['sequence_key']];threshold=THRESHOLDS[t['antigen_type']]
            if min(len(q),len(target))/max(len(q),len(target))<threshold-1e-12:
                continue
            if bounds is not None and bounds[index]<threshold*max(len(q),len(target))-1e-9:
                rejected+=1;continue
            pairs+=1;cells+=len(q)*len(target)
            ident=identity(q,target,a)
            if ident>=threshold-1e-12:
                hits.append(dict(target=t['sequence_key'],antigen_type=t['antigen_type'],
                                 identity=ident,components=t['components'],reference_ids=t['reference_ids']))
        if hits:out.append(dict(sequence_key=qkey,hits=hits))
    return dict(queries=len(keys),pairs=pairs,cells=cells,wall_s=time.monotonic()-started,hits=out,lcs_rejected=rejected)


def prepare(args):
    require_compute(threads=args.workers,allow_comp1400=True)
    root=args.inventory;out=args.output
    if out.exists():raise FileExistsError('Choose a new antigen audit root')
    out.mkdir(parents=True)
    manifest=read(root/'inventory-manifest.json')
    units=pq.read_table(root/'units.parquet').to_pylist();by_unit={r['unit_id']:r for r in units}
    membership=pq.read_table(root/'sequence-membership.parquet').to_pylist()
    sequences=fasta(root/'sequences.fasta')
    raw=read(root/'numbering-cached.json')['numbered'];new=read(root/'numbering-new.json')['numbered'];raw.update(new)
    numbered_ig={k for k,dd in raw.items() if dd}
    source_queries=set();validation_queries=set();member_evidence=[]
    for m in membership:
        u=by_unit[m['unit_id']];k=m['sequence_key']
        if u['source']=='sabdab':
            antigen=k in set(u['antigen_keys'])
        elif m['annotation_role'].startswith('antibody'):
            antigen=False
        else:
            antigen=(k not in numbered_ig or m['annotation_role'] in {'antigen','antigen_context'})
        if not antigen:continue
        if u['original_role']=='train':source_queries.add(k)
        if u['source']!='sabdab' and u['original_role']=='validation' and (
                u['stratum']=='Ab/Ag' or u['source_annotation'] in {'accepted','antibody_only'}):
            validation_queries.add(k)
        member_evidence.append(dict(unit_id=u['unit_id'],sequence_key=k,annotation_role=m['annotation_role']))
    targets=pq.read_table(root/'sabdab-antigen-targets.parquet').to_pylist()
    grouped=defaultdict(lambda:dict(components=set(),reference_ids=set()))
    for t in targets:
        z=grouped[(t['sequence_key'],t['antigen_type'])]
        z['components'].add(t['component']);z['reference_ids'].add(t['unit_id'])
    all_targets=[dict(sequence_key=k,antigen_type=typ,components=sorted(v['components']),
                      reference_ids=sorted(v['reference_ids'])) for (k,typ),v in grouped.items()]
    test_groups={u['sabdab_component'] for u in units if u['source']=='sabdab' and u['original_role']=='test'}
    test_targets=[t for t in all_targets if set(t['components'])&test_groups]
    # Recover actual antigen partners of existing external Ab/Ag references;
    # do not relabel every unrelated benchmark protein as an antigen.
    universe=Path(manifest['runtime'])/'universe/combined-split-v1/index.json'
    heldout_path=Path(manifest['runtime'])/'audits/ab-cdr-gap-v1/heldout.fasta'
    heldout=fasta(heldout_path);held_by_id=defaultdict(set)
    for name,sequence in heldout.items():held_by_id[name.split('|')[0]].add(sequence)
    external=defaultdict(set)
    for c in read(universe)['candidates']:
        actual=held_by_id.get(c['complex_id'])
        if not actual:continue
        chains={x['chain']:x['sequence'] for x in c['chains']}
        ab=[chains[x] for x in c['partner_A_chains'] if x in chains]
        if not any(skey(s) in numbered_ig for s in ab) and not c.get('sabdab'):
            continue
        typ='PEPTIDE' if (c.get('sabdab') or {}).get('antigen_type')=='PEPTIDE' else 'PROTEIN'
        for ch in c['partner_B_chains']:
            if ch not in chains or chains[ch] not in actual:continue
            sequence=chains[ch];k=skey(sequence);sequences[k]=sequence
            external[(k,typ)].add(c['complex_id'])
    external_targets=[dict(sequence_key=k,antigen_type=typ,components=[],reference_ids=sorted(v))
                      for (k,typ),v in external.items()]
    atomic_json(out/'inputs.json',dict(sequences=sequences,source_queries=sorted(source_queries),
                 validation_queries=sorted(validation_queries),test_targets=test_targets,
                 external_targets=external_targets,all_sabdab_targets=all_targets))
    pq.write_table(pa.Table.from_pylist(member_evidence),out/'antigen-sequence-membership.parquet',compression='zstd')
    receipt=dict(status='antigen_inputs_prepared',inventory_manifest_sha256=digest(root/'inventory-manifest.json'),
                 source_queries=len(source_queries),validation_queries=len(validation_queries),
                 test_targets=len(test_targets),external_targets=len(external_targets),all_sabdab_targets=len(all_targets),
                 universe_sha256=digest(universe),heldout_fasta_sha256=digest(heldout_path),
                 thresholds=THRESHOLDS,scoring=dict(mode='global',match=1,mismatch=0,gap_open=-2,gap_extend=-1),
                 query_type_policy='Existing polypeptide context chains are checked against both target types conservatively; no length-based retyping.',
                 sequence_policy='Union of published resolved and expected antigen sequences; stronger context audit than cropped-ML membership alone.',
                 no_teacher_or_features=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(out/'manifest.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


def scopes(data,scope):
    if scope=='test':return data['source_queries'],data['test_targets']
    if scope=='external':return data['source_queries'],data['external_targets']
    if scope=='validation':return data['validation_queries'],data['all_sabdab_targets']
    raise ValueError(scope)


def candidate_work(sequences,keys,targets):
    total=0;cells=0
    for typ,threshold in THRESHOLDS.items():
        lengths=np.array(sorted(len(sequences[t['sequence_key']]) for t in targets if t['antigen_type']==typ),dtype=np.int64)
        cumulative=np.r_[0,np.cumsum(lengths)]
        for k in keys:
            n=len(sequences[k]);lo=np.searchsorted(lengths,n*threshold-1e-9,'left');hi=np.searchsorted(lengths,n/threshold+1e-9,'right')
            total+=int(hi-lo);cells+=int((cumulative[hi]-cumulative[lo])*n)
    return dict(candidate_pairs=total,candidate_DP_cells=cells)


def pilot(args):
    require_compute(threads=args.workers,allow_comp1400=True)
    data=read(args.root/'inputs.json');sequences=data['sequences'];rng=random.Random(20261122)
    reports={}
    for scope in ['test','external','validation']:
        keys,targets=scopes(data,scope);work=candidate_work(sequences,keys,targets)
        # Stratify a small, bounded pair sample by query and target lengths.
        ordered_q=sorted(keys,key=lambda k:(len(sequences[k]),k));ordered_t=sorted(targets,key=lambda t:(len(sequences[t['sequence_key']]),t['sequence_key']))
        pairs=[]
        if ordered_q and ordered_t:
            for qi in np.linspace(0,len(ordered_q)-1,min(48,len(ordered_q)),dtype=int):
                for ti in np.linspace(0,len(ordered_t)-1,min(24,len(ordered_t)),dtype=int):
                    q=ordered_q[int(qi)];t=ordered_t[int(ti)];a,b=sequences[q],sequences[t['sequence_key']]
                    if min(len(a),len(b))/max(len(a),len(b))>=THRESHOLDS[t['antigen_type']]:pairs.append((a,b))
        rng.shuffle(pairs);pairs=pairs[:512];a=aligner();cells=0;start=time.monotonic()
        for q,t in pairs:identity(q,t,a);cells+=len(q)*len(t)
        wall=time.monotonic()-start
        cpu_s=work['candidate_DP_cells']*wall/max(cells,1)
        reports[scope]=dict(queries=len(keys),targets=len(targets),**work,pilot_pairs=len(pairs),
                            pilot_DP_cells=cells,pilot_wall_s=wall,estimated_serial_s=cpu_s,
                            projected_64_CPU_s=cpu_s/64,projected_96_CPU_s=cpu_s/96,
                            caveat='Projection from a bounded length-spread timing sample; excludes process scheduling and output overhead.')
    atomic_json(args.root/'pilot.json',reports);print(json.dumps(reports,indent=2),flush=True)


def run(args):
    require_compute(threads=args.workers,allow_comp1400=True)
    data=read(args.root/'inputs.json');keys,targets=scopes(data,args.scope)
    folder=args.root/args.scope
    if folder.exists():raise FileExistsError('Completed or partial scope exists: '+str(folder))
    folder.mkdir()
    chunk_size=max(1,min(64,(len(keys)+args.workers*8-1)//(args.workers*8)))
    chunks=[keys[i:i+chunk_size] for i in range(0,len(keys),chunk_size)];started=time.monotonic();pairs=cells=processed=hit_queries=0
    with gzip.open(folder/'hits.jsonl.gz','wt') as output:
        with ProcessPoolExecutor(args.workers,initializer=initialise,initargs=(data['sequences'],targets)) as pool:
            for i,r in enumerate(pool.map(compare_chunk,chunks)):
                pairs+=r['pairs'];cells+=r['cells'];processed+=r['queries'];hit_queries+=len(r['hits'])
                for hit in r['hits']:output.write(json.dumps(hit)+'\n')
                if i%20==0:print(json.dumps(dict(scope=args.scope,queries=processed,total=len(keys),wall_s=time.monotonic()-started)),flush=True)
    import Bio
    receipt=dict(status='global_antigen_alignment_complete',scope=args.scope,queries=processed,
                 hit_queries=hit_queries,pairs=pairs,DP_cells=cells,wall_s=time.monotonic()-started,
                 input_sha256=digest(args.root/'inputs.json'),code_sha256=digest(__file__),
                 biopython_version=Bio.__version__,thresholds=THRESHOLDS,
                 slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(folder/'summary.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--inventory',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--workers',type=int,default=4)
    for name in ['pilot','run']:
        q=sub.add_parser(name);q.add_argument('--root',type=Path,required=True);q.add_argument('--workers',type=int,default=4)
        if name=='run':q.add_argument('--scope',choices=['test','external','validation'],required=True)
    args=p.parse_args();{'prepare':prepare,'pilot':pilot,'run':run}[args.action](args)
