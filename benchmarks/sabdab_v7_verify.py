"""Independent coverage, CDR and global-alignment verification for the audit."""
import argparse
import ast
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import random

import pyarrow.parquet as pq
from Bio.Align import PairwiseAligner

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.source_policy import SourcePolicy
from sabdab_pool_overlap import CDRS,compare,domain
from sabdab_v7_exposure import read,fasta
from sabdab_v7_screen import merge_numbering,domain_cdrs


def run(args):
    require_compute(threads=args.workers,allow_comp1400=True)
    manifest=read(args.inventory/'inventory-manifest.json')
    for name,pin in manifest['input_pins'].items():
        assert digest(pin['path'])==pin['sha256'],name
    units=pq.read_table(args.inventory/'units.parquet').to_pylist()
    rows=pq.read_table(args.report/'strict-proposed-decisions.parquet').to_pylist()
    alt=pq.read_table(args.report/'preserve-pkpdb-counterfactual.parquet').to_pylist()
    assert len(units)==len(rows)==len(alt)==96922
    assert len({r['unit_id'] for r in units})==len(units)
    original={r['unit_id']:(r['source'],r['original_role']) for r in units}
    assert all(original[r['unit_id']]==(r['source'],r['original_role']) for r in rows+alt)
    assert sum(u['source']!='sabdab' for u in units)==86044
    assert all(r['prepared_key'] is None and not r['release_active'] for r in rows+alt)
    assert not any('full_antigen_global_homology_pending' in r['review_flags'] for r in rows)
    for r in rows:
        if r['original_role']=='train' and r['exclusion_reasons']:
            assert r['proposed_action']=='exclude_training'
    strict_by_id={r['unit_id']:r for r in rows}
    preserved=0
    for r in alt:
        if r['proposed_action']=='candidate_retain_pkpdb_exclude_affected_test_components':
            preserved+=1;assert r['source']=='pkpdb' and r['original_role']=='train'
            assert strict_by_id[r['unit_id']]['proposed_action']=='exclude_training'
            assert all(x.startswith('sabdab_test:') for x in r['exclusion_reasons'])
    policy=SourcePolicy.load(args.inventory/'source-policy.json')
    assert read(args.report/'summary.json')['source_policy_sha256']==policy.sha256
    # Compile the actual immutable legacy CDR function, not another copy of the
    # matrix implementation being tested.
    tree=ast.parse(args.legacy_cdr_source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cdr_identity')
    ns={'IMGT':CDRS};exec(compile(ast.Module(body=[fn],type_ignores=[]),'legacy_cdr_identity','exec'),ns)
    verification_folder=args.report/'verification-work';verification_folder.mkdir()
    raw,attempted,mismatches=merge_numbering(args.inventory,verification_folder)
    assert not mismatches
    seqs=fasta(args.inventory/'sequences.fasta')
    existing={k for u in units if u['source']!='sabdab' for k in u['sequence_keys']}
    assert existing<=attempted
    domains=[domain_cdrs(d) for k,dd in raw.items() if k in seqs for d in dd]
    rng=random.Random(20261122);qq=rng.sample(domains,60);rr=rng.sample(domains,60)
    scores=compare(qq,rr)
    def loops(d):return {c:{p:a for p,a in d.items() if lo<=p[0]<=hi} for c,(lo,hi) in CDRS.items()}
    for i,q in enumerate(qq):
        for j,r in enumerate(rr):
            a,b=ns['cdr_identity'](loops(q),loops(r))
            assert abs(a-float(scores['cdr3'][i,j]))<1e-6
            assert abs(b-float(scores['cdrs'][i,j]))<1e-6
    data=read(args.antigen/'inputs.json');pilot=read(args.antigen/'pilot.json');alignment_checks=0;exact_checks=0
    aligner=PairwiseAligner(mode='global',match_score=1,mismatch_score=0,open_gap_score=-2,extend_gap_score=-1)
    for scope in ['test','external','validation']:
        receipt=read(args.antigen/scope/'summary.json')
        assert receipt['pairs']==pilot[scope]['candidate_pairs']
        assert receipt['DP_cells']==pilot[scope]['candidate_DP_cells']
        assert receipt['queries']==pilot[scope]['queries']
        hits={}
        with gzip.open(args.antigen/scope/'hits.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);assert r['sequence_key'] not in hits;hits[r['sequence_key']]=r['hits']
        assert len(hits)==receipt['hit_queries']
        targets=data[{'test':'test_targets','external':'external_targets','validation':'all_sabdab_targets'}[scope]]
        keys=data['validation_queries' if scope=='validation' else 'source_queries']
        for k in set(keys)&{r['sequence_key'] for r in targets}:
            assert any(h['target']==k and h['identity']==1 for h in hits[k]);exact_checks+=1
        examples=[(k,h) for k,hh in hits.items() for h in hh]
        for q,h in rng.sample(examples,min(128,len(examples))):
            a,b=data['sequences'][q],data['sequences'][h['target']]
            al=aligner.align(a,b)[0];left,right=al[0],al[1]
            identity=sum(x==y and x!='-' for x,y in zip(left,right))/max(len(a),len(b))
            assert abs(identity-h['identity'])<1e-12
            assert identity>={'PROTEIN':.4,'PEPTIDE':.8}[h['antigen_type']]-1e-12
            alignment_checks+=1
    with Path(manifest['input_pins']['sabdab_csv']['path']).open() as f:sab=list(csv.DictReader(f))
    comps={s:{r['ab_ag_cluster'] for r in sab if r['ab_ag_split']==s} for s in ['train','test']}
    assert not comps['train']&comps['test']
    receipt=dict(status='passed',release_ready=False,original_input_hashes_unchanged=True,
                 inventory_units=len(units),existing_pool_entries=86044,existing_sequence_numbering_coverage='complete_attempted',
                 legacy_CDR_pair_checks=3600,numbering_reproduction_mismatches=0,
                 exhaustive_global_pair_counts_verified=True,global_alignment_traceback_checks=alignment_checks,
                 exact_antigen_self_matches_verified=exact_checks,original_official_component_overlap=0,
                 preserved_pkpdb_counterfactual_units=preserved,
                 no_active_pool_or_label_changes=True,slurm_job_id=os.environ['SLURM_JOB_ID'],
                 summary_sha256=digest(args.report/'summary.json'),code_sha256=digest(__file__))
    atomic_json(args.report/'verification.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','screen','antigen','report','legacy-cdr-source']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);run(p.parse_args())
