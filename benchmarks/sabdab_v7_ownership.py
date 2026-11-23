"""Resolve native copy sampling separately from exact teacher-cache aliases."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore
from sabdab_v7_teacher_context import normalise_pdb

STORE=None


def read(path):return json.loads(Path(path).read_text())


def initialise(path):
    global STORE
    STORE=AugmentationStore(path)


@lru_cache(maxsize=48)
def context(uid):
    frame=STORE.raw(uid);data=frame['metadata']
    mapping=data['request']['mappings']['AB'];keys={(c,int(n),i):(c,int(n),i) for _,_,c,n,i in mapping['residues']}
    atoms=normalise_pdb(frame['arrays']['teacher_pdb_AB'].tobytes().decode(),mapping,keys)
    masks={(s['chain'],s['resnum'],s['icode'],s['group']):(bool(s['train_mask']),bool(s['eval_mask']),bool(s['interface'])) for s in data['sites']}
    return data,atoms,masks


def compare(pair):
    left=pair['unit_id'];right=pair['candidate_parent'];a,x,am=context(left);b,y,bm=context(right)
    chain={r['left']:r['right'] for r in pair['chain_map']}
    def key(k):return (chain[k[0]],*k[1:])
    xa={key(k):v for k,v in x.items()}
    real=lambda data:{(r['chain'],r['number'],r['insertion'],r['group']) for r in data['true_termini']}
    terminal={key(k) for k in real(a)}==real(b)
    masks={key(k):v for k,v in am.items()}==bm
    side={s['chain']:s['partner'] for s in a['sites']};other={s['chain']:s['partner'] for s in b['sites']}
    biological=all(other.get(chain[c])==s for c,s in side.items())
    chemistry=xa.keys()==y.keys();target_keys=True;max_diff=0.
    for model in ['pkai','pkai_plus']:
        for state in ['AB','A','B']:
            av={key(tuple(r[:4])):float(r[4]) for r in a['labels'][model][state]}
            bv={tuple(r[:4]):float(r[4]) for r in b['labels'][model][state]}
            target_keys &= av.keys()==bv.keys()
            if av.keys()==bv.keys() and av:max_diff=max(max_diff,max(abs(av[k]-bv[k]) for k in av))
    max_atom=None;rmsd=None
    if chemistry:
        order=sorted(y);xx=np.asarray([xa[k] for k in order]);yy=np.asarray([y[k] for k in order])
        xx-=xx.mean(0);yy-=yy.mean(0);u,_,vh=np.linalg.svd(xx.T@yy);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vh)
        distances=np.linalg.norm(xx@(u@fix@vh)-yy,axis=1);max_atom=float(distances.max());rmsd=float(np.sqrt(np.mean(distances**2)))
    # Near-copy membership is a sampling decision, not a teacher reuse claim.
    # It uses the previously verified all-native-heavy-atom geometry, same
    # reference component/role, chemistry, biological sides, termini and masks.
    native_same=(a['result']['native_component']==b['result']['native_component'] and
        a['result']['operational_role']==b['result']['operational_role'] and left.split('_')[1]==right.split('_')[1])
    redundant=bool(native_same and biological and terminal and chemistry and masks and target_keys
        and pair['max_atom_error_A']<=.005 and pair['rmsd_A']<=.002)
    exact=bool(redundant and max_atom is not None and max_atom<1e-8 and max_diff==0.)
    return dict(unit_id=left,peer=right,same_published_context=native_same,same_biological_sides=biological,
        same_true_termini=terminal,same_teacher_chemistry=chemistry,same_query_masks=masks,
        same_target_keys=target_keys,max_teacher_target_difference=max_diff,teacher_atom_max_error_A=max_atom,
        teacher_atom_rmsd_A=rmsd,native_max_atom_error_A=pair['max_atom_error_A'],native_rmsd_A=pair['rmsd_A'],
        redundant_native_copy=redundant,exact_teacher_equivalence=exact)


def select_representatives(ids,proofs):
    """Complete-link groups: no tolerance chaining or cross-role merging."""
    edge={frozenset([r['unit_id'],r['peer']]):r for r in proofs if r['redundant_native_copy']}
    groups=[]
    for uid in sorted(ids):
        compatible=[g for g in groups if all(frozenset([uid,member]) in edge for member in g)]
        if compatible:compatible[0].append(uid)
        else:groups.append([uid])
    result={}
    for members in groups:
        owner=members[0]
        for uid in members:
            proof=edge.get(frozenset([uid,owner]))
            action='include_native' if uid==owner else ('alias_exact_context' if proof['exact_teacher_equivalence'] else 'withhold_redundant_native_copy')
            result[uid]=dict(unit_id=uid,representative_id=owner,membership_action=action,
                teacher_reuse_allowed=action=='alias_exact_context',group_members=members)
    return result


def run(args):
    require_compute(threads=args.workers+1)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    plan_path=args.audit/'build-manifest-v2/units.parquet';plan={r['unit_id']:r for r in pq.read_table(plan_path).to_pylist()
        if r['source']=='sabdab' and r['eligible_for_build']}
    label_summary=read(args.labels/'summary.json');verification=read(args.labels/'independent-verification.json')
    if verification['status']!='passed' or verification['summary_sha256']!=digest(args.labels/'summary.json'):
        raise ValueError('Unverified native teachers')
    candidates=[r for r in pq.read_table(args.audit/'reuse/context-comparisons.parquet').to_pylist()
        if r['status']=='rigid_heavy_atom_identity' and r['unit_id'] in plan and r['candidate_parent'] in plan]
    with ProcessPoolExecutor(args.workers,initializer=initialise,initargs=(args.labels/'labels',)) as pool:
        proofs=list(pool.map(compare,candidates,chunksize=8))
    touched={r['unit_id'] for r in proofs}|{r['peer'] for r in proofs}
    selected=select_representatives(touched,proofs)
    rows=[]
    for uid,r in sorted(plan.items()):
        decision=selected.get(uid,dict(unit_id=uid,representative_id=uid,membership_action='include_native',
            teacher_reuse_allowed=False,group_members=[uid]))
        rows.append(dict(decision,operational_role=r['operational_role'],n_res=r['n_res'],ownership_ready=True,
            previous_review_flag=not r['ownership_ready'],release_active=False))
    pq.write_table(pa.Table.from_pylist(rows),args.output/'membership.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(proofs),args.output/'pair-checks.parquet',compression='zstd')
    policy=dict(id='v7-native-copy-representatives-v1',scope='same published SAbDab component/role and same parent PDB',
        native_heavy_atom_rmsd_max_A=.002,native_heavy_atom_max_error_A=.005,
        required=['same entire observed assembly atom inventory','same biological-side partition','same true termini',
            'same qualified query masks and teacher target identities'],
        grouping='complete-link; deterministic lexicographic source-native representative',
        exact_alias='teacher atom sets congruent below 1e-8 A AND both model target values exactly equal',
        numerical_copy='native representative selected; excluded copies keep their own labels; no cache equivalence asserted',
        label_values_used_to_select_representative=False,new_inverse_group_weights=False,
        source_precedence=['pkpdb','sabdab','pinder'],cross_source_ownership_unchanged=True)
    atomic_json(args.output/'policy.json',policy)
    includes=[r for r in rows if r['membership_action']=='include_native']
    report=dict(status='native_copy_membership_review_complete',candidates=len(rows),pair_checks=len(proofs),
        previous_review_flags=sum(r['previous_review_flag'] for r in rows),unresolved=0,
        actions=dict(Counter(r['membership_action'] for r in rows)),
        selected_by_role=dict(Counter(r['operational_role'] for r in includes)),
        target_difference_max=max((r['max_teacher_target_difference'] for r in proofs),default=0),
        distinct_context_pairs=sum(not r['redundant_native_copy'] for r in proofs),
        policy_sha256=digest(args.output/'policy.json'),membership_sha256=digest(args.output/'membership.parquet'),
        pair_checks_sha256=digest(args.output/'pair-checks.parquet'),
        input_pins={str(p):digest(p) for p in [plan_path,args.labels/'summary.json',args.labels/'independent-verification.json',
            args.labels/'labels/manifest.json',args.audit/'reuse/context-comparisons.parquet']},release_active=False)
    atomic_json(args.output/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['audit','labels','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);run(p.parse_args())
