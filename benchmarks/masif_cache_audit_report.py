"""Consolidate geometry, grouping and preparation findings into a redo plan."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
from pkabench.runtime import atomic_json,require_compute,digest


def report(root,old):
    root=Path(root);old=Path(old);summary=json.loads((root/'full/summary.json').read_text())
    previous=json.loads((old/'publication-hold.json').read_text());known=set(previous['additional_mesh_recovery_ids'])
    pending=json.loads((old/'remaining-failures-20261118.json').read_text());regroup={};hard=[];uncertain=[];no_add=[];terminal=[];superseded=[]
    reasons=Counter();topology=Counter()
    with (root/'full/records.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);rid=row['id']
            if row.get('superseded_by_grouped_abag'):superseded.append(rid)
            if row['status']!='audited':uncertain.append(dict(id=rid,status=row['status'],error=row.get('error')))
            if row.get('hard_rebuild_reasons'):
                hard.append(dict(id=rid,cid=row['cid'],reasons=row['hard_rebuild_reasons'],
                    already_replaced_by_grouped_cache=row.get('superseded_by_grouped_abag',False)))
                reasons.update(row['hard_rebuild_reasons'])
            topology.update(row.get('review_reasons',[]))
            if row.get('intact_antibody_regrouping'):regroup[row['cid']]=row['intact_antibody_regrouping']
            policy=row.get('preparation_compatibility','unverified')
            if policy=='no_terminal_addition_required' and not row.get('hard_rebuild_reasons') and row['status']=='audited':no_add.append(rid)
            if policy!='no_terminal_addition_required':terminal.append(dict(id=rid,policy=policy,
                added_terminal_oxygens=len(row.get('new_policy_added_OXT_canonical_ids',[])),error=row.get('terminal_error')))
    if not known<={r['id'] for r in hard}:raise ValueError('complete audit did not recover every confirmed malformed mesh')
    findings=dict(status='audit_report_complete',legacy_records=summary['records'],
        existing_unresolved_records=len(pending),geometry_rebuild_records=len(hard),
        newly_identified_geometry_rebuild_records=len({r['id'] for r in hard}-known),
        geometry_reasons=dict(reasons),source_or_read_uncertainties=len(uncertain),
        already_replaced_chain_records=len(superseded),
        geometry_rebuilds_not_already_replaced=sum(not r['already_replaced_by_grouped_cache'] for r in hard),
        additional_intact_antibody_assemblies=len(regroup),
        additional_intact_antibody_components=sum(len(groups) for groups in regroup.values()),
        valid_geometry_without_terminal_additions=len(no_add),
        preparation_review_counts=dict(Counter(r['policy'] for r in terminal)),topology_review_counts=dict(topology),
        source_audit_results_sha256=digest(root/'full/records.jsonl'),
        recommendation='Rebuild failed geometry and intact-antibody groupings; keep preparation upgrades separate from corruption. Do not promote the old cache into a homogeneous current-policy release without resolving preparation equivalence.',
        limitations=summary['limitations'])
    comparison_path=root/'preparation-comparison.json'
    if comparison_path.exists():
        c=json.loads(comparison_path.read_text())['cases'];groups={}
        for key in ['old_to_current','old_to_old_repeat']:
            rows=[r[key] for r in c if key in r]
            groups[key]=dict(cases=len(rows),median_case_matched_fraction=float(np.median([r['matched_fraction'] for r in rows])),
                median_case_surface_p95_A=float(np.median([r['surface_distance_p95_A'] for r in rows])),
                median_case_descriptor_cosine=float(np.median([r['all']['median_cosine'] for r in rows if r['all']['median_cosine'] is not None])),
                median_case_descriptor_p05_cosine=float(np.median([r['all']['p05_cosine'] for r in rows if r['all']['p05_cosine'] is not None])),
                median_case_far_terminal_cosine=float(np.median([r['far_from_changed_termini_16A']['median_cosine'] for r in rows if r['far_from_changed_termini_16A']['median_cosine'] is not None])))
        findings['paired_preparation_comparison']=groups
    else:findings['paired_preparation_comparison_status']='not_completed; geometry results remain valid but preparation equivalence is unresolved'
    atomic_json(root/'redo-plan.json',dict(summary=findings,geometry_rebuild=hard,existing_failed=pending,
        additional_antibody_regrouping=[dict(cid=cid,chain_index_groups=groups) for cid,groups in sorted(regroup.items())],
        source_or_read_uncertainties=uncertain,already_replaced_chain_ids=sorted(superseded),
        no_terminal_addition_candidates=no_add,preparation_review=terminal))
    text='pKPDB cache audit and redo plan\n==============================\n\n'
    text+='Read-only audit of all {:,} legacy successful training-chain records. Original payloads and membership tables are unchanged.\n\n'.format(summary['records'])
    text+='* Existing unresolved input/pipeline records: {:,}.\n'.format(len(pending))
    text+='* Cached records with geometry/mapping/integrity defects: {:,}, including {:,} beyond the original 17.\n'.format(len(hard),findings['newly_identified_geometry_rebuild_records'])
    text+='* Geometry rebuild targets not already replaced by the grouped Ab/Ag cache: {:,}.\n'.format(findings['geometry_rebuilds_not_already_replaced'])
    text+='* Additional antibody-only assemblies needing intact grouping: {:,} ({:,} contact components).\n'.format(len(regroup),findings['additional_intact_antibody_components'])
    text+='* Valid cached geometry with no new terminal OXT addition required: {:,}. These remain reuse candidates, not a blanket certification of the current preparation profile.\n'.format(len(no_add))
    text+='* Source/read uncertainties requiring review: {:,}.\n\n'.format(len(uncertain))
    text+='Preparation changes are not counted as numerical corruption. Paired old/current outputs and unchanged-preparation repeats are recorded in preparation-comparison.json. Statistics are per-case summaries, not a claim of identical descriptors everywhere.\n\n'
    text+='Full raw-MSMS coverage is not regenerated for all records. Convex-hull checks detect exterior surface loss but cannot rule out every missing concave patch. Boundary/nonmanifold edges are review flags; current cleanup can also create boundaries. The bulk cache remains held for a publication decision.\n\n'
    text+='redo-plan.json contains exact geometry IDs, existing failed IDs, additional antibody chain groups, preparation-review IDs and already-replaced IDs. Full per-record metrics are in full/records.jsonl.\n'
    (root/'report.rst').write_text(text);print(json.dumps(findings),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True);report(a.root,a.old)
