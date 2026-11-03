"""Read-only projection and saved-pilot audit of a small 0.5 A ANM candidate pool."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from pkabench.runtime import require_compute,atomic_json,digest
from augmentation_surface_filter import farthest


def main(args):
    require_compute(threads=1);args.output.mkdir(parents=True,exist_ok=True)
    units=json.loads((args.pilot/'native-large/unit-summary.json').read_text())
    original=0.;reduced=0.;removed=0.
    for unit in units:
        original+=unit['weight']*unit['production_s']
        report=json.loads((args.pilot/f"native-large/units/{unit['index']:03d}/report.json").read_text())
        saving=sum(m['total_s']+m.get('graph_s',0) for m in report.get('metrics',[]) if m['candidate']>args.candidates)
        saving+=sum(f.get('elapsed_s',0) for f in report.get('failures',[]) if isinstance(f.get('candidate'),int) and f['candidate']>args.candidates)
        if saving>unit['production_s']:raise ValueError('timing components exceed original total')
        removed+=unit['weight']*saving;reduced+=unit['weight']*(unit['production_s']-saving)
    with (args.full/'candidate-filter.csv').open() as f:decisions=list(csv.DictReader(f))
    accepted={(r['parent'],int(r['candidate'])):r['accepted']=='True' for r in decisions}
    rows=[]
    for file in sorted((args.exhaustive/'parents').glob('*/result.json')):
        result=json.loads(file.read_text())
        if not result['evaluation']:continue
        candidates=[r['candidate'] for r in result['candidates'] if r['geometry_valid'] and r['candidate']<=args.candidates]
        with np.load(file.parent/'candidate-ca.npz') as coords:
            selected=farthest(candidates,coords)
        passed=sum(accepted[result['parent'],c] for c in selected)
        rows.append(dict(parent=result['parent'],dataset=result['dataset'],selected=selected,
            selected_pass_strict_surface_audit=passed,all_three_pass_strict_audit=len(selected)==3 and passed==3))
    eta={str(c):dict(original_24_candidate_ideal_h=original/c/3600,candidate_ideal_h=reduced/c/3600,
        candidate_planning_h_with_30pct_margin=reduced/c/3600*1.3) for c in (240,288,380)}
    summary=dict(candidate_budget=args.candidates,parents=len(rows),yield_three=sum(len(r['selected'])==3 for r in rows),
        selected_candidates=sum(len(r['selected']) for r in rows),
        selected_pass_strict_surface_audit=sum(r['selected_pass_strict_surface_audit'] for r in rows),
        parents_all_three_pass_strict_surface_audit=sum(r['all_three_pass_strict_audit'] for r in rows),
        eta=eta,results=rows,
        method=f'Five ANM modes, {args.candidates} directions at 0.5 A; existing native transport, cleanup and geometry gates; select three by aligned global RMSD',
        limitations=['Counterfactual timing from measured 500-parent components, not a new end-to-end run',
            'Retains all fixed ANM/preparation overhead and subtracts only measured omitted candidate work',
            'Ideal CPU scaling is not a memory-qualified full-production throughput guarantee',
            '12 enriched pilot parents; strict surface audit used only for diagnosis, never selection',
            'Cheap selection cannot guarantee strict local surface chemistry preservation; queues and transfers excluded'])
    atomic_json(args.output/'summary.json',summary)
    atomic_json(args.output/'provenance.json',dict(code_sha256=digest(__file__),
        unit_summary_sha256=digest(args.pilot/'native-large/unit-summary.json'),audit_decisions_sha256=digest(args.full/'candidate-filter.csv')))
    text=['Cheap coordinate-only augmentation baseline','==========================================','',
        f"Three-view yield: {summary['yield_three']}/{len(rows)}.",
        f"Selected views passing strict surface audit: {summary['selected_pass_strict_surface_audit']}/{summary['selected_candidates']}.",'']
    text += [f"* {c} CPUs: ideal {e['candidate_ideal_h']:.2f} h; planning {e['candidate_planning_h_with_30pct_margin']:.2f} h." for c,e in eta.items()]
    text += ['']+['* '+v for v in summary['limitations']]
    (args.output/'report.rst').write_text('\n'.join(text)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('pilot','exhaustive','full','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidates',type=int,default=8,choices=(4,8))
    main(p.parse_args())
