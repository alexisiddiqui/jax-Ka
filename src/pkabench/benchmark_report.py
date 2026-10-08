"""Consolidated full-set report from verified score artifacts; no rescoring."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from collections import Counter
from .runtime import require_compute,atomic_json,digest

NAMES={'propka':'PROPKA','pkai':'pKAI','pkai_plus':'pKAI+','null':'Zero shift'}


def plots(campaign):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    campaign=Path(campaign); out=campaign/'report_figures'; out.mkdir(exist_ok=True)
    scores=json.loads((campaign/'scores_set1.json').read_text())
    methods=list(NAMES)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,split in zip(axes,('train','val','test')):
        for i,m in enumerate(methods):
            r=next(r for r in scores if r['method']==m and r['split']==split and r['scope']=='all_method_common' and r['subset']=='interface')
            ci=r['mae_ci95']; ax.plot(i,r['mae'],'o',color='navy')
            if ci: ax.vlines(i,*ci,color='navy')
        ax.set(xticks=range(len(methods)),xticklabels=[NAMES[m] for m in methods],ylabel='Group-macro ΔpKa MAE',title=split)
        ax.tick_params(axis='x',rotation=25); ax.set_ylim(bottom=0)
    fig.suptitle('Full frozen structural benchmark: common interface sites\nAgreement with current PypKa; 95% sequence-group bootstrap intervals')
    for ext in ('png','pdf'): fig.savefig(out/f'interface_mae.{ext}',dpi=180)
    plt.close(fig)
    with (campaign/'coverage.csv').open() as f: coverage=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained'); x=np.arange(len(methods))
    for j,split in enumerate(('train','val','test')):
        values=[float(next(r for r in coverage if r['method']==m and r['split']==split and r['subset']=='interface')['method_coverage']) for m in methods]
        ax.bar(x+(j-1)*.25,values,.25,label=split)
    ax.set(xticks=x,xticklabels=[NAMES[m] for m in methods],ylim=(0,1.05),ylabel='Method-valid / mask-eligible interface sites',title='Coverage is reported separately from accuracy'); ax.legend()
    for ext in ('png','pdf'): fig.savefig(out/f'interface_coverage.{ext}',dpi=180)
    plt.close(fig)


def run(campaign,output):
    require_compute(); campaign=Path(campaign); output=Path(output)
    verify=json.loads((campaign/'verification.json').read_text()); assert verify['passed']
    assert digest(campaign/'predictions.parquet')==verify['prediction_sha256']
    assert digest(campaign/'site_masks.parquet')==verify['mask_sha256']
    summary=json.loads((campaign/'scoring_report.json').read_text()); completion=json.loads((campaign/'completion.json').read_text())
    scores=json.loads((campaign/'scores_set1.json').read_text()); coverage=json.loads((campaign/'coverage_gate.json').read_text())
    with (campaign/'coverage.csv').open() as f: cov=list(csv.DictReader(f))
    def selected(method,split,subset='interface'):
        return next(r for r in scores if r['method']==method and r['split']==split and r['scope']=='all_method_common' and r['subset']==subset)
    def ci(r):
        v=r['mae_ci95']; return f"{r['mae']:.3f}"+(f" ({v[0]:.3f}–{v[1]:.3f})" if v else ' (CI unavailable)')
    lines=['# Full structural benchmark report — initial non-JAX round','',
        'Date: 2026-10-05. All 1,452 frozen complexes have result receipts for PypKa, PROPKA, pKAI, pKAI+ and the zero-shift baseline. The report is complete with explicit method failures; JAX-Ka was deferred at the user’s request. No model training has been performed for these scores.','',
        '## Main outcome','',
        f"**{coverage['teacher_valid_interface_pairs']:,}/1,452 complexes have usable teacher interface labels ({100*coverage['teacher_interface_coverage_fraction']:.1f}%).** There are **{summary['teacher_paired_sites']:,} usable teacher paired sites**, of {summary['expected_eligible_sites']:,} mask-eligible sites, and **{summary['common_paired_sites']:,} sites shared by all retained comparison methods and the teacher**. Counts span the frozen training, validation and test splits; only training-eligible sites can enter optimization.",'',
        'The dataset and split remain 778 training / 151 validation / 523 test complexes. The methods are compared to newly generated PypKa 2.10.0 labels, not experimental ground truth. Historical pKPDB configuration equivalence remains unresolved. pKAI and pKAI+ share the PypKa training lineage, so their agreement must not be interpreted as independent physical validation.','',
        '## Computation and failures','', '| Method | Complete complex calculations | Calculations with failures |','|---|---:|---:|']
    for m in ['pypka',*NAMES]:
        counts=completion['process_status_counts']; lines.append(f"| {NAMES.get(m,'PypKa reference')} | {counts.get(m+':complete',0):,} | {counts.get(m+':failed',0):,} |")
    errors=Counter(e['type'] for r in completion['failed_jobs'] for e in r['errors'].values())
    lines+=['',f"State-level exception counts: {dict(errors)}. A complex can fail in multiple states; these counts are not counts of distinct complexes. Failed, unreported and out-of-range midpoints are excluded explicitly. Partial state successes remain usable only when the matching bound/free pair is valid.",'',
        'The 20 missing receipts left after worker retirement were recovered before collection. Final collection checked all 7,260 complex-method receipts and independently recomputed 2,952 group macro metrics. It completed in 10 min 44 s with a 32 GB allocation. Original failures and retry provenance are retained.','',
        '## Interface coverage','',
        '| Split | Mask-eligible sites | Teacher-valid sites | All-method common sites | Common-support groups |','|---|---:|---:|---:|---:|']
    for split in ('train','val','test'):
        c=next(r for r in cov if r['method']=='propka' and r['split']==split and r['subset']=='interface'); s=selected('propka',split)
        lines.append(f"| {split} | {int(c['eligible_sites']):,} | {int(c['teacher_sites']):,} | {int(c['all_method_common_sites']):,} | {s['groups']} |")
    lines+=['','Interface membership uses residue ΔSASA >10 Å². Coverage is after the frozen, split-aware uncertainty masks. Method-specific and teacher-matched coverage remain available in coverage.csv; common support excludes every site missing any retained method.','',
        '## Bound-minus-free pKa agreement','',
        'The target is pKa(AB) − pKa(free partner). Metrics are computed per complex, averaged within sequence groups and then equally across groups. Confidence intervals bootstrap sequence groups, not individual sites. The tables below use all-method common interface support. No held-out result was used to tune these pretrained methods.','',
        '| Method | Train MAE (95% CI) | Validation MAE (95% CI) | Test MAE (95% CI) | Test macro RMSE | Test macro skill |','|---|---:|---:|---:|---:|---:|']
    for m in NAMES:
        t=selected(m,'test'); lines.append(f"| {NAMES[m]} | {ci(selected(m,'train'))} | {ci(selected(m,'val'))} | {ci(t)} | {t['rmse']:.3f} | {t['skill']:.3f} |")
    lines+=['',
        'Skill is 1 − MSE(predicted shift)/mean(reference shift²), evaluated per complex before group averaging. Small reference-shift denominators can produce large negative skill; a lower macro MAE need not imply positive macro skill. RMSE is also a macro average, not pooled site RMSE. The zero-shift baseline has skill zero by construction. These definitions were not changed after seeing results.','',
        '## Antibody versus general protein interfaces','',
        '| Test role | Method | Sites | Groups | Macro MAE (95% CI) |','|---|---|---:|---:|---:|']
    for role in ('antibody_antigen','general'):
        for m in NAMES:
            rr=[r for r in scores if r['method']==m and r['split']=='test' and r['scope']=='all_method_common' and r['subset']=='interface:role:'+role]
            if rr:
                r=rr[0]; lines.append(f"| {role} | {NAMES[m]} | {r['sites']} | {r['groups']} | {ci(r)} |")
    lines+=['',
        'Antibody/antigen grouping follows the frozen antigen-family split with separate antibody-novelty annotations. Roles and groups are not rebalanced to improve this table. Missing role rows mean no scored common support under that exact stored role label.','',
        '## Linkage limits and figures','',
        f"Complete accepted charge coverage permits linkage for {summary['linkage_statuses'].get('ok',0)} of 7,260 complex-method combinations. {summary['linkage_statuses'].get('masked_uncertain_charge_coverage',0)} are blocked by masks and {summary['linkage_statuses'].get('incomplete_charge_coverage',0)} by incomplete charge coverage. Partial masked charge sums are not reported as full binding free-energy curves.",'',
        f"![Interface MAE]({campaign}/report_figures/interface_mae.png)",'',f"![Interface coverage]({campaign}/report_figures/interface_coverage.png)",'',
        '## Experiment 02 handoff and next work','',
        'The fixed 500-complex training pilot currently contains 458 complexes with usable teacher interface labels, 24,646 teacher paired sites and 23,818 sites shared by the retained methods. Those are coverage results, not permission to replace failed pilot complexes with validation/test examples. The handoff adds the 151 frozen validation complexes and excludes test inputs and labels.','',
        'Paired pKAI features must reproduce the frozen model predictions before fitting. CatBoost residual targets are PypKa ΔpKa minus PROPKA ΔpKa; subtracting PypKa from itself would leak the target. Density, formal-charge and donor/acceptor-proximity features are explicitly geometric proxies. Scaling and weighting are fitted on training rows only.','',
        'A separate recovery run retries 61 timed-out states across 26 pilot complexes with a three-hour per-state budget. Completed states and the original label version remain unchanged. The two preparation-failed states of one complex are handled separately. A recovery overlay must be verified and versioned before it can replace handoff labels.','',
        'Remaining benchmark work includes experimental Set 2 curation and independent overlap checks. JAX-Ka recovery is deferred. The full benchmark has not established experimental accuracy, and experiment 02 training outcomes are not part of this report.','',
        '## Artifacts and provenance','',
        f"- [Verification]({campaign}/verification.json)",f"- [Complete scores and intervals]({campaign}/scores_set1.csv)",f"- [Coverage]({campaign}/coverage.csv)",f"- [Representability by shell]({campaign}/representability.csv)",f"- [Failure receipts]({campaign}/completion.json)",f"- [Manifest]({campaign}/manifest.json)",'',
        f"Manifest SHA-256: `{digest(campaign/'manifest.json')}`.",
        'The inherited scoring_report.json interpretation sentence still says “engineering smoke”; that is a reused metadata template. This report’s scope is the complete frozen 1,452-complex structural benchmark. Numeric artifacts were not altered to correct that wording.']
    output.write_text('\n'.join(lines)+'\n')
    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/radial-plots/bin/python'),'-m','pkabench.benchmark_report','plots',str(campaign)],check=True)
    atomic_json(campaign/'consolidated_report.json',{'markdown':str(output),'markdown_sha256':digest(output),'code_sha256':digest(Path(__file__)),
        'figures_sha256':{p.name:digest(p) for p in (campaign/'report_figures').iterdir()}})
    print(str(output),flush=True)

if __name__=='__main__': plots(Path(sys.argv[2]))
