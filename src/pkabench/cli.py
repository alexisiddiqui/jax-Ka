"""All workload entrypoints reject execution on login nodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .runtime import require_compute, atomic_json


def main():
    require_compute()
    parser = argparse.ArgumentParser(prog="pkabench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser('production-nojax')
    p.add_argument('--scope',choices=['pilot','full'],required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p = sub.add_parser('production')
    p.add_argument('stage',choices=['init','gate','pool','status','collect'])
    p.add_argument('--campaign',type=Path,required=True)
    p = sub.add_parser('production-full-v2')
    p.add_argument('--campaigns',type=Path,required=True)
    p = sub.add_parser('production-jax-v2')
    p.add_argument('stage',choices=['init','validate','gate','pool','status','collect'])
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--complex-id')
    p = sub.add_parser('isolated-solver')
    p.add_argument('stage',choices=['init','run','teacher-report','collect'])
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--complex-id')
    p = sub.add_parser('numerical-followup')
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--complex-id',required=True)
    p.add_argument('--method',choices=['pypka','jaxka','jaxka-retry','review'],required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--count", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261003); p.add_argument("--completion-executable")
    p = sub.add_parser("validate"); p.add_argument("--tests", nargs="*", default=["tests/test_pkabench.py"])
    p = sub.add_parser("run"); p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--complex-id", required=True); p.add_argument("--method", required=True)
    p.add_argument("--timeout", type=int, default=1800)
    p = sub.add_parser("merge"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--methods", nargs="+", required=True)
    p = sub.add_parser("score"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--methods", nargs="+", required=True)
    p = sub.add_parser("gates"); p.add_argument("--campaign", type=Path, required=True)
    p = sub.add_parser("dispatch"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--diagnostic",action="store_true")
    p = sub.add_parser("discover")
    p = sub.add_parser("curation-pilot"); p.add_argument('stage',choices=['init','scan','collect'])
    p.add_argument('--campaign',type=Path,required=True); p.add_argument('--audit',type=Path)
    p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=16)
    p = sub.add_parser('curation-report'); p.add_argument('--campaign',type=Path,required=True)
    p = sub.add_parser('sensitivity'); p.add_argument('stage',choices=['init','init-altloc','collect'])
    p.add_argument('--campaign',type=Path); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('radial-error'); p.add_argument('stage',choices=['init','collect'])
    p.add_argument('--campaign',type=Path); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=3)
    p.add_argument('--completed-only',action='store_true')
    p = sub.add_parser('expanded-radial'); p.add_argument('stage',choices=['plan','prepare','assemble','dispatch','monitor','status','eta','plots','anchors','internal-anchors'])
    p.add_argument('--campaigns',type=Path,nargs='+'); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=30); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=30)
    p = sub.add_parser("download-metadata"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser('mask-policy'); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('anchor-tiers'); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('ligand-audit'); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('glycan-buffer-policy'); p.add_argument('stage',choices=['init','scan','collect','recount','verify']); p.add_argument('--source',type=Path); p.add_argument('--campaign',type=Path); p.add_argument('--campaigns',type=Path,nargs='+'); p.add_argument('--out',type=Path); p.add_argument('--smoke',action='store_true'); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=32)
    p = sub.add_parser('stripped-policy'); p.add_argument('stage',choices=['init','scan','collect']); p.add_argument('--source',type=Path); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=16)
    p = sub.add_parser('pool-shards'); p.add_argument('stage',choices=['index','collect']); p.add_argument('--out',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=16)
    p = sub.add_parser('glycan-pilot'); p.add_argument('stage',choices=['init','init-ccd','run','collect','plots','verify']); p.add_argument('--out',type=Path,required=True); p.add_argument('--source',type=Path); p.add_argument('--count',type=int,default=30); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=8)
    p = sub.add_parser('frozen-smoke'); p.add_argument('stage',choices=['init','gates','dispatch','run','collect','verify','retry-jax','diagnostics']); p.add_argument('--freeze',type=Path); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--complex-id')
    p = sub.add_parser('freeze-split'); p.add_argument('stage',choices=['run','verify']); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('buffer-mask-revision'); p.add_argument('stage',choices=['apply','recount']); p.add_argument('--source',type=Path); p.add_argument('--out',type=Path,required=True); p.add_argument('--campaigns',type=Path,nargs='+')
    p = sub.add_parser('buffer-pilot'); p.add_argument('stage',choices=['init','run','collect','plots','verify']); p.add_argument('--out',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=30)
    p = sub.add_parser('propka-components'); p.add_argument('stage',choices=['init','run','collect','plots']); p.add_argument('--out',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=8)
    p = sub.add_parser('component-pilot'); p.add_argument('stage',choices=['inventory','collect','gate']); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=8)
    p = sub.add_parser('ligand-review'); p.add_argument('stage',choices=['run','collect','sasa','grid']); p.add_argument('--audit',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=8)
    p = sub.add_parser('long-tails'); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('coverage-audit'); p.add_argument('--campaign',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--count',type=int,default=12)
    p = sub.add_parser('crystal-compare'); p.add_argument('stage',choices=['init','run','collect']); p.add_argument('--audit',type=Path); p.add_argument('--out',type=Path,required=True); p.add_argument('--index',type=int,default=0)
    p = sub.add_parser('candidate-pool'); p.add_argument('stage',choices=['init','download','index','sequence','report'])
    p.add_argument('--exclude-pool',type=Path)
    p.add_argument('--metadata',type=Path); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--campaign',type=Path)
    p.add_argument('--count',type=int,default=1000); p.add_argument('--antibody-count',type=int,default=250)
    p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=8)
    p = sub.add_parser("download-sample"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("download-file"); p.add_argument("--url", required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("dataset-audit")
    p.add_argument("stage", choices=["index", "install-mmseqs", "scan", "sequence", "summarize", "defects"])
    p.add_argument("--root", type=Path, required=True); p.add_argument("--manifest", type=Path); p.add_argument("--archive", type=Path); p.add_argument("--smoke", type=Path)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--shards", type=int, default=32)
    p = sub.add_parser("audit-prep"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("audit-sources"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("audit-teacher"); p.add_argument("--source-root", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("audit-findings"); p.add_argument("--campaign", type=Path, required=True); p.add_argument("--prep-audit", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("inspect"); p.add_argument("--campaign",type=Path,required=True)
    p = sub.add_parser("preflight"); p.add_argument("--campaign",type=Path,required=True); p.add_argument("--methods",nargs="+",default=["propka","pkai","pkai_plus","kaml"])
    p = sub.add_parser("report"); p.add_argument("--campaign",type=Path,required=True)
    p = sub.add_parser("deposit-evidence")
    p = sub.add_parser("verify-deposit"); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--source-url",required=True)
    p = sub.add_parser("finalize"); p.add_argument("--campaign",type=Path,required=True); p.add_argument("--methods",nargs="+",required=True)
    p = sub.add_parser('split-audit'); p.add_argument('--campaigns', type=Path, nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('split-diagnose'); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('antigen-split'); p.add_argument('--out', type=Path, required=True); p.add_argument('--verify',action='store_true')
    p = sub.add_parser('split-targets'); p.add_argument('stage',choices=['prepare','references','allocate']); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('split-review'); p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('split-leakage'); p.add_argument('--out',type=Path,required=True); p.add_argument('--scoped',action='store_true')
    p = sub.add_parser('reference-inventory-check'); p.add_argument('--out',type=Path,required=True); p.add_argument('--download-all',action='store_true'); p.add_argument('--reconcile',action='store_true'); p.add_argument('--check-delta',action='store_true')
    p = sub.add_parser('set2-screen'); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'split-audit':
        from .split_audit import run
        run(args.campaigns, args.out)
    elif args.command == 'set2-screen':
        from .set2_screen import run
        run(args.manifest,args.out)
    elif args.command == 'reference-inventory-check':
        from .reference_inventory_check import run,download_all,reconcile,check_delta
        (check_delta if args.check_delta else reconcile if args.reconcile else download_all if args.download_all else run)(args.out)
    elif args.command == 'split-leakage':
        from .split_leakage import run
        run(args.out,args.scoped)
    elif args.command == 'split-review':
        from .split_review import run
        run(args.out)
    elif args.command == 'split-targets':
        from .split_targets import prepare,references,allocate
        {'prepare':prepare,'references':references,'allocate':allocate}[args.stage](args.out)
    elif args.command == 'antigen-split':
        from .antigen_split import run,verify
        (verify if args.verify else run)(args.out)
    elif args.command == 'split-diagnose':
        from .split_diagnose import run
        run(args.out)
    elif args.command == 'crystal-compare':
        from .crystal_compare import initialise,run,collect
        if args.stage=='init': initialise(args.audit,args.out)
        elif args.stage=='run': run(args.out,args.index)
        else: collect(args.out)
    elif args.command == 'coverage-audit':
        from .coverage_audit import audit
        audit(args.campaign,args.out,args.count)
    elif args.command == 'long-tails':
        from .long_tails import start
        start(args.source,args.out)
    elif args.command == 'ligand-review':
        from .ligand_review import run,collect,sasa,grid
        if args.stage=='run': run(args.audit,args.out,args.shard,args.shards)
        elif args.stage=='sasa': sasa(args.audit,args.out)
        elif args.stage=='grid': grid(args.audit,args.out)
        else: collect(args.audit,args.out)
    elif args.command == 'component-pilot':
        from .component_pilot import inventory,collect,gate
        if args.stage=='inventory': inventory(args.campaign,args.out,args.shard,args.shards)
        elif args.stage=='gate': gate(args.out)
        else: collect(args.campaign,args.out,args.shards)
    elif args.command == 'glycan-pilot':
        from .glycan_pilot import initialise,initialise_ccd,run,collect,plots,verify
        if args.stage=='init': initialise(args.out,args.count)
        elif args.stage=='init-ccd': initialise_ccd(args.out,args.source)
        elif args.stage=='run': run(args.out,args.shard,args.shards)
        elif args.stage=='collect': collect(args.out)
        elif args.stage=='verify': verify(args.out)
        else: plots(args.out)
    elif args.command == 'production-nojax':
        from .production_nojax import run
        run(args.campaign,args.scope)
    elif args.command == 'production':
        from .production import initialise,gate,pool,status,collect
        {'init':initialise,'gate':gate,'pool':pool,'status':status,'collect':collect}[args.stage](args.campaign)
    elif args.command == 'production-full-v2':
        from .production_full_v2 import run
        run(args.campaigns)
    elif args.command == 'production-jax-v2':
        from . import production_jax_v2 as v2
        if args.stage == 'init': v2.initialise(args.campaign)
        elif args.stage == 'validate': v2.validate(args.campaign,args.complex_id)
        else: {'gate':v2.gate,'pool':v2.pool,'status':v2.status,'collect':v2.collect}[args.stage](args.campaign)
    elif args.command == 'isolated-solver':
        from .isolated_solver import initialise,run,teacher_report
        if args.stage=='init': initialise(args.campaign)
        elif args.stage=='run': run(args.campaign,args.complex_id)
        elif args.stage=='collect':
            from .solver_report import collect
            collect(args.campaign)
        else: teacher_report(args.campaign)
    elif args.command == 'numerical-followup':
        from .numerical_followup import run
        run(args.campaign,args.complex_id,args.method)
    elif args.command == 'frozen-smoke':
        from .frozen_smoke import initialise,preparation_gate,dispatch,run,collect,verify,retry_jax,diagnostics
        if args.stage=='init': initialise(args.freeze,args.campaign)
        elif args.stage=='gates': preparation_gate(args.campaign)
        elif args.stage=='dispatch': dispatch(args.campaign)
        elif args.stage=='run': run(args.campaign,args.complex_id)
        elif args.stage=='verify': verify(args.campaign)
        elif args.stage=='retry-jax': retry_jax(args.campaign,args.complex_id)
        elif args.stage=='diagnostics': diagnostics(args.campaign)
        else: collect(args.campaign)
    elif args.command == 'freeze-split':
        from .freeze_split import run,verify
        if args.stage=='run': run(args.out)
        else: verify(args.out)
    elif args.command == 'buffer-mask-revision':
        from .buffer_mask_revision import apply,recount
        if args.stage=='apply': apply(args.source,args.out)
        else: recount(args.campaigns,args.out)
    elif args.command == 'buffer-pilot':
        from .buffer_pilot import initialise,run,collect,plots
        if args.stage=='verify':
            from .buffer_verify import verify
            verify(args.out)
        elif args.stage=='init': initialise(args.out)
        elif args.stage=='run': run(args.out,args.shard,args.shards)
        elif args.stage=='collect': collect(args.out)
        else: plots(args.out)
    elif args.command == 'propka-components':
        from .propka_components import initialise,run,collect
        if args.stage=='init': initialise(args.out)
        elif args.stage=='run': run(args.out,args.shard,args.shards)
        elif args.stage=='plots':
            from .component_plots import launch
            launch(args.out)
        else: collect(args.out)
    elif args.command == 'pool-shards':
        from .pool_shards import index,collect
        if args.stage=='index': index(args.out,args.shard,args.shards)
        else: collect(args.out,args.shards)
    elif args.command == 'glycan-buffer-policy':
        from .glycan_buffer_policy import initialise,scan,collect,recount
        if args.stage=='init': initialise(args.source,args.campaign,args.smoke)
        elif args.stage=='scan': scan(args.campaign,args.shard,args.shards)
        elif args.stage=='collect': collect(args.campaign)
        elif args.stage=='verify':
            from .glycan_buffer_verify import run
            run(args.out)
        else: recount(args.campaigns,args.out)
    elif args.command == 'stripped-policy':
        from .stripped_policy import initialise,scan,collect
        if args.stage=='init': initialise(args.source,args.campaign)
        elif args.stage=='scan': scan(args.campaign,args.shard,args.shards)
        else: collect(args.campaign)
    elif args.command == 'ligand-audit':
        from .ligand_audit import audit
        audit(args.campaign,args.out)
    elif args.command == 'anchor-tiers':
        from .anchor_tiers import apply
        apply(args.campaign,args.out)
    elif args.command == 'mask-policy':
        from .mask_policy import apply
        apply(args.campaign,args.out)
    elif args.command == 'expanded-radial':
        from .expanded import plan, prepare, assemble, dispatch, monitor, status
        if args.stage=='plan': plan(args.campaigns,args.out,args.count)
        elif args.stage=='prepare': prepare(args.out,args.shard,args.shards)
        elif args.stage=='assemble': assemble(args.out)
        elif args.stage=='dispatch': dispatch(args.out)
        elif args.stage=='monitor': monitor(args.out)
        elif args.stage=='status': status(args.out)
        elif args.stage=='internal-anchors':
            from .internal_anchors import prepare as prepare_internal
            prepare_internal(args.out)
        elif args.stage=='anchors':
            from .anchor_analysis import prepare as prepare_anchors
            prepare_anchors(args.out)
        elif args.stage=='plots':
            from .radial_plots import prepare as prepare_plots
            prepare_plots(args.out)
        else:
            from .eta import estimate
            estimate(args.out)
    elif args.command == 'candidate-pool':
        from .pool import initialise, download, index, sequence, report
        if args.stage=='init': initialise(args.metadata,args.out,args.count,args.antibody_count,exclude_pool=args.exclude_pool)
        elif args.stage=='download': download(args.out,args.shard,args.shards)
        elif args.stage=='index': index(args.out)
        elif args.stage=='sequence': sequence(args.out)
        else: report(args.out,args.campaign)
    elif args.command == 'radial-error':
        from .radial import initialise, collect
        if args.stage=='init': initialise(args.campaign,args.out,args.count)
        else: collect(args.out,args.completed_only)
    elif args.command == 'sensitivity':
        from .sensitivity import initialise, initialise_altloc, collect
        if args.stage=='init': initialise(args.campaign,args.out)
        elif args.stage=='init-altloc': initialise_altloc(args.campaign,args.out)
        else: collect(args.out)
    elif args.command == 'curation-report':
        from .curation_report import report
        report(args.campaign)
    elif args.command == 'curation-pilot':
        from .curation import initialise, scan, collect
        if args.stage=='init': initialise(args.audit,args.campaign)
        elif args.stage=='scan': scan(args.campaign,args.shard,args.shards)
        else: collect(args.campaign)
    elif args.command == "download-sample":
        from .download import coordinate_sample
        coordinate_sample(args.out)
    elif args.command == "download-file":
        from .download import fetch
        atomic_json(args.out.with_suffix(args.out.suffix+'.receipt.json'), fetch(args.url,args.out))
    elif args.command == "download-metadata":
        from .download import metadata
        metadata(args.out)
    elif args.command == "dataset-audit":
        from .dataset_audit import main as dataset_main
        dataset_main(args)
    elif args.command == "audit-findings":
        from .audit import audit_findings
        audit_findings(args.campaign, args.prep_audit, args.out)
    elif args.command == "audit-teacher":
        from .audit import audit_teacher
        audit_teacher(args.source_root, args.out)
    elif args.command == "audit-prep":
        from .audit import audit_prep
        audit_prep(args.campaign, args.out)
    elif args.command == "audit-sources":
        from .audit import audit_sources
        audit_sources(args.out)
    elif args.command == "prepare":
        from .prep import prepare_campaign
        prepare_campaign(args.manifest, args.archive, args.out, args.count, args.seed, args.completion_executable)
    elif args.command == "preflight":
        from .preflight import preflight
        preflight(args.campaign,args.methods)
    elif args.command == "report":
        from .report import summarize
        summarize(args.campaign)
    elif args.command == "finalize":
        from .jobs import merge
        from .score import score_campaign
        from .report import summarize
        from .gates import report_gates
        merge(args.campaign,["pypka",*args.methods])
        score_campaign(args.campaign,args.methods)
        report_gates(args.campaign)
        summarize(args.campaign)
    elif args.command == "deposit-evidence":
        import urllib.request
        import re
        root=Path(os.environ["PKABENCH_RUNTIME"])/"sources/deposit-evidence"; root.mkdir(exist_ok=True)
        receipts=[]
        for i,url in enumerate(("https://www.pypka.org/pKPDB","https://pkpdb.pypka.org")):
            try:
                with urllib.request.urlopen(url,timeout=30) as response:
                    raw=response.read(2*1024*1024)
                    (root/f"landing-{i}.html").write_bytes(raw)
                    receipts.append({"url":url,"resolved":response.url,"status":response.status,
                        "links":re.findall(r'href=[\"\x27]([^\"\x27]+)',raw.decode(errors="replace"))})
            except Exception as exc: receipts.append({"url":url,"error":str(exc)})
        atomic_json(root/"receipts.json",receipts); print(json.dumps(receipts,indent=2))
    elif args.command == "verify-deposit":
        from .gates import verify_deposit
        verify_deposit(args.manifest,args.source_url)
    elif args.command == "validate":
        import sys
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", *args.tests], cwd=os.environ["PKABENCH_SOURCE"])
        if result.returncode: raise SystemExit(result.returncode)
        atomic_json(Path(os.environ["PKABENCH_RUNTIME"])/"receipts/runner.validated.json",
            {"status": "passed", "tests": args.tests, "job": os.environ["SLURM_JOB_ID"], "node": os.environ["SLURMD_NODENAME"]})
    elif args.command == "run":
        from .jobs import run_job
        run_job(args.campaign, args.complex_id, args.method, args.timeout)
    elif args.command == "merge":
        from .jobs import merge
        merge(args.campaign, args.methods)
    elif args.command == "score":
        from .score import score_campaign
        score_campaign(args.campaign, args.methods)
    elif args.command == "gates":
        from .gates import report_gates
        report_gates(args.campaign)
    elif args.command == "dispatch":
        from .jobs import dispatch
        dispatch(args.campaign, args.methods, args.diagnostic)
    elif args.command == "discover":
        import urllib.request
        import tarfile
        import io
        root=Path(os.environ["PKABENCH_RUNTIME"])/"sources"
        for name, repo in (("pKPDB", "mms-fcul/pKPDB"), ("pKAI", "bayer-science-for-a-better-life/pKAI"), ("KaMLs", "JanaShenLab/KaMLs")):
            if (root/f"{name}.json").exists(): continue
            with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/commits?per_page=1", timeout=60) as stream:
                commit=json.load(stream)[0]["sha"]
            with urllib.request.urlopen(f"https://codeload.github.com/{repo}/tar.gz/{commit}", timeout=120) as stream:
                raw=stream.read()
            (root/f"{name}.tar.gz").write_bytes(raw)
            target=root/name; target.mkdir(exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                for member in tar.getmembers():
                    relative=Path(*Path(member.name).parts[1:])
                    if not member.isfile() or '..' in relative.parts: continue
                    dest=target/relative; dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
            from .runtime import digest
            atomic_json(root/f"{name}.json", {"url":f"https://github.com/{repo}","commit":commit,"archive_sha256":digest(root/f"{name}.tar.gz")})
    elif args.command == "inspect":
        from collections import Counter
        from .schema import read_table
        rejected=read_table(args.campaign/"rejections.parquet")
        report={"histogram":dict(Counter(r["code"] for r in rejected)),"rejections":rejected,
            "accepted":[{k:s[k] for k in ("complex_id","pdb_id","n_residues")} for s in read_table(args.campaign/"structures.parquet")]}
        atomic_json(args.campaign/"inspection.json",report)
        print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
