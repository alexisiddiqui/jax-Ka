"""Evidence ledger: missing evidence stays unresolved, never an assumed pass."""
from collections import Counter
import json
import os
from pathlib import Path
from .runtime import atomic_json, require_compute, digest, config_hash
from .schema import read_table


def compare_deposit(data,expected):
    """Compare actual deposit settings, never substitute current package defaults."""
    flattened={}
    for section in (data,data.get('pypka_params',{}),data.get('delphi_params',{}),data.get('mc_params',{})):
        for name,value in section.items():
            if isinstance(value,dict): continue
            name={'pbc_dim':'pbc_dimensions'}.get(name,name)
            if name in flattened and flattened[name]!=value:
                raise ValueError(f'conflicting deposit setting {name}')
            flattened[name]=value
    comparisons={}
    for name,value in expected.items():
        actual=flattened.get(name)
        if actual is not None:
            if isinstance(value,bool):
                if isinstance(actual,str): actual={'true':True,'false':False}.get(actual.lower(),actual)
            elif isinstance(value,(float,int)):
                try: actual=float(actual)
                except (TypeError,ValueError): pass
        comparisons[name]={'expected':value,'deposited':actual,'status':'missing' if actual is None else 'match' if actual==value else 'mismatch'}
    return {'status':'pass' if all(v['status']=='match' for v in comparisons.values()) else 'fail' if any(v['status']=='mismatch' for v in comparisons.values()) else 'unresolved','comparison':comparisons}


def verify_deposit(manifest,source_url):
    require_compute()
    from .adapters.base import TEACHER
    manifest=Path(manifest).resolve()
    result=compare_deposit(json.loads(manifest.read_text()),TEACHER)
    result.update(manifest=str(manifest),manifest_sha256=digest(manifest),source_url=source_url,
        config_sha256=config_hash(TEACHER),evidence_kind='supplied deposit settings export')
    atomic_json(Path(os.environ['PKABENCH_RUNTIME'])/'sources/teacher-deposit-verification.json',result)
    print(json.dumps(result,indent=2))


def report_gates(campaign):
    require_compute(); campaign=Path(campaign); runtime=Path(os.environ["PKABENCH_RUNTIME"])
    structures=read_table(campaign/"structures.parquet")
    if not structures: raise RuntimeError("no accepted smoke pair for G1")
    target=min(structures,key=lambda s:s["n_residues"])
    cid=target["complex_id"]
    from .jobs import run_job, inputs
    side=campaign/"jobs/pypka"/f"{cid}.json"
    if not side.exists() or json.loads(side.read_text())["status"] != "complete": run_job(campaign,cid,"pypka",1800)
    receipt=json.loads(side.read_text()); predictions=read_table(side.with_suffix(".parquet"))
    if receipt["status"]=="complete" and receipt["inputs"]!=inputs(campaign,cid,"pypka"):
        raise ValueError("G1 evidence is stale for this code/configuration/environment; use a fresh campaign")
    native=all(any(r["state"]==s and r["curve_source"]=="native" and r["curve"] is not None for r in predictions) for s in ("AB","A","B"))
    g1=receipt["status"]=="complete" and native
    if g1:
        atomic_json(runtime/"receipts/pypka.validated.json",{"status":"pass","gate":"G1","evidence":str(side),"job":receipt["job"],"node":receipt["node"]})
    report={"G1":{"status":"pass" if g1 else "fail","complex_id":cid,"evidence":str(side),"errors":receipt["errors"]},
        "G1b":{"status":"pass" if all(x.get("intermediates") and x.get("intrinsic_tautomers_available") for x in receipt["extra"].values()) and len(receipt["extra"])==3 else "partial" if any(x.get("intermediates") for x in receipt["extra"].values()) else "unresolved",
            "detail":"Intrinsic pKa and interactions retained at native tautomer-state resolution; scalar site-pair reduction is not assumed."},
        "G2":{"status":"unresolved","detail":"Requires deposit-associated configuration evidence; no production release from defaults alone."},
        "G3":{"status":"unresolved","detail":"Joint-system equivalence controls required for each retained method."},
        "G4":{"status":"unresolved","detail":"Weight load and training update required."},
        "G5":{"status":"fallback","detail":"Titratable-only rejection when rebuild unavailable or invalid; per-structure completion verified."}}
    rejections=read_table(campaign/"rejections.parquet")
    curation=json.loads((campaign/"curation_report.json").read_text())
    report["curation"]={**curation,"rejection_histogram":dict(Counter(r["code"] for r in rejections))}
    report["production_allowed"]=False
    if (campaign/"preflight.json").exists():
        preflight=json.loads((campaign/"preflight.json").read_text())
        for gate in ("G3","G4","G5"): report[gate]=preflight[gate]
    deposit_source=runtime/"sources/pKPDB/src/fill.py"
    if deposit_source.exists():
        report["G2"]["evidence"]={"source":str(deposit_source),"confirmed_explicit":{"epsin":15,"ionicstr":.1,"pbc_dimensions":0},
            "deposit_landing_receipts":str(runtime/"sources/deposit-evidence/receipts.json"),
            "missing":"Deposit sim_settings export for solvent dielectric, grid, force field, protonation settings, ions, temperature; SER/THR titration also needs reconciliation with nine-group schema"}
    verification=runtime/'sources/teacher-deposit-verification.json'
    teacher_audit = runtime/'sources/teacher-audit.json'
    if teacher_audit.exists():
        from .adapters.base import TEACHER
        audit = json.loads(teacher_audit.read_text())
        evidence_path = Path(audit['path'])
        if digest(evidence_path) != audit['sha256']: raise ValueError('teacher audit evidence changed')
        evidence = json.loads(evidence_path.read_text())
        source_settings = json.loads((evidence_path.parent/'server-pkpdb-params.json').read_text())
        report['G2']['server_evidence'] = {**audit, 'comparison': compare_deposit(source_settings, TEACHER),
            'limitation': evidence['evidence_limit'], 'historical_version': evidence['declared_historical_version']}
        report['G2']['evidence']['missing'] = 'Per-simulation historical provenance, temperature and hydrogen optimisation; server metadata declares PypKa 2.1.0 while installed teacher is 2.10.0'
        report['G2']['detail'] = 'Official server metadata corroborates core settings and SER/THR off; it is not a per-simulation deposit export.'
    if verification.exists():
        from .adapters.base import TEACHER
        evidence=json.loads(verification.read_text())
        if digest(evidence['manifest'])!=evidence['manifest_sha256'] or evidence['config_sha256']!=config_hash(TEACHER):
            raise ValueError('stale deposit configuration evidence')
        report['G2']=evidence
    report["stop_reasons"]=([] if g1 else ["G1 failed; no fallback teacher"])+([] if curation["progression_allowed"] else ["prep acceptance below 50% or pipeline errors"])
    if report['G2']['status']!='pass': report['stop_reasons'].append('G2: full deposit-associated teacher configuration is unverified')
    atomic_json(campaign/"gates.json",report)
    print(json.dumps(report,indent=2))
