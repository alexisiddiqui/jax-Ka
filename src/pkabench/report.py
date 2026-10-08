"""Resource calibration and candidate accounting without resampling rejections."""
from collections import Counter
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from biotite.structure.io import pdbx
import biotite.structure as struc
from .runtime import require_compute, atomic_json
from .schema import read_table


def summarize(campaign):
    require_compute(); campaign=Path(campaign)
    manifest=json.loads((campaign/"manifest.json").read_text())
    accepted={s["complex_id"] for s in read_table(campaign/"structures.parquet")}
    rejections=read_table(campaign/"rejections.parquet")
    sizes=[]
    for row in manifest["candidates"]:
        path=campaign/"structures"/row["complex_id"]/"source.cif"
        if not path.exists(): continue
        atoms=pdbx.get_structure(pdbx.CIFFile.read(path),model=1,altloc="occupancy",use_author_fields=False)
        atoms=atoms[np.isin(atoms.chain_id,row["partner_A_chains"]+row["partner_B_chains"])]
        sizes.append({"complex_id":row["complex_id"],"observed_residues":int(struc.get_residue_count(atoms)),"accepted":row["complex_id"] in accepted})
    jobs=[]
    for path in sorted((campaign/"jobs").glob("*/*.json")):
        value=json.loads(path.read_text())
        jobs.append({"method":path.parent.name,"complex_id":path.stem,"wall_seconds":value["wall_seconds"],
            "maxrss_kib":value["maxrss_kib"],"status":value["status"],"node":value["node"],"job":value["job"],"errors":value["errors"]})
    calibration={}
    for method in sorted({j["method"] for j in jobs}):
        successful=[j for j in jobs if j["method"]==method and j["status"]=="complete"]
        if not successful: continue
        times=[j["wall_seconds"] for j in successful]
        memory=max(j["maxrss_kib"] for j in successful)
        calibration[method]={"n":len(times),"median_seconds":float(np.median(times)),"p98_seconds":float(np.quantile(times,.98)),
            "max_seconds":max(times),"provisional_timeout_seconds":math.ceil(max(times)*1.25),
            "maxrss_kib":memory,"provisional_reserved_cpus":max(2,2*math.ceil(1.25*memory/(4*1024*1024))),
            "memory_policy":"2 GiB per requested CPU; round CPU reservations to even counts",
            "limitation":"Small accepted smoke sample; excludes censored failures and does not establish a <=2% population timeout rate"}
    report={"candidates":len(manifest["candidates"]),"accepted":len(accepted),"rejection_histogram":dict(Counter(r["code"] for r in rejections)),
        "sizes":sizes,"jobs":jobs,"calibration":calibration,"production_allowed":False}
    if jobs:
        job_ids=','.join(sorted({j['job'].split(';')[0] for j in jobs}))
        result=subprocess.run(['sacct','-n','-P','-j',job_ids,'-o','JobID,State,AllocCPUS,ReqMem,MaxRSS,Elapsed'],capture_output=True,text=True)
        (campaign/'slurm-accounting.tsv').write_text('job_id\tstate\tallocated_cpus\trequested_memory\tmax_rss\telapsed\n'+result.stdout.replace('|','\t'))
        report['scheduler_accounting_error']=result.stderr if result.returncode else None
    ledger=campaign/'submission.json'
    if ledger.exists():
        entries=list(json.loads(ledger.read_text()).values())
        report['maximum_observed_user_cpu_requests']=max((e.get('queued_cores_before',0)+e.get('requested_cpus',0) for e in entries),default=0)
    atomic_json(campaign/"smoke_report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"sizes","jobs"}},indent=2))
