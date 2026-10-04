"""One complex × method shard; conservative queue accounting and strict merge."""
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from .runtime import atomic_json, config_hash, digest, require_compute
from .schema import read_table, write_table, KEY


def inputs(campaign, complex_id, method):
    work=campaign/"structures"/complex_id
    scientific=("prep.py","annotate.py","schema.py","supervision.py","curation.py","conformers.py","adapters/base.py","adapters/worker.py")
    module_root=Path(__file__).parent
    implementations={name:digest(module_root/name) for name in scientific}
    implementations.update({f"jaxpropka/{p.name}":digest(p) for p in (module_root.parent/"jaxpropka").glob("*.py")})
    from .adapters.base import Adapter
    runtime=Path(os.environ["PKABENCH_RUNTIME"])
    native_manifest=runtime/"manifests/fortran-explicit.txt"
    return {"states":{s:digest(work/f"{s}.cif") for s in ("AB","A","B")},
        "config":config_hash(Adapter(method,[],runtime).config),
        "native_runtime":digest(native_manifest) if method=="pypka" and native_manifest.exists() else None,
        "weights":{p.name:digest(p) for p in (runtime/"envs/pkai/lib/python3.11/site-packages/pkai/models").glob("*.pt")} if method in {"pkai","pkai_plus"} else {},
        "sites":digest(work/"sites.parquet"), "manifest":digest(campaign/"manifest.json"), "method":method,
        "implementation": config_hash(implementations),
        "environment": digest(Path(os.environ["PKABENCH_RUNTIME"])/"manifests"/f'{ {"pypka":"pypka","pkai":"pkai","pkai_plus":"pkai","kaml":"kaml"}.get(method,"runner")}.requirements.lock')}


def run_job(campaign, complex_id, method, timeout=1800):
    require_compute(); campaign=Path(campaign).resolve()
    from .adapters.base import Adapter
    base=campaign/"jobs"/method; base.mkdir(parents=True,exist_ok=True)
    parquet=base/f"{complex_id}.parquet"; sidecar=base/f"{complex_id}.json"
    with (base/f"{complex_id}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        provenance=inputs(campaign,complex_id,method)
        adapter=Adapter(method,read_table(campaign/"structures"/complex_id/"sites.parquet"),os.environ["PKABENCH_RUNTIME"],timeout)
        provenance["config"]=config_hash(adapter.config)
        if sidecar.exists():
            previous=json.loads(sidecar.read_text())
            if previous["inputs"] != provenance:
                if previous["status"] == "complete":
                    raise ValueError("existing completed shard has incompatible hashes; use a new campaign or archive that shard")
                # Retain failed attempts before retrying corrected code/environment.
                archived=base/complex_id/f"failed-{previous['job']}"; archived.mkdir(parents=True,exist_ok=True)
                sidecar.rename(archived/"sidecar.json")
                if parquet.exists(): parquet.rename(archived/"predictions.parquet")
            if previous["status"] == "complete" and parquet.exists() and digest(parquet)==previous["output_sha256"]: return
        work=base/complex_id/f"attempt-{os.environ['SLURM_JOB_ID']}"; work.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        rows=adapter.run({s:campaign/"structures"/complex_id/f"{s}.cif" for s in ("AB","A","B")},work)
        write_table(parquet,"predictions",rows)
        atomic_json(sidecar,{"inputs":provenance,"output_sha256":digest(parquet),"status":"failed" if adapter.errors else "complete",
            "errors":adapter.errors,"state_seconds":adapter.timings,"wall_seconds":time.monotonic()-started,
            "maxrss_kib":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,"extra":adapter.extras,
            "node":os.environ["SLURMD_NODENAME"],"job":os.environ["SLURM_JOB_ID"],"workdir":str(work)})


def merge(campaign, methods):
    require_compute(); campaign=Path(campaign); rows=[]; missing=[]; statuses=[]
    if len(methods)!=len(set(methods)): raise ValueError("duplicate methods")
    structures=read_table(campaign/"structures.parquet")
    for method in methods:
        for structure in structures:
            cid=structure["complex_id"]; base=campaign/"jobs"/method
            sidecar=base/f"{cid}.json"; parquet=base/f"{cid}.parquet"
            if not sidecar.exists() or not parquet.exists():
                reconcile_scheduler_failure(campaign,cid,method)
                if not sidecar.exists() or not parquet.exists(): missing.append([cid,method]); continue
            side=json.loads(sidecar.read_text()); current=inputs(campaign,cid,method)
            if any(side["inputs"][k]!=v for k,v in current.items()) or digest(parquet)!=side["output_sha256"]:
                raise ValueError(f"hash mismatch: {cid}/{method}")
            shard=read_table(parquet)
            if any(r["complex_id"]!=cid or r["method"]!=method for r in shard): raise ValueError("wrong shard identity")
            if any(r["config_sha256"]!=side["inputs"]["config"] for r in shard): raise ValueError("row configuration differs from sidecar")
            expected={(s["chain"],s["resnum"],s["icode"],s["group"],state) for s in read_table(campaign/"structures"/cid/"sites.parquet") for state in ("AB",s["partner"])}
            actual={(s["chain"],s["resnum"],s["icode"],s["group"],s["state"]) for s in shard}
            if actual!=expected: raise ValueError("missing/extra state-site rows")
            rows.extend(shard); statuses.append({"complex_id":cid,"method":method,"status":side["status"],"errors":side["errors"]})
    atomic_json(campaign/"merge_report.json",{"missing":missing,"jobs":statuses})
    if missing: raise ValueError("incomplete campaign: missing shards")
    write_table(campaign/"predictions.parquet","predictions",rows)
    write_table(campaign/"pairs.parquet","pairs",[])


def reconcile_scheduler_failure(campaign,cid,method):
    """Account for Slurm-level OOM/timeout before the worker could write a shard."""
    ledger=campaign/"submission.json"
    if not ledger.exists(): return
    task=json.loads(ledger.read_text()).get(f"{cid}/{method}")
    if not task: return
    job=task["job"].split(";")[0]
    output=subprocess.check_output(["sacct","-X","-n","-P","-j",job,"-o","JobIDRaw,State,ElapsedRaw"],text=True)
    match=[line.split("|") for line in output.splitlines() if line.split("|")[0]==job]
    if not match: return
    _,state,seconds=match[0][:3]; state=state.split()[0]
    if state not in {"OUT_OF_MEMORY","TIMEOUT","FAILED","CANCELLED","NODE_FAIL","PREEMPTED"}: return
    provenance=inputs(campaign,cid,method); rows=[]
    for site in read_table(campaign/"structures"/cid/"sites.parquet"):
        for s in ("AB",site["partner"]):
            rows.append({**{k:site[k] for k in KEY},"state":s,"method":method,"method_version":"unavailable",
                "config_sha256":provenance["config"],"status":"failed","pka":None,"curve":None})
    base=campaign/"jobs"/method; base.mkdir(parents=True,exist_ok=True)
    parquet=base/f"{cid}.parquet"; write_table(parquet,"predictions",rows)
    atomic_json(base/f"{cid}.json",{"inputs":provenance,"output_sha256":digest(parquet),"status":"failed",
        "errors":{"scheduler":{"code":state.lower(),"job":job}},"extra":{},"state_seconds":{},
        "wall_seconds":float(seconds),"maxrss_kib":0,"job":job,"node":None,"workdir":None})


def queued_cores(text):
    """Input must be squeue -r: one line per array task, including pending tasks."""
    total=0
    for line in text.splitlines():
        if line.strip(): total+=int(line.strip())
    return total


def dispatch(campaign, methods, diagnostic=False):
    require_compute(); campaign=Path(campaign).resolve()
    from .gates import report_gates
    report_gates(campaign)
    gates=json.loads((campaign/"gates.json").read_text())
    if gates["G1"]["status"]!="pass": raise RuntimeError("G1 has not passed; no campaign dispatch")
    if not diagnostic and (not gates["curation"]["progression_allowed"] or gates["G2"]["status"]!="pass"):
        raise RuntimeError("preparation/configuration gates block release; --diagnostic permits only the frozen smoke candidates")
    if diagnostic:
        manifest=json.loads((campaign/"manifest.json").read_text())
        if len(manifest["candidates"])>50: raise RuntimeError("diagnostic bypass is limited to the <=50 frozen smoke candidates")
        atomic_json(campaign/"diagnostic.json",{"production_allowed":False,"gates":gates,"purpose":"complete smoke diagnostics without expanding candidate universe"})
    for method in methods:
        if method not in {"pypka","null"} and gates.get("G3",{}).get(method,{}).get("status")!="pass":
            raise RuntimeError(f"method {method} has not passed G3")
    runtime=Path(os.environ["PKABENCH_RUNTIME"])
    tasks=[(s["complex_id"],m) for s in read_table(campaign/"structures.parquet") for m in methods]
    ledger_path=campaign/"submission.json"
    with (runtime/"submission.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        for cid,method in tasks:
            identity=f"{cid}/{method}"
            if identity in ledger: continue
            used=queued_cores(subprocess.check_output(["squeue","-r","-h","-u",os.environ["USER"],"-o","%C"],text=True))
            if used+2>400: break
            state=subprocess.check_output(["sinfo","-N","-h","-p","generalaccess,amd96","-o","%N %t"],text=True)
            excluded={"comp1400"}|{line.split()[0] for line in state.splitlines() if line.split()[1] not in {"idle","mix","alloc"}}
            command=["sbatch","--parsable",f"--exclude={','.join(sorted(excluded))}",f"--output={runtime}/logs/%x-%j.out",
                "/home/coulson/oc/lina4225/_HPC/submission/jax-Ka/pkabench/work.sbatch","run","--campaign",str(campaign),"--complex-id",cid,"--method",method]
            job=subprocess.check_output(command,text=True).strip()
            ledger[identity]={"job":job,"queued_cores_before":used,"requested_cpus":2,"requested_memory_gib":4}
            atomic_json(ledger_path,ledger)
        atomic_json(campaign/"unsent.json",[t for t in tasks if f"{t[0]}/{t[1]}" not in ledger])
