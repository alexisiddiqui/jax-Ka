"""Shared normalization, site accounting, and process-group timeouts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
from . import worker
from ..prep import read_cif, export_pdb
from ..runtime import atomic_json, config_hash
from ..schema import PH, KEY, NULL_PKA, GROUPS

TEACHER = {"epsin":15., "epssol":80., "ionicstr":.1, "gsize":81,
    "pbc_dimensions":0, "ffID":"G54A7", "pdb2pqr_h_opt":True, "keep_ions":False, "temp":298.15,
    "ser_thr_titration":False}


def execute(command, work, timeout, env=None):
    started=time.monotonic()
    with (work/"stdout.log").open("w") as stdout, (work/"stderr.log").open("w") as stderr:
        proc=subprocess.Popen(command, cwd=work, stdout=stdout, stderr=stderr, start_new_session=True,env=env)
        try:
            code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
    if code: raise RuntimeError(f"worker exited {code}; see {work}/stderr.log")
    return time.monotonic()-started


class Adapter:
    def __init__(self, method, sites, runtime, timeout=1800):
        if method not in {"pypka","propka","jaxka","pkai","pkai_plus","kaml","null"}: raise ValueError(method)
        self.method=method; self.sites=sites; self.runtime=Path(runtime); self.timeout=timeout
        self.config=TEACHER if method == "pypka" else {"model":method,"grid":[-2,16,.25],"gap_policy":"cap" if method == "jaxka" else "method_internal"}
        self.timings={}; self.errors={}; self.extras={}

    def run(self, state_files: dict[str, Path], workdir: Path):
        rows=[]; started=time.monotonic()
        for state, path in state_files.items():
            expected=[s for s in self.sites if state == "AB" or s["partner"] == state]
            work=workdir/state; work.mkdir(parents=True, exist_ok=True)
            output={}; version="unavailable"; status=None
            try:
                if self.method == "null":
                    version="1"
                    output={(s["chain"],s["resnum"],s["icode"],s["group"]): {"pka":NULL_PKA[s["group"]]} for s in expected}
                else:
                    mapping=export_pdb(read_cif(path), work/"input.pdb")
                    atomic_json(work/"mapping.json", [{"chain":c,"resnum":n,"original":v} for (c,n),v in mapping.items()])
                    env={"pypka":"pypka","pkai":"pkai","pkai_plus":"pkai","kaml":"kaml"}.get(self.method,"runner")
                    python=self.runtime/"envs"/env/"bin/python"
                    req={"method":self.method,"config":self.config,"pdb":str((work/"input.pdb").resolve()),"cif":str(path.resolve())}
                    atomic_json(work/"request.json",req)
                    remaining=self.timeout-(time.monotonic()-started)
                    if remaining <= 0: raise subprocess.TimeoutExpired([self.method], self.timeout)
                    environment=os.environ.copy()
                    if self.method == "pypka": environment["LD_LIBRARY_PATH"]=str(self.runtime/"fortran/lib")
                    self.timings[state]=execute([str(python), "-m", "pkabench.adapters.worker", str((work/"request.json").resolve()), str((work/"result.json").resolve())],work,remaining,environment)
                    result=json.loads((work/"result.json").read_text()); version=result["version"]
                    self.extras[state]={k:v for k,v in result.items() if k != "rows"}
                    self.extras[state]["unsupported_groups"] = sorted({r["group"] for r in result["rows"] if r["group"] not in GROUPS})
                    self.extras[state]["intrinsic_tautomers_available"] = any(r.get("intrinsic_tautomers") for r in result["rows"])
                    for item in result["rows"]:
                        if item["group"] not in GROUPS: continue
                        if self.method == "jaxka": original=(item["chain"],item["resnum"],item.get("icode",""))
                        else: original=mapping[(item["chain"],item["resnum"])]
                        k=(*original,item["group"])
                        if k in output: raise ValueError(f"duplicate output site {k}")
                        output[k]=item
            except Exception as exc:
                status="failed"; output={}
                self.errors[state]={"type":type(exc).__name__,"detail":str(exc),
                    "code": "teacher_timeout" if self.method == "pypka" and isinstance(exc,subprocess.TimeoutExpired) else "teacher_failed" if self.method == "pypka" else "method_failed"}
            for site in expected:
                item=output.get((site["chain"],site["resnum"],site["icode"],site["group"]))
                row={k:site[k] for k in KEY}
                row.update(state=state,method=self.method,method_version=version,config_sha256=config_hash(self.config),
                    status=status or "not_reported",pka=None,curve=None,curve_source=None,intrinsic_pka=None)
                if item:
                    pka=item.get("pka")
                    pka=float(pka) if pka is not None and np.isfinite(pka) else None
                    inferred="ok" if pka is not None and -2 <= pka <= 16 else "out_of_range"
                    row.update(pka=pka,status=item.get("status",inferred),intrinsic_pka=item.get("intrinsic_pka"))
                    if item.get("curve") is not None:
                        row.update(curve=item["curve"],curve_source=item["curve_source"])
                    elif pka is not None:
                        row.update(curve=(1/(1+10**(PH-pka))).tolist(),curve_source="hh")
                rows.append(row)
        return rows
