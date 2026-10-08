"""Smoke diagnostics: no inferential claims before a frozen component split."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .schema import key, read_table
from .runtime import atomic_json, require_compute
from .linkage import linkage


def paired(rows, sites, method):
    index={(key(r),r["state"]):r for r in rows if r["method"]==method}
    out={}
    for s in sites:
        if s["is_break_terminus"] or s.get('supervision_mask') is False or s["min_partner_distance"]>20: continue
        k=key(s); ab=index.get((k,"AB")); free=index.get((k,s["partner"]))
        if ab and free and ab["status"]==free["status"]=="ok":
            out[k]=(ab["pka"]-free["pka"],ab["pka"],free["pka"])
    return out


def correlation(a,b,rank=False):
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0: return None
    return float(spearmanr(a,b).statistic if rank else np.corrcoef(a,b)[0,1])


def metrics(reference,predicted):
    a=np.asarray(reference,dtype=float); b=np.asarray(predicted,dtype=float)
    if not len(a): return {"n":0,"skill":None,"spearman":None,"sign_accuracy":None}
    denom=np.mean(a*a); mask=np.abs(a)>=.5
    return {"n":len(a),"skill":float(1-np.mean((a-b)**2)/denom) if denom else None,
        "spearman":correlation(a,b,True),"sign_accuracy":float(np.mean(np.sign(a[mask])==np.sign(b[mask]))) if mask.any() else None}


def csv_write(path, rows, fields):
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def score_campaign(campaign,methods):
    require_compute(); campaign=Path(campaign)
    manifest_path=campaign/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get("freeze_manifest_sha256"):
            expected=set(manifest["methods"])-{"pypka"}
            if set(methods)!=expected or len(methods)!=len(set(methods)):
                raise ValueError("Frozen scoring methods must match the manifest")
            from .frozen_score import score
            from .frozen_score_secondary import run as secondary
            score(campaign); secondary(campaign)
            return
    if "pypka" in methods: raise ValueError("teacher is the reference, not a scored smoke method")
    rows=read_table(campaign/"predictions.parquet"); sites=read_table(campaign/"sites.parquet")
    ref=paired(rows,sites,"pypka"); models={m:paired(rows,sites,m) for m in methods}
    common=set(ref)
    for model in models.values(): common &= set(model)
    annotations={key(s):s for s in sites}; scores=[]; represent=[]
    for method,model in models.items():
        coverage=len(set(ref)&set(model))/len(ref) if ref else None
        for label,select in [("interface",lambda s:s["residue_delta_sasa"]>10),("shell_0_20",lambda s:True)]:
            keys=sorted(k for k in common if select(annotations[k]))
            result=metrics([ref[k][0] for k in keys],[model[k][0] for k in keys])
            errors_ab=[model[k][1]-ref[k][1] for k in keys]; errors_free=[model[k][2]-ref[k][2] for k in keys]
            scores.append({"method":method,"subset":label,**result,"error_cancellation":correlation(errors_ab,errors_free),"coverage":coverage})
        for lo,hi in ((0,5),(5,10),(10,15),(15,20)):
            keys=[k for k in set(ref)&set(model) if (lo<=annotations[k]["min_partner_distance"]<hi or hi==20 and annotations[k]["min_partner_distance"]==20) and abs(ref[k][0])>=.1]
            represent.append({"method":method,"shell":f"{lo}-{hi}","coverage":coverage,"n":len(keys),
                "structural_zeros":float(np.mean([abs(model[k][0])<.01 for k in keys])) if keys else None})
    csv_write(campaign/"scores_smoke.csv",scores,["method","subset","n","skill","spearman","sign_accuracy","error_cancellation","coverage"])
    csv_write(campaign/"representability.csv",represent,["method","shell","coverage","n","structural_zeros"])
    curves=[]
    for cid in sorted({s["complex_id"] for s in sites}):
        for method in ["pypka",*methods]:
            result=linkage([r for r in rows if r["complex_id"]==cid and r["method"]==method],[s for s in sites if s["complex_id"]==cid])
            sidecar=campaign/"jobs"/method/f"{cid}.json"
            if sidecar.exists():
                extra=json.loads(sidecar.read_text()).get("extra",{})
                unsupported=sorted({g for value in extra.values() for g in value.get("unsupported_groups",[])})
                if unsupported:
                    result={"status":"incomplete_charge_coverage","unsupported_groups":unsupported,"delta_q":None,"delta_g":None}
            curves.append({"complex_id":cid,"method":method,**result})
    atomic_json(campaign/"linkage_smoke.json",curves)
    atomic_json(campaign/"scoring_policy.json",{"label":"agreement with PB; smoke diagnostics only","common_sites":len(common),
        "reference_valid_sites":len(ref),"bootstrap":None,"reason":"no frozen split components for smoke set",
        "pkai_caveat":"pKAI and pKAI+ were distilled from pKPDB/PypKa labels; PB agreement measures self-consistency, not independent experimental accuracy",
        "teacher_configuration":"G2 remains authoritative; unresolved deposit settings make these provisional smoke diagnostics"})
