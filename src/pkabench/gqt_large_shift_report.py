"""CPU-only report for the completed large-shift diagnostics."""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .runtime import atomic_json, digest, require_compute

BINS=((0.,.5,"<0.5"),(.5,1.,"0.5-1"),(1.,2.,"1-2"),(2.,np.inf,">=2"))
VARIANTS=("original","site_self_only","remove_arg","remove_orientation",
          "remove_site_geometry","remove_0_6A","remove_6_10A","remove_10_15A",
          "remove_15_20A","remove_pair_bias")

def read(path): return json.loads(Path(path).read_text())

def report(out):
    require_compute(threads=4,allow_comp1400=True)
    out=Path(out);sites=pq.read_table(out/"sites.parquet").to_pylist();attention=pq.read_table(out/"attention.parquet").to_pylist()
    gradients=read(out/"gradients.json");summaries=[];causal=[];att=[]
    for split in ("train","val"):
        for _,_,name in BINS:
            rr=[r for r in sites if r["split"]==split and r["shift_bin"]==name]
            target=np.asarray([r["teacher_shift"] for r in rr]);pred=np.asarray([r["predicted_shift"] for r in rr]);late=np.asarray([r["late_predicted_shift"] for r in rr])
            slope=lambda y:float(np.cov(target,y,ddof=0)[0,1]/np.var(target)) if np.var(target)>0 else None
            summaries.append(dict(split=split,bin=name,sites=len(rr),mae=float(np.mean(abs(pred-target))),
                late_mae=float(np.mean(abs(late-target))),slope=slope(pred),late_slope=slope(late),
                prediction_sd=float(pred.std()),teacher_sd=float(target.std()),mean_magnitude_ratio=float(np.mean([r["magnitude_ratio"] for r in rr])),
                mean_tanh_derivative=float(np.mean([r["tanh_derivative"] for r in rr]))))
            if split=="val":
                for variant in VARIANTS[1:]:
                    changed=np.asarray([r["predicted_"+variant] for r in rr])
                    causal.append(dict(bin=name,variant=variant,mean_absolute_effect=float(np.mean(abs(changed-pred))),ablated_mae=float(np.mean(abs(changed-target)))))
                aa=[r for r in attention if r["shift_bin"]==name]
                for metric in ("effective","maximum","mass_self","mass_arg","mass_opposite_class","mass_0_6","mass_6_10","mass_10_15","mass_15_20","neighbor_count"):
                    att.append(dict(bin=name,metric=metric,mean=float(np.mean([r[metric] for r in aa]))))
    results=dict(summary=summaries,causal=causal,attention=att,gradients=gradients);atomic_json(out/"results.json",results)
    val={r["bin"]:r for r in summaries if r["split"]=="val"};train={r["bin"]:r for r in summaries if r["split"]=="train"}
    lines=["# Why does the site-token GQT miss large shifts?","",
        "Selected epoch 9 and late epoch 17 of the backbone-only site-orientation model. Predictions, interventions and gradients use full-float32 Triton. Native attention is used only to expose parity-validated weights. No test data were read.","",
        "| Shift bin | Train sites | Train MAE | Validation sites | Validation MAE | Predicted/teacher magnitude | Shift slope | Late validation MAE |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _,_,name in BINS:
        a,b=train[name],val[name];lines.append(f"| {name} | {a['sites']:,} | {a['mae']:.4f} | {b['sites']:,} | {b['mae']:.4f} | {b['mean_magnitude_ratio']:.3f} | {b['slope']:.3f} | {b['late_mae']:.4f} |")
    lines += ["","## Causal context tests","","| Shift bin | Intervention | Mean absolute prediction change | Ablated MAE |","|---|---|---:|---:|"]
    for row in causal:lines.append(f"| {row['bin']} | {row['variant']} | {row['mean_absolute_effect']:.4f} | {row['ablated_mae']:.4f} |")
    lines += ["","## Gradient path","","| Shift bin | Sites | MSE | Total | Head | Site block | Query | Encoder 0 | Encoder 1 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in gradients["bins"]:lines.append(f"| {row['bin']} | {row['sites']:,} | {row['mse']:.4f} | {row['total']:.3g} | {row['head']:.3g} | {row['site']:.3g} | {row['query']:.3g} | {row['encoder_0']:.3g} | {row['encoder_1']:.3g} |")
    lines += ["","Gradient cosines and complete attention summaries are in `results.json`."]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    verification=read(out/"verification.json");verification.update(report_sha256=digest(out/"report.md"),results_sha256=digest(out/"results.json"),report_complete=True);atomic_json(out/"verification.json",verification)

if __name__=="__main__":
    root=Path(os.environ["PKABENCH_RUNTIME"])
    report(root/"audits/gqt-large-shift-diagnostics-v2")
