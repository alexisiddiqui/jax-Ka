"""Frozen bound-versus-disconnected interface scores on matched sites."""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import time
import jax
import numpy as np
from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.gqt_auxiliary_pilot import read, _interface_metrics
from pkatrain.gqt_paired_pinder import Loader, _bucket_n, _prefetched
from pkatrain.gqt_query_norm_pilot import setup
from pkatrain.trainer import load_checkpoint


def run(runtime):
    require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True)
    started=time.monotonic()
    base,root,manifest,p,state,engine=setup(Path(runtime),"separate")
    output=root/"partner-removal-scores-v1"; output.mkdir(exist_ok=False)
    verification=read(root/"separate/verification.json")
    checkpoint=root/"separate/checkpoints"/f"epoch-{verification['best']['epoch']:03d}"
    p,_,_=load_checkpoint(checkpoint,(p,state))
    mask_audit=read(root/"edge-corruption-v1/partner-mask-audit.json")
    if not mask_audit["passed"]:raise AssertionError("partner mask audit failed")
    records=[r for r in manifest["records"] if r["split"]=="val"]
    lookup={r["id"]:r for r in records}; buckets=defaultdict(list)
    for r in records:buckets[_bucket_n(r["n"])].append(r["id"])
    plans=[ids[i:i+manifest["batch_sizes"][bucket]] for bucket,ids in sorted(buckets.items(),key=lambda x:int(x[0]))
           for i in range(0,len(ids),manifest["batch_sizes"][bucket])]
    atomic_json(output/"protocol.json",{"checkpoint_sha256":digest(checkpoint/"state.npz"),"code_sha256":digest(__file__),
        "manifest_sha256":digest(base/"manifest.json"),"complexes":len(records),"retraining":False,"test_data_included":False,
        "removal":"all interpartner residue and site edges disconnected; own-partner geometry, identity and terminal flags retained",
        "caveat":"interface head supervised on bound branch only; free scores are an unsupervised behavior diagnostic, not validated free-state probability"})
    loader=Loader(base,manifest);rows=[]
    for number,(ids,batch) in enumerate(_prefetched(loader,plans),1):
        graph,_,mask,_,interface,metadata=batch
        predicted=jax.tree.map(np.asarray,engine.predictions(p,graph))
        for bi,cid in enumerate(ids):
            for qi,key in enumerate(lookup[cid]["keys"]):
                if not mask[bi,qi]:continue
                bound=float(.05+.95*predicted["interface"][bi,0,qi])
                free=float(.05+.95*predicted["interface"][bi,1,qi])
                rows.append({"complex_id":cid,**dict(zip(("chain","resnum","icode","group"),key)),
                    "interface":bool(metadata["interface"][bi,qi]),"distance":float(metadata["partner_distance_A"][bi,qi]),
                    "target_bound":float(interface[bi,qi]*manifest["normalization"]["interface"]),
                    "bound_score":bound,"free_score":free,"drop":bound-free,
                    "burial_bound":float(predicted["burial"][bi,0,qi]),"burial_free":float(predicted["burial"][bi,1,qi])})
        if number%20==0:print(json.dumps({"batch":number,"total":len(plans),"seconds":time.monotonic()-started}),flush=True)
    loader.close()
    with (output/"predictions.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def stats(items):
        if not items:return {"sites":0}
        drops=np.asarray([r["drop"] for r in items]); complex_means=defaultdict(list)
        for r in items:complex_means[r["complex_id"]].append(r["drop"])
        means=np.asarray([np.mean(v) for v in complex_means.values()]);rng=np.random.default_rng(17)
        bootstrap=np.mean(means[rng.integers(len(means),size=(2000,len(means)))],axis=1)
        return {"sites":len(items),"complexes":len(means),"mean_bound":float(np.mean([r["bound_score"] for r in items])),
            "mean_free":float(np.mean([r["free_score"] for r in items])),"mean_drop":float(drops.mean()),
            "median_drop":float(np.median(drops)),"fraction_drop_gt_0_01":float(np.mean(drops>.01)),
            "fraction_drop_gt_0_05":float(np.mean(drops>.05)),"fraction_increase_gt_0_01":float(np.mean(drops<-.01)),
            "equal_complex_mean_drop":float(means.mean()),"equal_complex_drop_CI95":np.quantile(bootstrap,[.025,.975]).tolist(),
            "mean_abs_burial_change":float(np.mean([abs(r["burial_bound"]-r["burial_free"]) for r in items]))}
    strata={"all":rows,"stored_interface_zone":[r for r in rows if r["interface"]],
        "noninterface":[r for r in rows if not r["interface"]],"distance_le_4":[r for r in rows if r["distance"]<=4],
        "distance_4_10":[r for r in rows if 4<r["distance"]<=10],"distance_gt_10":[r for r in rows if r["distance"]>10]}
    result={"strata":{k:stats(v) for k,v in strata.items()},"ranking_original_bound_interface":{
        field:_interface_metrics(rows,field) for field in ("bound_score","free_score")},"wall_seconds":time.monotonic()-started}
    atomic_json(output/"summary.json",result)
    lines=["# Interface scores with and without the binding partner","",
        "Frozen seed-17 separate-normalization oGQT; all 400 validation complexes. Scores are raw w_interface, not calibrated contact probabilities.","",
        "| Sites | Count | With partner | Without partner | Mean decrease | Decrease >0.05 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name,s in result["strata"].items():
        lines.append(f"| {name} | {s['sites']} | {s['mean_bound']:.5f} | {s['mean_free']:.5f} | {s['mean_drop']:.5f} | {s['fraction_drop_gt_0_05']:.1%} |")
    lines.extend(["","Both residue and site cross-partner edges are disconnected. Existing mask alignment checks passed. Interface supervision was bound-only, so free-state behavior was not directly trained.","",
        "No alternative real partners were evaluated here. Rankings against the original bound-interface flag diagnose persistence of the original pattern; they do not label an interface in the free state.","",
        f"Completed in {result['wall_seconds']:.1f} seconds. Predictions, complex-bootstrap intervals, and ranking diagnostics are saved alongside this report."])
    (output/"report.md").write_text("\n".join(lines)+"\n")
    atomic_json(output/"verification.json",{"passed":True,"complexes":len(records),"sites":len(rows),"wall_seconds":result["wall_seconds"]})


if __name__=="__main__":run(os.environ["PKABENCH_RUNTIME"])
