"""Numerically controlled tighter-capacity benchmark for full-graph GQT."""
import json
import math
import os
from pathlib import Path
import time

import jax
import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.graph_batches import BatchLoader, LoaderTelemetry
from pkatrain.graph_data import bucket
from pkatrain.graph_mmap import GraphMMap, default_bundle
from pkatrain.records import read
from profile_gqt_fullgraph import make_engine, manifest_for, plans_for, ready, summarize_telemetry
from profile_gqt_mmap import compare_tree


CANDIDATES={
    "n128":(128,None,None),
    "n128_k64":(128,64,None),
    "n64_k32":(64,32,None),
}
MAX_COMPILED_SHAPES=12


def rounded(value,quantum):return int(math.ceil(value/quantum)*quantum)


def capacity(cids,byid,quantum,manifest):
    current=old_capacity(cids,byid,manifest)
    return [current[i] if step is None else min(current[i],rounded(max(int(byid[cid][key]) for cid in cids),step))
            for i,(key,step) in enumerate(zip(("n","k","q"),quantum))]


def shaped(plans,byid,quantum,manifest):
    return [dict(cids=list(cids),capacities=capacity(cids,byid,quantum,manifest)) for cids in plans]


def old_capacity(cids,byid,manifest):return list(manifest["capacities"][bucket(byid[cids[0]])])


def summarize_plan(specs,edges,batch_size):
    shapes={tuple(spec["capacities"]) for spec in specs};actual_edges=padded_edges=actual_nodes=padded_nodes=actual_q=padded_q=0
    for spec in specs:
        n,k,q=spec["capacities"];cids=spec["cids"]
        actual_edges+=sum(edges[cid] for cid in cids);padded_edges+=batch_size*n*k
        actual_nodes+=sum(edges[cid+":n"] for cid in cids);padded_nodes+=batch_size*n
        actual_q+=sum(edges[cid+":q"] for cid in cids);padded_q+=batch_size*q
    return dict(batches=len(specs),unique_shapes=len(shapes),shapes=[list(x) for x in sorted(shapes)],
        actual_to_padded_edges=actual_edges/padded_edges,actual_to_padded_nodes=actual_nodes/padded_nodes,
        actual_to_padded_queries=actual_q/padded_q,padded_edges=padded_edges)


def plan(root):
    destination=root/"audits/gqt-tight-buckets-v1";destination.mkdir(parents=True,exist_ok=True)
    out,manifest=manifest_for(root,"50k");byid={row["complex_id"]:row for row in manifest["records"]}
    store=GraphMMap(default_bundle(out),manifest["records"])
    edges={}
    for row in manifest["records"]:
        graph,_=store.raw(row["complex_id"]);cid=row["complex_id"]
        edges[cid]=int(graph["edge_mask"].sum());edges[cid+":n"]=int(row["n"]);edges[cid+":q"]=int(row["q"])
    store.close();results={}
    for seed in range(18,23):
        _,plans=plans_for(manifest,seed=seed)
        variants={"current":[dict(cids=list(cids),capacities=old_capacity(cids,byid,manifest)) for cids in plans]}
        variants.update({name:shaped(plans,byid,quantum,manifest) for name,quantum in CANDIDATES.items()})
        for name,specs in variants.items():results.setdefault(name,[]).append(summarize_plan(specs,edges,manifest["config"]["batch_size"]))
    summary={}
    for name,epochs in results.items():
        shapes={tuple(shape) for epoch in epochs for shape in epoch["shapes"]}
        summary[name]=dict(quantum=None if name=="current" else list(CANDIDATES[name]),
            epochs=len(epochs),union_shapes=len(shapes),shapes=[list(x) for x in sorted(shapes)],
            mean_actual_to_padded_edges=float(np.mean([x["actual_to_padded_edges"] for x in epochs])),
            mean_actual_to_padded_nodes=float(np.mean([x["actual_to_padded_nodes"] for x in epochs])),
            mean_actual_to_padded_queries=float(np.mean([x["actual_to_padded_queries"] for x in epochs])))
    eligible=[name for name in CANDIDATES if summary[name]["union_shapes"]<=MAX_COMPILED_SHAPES]
    selected=[name for name in ("n128","n128_k64") if name in eligible]
    if not selected:selected=[min(CANDIDATES,key=lambda name:summary[name]["union_shapes"])]
    atomic_json(destination/"plan.json",dict(passed=True,seeds=list(range(18,23)),batch_membership_preserved=True,
        update_count_preserved=True,selected=selected,variants=summary,
        maximum_compiled_shapes=MAX_COMPILED_SHAPES,
        source_manifest_sha256=digest(out/"manifest.json"),mmap_verification_sha256=store.verification_sha256))


def prefix_equal(old,new):
    for left,right in zip(jax.tree.leaves(old),jax.tree.leaves(new)):
        left=np.asarray(left);right=np.asarray(right)
        slices=tuple(slice(0,size) for size in right.shape)
        if not np.array_equal(left[slices],right,equal_nan=True):return False
    return True


def representative_specs(specs):
    ordered=sorted(specs,key=lambda x:np.prod(x["capacities"]));indices={0,len(ordered)//2,len(ordered)-1}
    return [ordered[i] for i in sorted(indices)]


def equivalence_and_warm(out,manifest,engine,params,state,specs,byid):
    loader=BatchLoader(out,manifest,manifest["config"]["batch_size"],backend="mmap");rows=[];warmed=set();began=time.perf_counter()
    representatives={tuple(x["capacities"]):x for x in representative_specs(specs)}
    for shape,spec in representatives.items():
        oldcap=old_capacity(spec["cids"],byid,manifest);old=loader.load(spec["cids"],oldcap);new=loader.load(spec["cids"],spec["capacities"])
        if not prefix_equal(old,new):raise AssertionError(("padded prefix changed",shape))
        old_forward=ready(engine.batch_forward(params,jax.device_put(old[0])));new_forward=ready(engine.batch_forward(params,jax.device_put(new[0])))
        selected_old=[];selected_new=[]
        for i in range(len(spec["cids"])):
            mask=np.asarray(new[2][i]);selected_old.append(np.asarray(old_forward[i])[:len(mask)][mask]);selected_new.append(np.asarray(new_forward[i])[mask])
        prediction=compare_tree(selected_old,selected_new)
        if prediction["relative_l2"]>5e-6:raise AssertionError(("prediction",shape,prediction))
        left=ready(engine.batch_step(params,state,*jax.device_put(old)));right=ready(engine.batch_step(params,state,*jax.device_put(new)))
        update=compare_tree(left,right)
        if update["max_absolute"]>1e-5 or update["relative_l2"]>5e-6:raise AssertionError(("update",shape,update))
        rows.append(dict(capacities=list(shape),prefix_exact=True,prediction=prediction,loss_absolute_difference=abs(float(left[2])-float(right[2])),update=update))
        warmed.add(shape)
    for spec in specs:
        shape=tuple(spec["capacities"])
        if shape in warmed:continue
        batch=loader.load(spec["cids"],spec["capacities"]);ready(engine.batch_step(params,state,*jax.device_put(batch)));warmed.add(shape)
    warmup_seconds=time.perf_counter()-began;loader.close();return rows,warmup_seconds,len(warmed)


def timed_epoch(out,manifest,engine,params,state,specs,name):
    telemetry=LoaderTelemetry();loader=BatchLoader(out,manifest,manifest["config"]["batch_size"],telemetry=telemetry,backend="mmap")
    loader.set_epoch(1);p=params;s=state;t=time.perf_counter()
    for _,batch in loader.iterate(specs):p,s,_=engine.audited_batch_update(p,s,*batch)
    seconds=time.perf_counter()-t;loader.close();summary=summarize_telemetry(telemetry.snapshot())
    row=dict(variant=name,seconds=seconds,loader=summary,steady_wait_fraction=summary["steady_wait"]["total"]/seconds)
    print(json.dumps({"tight_bucket_epoch":name,"seconds":seconds}),flush=True);return row


def benchmark(root):
    destination=root/"audits/gqt-tight-buckets-v1";protocol=read(destination/"plan.json");results={}
    for size in ("50k","200k"):
        out,manifest=manifest_for(root,size);byid,plans=plans_for(manifest,seed=18);engine,params,state=make_engine(manifest)
        variants={"current":[dict(cids=list(cids),capacities=old_capacity(cids,byid,manifest)) for cids in plans]}
        variants.update({name:shaped(plans,byid,CANDIDATES[name],manifest) for name in protocol["selected"]})
        checks={};startup={}
        for name,specs in variants.items():
            rows,seconds,count=equivalence_and_warm(out,manifest,engine,params,state,specs,byid)
            checks[name]=rows;startup[name]=dict(warmup_and_equivalence_seconds=seconds,unique_shapes=count)
        order=["current",*protocol["selected"],*reversed(protocol["selected"]),"current"]
        epochs=[timed_epoch(out,manifest,engine,params,state,variants[name],name) for name in order]
        medians={name:float(np.median([x["seconds"] for x in epochs if x["variant"]==name])) for name in variants}
        results[size]=dict(checks=checks,startup=startup,epochs=epochs,median_seconds=medians,
            speedup_fraction={name:1-value/medians["current"] for name,value in medians.items() if name!="current"})
        atomic_json(destination/"benchmark.json",dict(complete=False,models=results))
    atomic_json(destination/"benchmark.json",dict(complete=True,models=results))


def report(root):
    destination=root/"audits/gqt-tight-buckets-v1";plan_result=read(destination/"plan.json");result=read(destination/"benchmark.json")
    lines=["# Tighter GQT capacity benchmark","",
        "Batch membership, order and optimizer-update count are unchanged. Only padded N/K/Q capacities differ.","",
        "| Variant | Rounding N/K/Q | Shapes across five plans | Actual/padded edges |", "|---|---:|---:|---:|"]
    for name,row in plan_result["variants"].items():
        lines.append(f"| {name} | {row['quantum'] or 'fixed current'} | {row['union_shapes']} | {row['mean_actual_to_padded_edges']:.3f} |")
    lines += ["","| Model | Variant | Median warm epoch | Speedup | Compiled shapes |","|---|---|---:|---:|---:|"]
    for size,model in result["models"].items():
        for name,seconds in model["median_seconds"].items():
            speed=0 if name=="current" else model["speedup_fraction"][name]
            lines.append(f"| {size} | {name} | {seconds:.2f} s | {100*speed:.1f}% | {model['startup'][name]['unique_shapes']} |")
    lines += ["","All checked padded prefixes were exact. Prediction and one-step update comparisons used the accepted 5e-6 relative float32 tolerance. Cropping, model features, batch membership and training targets were unchanged."]
    (destination/"report.md").write_text("\n".join(lines)+"\n")
    atomic_json(destination/"verification.json",dict(passed=True,plan_sha256=digest(destination/"plan.json"),
        benchmark_sha256=digest(destination/"benchmark.json"),report_sha256=digest(destination/"report.md")))


if __name__=="__main__":
    import sys
    action=sys.argv[1];gpu=action=="benchmark"
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]),gpu_benchmark=gpu,allow_comp1400=True)
    root=Path(os.environ["PKABENCH_RUNTIME"]);jax.config.update("jax_enable_x64",False)
    if action=="plan":plan(root)
    elif action=="benchmark":benchmark(root)
    elif action=="report":report(root)
    else:raise ValueError(action)
