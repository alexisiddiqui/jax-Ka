"""Build and benchmark the exact-array mmap backend against compressed NPZ."""
import json
import os
from pathlib import Path
import time

import jax
import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.graph_batches import BatchLoader, LoaderTelemetry
from pkatrain.graph_mmap import build_bundle
from pkatrain.records import read
from profile_gqt_fullgraph import (
    MODELS, make_engine, manifest_for, plans_for, ready, summarize_telemetry,
)


def convert(root):
    out, manifest = manifest_for(root, "50k")
    began = time.perf_counter(); destination = build_bundle(out, manifest)
    receipt = read(destination / "verification.json")
    atomic_json(root / "audits/gqt-fullgraph-profile-v1/mmap-conversion.json", dict(
        passed=True, destination=str(destination), seconds=time.perf_counter() - began,
        bytes=receipt["bytes"], records=len(manifest["records"]),
        all_records_and_fields_bit_identical=receipt["all_records_and_fields_bit_identical"],
        verification_sha256=digest(destination / "verification.json")))


def compare_tree(left, right):
    leaves = list(zip(jax.tree.leaves(left), jax.tree.leaves(right)))
    exact = all(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) for a, b in leaves)
    largest = 0.; squared_delta = 0.; squared_scale = 0.
    for a, b in leaves:
        a = np.asarray(a); b = np.asarray(b)
        if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):continue
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            av=a[finite].astype(np.float64);bv=b[finite].astype(np.float64);delta=av-bv
            largest=max(largest,float(np.max(np.abs(delta))))
            squared_delta+=float(np.sum(delta**2));squared_scale+=max(float(np.sum(av**2)),float(np.sum(bv**2)))
    return dict(exact=exact,max_absolute=largest,
        relative_l2=float(np.sqrt(squared_delta/max(squared_scale,1e-60))))


def equivalence(out, manifest, engine, params, state):
    byid, plans = plans_for(manifest); rows=[]
    for name in sorted(manifest["capacities"], key=int):
        members = next(batch for batch in plans if str(384 if byid[batch[0]]["n"] <= 384 else 768 if byid[batch[0]]["n"] <= 768 else 100000) == name)
        native = BatchLoader(out, manifest, manifest["config"]["batch_size"], backend="npz")
        mapped = BatchLoader(out, manifest, manifest["config"]["batch_size"], backend="mmap")
        native.set_epoch(1); mapped.set_epoch(1)
        a = native.load(members); b = mapped.load(members)
        native.close(); mapped.close()
        host = compare_tree(a, b)
        if not host["exact"]:raise AssertionError((name, host))
        forward_left=ready(engine.batch_forward(params,jax.device_put(a[0])))
        forward_right=ready(engine.batch_forward(params,jax.device_put(b[0])))
        forward=compare_tree(forward_left,forward_right)
        if not forward["exact"]:raise AssertionError((name,"forward",forward))
        left = ready(engine.batch_step(params, state, *jax.device_put(a)))
        repeated = ready(engine.batch_step(params, state, *jax.device_put(a)))
        right = ready(engine.batch_step(params, state, *jax.device_put(b)))
        device = compare_tree(left, right)
        repeat_floor=compare_tree(left,repeated)
        if not np.array_equal(np.asarray(left[2]),np.asarray(right[2])):
            raise AssertionError((name,"loss changed",left[2],right[2]))
        # GPU gradient reductions are nondeterministic even when the same
        # device batch is repeated. This gate is no looser than the accepted
        # float32-vs-float64 gradient comparison (4.9e-6 relative).
        if device["max_absolute"]>1e-5 or device["relative_l2"]>5e-6:
            raise AssertionError((name,"above float32 gradient gate",device,repeat_floor))
        rows.append(dict(bucket=name,complex_ids=members,host=host,forward=forward,
            loss_exact=True,fused_update=device,same_input_gpu_repeat=repeat_floor,
            allowed_max_absolute=1e-5,allowed_relative_l2=5e-6))
    return rows


def one_epoch(out, manifest, engine, initial_params, initial_state, backend):
    _, plans = plans_for(manifest, seed=18); telemetry = LoaderTelemetry()
    loader = BatchLoader(out, manifest, manifest["config"]["batch_size"], telemetry=telemetry, backend=backend)
    loader.set_epoch(1); params=initial_params; state=initial_state
    began=time.perf_counter()
    for _, batch in loader.iterate(plans):
        params,state,_=engine.audited_batch_update(params,state,*batch)
    seconds=time.perf_counter()-began;loader.close()
    row=dict(backend=backend,seconds=seconds,loader=summarize_telemetry(telemetry.snapshot()),
        parameter_checksum=float(sum(np.asarray(x,dtype=np.float64).sum() for x in jax.tree.leaves(params))))
    print(json.dumps({"mmap_epoch":backend,"seconds":seconds,"parameter_checksum":row["parameter_checksum"]}),flush=True)
    return row


def benchmark(root):
    destination=root/"audits/gqt-fullgraph-profile-v1"; profile=read(destination/"profile.json")
    results={}
    for size in ("50k","200k"):
        out,manifest=manifest_for(root,size);engine,params,state=make_engine(manifest)
        checks=equivalence(out,manifest,engine,params,state)
        # ABBA order controls for filesystem cache and run-order effects.
        epochs=[]
        for name in ("npz","mmap","mmap","npz"):
            epochs.append(one_epoch(out,manifest,engine,params,state,name))
            atomic_json(destination/"mmap-benchmark-progress.json",dict(model=size,epochs=epochs))
        checksums=np.asarray([row["parameter_checksum"] for row in epochs])
        checksum_relative_spread=float(np.ptp(checksums)/max(np.max(np.abs(checksums)),1e-30))
        medians={name:float(np.median([row["seconds"] for row in epochs if row["backend"]==name])) for name in ("npz","mmap")}
        validation=float(profile["models"][size]["validation"]["normal_pipeline_seconds"])
        speedup=1-medians["mmap"]/medians["npz"]
        total_speedup=1-(medians["mmap"]+validation)/(medians["npz"]+validation)
        results[size]=dict(equivalence=checks,epochs=epochs,median_seconds=medians,
            train_speedup_fraction=speedup,end_to_end_speedup_fraction=total_speedup,
            validation_seconds_unchanged=validation,parameter_checksum_relative_spread=checksum_relative_spread)
        atomic_json(destination/"mmap-benchmark.json",dict(complete=False,models=results))
    passed=all(row["end_to_end_speedup_fraction"]>=.10 for row in results.values())
    atomic_json(destination/"mmap-benchmark.json",dict(complete=True,adoption_gate_passed=passed,
        gate="at least 10% train-plus-validation epoch speedup for both 50k and 200k",models=results))


def report(root):
    destination=root/"audits/gqt-fullgraph-profile-v1"
    conversion=read(destination/"mmap-conversion.json"); result=read(destination/"mmap-benchmark.json")
    lines=["# Exact-array mmap A/B benchmark","",
        f"Converted {conversion['records']:,} structures ({conversion['bytes']/2**30:.1f} GiB) in {conversion['seconds']/60:.1f} minutes. Every field of every structure was checked bit-for-bit against its source NPZ.","",
        "| Model | NPZ epoch | mmap epoch | Train speedup | Train + validation speedup |", "|---|---:|---:|---:|---:|"]
    for size,row in result["models"].items():
        lines.append(f"| {size} | {row['median_seconds']['npz']:.2f} s | {row['median_seconds']['mmap']:.2f} s | {100*row['train_speedup_fraction']:.1f}% | {100*row['end_to_end_speedup_fraction']:.1f}% |")
    lines += ["",f"Adoption gate: **{'passed' if result['adoption_gate_passed'] else 'not passed'}**. The gate requires at least 10% train-plus-validation epoch improvement for both model sizes.",
        "Every raw array in all structures was bit-identical. Representative padded host batches, model forwards and scalar losses were also bit-identical in every capacity bucket. Gradient-derived one-step updates stayed below 1e-5 absolute and 5e-6 relative L2, matching the accepted float32 gradient gate. Full-epoch parameter checksums are diagnostic only, because repeated runs of the same backend diverge as nondeterministic CUDA reductions compound through Adam. The mmap arrays remain read-only, and augmentation copies only fields it mutates before padding.",
        "Cropping and graph-shape changes were not introduced."]
    (destination/"mmap-report.md").write_text("\n".join(lines)+"\n")
    atomic_json(destination/"mmap-verification.json",dict(passed=True,
        adoption_gate_passed=result["adoption_gate_passed"],conversion_sha256=digest(destination/"mmap-conversion.json"),
        benchmark_sha256=digest(destination/"mmap-benchmark.json"),report_sha256=digest(destination/"mmap-report.md")))


if __name__=="__main__":
    import sys
    action=sys.argv[1];gpu=action=="benchmark"
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]),gpu_benchmark=gpu,allow_comp1400=True)
    root=Path(os.environ["PKABENCH_RUNTIME"]);jax.config.update("jax_enable_x64",False)
    if action=="convert":convert(root)
    elif action=="benchmark":benchmark(root)
    elif action=="report":report(root)
    else:raise ValueError(action)
