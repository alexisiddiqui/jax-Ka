"""End-to-end 200k GQT training-step benchmark for indexed Triton attention."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from pkanet import model
from pkanet.model import PKPDB_PK_MOD, initialize, predict_shift_indexed
from pkabench.runtime import atomic_json, require_compute
from pkatrain.graph_batches import BatchLoader
from pkatrain.graph_pkmod_compare import BINS
from pkatrain.records import read
from profile_gqt_fullgraph import plans_for, ready
from profile_gqt_tight_buckets import CANDIDATES, shaped


MODEL_PATHS={
    "50k":"pretraining/gqt-backbone-5k-pkmod-v1/unweighted",
    "200k":"pretraining/gqt-pkai-parameter-sweep-v1/gqt/200k/unweighted",
    "800k":"pretraining/gqt-pkai-parameter-sweep-v1/gqt/800k/unweighted",
}


def make_loss(predict, bin_weights):
    baseline = jnp.asarray(PKPDB_PK_MOD, jnp.float32)
    bins = jnp.asarray(BINS)
    weights = jnp.asarray(bin_weights, jnp.float32)

    def one(params, graph, target, eligible):
        target_shift = target - baseline[graph["query_group"]]
        predicted = predict(params, graph)
        index = jnp.sum(jnp.abs(target_shift)[..., None] >= bins, axis=-1)
        weight = jnp.where(eligible, weights[index], 0.0)
        error = jnp.where(eligible, predicted - target_shift, 0.0)
        return jnp.sum(weight * error**2) / jnp.maximum(jnp.sum(weight), 1e-8)

    def batch(params, graphs, target, eligible, valid):
        losses = jax.vmap(one, in_axes=(None, 0, 0, 0))(
            params, graphs, target, eligible)
        return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(valid.sum(), 1)
    return batch


def tree_relative(left, right):
    numerator = denominator = maximum = 0.0
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        numerator += float(np.sum((a-b)**2)); denominator += float(np.sum(a**2))
        maximum = max(maximum, float(np.max(np.abs(a-b))))
    return dict(relative_l2=float(np.sqrt(numerator/max(denominator,1e-30))),max_absolute=maximum)


def timed(compiled, args, repeats):
    ready(compiled(*args));values=[]
    for _ in range(repeats):
        start=time.perf_counter();ready(compiled(*args));values.append(time.perf_counter()-start)
    return dict(median=float(np.median(values)),minimum=float(np.min(values)),
                p95=float(np.quantile(values,.95)),repeats=repeats)


def memory(compiled):
    value=compiled.memory_analysis()
    return {key:int(getattr(value,key)) for key in (
        "argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")}


def benchmark_case(out, manifest, params, spec):
    loader=BatchLoader(out,manifest,manifest["config"]["batch_size"],backend="mmap")
    batch=tuple(jax.device_put(x) for x in loader.load(spec["cids"],spec["capacities"]));loader.close()
    graphs,target,eligible,valid=batch
    native_loss=make_loss(model.predict_shift,manifest["config"]["shift_bin_weights"])
    triton_loss=make_loss(predict_shift_indexed,manifest["config"]["shift_bin_weights"])
    native=jax.jit(jax.value_and_grad(native_loss));custom=jax.jit(jax.value_and_grad(triton_loss))
    args=(params,graphs,target,eligible,valid)
    native_value=ready(native(*args));custom_value=ready(custom(*args))
    native_compiled=native.lower(*args).compile();custom_compiled=custom.lower(*args).compile()
    return dict(capacities=spec["capacities"],complex_ids=spec["cids"],
        loss=dict(native=float(native_value[0]),triton=float(custom_value[0]),
                  absolute_difference=abs(float(native_value[0])-float(custom_value[0]))),
        gradient=tree_relative(native_value[1],custom_value[1]),
        timing=dict(native=timed(native_compiled,args,20),triton=timed(custom_compiled,args,20)),
        memory=dict(native=memory(native_compiled),triton=memory(custom_compiled)))


def main():
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]),gpu_benchmark=True,allow_comp1400=True)
    root=Path(os.environ["PKABENCH_RUNTIME"]);destination=root/"audits/gqt-triton-attention-v1"
    size=os.environ.get("GQT_SIZE","200k");out=root/MODEL_PATHS[size];manifest=read(out/"manifest.json")
    targets=((384,192),(512,256));selected=[]
    for n,k in targets:
        matches=[]
        for seed in range(18,23):
            byid,plans=plans_for(manifest,seed=seed)
            specs=shaped(plans,byid,CANDIDATES["n128_k64"],manifest)
            matches=[x for x in specs if tuple(x["capacities"][:2])==(n,k)]
            if matches:break
        if not matches:raise RuntimeError(f"No real batch with capacity {(n,k)}")
        selected.append(matches[0])
    cfg=manifest["config"];params=initialize(jax.random.PRNGKey(cfg["seed"]),**cfg["architecture"])
    rows=[]
    for spec in selected:
        rows.append(benchmark_case(out,manifest,params,spec))
        atomic_json(destination/"fullstep-progress.json",dict(complete=False,cases=rows))
    result=dict(complete=True,model=size,parameters=cfg["parameter_count"],batch_size=cfg["batch_size"],
        backend_scope="two same-node-set encoder attention blocks; native query attention",
        gates=dict(loss_absolute=1e-5,gradient_relative_l2=2e-4,faster_training_step=True),cases=rows)
    atomic_json(destination/f"fullstep-{size}.json",result)
    if size=="200k":atomic_json(destination/"fullstep.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
