"""A40 equivalence and throughput gate on real raw-arm graphs in every bucket."""
import json
import os
from pathlib import Path
import time
import jax
import numpy as np
from pkanet.model import initialize,predict
from pkatrain.trainer import ScalarEngine
from pkatrain.graph_batches import BatchLoader
from pkatrain.graph_data import bucket
from pkatrain.records import read
from pkabench.runtime import require_compute,atomic_json


def ready(tree):
    return jax.tree.map(lambda x:x.block_until_ready(),tree)


def main():
    require_compute(threads=8,gpu_benchmark=True)
    root=Path(os.environ['PKABENCH_RUNTIME'])/'pretraining/pkpdb-5k-comparison-v1'
    out=root/'gqt-raw';m=read(out/'manifest.json');loader=BatchLoader(out,m,8)
    params=initialize(jax.random.PRNGKey(17),**m['config']['architecture']);engine=ScalarEngine(predict)
    rows=[]
    for b in sorted(m['capacities'],key=int):
        members=[r for r in m['records'] if r['split']=='train' and bucket(r)==b]
        # Largest eight in each bucket exercise its memory and padded capacities.
        members=sorted(members,key=lambda r:r['n'],reverse=True)[:8]
        batch=loader.load([r['complex_id'] for r in members]);g,y,mask,valid=batch
        t=time.monotonic();loss,grad=ready(engine.batch_value_grad(params,*batch));compile_seconds=time.monotonic()-t
        serial=[];serial_losses=[]
        for i in range(len(members)):
            value,gradient=ready(engine.value_grad(params,jax.tree.map(lambda x:x[i],g),y[i],mask[i]))
            serial.append(gradient);serial_losses.append(float(value))
        mean=jax.tree.map(lambda *xs:np.mean(np.stack(xs),axis=0),*serial)
        diff=np.sqrt(sum(np.sum((np.asarray(a)-np.asarray(b))**2) for a,b in zip(jax.tree.leaves(grad),jax.tree.leaves(mean))))
        norm=np.sqrt(sum(np.sum(np.asarray(a)**2) for a in jax.tree.leaves(mean)))
        relative=float(diff/max(norm,1e-12));assert relative<2e-4,relative
        np.testing.assert_allclose(loss,np.mean(serial_losses),rtol=2e-5,atol=2e-5)
        state=engine.optimizer.init(params);ready(engine.batch_step(params,state,*batch))
        batch_times=[];serial_times=[]
        for _ in range(3):
            t=time.monotonic();ready(engine.batch_step(params,state,*batch));batch_times.append(time.monotonic()-t)
            t=time.monotonic()
            gradients=[engine.value_grad(params,jax.tree.map(lambda x:x[i],g),y[i],mask[i])[1] for i in range(len(members))]
            ready(engine.update(params,state,gradients));serial_times.append(time.monotonic()-t)
        result=dict(bucket=b,residues=[r['n'] for r in members],capacities=m['capacities'][b],
            batch_size=len(members),compile_seconds=compile_seconds,gradient_relative_difference=relative,
            batched_step_seconds=float(np.median(batch_times)),serial_eight_seconds=float(np.median(serial_times)),
            speedup=float(np.median(serial_times)/np.median(batch_times)),
            matmul_precision=str(jax.config.jax_default_matmul_precision),gpu_memory=jax.devices()[0].memory_stats())
        rows.append(result);print(json.dumps(result),flush=True)
        atomic_json(root/'batch-probe.json',dict(passed=len(rows)==len(m['capacities']),rows=rows))
    loader.close()


if __name__=='__main__':main()
