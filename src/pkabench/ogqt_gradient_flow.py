"""Frozen-checkpoint gradient-flow diagnostic; never updates model parameters."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pkanet.model import HEADS, PKPDB_PK_MOD, linear, norm
from pkanet.ogqt import initialize_auxiliary, predict_multi
from pkanet.site_model import _site_bias, attend_sites_indexed
from pkanet.triton_attention import indexed_attention_one
from pkabench.runtime import atomic_json, require_compute
from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine, _diagnostic_plan
from pkatrain.gqt_paired_pinder import Loader, _plans, _prefetched
from pkatrain.trainer import load_checkpoint

VERSION = "gradient-flow-v1"
STAGES = ("embedding", "encoder_1", "encoder_2", "initial_site", "local_query", "final_site")
LOSSES = ("state", "paired", "burial", "interface")
GATES = tuple(f"{block}_{part}" for block in ("encoder_1", "encoder_2", "local_query", "site")
              for part in ("attention", "ff")) + ("repeat_site", "initial_site_skip")


def read(path):
    return json.loads(Path(path).read_text())


def attention(p, x, context, neighbors, edge, mask, switch, gain, active, *, site_types=None, native=False):
    """Same arithmetic/backend as production, with explicit residual gains."""
    width = x.shape[-1]
    q = linear(p["q"], norm(p["norm1"], x)).reshape((-1, HEADS, width // HEADS))
    c = norm(p["norm1"], context)
    k = linear(p["k"], c).reshape((-1, HEADS, width // HEADS))
    v = linear(p["v"], c).reshape((-1, HEADS, width // HEADS))
    bias = (linear(p["edge"], edge) if site_types is None else
            _site_bias(p, edge, site_types, site_types[neighbors]))
    if native:
        logits = jnp.einsum("nhd,nkhd->nkh", q, k[neighbors]) / jnp.sqrt(float(width // HEADS)) + bias
        weights = jax.nn.softmax(jnp.where(mask[..., None], logits, -1e9), axis=1) * mask[..., None] * switch[..., None]
        weights = weights / jnp.maximum(weights.sum(1, keepdims=True), 1e-8)
        message = jnp.einsum("nkh,nkhd->nhd", weights, v[neighbors]).reshape((-1, width))
    else:
        message = indexed_attention_one(q, k, v, neighbors, bias, mask, switch).reshape((-1, width))
    update = linear(p["o"], message)
    middle = x + gain[0] * update
    ff = linear(p["down"], jax.nn.gelu(linear(p["up"], norm(p["norm2"], middle))))
    rms = lambda z: jnp.sqrt(jnp.sum(z*z*active[:, None]) / jnp.maximum(active.sum()*width, 1))
    ratios = jnp.stack((rms(update)/jnp.maximum(rms(x), 1e-12), rms(ff)/jnp.maximum(rms(middle), 1e-12)))
    return (middle + gain[1]*ff)*active[:, None], ratios


def forward(p, g, taps, gates):
    activations = []; ratios = []
    def tap(x, index, mask):
        value = x + taps[index]*mask[:, None]
        activations.append(value)
        return value
    h = tap(linear(p["embed"], g["nodes"]), 0, g["node_mask"])
    for index, block in enumerate(p["blocks"]):
        h, r = attention(block, h, h, g["neighbors"], g["edge"], g["edge_mask"], g["switch"],
                         gates[index*2:index*2+2], g["node_mask"])
        ratios.append(r); h = tap(h, index+1, g["node_mask"])
    sr = g["site_residue"]; sm = g["site_mask"]
    initial = tap((h[sr]+p["groups"][g["site_type"]])*sm[:, None], 3, sm)
    local, r = attention(p["query"], initial, h, g["neighbors"][sr], g["edge"][sr],
                         g["edge_mask"][sr], g["switch"][sr], gates[4:6], sm, native=True)
    ratios.append(r); local = tap(local, 4, sm)
    final, r = attention(p["site"], local, local, g["site_neighbors"], g["site_edge"],
                         g["site_edge_mask"], g["site_switch"], gates[6:8], sm, site_types=g["site_type"])
    ratios.append(r)
    # rho=0 exactly restores the trained architecture. Repeated block shares weights.
    repeat = attend_sites_indexed(p["site"], final, g)-final
    normalized_skip = initial * jax.lax.rsqrt(jnp.mean(initial*initial, -1, keepdims=True)+1e-5)
    final = tap((final+gates[8]*repeat+gates[9]*normalized_skip)*sm[:, None], 5, sm)
    qs = g["query_site"]
    out = {"shift": (8*jnp.tanh(linear(p["head"], final)[:, 0]))[qs]}
    for name in ("burial", "interface"):
        out[name] = jax.nn.sigmoid(linear(p["auxiliary"][name], final)[:, 0])[qs]
    return out, (tuple(activations), jnp.concatenate(ratios))


def zeros(g, width):
    n, s = g["nodes"].shape[1], g["site_mask"].shape[1]
    return tuple(jnp.zeros((2, size, width), jnp.float32) for size in (n,n,n,s,s,s))


def base_gates():
    return jnp.asarray([1.]*8+[0.,0.], jnp.float32)


def predicted(p, g, taps, gates):
    return jax.vmap(forward, in_axes=(None,0,0,None))(p, g, taps, gates)


def loss_vector(out, g, target, mask, burial, interface):
    expected = target-PKPDB_PK_MOD[g["query_group"][0]][None, :]
    count = jnp.maximum(mask.sum(), 1)
    return jnp.stack((jnp.sum((out["shift"]-expected)**2*mask)/(2*count),
        jnp.sum(((out["shift"][0]-out["shift"][1])-(expected[0]-expected[1]))**2*mask)/count,
        jnp.sum((out["burial"][1]-burial)**2*mask)/count,
        jnp.sum((out["interface"][0]-interface)**2*mask)/count))


def losses(p, g, target, mask, burial, interface, taps, gates):
    out, trace = predicted(p,g,taps,gates)
    return loss_vector(out,g,target,mask,burial,interface), trace


@jax.jit
def diagnose(p,g,target,mask,burial,interface):
    taps = zeros(g,p["embed"]["w"].shape[1]); gates = base_gates()
    def objective(t, a): return losses(p,g,target,mask,burial,interface,t,a)
    # The Triton backward accepts scalar VJPs, not jacrev's extra cotangent vmap.
    gradients=[]; gate_gradients=[]
    for loss_index in range(4):
        def scalar(t,a):
            values, trace = objective(t,a)
            return values[loss_index], trace
        (grad, gate_grad), trace = jax.grad(scalar,argnums=(0,1),has_aux=True)(taps,gates)
        gradients.append(grad); gate_gradients.append(gate_grad)
    gradient=jax.tree.map(lambda *x:jnp.stack(x),*gradients)
    gate_gradient=jnp.stack(gate_gradients)
    activations, ratios = trace
    metrics = []
    for index, (value, grad) in enumerate(zip(activations, gradient)):
        active = g["node_mask"] if index < 3 else g["site_mask"]
        denom = jnp.maximum(active.sum()*value.shape[-1],1)
        rms = jnp.sqrt(jnp.sum(value**2*active[...,None])/denom)
        grms = jnp.sqrt(jnp.sum(grad**2*active[None,...,None],axis=(1,2,3))/denom)
        metrics.append(jnp.stack((jnp.repeat(rms,4), grms),axis=1))
    value, _ = objective(taps,gates)
    return value, jnp.stack(metrics), ratios, gate_gradient


def stats(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "p10": float(np.quantile(values,.1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values,.9))}


def verify(p,g,args):
    taps=zeros(g,p["embed"]["w"].shape[1]); gates=base_gates()
    production=jax.jit(lambda x: jax.vmap(predict_multi,in_axes=(None,0))(x,g))
    instrument=jax.jit(lambda x: predicted(x,g,taps,gates)[0])
    a=production(p); b=instrument(p)
    forward_error=max(float(jnp.max(jnp.abs(a[k]-b[k]))) for k in a)
    for k in a: np.testing.assert_allclose(np.asarray(a[k]),np.asarray(b[k]),rtol=2e-5,atol=1e-5)
    def primary(out): return loss_vector(out,g,*args)[:2].sum()
    ga=jax.jit(jax.grad(lambda x: primary(production(x))))(p)
    gb=jax.jit(jax.grad(lambda x: primary(instrument(x))))(p)
    av=np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(ga)])
    bv=np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(gb)])
    relative=float(np.linalg.norm(av-bv)/max(np.linalg.norm(av),1e-12))
    if not np.isfinite(relative) or relative>2e-4: raise AssertionError(("parameter gradient parity",relative))
    f=jax.jit(lambda gate: losses(p,g,*args,taps,gate)[0])
    analytical=np.asarray(jax.jit(lambda a:jnp.stack([jax.grad(lambda x:f(x)[i])(a) for i in range(4)]))(gates)); checks=[]
    for index in (8,9):
        for step in (.01,.003):
            delta=jnp.zeros_like(gates).at[index].set(step)
            fd=np.asarray((f(gates+delta)-f(gates-delta))/(2*step))
            np.testing.assert_allclose(fd,analytical[:,index],rtol=.03,atol=5e-4)
            checks.append({"gate":GATES[index],"step":step,"analytical":analytical[:,index].tolist(),"finite_difference":fd.tolist()})
    # Masked taps on padded residues/sites must not affect predictions or gradients.
    padding=tuple(jnp.broadcast_to((~(g["node_mask"] if i<3 else g["site_mask"]))[...,None],x.shape).astype(jnp.float32)*3 for i,x in enumerate(taps))
    padded=jax.jit(lambda: predicted(p,g,padding,gates)[0])()
    for k in b: np.testing.assert_allclose(np.asarray(b[k]),np.asarray(padded[k]),rtol=0,atol=1e-6)
    return {"passed":True,"max_prediction_difference":forward_error,"parameter_gradient_relative_difference":relative,"finite_difference":checks,"padding_invariance":True}


def run(runtime):
    require_compute(threads=8,gpu_benchmark=True,allow_comp1400=True)
    started=time.monotonic(); base=Path(runtime)/"training/ogqt-auxiliary-pilot-v1"; manifest=read(base/"manifest.json")
    output=base/VERSION; output.mkdir(exist_ok=True)
    p=initialize_auxiliary(jax.random.PRNGKey(17),**manifest["architecture"])
    engine=AuxiliaryEngine(p,0,{"burial":1,"interface":1})
    selected=read(base/"standard/seed-17/verification.json")["selected_epoch"]
    checkpoint=base/f"standard/seed-17/checkpoints/epoch-{selected:03d}"
    p,_,_=load_checkpoint(checkpoint,(p,engine.optimizer.init(p)))
    plans={"train":_diagnostic_plan([r for r in manifest["records"] if r["split"]=="train"],manifest), "val":_plans([r for r in manifest["records"] if r["split"]=="val"],np.random.default_rng(17017),manifest)[:8]}
    atomic_json(output/"protocol.json",{"checkpoint":str(checkpoint),"plans":plans,"plan_sha256":hashlib.sha256(json.dumps(plans,sort_keys=True).encode()).hexdigest(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"test_data_included":False,"losses":LOSSES,"gates":GATES,"stages":STAGES,"normalization":"per-complex loss; RMS excludes padding; equal-complex summaries","note":"frozen local derivatives, not evidence of retraining improvement"})
    loader=Loader(base,manifest); rows=[]; verified=False
    for split, plan in plans.items():
        for number,(ids,batch) in enumerate(_prefetched(loader,plan),1):
            targets,mask,burial,interface=engine.targets(batch,manifest["normalization"])
            for index,cid in enumerate(ids):
                g=jax.tree.map(lambda x:jnp.asarray(x[index]),batch[0])
                args=tuple(jnp.asarray(x[index]) for x in (targets,mask,burial,interface))
                if not verified:
                    atomic_json(output/"verification.json",verify(p,g,args)); verified=True
                    print(json.dumps({"verification":"passed","elapsed_s":time.monotonic()-started}),flush=True)
                value,flow,ratios,gates=jax.device_get(diagnose(p,g,*args))
                if not all(np.isfinite(x).all() for x in (value,flow,ratios,gates)): raise FloatingPointError(cid)
                rows.append({"id":cid,"split":split,"loss":value.tolist(),"flow":flow.tolist(),"residual_ratios":ratios.tolist(),"gate_derivatives":gates.tolist()})
            atomic_json(output/"progress.json",{"split":split,"batch":number,"batches":len(plan),"complexes":len(rows),"elapsed_s":time.monotonic()-started})
            print(json.dumps(read(output/"progress.json")),flush=True)
    loader.close(); atomic_json(output/"per-complex.json",rows)
    result={}
    for split in plans:
        rr=[r for r in rows if r["split"]==split]; flow=np.asarray([r["flow"] for r in rr]); gates=np.asarray([r["gate_derivatives"] for r in rr]); ratios=np.asarray([r["residual_ratios"] for r in rr])
        result[split]={"complexes":len(rr),"flow":{stage:{loss:{"activation_rms":stats(flow[:,s,l,0]),"gradient_rms":stats(flow[:,s,l,1])} for l,loss in enumerate(LOSSES)} for s,stage in enumerate(STAGES)},
            "gate_derivatives":{name:{loss:{**stats(gates[:,l,k]),"negative_fraction":float(np.mean(gates[:,l,k]<0))} for l,loss in enumerate(LOSSES)} for k,name in enumerate(GATES)},
            "primary_gate_derivatives":{name:{**stats(gates[:,0,k]+gates[:,1,k]),"negative_fraction":float(np.mean(gates[:,0,k]+gates[:,1,k]<0))} for k,name in enumerate(GATES)},
            "residual_ratios":{name:stats(ratios[:,:,k].mean(1)) for k,name in enumerate(GATES[:8])}}
    atomic_json(output/"summary.json",{"results":result,"elapsed_s":time.monotonic()-started,"checkpoint":str(checkpoint),"test_data_included":False})


if __name__=="__main__": run(os.environ["PKABENCH_RUNTIME"])
