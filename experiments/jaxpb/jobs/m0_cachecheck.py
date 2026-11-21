"""M0 cache-tarball check incl. Pallas: phase 'write' compiles an XLA jit and a Pallas (Triton) kernel into an empty
node-local cache and tars it; phase 'read' (a new process) untars it and runs the same programs: real compiles must be 0
and both executables must be cache hits. usage: python m0_cachecheck.py <write|read> <tar> <out.json>"""
import sys, json, time
import numpy as np
from jaxpb.compile_budget import CompileCounter, cache_open, cache_close, cache_clear_local
phase, tar, out = sys.argv[1:4]
import os
if phase == "write" and os.path.exists(tar):
    os.remove(tar)
info = cache_open(tar if phase == "read" else None, local_dir=f"/tmp/jaxpb-cc-{os.environ.get('SLURM_JOB_ID','x')}-{phase}")
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

res = {"phase": phase, "jax": jax.__version__, "device": str(jax.devices()[0]), "cache_open": info}


def kern(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0 + y_ref[...]


def make_pallas():
    kw = dict(out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32))
    gpu = jax.devices()[0].platform == "gpu"
    # JAX 0.11 pallas_call has no `backend` kwarg and lowers to Mosaic GPU by default (fails pre-Hopper; on a GH200 it
    # would silently test Mosaic, not Triton): select the Triton lowering through its CompilerParams, as titrate.py does.
    if gpu:
        from jax.experimental.pallas import triton as pltriton
        cls = getattr(pltriton, "CompilerParams", None) or getattr(pltriton, "TritonCompilerParams")
    for extra in (({"compiler_params": cls(num_warps=1, num_stages=1)},) if gpu else ({"interpret": True},)):
        try:
            f = pl.pallas_call(kern, **kw, **extra)
            jax.jit(f).lower(jnp.ones(1024, jnp.float32), jnp.ones(1024, jnp.float32))
            return f, extra
        except Exception as e:
            res.setdefault("pallas_errors", []).append(repr(e)[:300])
    print(json.dumps(res, indent=1, default=str))
    raise RuntimeError("pallas unavailable")


@jax.jit
def xla_prog(x):
    return jnp.cumsum(jnp.sin(x) ** 2) @ jnp.ones_like(x)


pf, pextra = make_pallas()
res["pallas_kwargs"] = repr(pextra)
pal_prog = jax.jit(lambda a, b: pf(a, b) * 3.0)
x = np.linspace(0, 1, 1024, dtype=np.float32)
with CompileCounter() as cc:
    a = jax.device_put(x)
    r1 = float(xla_prog(a)); r2 = np.asarray(pal_prog(a, a))
res["compile"] = cc.summary()
res["values"] = [r1, float(r2.sum())]
hlo = jax.jit(lambda a, b: pf(a, b) * 3.0).lower(a, a).as_text()
res["pallas_custom_call_in_hlo"] = [t for t in ("__gpu$xla.gpu.triton", "triton_kernel_call", "mosaic_gpu", "custom_call") if t in hlo]
if phase == "write":
    res["cache_close"] = cache_close(tar, info["local_dir"])
else:
    res["pass"] = bool(cc.n_compiled == 0 and cc.n_cache_hits >= 2 and any("pal" in n or "lambda" in n for n in cc.hit_names))
cache_clear_local(info["local_dir"])
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
