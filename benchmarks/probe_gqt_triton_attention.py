"""CUDA-only prototype for gathered-neighbour GQT attention.

The Triton path keeps K/V once per node and gathers them inside the kernel.
It deliberately covers only the attention message; projections, residuals,
dropout and the feed-forward block remain ordinary JAX operations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
import numpy as np
import jax_triton as jt
import triton
import triton.language as tl

from pkabench.runtime import atomic_json, require_compute


EPS = 1e-8
WARPS = int(os.environ.get("TRITON_WARPS", "4"))
VARIANT = os.environ.get("TRITON_VARIANT", f"warps{WARPS}")


@triton.jit
def _indexed_attention_fwd(
    q_ptr, k_ptr, v_ptr, neighbors_ptr, bias_ptr, mask_ptr, switch_ptr,
    out_ptr,
    B: tl.constexpr, Q: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr, K: tl.constexpr,
    SCALE: tl.constexpr, EPSILON: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
):
    bn = tl.program_id(0)
    head = tl.program_id(1)
    batch = bn // Q
    node = bn - batch * Q
    kk = tl.arange(0, BLOCK_K)
    dd = tl.arange(0, BLOCK_D)
    k_valid = kk < K
    d_valid = dd < D

    q_off = ((batch * Q + node) * H + head) * D + dd
    q = tl.load(q_ptr + q_off, mask=d_valid, other=0.0)
    edge_off = (batch * Q + node) * K + kk
    neighbor = tl.load(neighbors_ptr + edge_off, mask=k_valid, other=0)
    edge_mask = tl.load(mask_ptr + edge_off, mask=k_valid, other=0).to(tl.int1)
    switch = tl.load(switch_ptr + edge_off, mask=k_valid, other=0.0)
    bias_off = edge_off * H + head
    bias = tl.load(bias_ptr + bias_off, mask=k_valid, other=0.0)

    kv_off = (((batch * N + neighbor[:, None]) * H + head) * D
              + dd[None, :])
    load_mask = k_valid[:, None] & d_valid[None, :]
    key = tl.load(k_ptr + kv_off, mask=load_mask, other=0.0)
    value = tl.load(v_ptr + kv_off, mask=load_mask, other=0.0)
    logits = tl.sum(key * q[None, :], axis=1) * SCALE + bias
    logits = tl.where(edge_mask, logits, -1.0e9)

    maximum = tl.max(tl.where(k_valid, logits, -float("inf")), axis=0)
    exponent = tl.where(k_valid, tl.exp(logits - maximum), 0.0)
    softmax = exponent / tl.sum(exponent, axis=0)
    weighted = softmax * switch * edge_mask.to(tl.float32)
    denominator = tl.maximum(tl.sum(weighted, axis=0), EPSILON)
    weights = weighted / denominator
    message = tl.sum(weights[:, None] * value, axis=0)
    tl.store(out_ptr + q_off, message, mask=d_valid)


@triton.jit
def _indexed_attention_bwd(
    q_ptr, k_ptr, v_ptr, neighbors_ptr, bias_ptr, mask_ptr, switch_ptr,
    gout_ptr,
    dq_ptr, dk_ptr, dv_ptr, dbias_ptr,
    B: tl.constexpr, Q: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr, K: tl.constexpr,
    SCALE: tl.constexpr, EPSILON: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
):
    bn = tl.program_id(0)
    head = tl.program_id(1)
    batch = bn // Q
    node = bn - batch * Q
    kk = tl.arange(0, BLOCK_K)
    dd = tl.arange(0, BLOCK_D)
    k_valid = kk < K
    d_valid = dd < D

    q_off = ((batch * Q + node) * H + head) * D + dd
    q = tl.load(q_ptr + q_off, mask=d_valid, other=0.0)
    gout = tl.load(gout_ptr + q_off, mask=d_valid, other=0.0)
    edge_off = (batch * Q + node) * K + kk
    neighbor = tl.load(neighbors_ptr + edge_off, mask=k_valid, other=0)
    edge_mask = tl.load(mask_ptr + edge_off, mask=k_valid, other=0).to(tl.int1)
    switch = tl.load(switch_ptr + edge_off, mask=k_valid, other=0.0)
    bias_off = edge_off * H + head
    bias = tl.load(bias_ptr + bias_off, mask=k_valid, other=0.0)

    kv_off = (((batch * N + neighbor[:, None]) * H + head) * D
              + dd[None, :])
    load_mask = k_valid[:, None] & d_valid[None, :]
    key = tl.load(k_ptr + kv_off, mask=load_mask, other=0.0)
    value = tl.load(v_ptr + kv_off, mask=load_mask, other=0.0)
    logits = tl.sum(key * q[None, :], axis=1) * SCALE + bias
    logits = tl.where(edge_mask, logits, -1.0e9)
    maximum = tl.max(tl.where(k_valid, logits, -float("inf")), axis=0)
    exponent = tl.where(k_valid, tl.exp(logits - maximum), 0.0)
    softmax = exponent / tl.sum(exponent, axis=0)
    weighted = softmax * switch * edge_mask.to(tl.float32)
    z = tl.sum(weighted, axis=0)
    denominator = tl.maximum(z, EPSILON)
    weights = weighted / denominator

    grad_weight = tl.sum(value * gout[None, :], axis=1)
    centered = tl.sum(grad_weight * weights, axis=0)
    grad_weighted = tl.where(z > EPSILON,
                             (grad_weight - centered) / z,
                             grad_weight / EPSILON)
    grad_softmax = grad_weighted * switch * edge_mask.to(tl.float32)
    softmax_center = tl.sum(grad_softmax * softmax, axis=0)
    grad_logits = softmax * (grad_softmax - softmax_center)
    grad_logits = tl.where(edge_mask & k_valid, grad_logits, 0.0)

    grad_q = tl.sum(grad_logits[:, None] * key, axis=0) * SCALE
    tl.store(dq_ptr + q_off, grad_q, mask=d_valid)
    tl.store(dbias_ptr + bias_off, grad_logits, mask=k_valid)
    tl.atomic_add(dk_ptr + kv_off, grad_logits[:, None] * q[None, :] * SCALE,
                  mask=load_mask)
    tl.atomic_add(dv_ptr + kv_off, weights[:, None] * gout[None, :],
                  mask=load_mask)


def _forward(q, k, v, neighbors, bias, mask, switch):
    b, queries, h, d = q.shape
    n = k.shape[1]
    assert k.shape == v.shape == (b, n, h, d)
    assert neighbors.shape[:2] == (b, queries)
    slots = neighbors.shape[-1]
    return jt.triton_call(
        q, k, v, neighbors, bias, mask, switch,
        kernel=_indexed_attention_fwd,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(b * queries, h), name="gqt_indexed_attention_fwd",
        B=b, Q=queries, N=n, H=h, D=d, K=slots,
        SCALE=float(1 / np.sqrt(d)),
        EPSILON=EPS,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_K=triton.next_power_of_2(slots),
        num_warps=WARPS, enable_fp_fusion=False,
    )


@jax.custom_vjp
def indexed_attention(q, k, v, neighbors, bias, mask, switch):
    return _forward(q, k, v, neighbors, bias, mask, switch)


def _fwd(q, k, v, neighbors, bias, mask, switch):
    out = _forward(q, k, v, neighbors, bias, mask, switch)
    return out, (q, k, v, neighbors, bias, mask, switch)


def _backward(q, k, v, neighbors, bias, mask, switch, gout):
    b, queries, h, d = q.shape
    n = k.shape[1]
    slots = neighbors.shape[-1]
    shapes = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(bias.shape, bias.dtype),
    )
    dq, dk, dv, dbias = jt.triton_call(
        q, k, v, neighbors, bias, mask, switch, gout,
        kernel=_indexed_attention_bwd,
        out_shape=shapes, grid=(b * queries, h),
        name="gqt_indexed_attention_bwd",
        B=b, Q=queries, N=n, H=h, D=d, K=slots,
        SCALE=float(1 / np.sqrt(d)),
        EPSILON=EPS,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_K=triton.next_power_of_2(slots),
        num_warps=WARPS, zeroed_outputs=(1, 2), enable_fp_fusion=False,
    )
    return dq, dk, dv, dbias


def _bwd(residual, gout):
    q, k, v, neighbors, bias, mask, switch = residual
    dq, dk, dv, dbias = _backward(
        q, k, v, neighbors, bias, mask, switch, gout)
    return dq, dk, dv, None, dbias, None, None


indexed_attention.defvjp(_fwd, _bwd)


@jax.custom_vjp
@custom_vmap
def indexed_attention_one(q, k, v, neighbors, bias, mask, switch):
    """Unbatched model boundary with one batched Triton call under ``vmap``."""
    return indexed_attention(
        q[None], k[None], v[None], neighbors[None], bias[None],
        mask[None], switch[None])[0]


@indexed_attention_one.def_vmap
def _indexed_attention_one_vmap(axis_size, in_batched, q, k, v, neighbors,
                                bias, mask, switch):
    del axis_size
    if not all(in_batched):
        raise ValueError("GQT Triton attention expects every graph tensor to be batched")
    return indexed_attention(q, k, v, neighbors, bias, mask, switch), True


def _one_fwd(q, k, v, neighbors, bias, mask, switch):
    out = indexed_attention_one(q, k, v, neighbors, bias, mask, switch)
    return out, (q, k, v, neighbors, bias, mask, switch)


@custom_vmap
def _backward_one(q, k, v, neighbors, bias, mask, switch, gout):
    return tuple(x[0] for x in _backward(
        q[None], k[None], v[None], neighbors[None], bias[None],
        mask[None], switch[None], gout[None]))


@_backward_one.def_vmap
def _backward_one_vmap(axis_size, in_batched, q, k, v, neighbors, bias,
                       mask, switch, gout):
    del axis_size
    if not all(in_batched):
        raise ValueError("GQT Triton backward expects every graph tensor to be batched")
    return _backward(q, k, v, neighbors, bias, mask, switch, gout), (True,) * 4


def _one_bwd(residual, gout):
    q, k, v, neighbors, bias, mask, switch = residual
    dq, dk, dv, dbias = _backward_one(
        q, k, v, neighbors, bias, mask, switch, gout)
    return dq, dk, dv, None, dbias, None, None


indexed_attention_one.defvjp(_one_fwd, _one_bwd)


def native_attention(q, k, v, neighbors, bias, mask, switch):
    """Reference with the exact operations used by pkanet.model.attend."""
    b = jnp.arange(q.shape[0])[:, None, None]
    gathered_k = k[b, neighbors]
    gathered_v = v[b, neighbors]
    logits = jnp.einsum("bnhd,bnkhd->bnkh", q, gathered_k)
    logits = logits / jnp.sqrt(float(q.shape[-1])) + bias
    logits = jnp.where(mask[..., None], logits, -1e9)
    weights = jax.nn.softmax(logits, axis=2) * mask[..., None] * switch[..., None]
    weights = weights / jnp.maximum(weights.sum(axis=2, keepdims=True), EPS)
    return jnp.einsum("bnkh,bnkhd->bnhd", weights, gathered_v)


def relative(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def ready(x):
    return jax.tree.map(lambda y: y.block_until_ready(), x)


def seconds(fn, args, repeats=20):
    ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return dict(median=float(np.median(samples)), minimum=float(np.min(samples)),
                p95=float(np.quantile(samples, .95)), repeats=repeats)


def memory(compiled):
    x = compiled.memory_analysis()
    return {name: int(getattr(x, name)) for name in (
        "argument_size_in_bytes", "output_size_in_bytes",
        "temp_size_in_bytes", "alias_size_in_bytes")}


def make_case(seed, b, n, slots, heads, width, mask_rate=.82):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(b, n, heads, width)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=q.shape).astype(np.float32)
    neighbors = rng.integers(0, n, size=(b, n, slots), dtype=np.int32)
    mask = rng.random((b, n, slots)) < mask_rate
    mask[:, 0] = False  # explicit empty-neighbour rows
    switch = (rng.random((b, n, slots), dtype=np.float32) ** 2) * mask
    bias = rng.normal(scale=.2, size=(b, n, slots, heads)).astype(np.float32)
    cotangent = rng.normal(size=q.shape).astype(np.float32)
    return tuple(map(jnp.asarray, (q, k, v, neighbors, bias, mask, switch))), jnp.asarray(cotangent)


def run_case(name, shape, repeats):
    args, cotangent = make_case(100 + shape[1], *shape)
    native_fwd = jax.jit(native_attention)
    triton_fwd = jax.jit(indexed_attention)
    native_grad = jax.jit(jax.grad(
        lambda q, k, v, neighbors, bias, mask, switch:
        jnp.vdot(native_attention(q, k, v, neighbors, bias, mask, switch), cotangent),
        argnums=(0, 1, 2, 4)))
    triton_grad = jax.jit(jax.grad(
        lambda q, k, v, neighbors, bias, mask, switch:
        jnp.vdot(indexed_attention(q, k, v, neighbors, bias, mask, switch), cotangent),
        argnums=(0, 1, 2, 4)))
    native_out = ready(native_fwd(*args)); triton_out = ready(triton_fwd(*args))
    native_g = ready(native_grad(*args)); triton_g = ready(triton_grad(*args))
    vmapped_fwd = jax.jit(jax.vmap(indexed_attention_one))
    vmapped_grad = jax.jit(jax.grad(
        lambda q, k, v, neighbors, bias, mask, switch:
        jnp.vdot(jax.vmap(indexed_attention_one)(
            q, k, v, neighbors, bias, mask, switch), cotangent),
        argnums=(0, 1, 2, 4)))
    vmapped_out = ready(vmapped_fwd(*args))
    vmapped_g = ready(vmapped_grad(*args))
    forward_rel = relative(native_out, triton_out)
    gradient_rel = {key: relative(a, b) for key, a, b in zip(
        ("q", "k", "v", "edge_bias"), native_g, triton_g)}
    empty_max = dict(native=float(np.max(np.abs(np.asarray(native_out)[:, 0]))),
                     triton=float(np.max(np.abs(np.asarray(triton_out)[:, 0]))))
    native_fwd_compiled = native_fwd.lower(*args).compile()
    triton_fwd_compiled = triton_fwd.lower(*args).compile()
    native_grad_compiled = native_grad.lower(*args).compile()
    triton_grad_compiled = triton_grad.lower(*args).compile()
    timing = dict(
        native_forward=seconds(native_fwd_compiled, args, repeats),
        triton_forward=seconds(triton_fwd_compiled, args, repeats),
        native_backward=seconds(native_grad_compiled, args, repeats),
        triton_backward=seconds(triton_grad_compiled, args, repeats),
    )
    return dict(name=name, shape=dict(batch=shape[0], nodes=shape[1], slots=shape[2],
                heads=shape[3], head_width=shape[4]),
        logical_gathered_kv_bytes=int(2*np.prod((shape[0],shape[1],shape[2],shape[3],shape[4]))*4),
        node_kv_bytes=int(2*np.prod((shape[0],shape[1],shape[3],shape[4]))*4),
        forward_relative_l2=forward_rel, gradient_relative_l2=gradient_rel,
        custom_vmap_forward_relative_l2=relative(triton_out, vmapped_out),
        custom_vmap_gradient_relative_l2={key: relative(a, b) for key, a, b in zip(
            ("q", "k", "v", "edge_bias"), triton_g, vmapped_g)},
        empty_row_max_abs=empty_max, timing=timing,
        memory=dict(native_forward=memory(native_fwd_compiled),triton_forward=memory(triton_fwd_compiled),
                    native_backward=memory(native_grad_compiled),triton_backward=memory(triton_grad_compiled)))


def floor_case():
    args, cotangent = make_case(7, 1, 4, 8, 4, 23, mask_rate=1.)
    q, k, v, neighbors, bias, mask, switch = args
    mask = mask.at[:, 0].set(True)
    switch = jnp.full_like(switch, 1e-12)
    args = q, k, v, neighbors, bias, mask, switch
    native = ready(jax.jit(native_attention)(*args))
    custom = ready(jax.jit(indexed_attention)(*args))
    return dict(forward_relative_l2=relative(native, custom),
        native_max_abs=float(np.max(np.abs(np.asarray(native)))),
        triton_max_abs=float(np.max(np.abs(np.asarray(custom)))),
        switch=float(switch[0, 0, 0]), empty_cotangent_checksum=float(jnp.sum(cotangent)))


def main():
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]),
                    gpu_benchmark=True, allow_comp1400=True)
    destination = Path(os.environ["PKABENCH_RUNTIME"]) / "audits/gqt-triton-attention-v1"
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    specifications = (
        ("representative-200k", (8, 384, 192, 4, 23), 20),
        ("largest-stated-200k", (8, 512, 256, 4, 23), 12),
    )
    for name, shape, repeats in specifications:
        cases.append(run_case(name, shape, repeats))
        atomic_json(destination / "progress.json", {"complete": False, "cases": cases})
    result = dict(complete=True, variant=VARIANT,num_warps=WARPS,
        versions=dict(jax=jax.__version__,jax_triton=jt.__version__,
        triton=triton.__version__), dtype="float32", epsilon=EPS,
        gates=dict(forward_relative_l2=5e-6,gradient_relative_l2=2e-4,
                   faster_forward_and_backward=True),
        floor_case=floor_case(), cases=cases)
    atomic_json(destination / f"benchmark-{VARIANT}.json", result)
    if VARIANT == "warps4":
        atomic_json(destination / "benchmark.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
