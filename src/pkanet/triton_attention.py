"""Optional CUDA indexed-attention backend for GQT.

The Triton path keeps K/V once per node and gathers them inside the kernel.
It covers only the attention message; projections, residuals, dropout and the
feed-forward block remain ordinary JAX operations.
"""
from __future__ import annotations

import jax
from jax.custom_batching import custom_vmap
import numpy as np
import jax_triton as jt
import triton
import triton.language as tl

EPS = 1e-8
WARPS = 8


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
    if q.dtype != np.float32 or k.dtype != np.float32 or v.dtype != np.float32:
        raise TypeError("GQT Triton attention requires float32 Q/K/V")
    b, queries, h, d = q.shape
    n = k.shape[1]
    assert k.shape == v.shape == (b, n, h, d)
    assert neighbors.shape[:2] == (b, queries)
    slots = neighbors.shape[-1]
    return jt.triton_call(
        q, k, v, neighbors, bias, mask, switch,
        kernel=_indexed_attention_fwd,
        out_type=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(b * queries, h), name="gqt_indexed_attention_fwd",
        B=b, Q=queries, N=n, H=h, D=d, K=slots,
        SCALE=float(1 / np.sqrt(d)),
        EPSILON=EPS,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_K=triton.next_power_of_2(slots),
        num_warps=WARPS, backend_options={"enable_fp_fusion": False},
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
        out_type=shapes, grid=(b * queries, h),
        name="gqt_indexed_attention_bwd",
        B=b, Q=queries, N=n, H=h, D=d, K=slots,
        SCALE=float(1 / np.sqrt(d)),
        EPSILON=EPS,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_K=triton.next_power_of_2(slots),
        num_warps=WARPS, zeroed_outputs=(1, 2), backend_options={"enable_fp_fusion": False},
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

