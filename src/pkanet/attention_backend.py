"""Indexed-attention backend selection for GQT.

Every indexed attention layer (residue blocks, query block, site blocks) calls
``indexed_attention_one``. On CUDA with ``jax_triton`` installed it uses the
Triton kernel (``pkanet.triton_attention``), as in training. Elsewhere, or with
``GQT_ATTENTION=jax``, it uses ``indexed_attention_reference``: the same
arithmetic in plain JAX (gathered keys and values, masked softmax over the
neighbour slots, multiplied by the edge mask and the cosine switch, then
renormalised with the same epsilon). The reference path is differentiable in
every input, including the switch weights.

``GQT_ATTENTION`` = ``auto`` (default), ``triton`` or ``jax``; ``set_backend``
overrides it in-process (before functions are traced).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

EPS = 1e-8
_BACKENDS = ("auto", "triton", "jax")
_override = None


def set_backend(name):
    """Select the indexed-attention backend for functions traced from now on."""
    global _override
    if name not in _BACKENDS:
        raise ValueError(f"attention backend must be one of {_BACKENDS}")
    _override = name


def backend():
    """'triton' or 'jax', resolving 'auto' from the default JAX platform."""
    name = _override or os.environ.get("GQT_ATTENTION", "auto")
    if name not in _BACKENDS:
        raise ValueError(f"GQT_ATTENTION must be one of {_BACKENDS}")
    if name != "auto":
        return name
    if jax.default_backend() != "gpu":
        return "jax"
    try:
        import jax_triton  # noqa: F401
    except ImportError:
        return "jax"
    return "triton"


def indexed_attention_reference(q, k, v, neighbors, bias, mask, switch):
    """Plain-JAX equivalent of the Triton forward for one graph.

    q (Q, H, D), k / v (N, H, D), neighbors / mask / switch (Q, K), bias (Q, K, H)
    -> (Q, H, D).
    """
    scale = 1.0 / jnp.sqrt(jnp.asarray(q.shape[-1], q.dtype))
    logits = jnp.einsum("qhd,qkhd->qkh", q, k[neighbors]) * scale + bias
    logits = jnp.where(mask[..., None], logits, -1.0e9)
    softmax = jax.nn.softmax(logits, axis=1)
    weighted = softmax * switch[..., None] * mask[..., None].astype(softmax.dtype)
    weights = weighted / jnp.maximum(weighted.sum(axis=1, keepdims=True), EPS)
    return jnp.einsum("qkh,qkhd->qhd", weights, v[neighbors])


def indexed_attention_one(q, k, v, neighbors, bias, mask, switch):
    if backend() == "triton":
        from .triton_attention import indexed_attention_one as triton_one
        return triton_one(q, k, v, neighbors, bias, mask, switch)
    return indexed_attention_reference(q, k, v, neighbors, bias, mask, switch)
