"""Opt-in initial-site residual bypass for the matched oGQT skip pilot."""
import jax
import jax.numpy as jnp

from .model import encode_indexed, linear
from .site_model import _site_tokens, attend_sites_indexed


def predict_multi(params, graph, *, enabled):
    residue = encode_indexed(params, graph)
    initial = (residue[graph["site_residue"]] + params["groups"][graph["site_type"]]) * graph["site_mask"][:, None]
    local = _site_tokens(params, graph, residue)
    tokens = attend_sites_indexed(params["site"], local, graph)
    if enabled:
        normalized = initial * jax.lax.rsqrt(jnp.mean(initial**2, axis=-1, keepdims=True) + 1e-5)
        tokens = (tokens + params["direct_skip"] * normalized) * graph["site_mask"][:, None]
    query = graph["query_site"]
    return {
        "shift": (8*jnp.tanh(linear(params["head"], tokens)[:, 0]))[query],
        "burial": jax.nn.sigmoid(linear(params["auxiliary"]["burial"], tokens)[:, 0])[query],
        "interface": jax.nn.sigmoid(linear(params["auxiliary"]["interface"], tokens)[:, 0])[query],
    }
