"""Explicit titratable-site tokens over the existing backbone GQT encoder."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .model import (
    HEADS, PKPDB_PK_MOD, attend, attend_with_trace, dropout, encode_indexed,
    encode_with_trace, initialize, linear, norm,
)


SITE_EDGE_DIM = 34
ORIENTATION_SLICE = slice(22, 31)


def _linear(key, inputs, outputs):
    return {"w": jax.random.normal(key, (inputs, outputs), dtype=jnp.float32) / jnp.sqrt(float(inputs)),
            "b": jnp.zeros(outputs, jnp.float32)}


def initialize_site(key, width=44, ff=88, node_dim=24):
    """Preserve the current model initialization and append one site block."""
    params = initialize(key, width=width, ff=ff, node_dim=node_dim)
    keys = iter(jax.random.split(jax.random.fold_in(key, 9137), 12))
    params["site"] = {
        "q": _linear(next(keys), width, width), "k": _linear(next(keys), width, width),
        "v": _linear(next(keys), width, width), "o": _linear(next(keys), width, width),
        "edge_up": _linear(next(keys), SITE_EDGE_DIM, width),
        "edge_down": _linear(next(keys), width, HEADS),
        "pair_bias": jax.random.normal(next(keys), (9, 9, HEADS), dtype=jnp.float32) * 0.01,
        "up": _linear(next(keys), width, ff), "down": _linear(next(keys), ff, width),
        "norm1": jnp.stack((jnp.ones(width), jnp.zeros(width))),
        "norm2": jnp.stack((jnp.ones(width), jnp.zeros(width))),
    }
    return params


def initialize_site_auxiliary(key, width=44, ff=88, node_dim=24):
    """Append burial/interface heads without changing any existing draw."""
    params = initialize_site(key, width=width, ff=ff, node_dim=node_dim)
    burial_key, interface_key = jax.random.split(jax.random.fold_in(key, 27183))
    params["auxiliary"] = {
        "burial": _linear(burial_key, width, 1),
        "interface": _linear(interface_key, width, 1),
    }
    return params


def _site_bias(block, edge, source_type, neighbor_type):
    geometry = linear(block["edge_down"], jax.nn.gelu(linear(block["edge_up"], edge)))
    return geometry + block["pair_bias"][source_type[:, None], neighbor_type]


def attend_sites_indexed(block, tokens, graph, *, key=None, dropout_rate=0.0):
    """One indexed site-to-site attention/FF block with directed pair bias."""
    from .triton_attention import indexed_attention_one
    width = block["q"]["w"].shape[0]
    normalized = norm(block["norm1"], tokens)
    query = linear(block["q"], normalized).reshape((-1, HEADS, width // HEADS))
    projected_key = linear(block["k"], normalized).reshape((-1, HEADS, width // HEADS))
    value = linear(block["v"], normalized).reshape((-1, HEADS, width // HEADS))
    neighbors = graph["site_neighbors"]
    neighbor_type = graph["site_type"][neighbors]
    bias = _site_bias(block, graph["site_edge"], graph["site_type"], neighbor_type)
    message = indexed_attention_one(
        query, projected_key, value, neighbors, bias, graph["site_edge_mask"], graph["site_switch"]
    ).reshape((-1, width))
    keys = jax.random.split(key, 2) if dropout_rate else (None, None)
    output = tokens + dropout(linear(block["o"], message), keys[0], dropout_rate)
    feed_forward = linear(block["down"], jax.nn.gelu(linear(block["up"], norm(block["norm2"], output))))
    output = output + dropout(feed_forward, keys[1], dropout_rate)
    return output * graph["site_mask"][:, None]


def _site_tokens(params, graph, residue, *, key=None, dropout_rate=0.0):
    site_residue = graph["site_residue"]
    site_type = graph["site_type"]
    tokens = (residue[site_residue] + params["groups"][site_type]) * graph["site_mask"][:, None]
    # Existing query block gives every candidate site its local residue context.
    tokens = attend(
        params["query"], tokens, residue,
        graph["neighbors"][site_residue], graph["edge"][site_residue],
        graph["edge_mask"][site_residue], graph["switch"][site_residue],
        key=key, dropout_rate=dropout_rate,
    )
    return tokens * graph["site_mask"][:, None]


def site_embeddings_indexed(params, graph, *, key=None, dropout_rate=0.0):
    """Return final candidate-site embeddings and the supervised site indices."""
    keys = jax.random.split(key, 3) if dropout_rate else (None, None, None)
    residue = encode_indexed(params, graph, key=keys[0], dropout_rate=dropout_rate)
    tokens = _site_tokens(params, graph, residue, key=keys[1], dropout_rate=dropout_rate)
    tokens = attend_sites_indexed(params["site"], tokens, graph, key=keys[2], dropout_rate=dropout_rate)
    return tokens, graph["query_site"]


def predict_site_shift_indexed(params, graph, *, key=None, dropout_rate=0.0):
    tokens, query_site = site_embeddings_indexed(params, graph, key=key, dropout_rate=dropout_rate)
    all_shifts = 8 * jnp.tanh(linear(params["head"], tokens)[:, 0])
    return all_shifts[query_site]


def predict_site_multi_indexed(params, graph, *, key=None, dropout_rate=0.0):
    """Return pKa shift and independent normalized structural predictions."""
    tokens, query_site = site_embeddings_indexed(params, graph, key=key, dropout_rate=dropout_rate)
    return {
        "shift": (8 * jnp.tanh(linear(params["head"], tokens)[:, 0]))[query_site],
        "burial": jax.nn.sigmoid(linear(params["auxiliary"]["burial"], tokens)[:, 0])[query_site],
        "interface": jax.nn.sigmoid(linear(params["auxiliary"]["interface"], tokens)[:, 0])[query_site],
    }


def predict_site_pkpdb_indexed(params, graph, *, key=None, dropout_rate=0.0):
    return PKPDB_PK_MOD[graph["query_group"]] + predict_site_shift_indexed(
        params, graph, key=key, dropout_rate=dropout_rate
    )


def _attend_sites_trace(block, tokens, graph):
    width = block["q"]["w"].shape[0]
    normalized = norm(block["norm1"], tokens)
    q = linear(block["q"], normalized).reshape((-1, HEADS, width // HEADS))
    k = linear(block["k"], normalized).reshape((-1, HEADS, width // HEADS))
    v = linear(block["v"], normalized).reshape((-1, HEADS, width // HEADS))
    neighbors = graph["site_neighbors"]
    neighbor_type = graph["site_type"][neighbors]
    bias = _site_bias(block, graph["site_edge"], graph["site_type"], neighbor_type)
    logits = jnp.einsum("nhd,nkhd->nkh", q, k[neighbors]) / jnp.sqrt(float(width // HEADS)) + bias
    logits = jnp.where(graph["site_edge_mask"][..., None], logits, -1e9)
    softmax = jax.nn.softmax(logits, axis=1) * graph["site_edge_mask"][..., None]
    weighted = softmax * graph["site_switch"][..., None]
    weights = weighted / jnp.maximum(weighted.sum(axis=1, keepdims=True), 1e-8)
    message = jnp.einsum("nkh,nkhd->nhd", weights, v[neighbors]).reshape((-1, width))
    output = tokens + linear(block["o"], message)
    output = output + linear(block["down"], jax.nn.gelu(linear(block["up"], norm(block["norm2"], output))))
    return output * graph["site_mask"][:, None], {"logits": logits, "weights": weights}


def predict_site_with_trace(params, graph):
    residue, encoder_trace = encode_with_trace(params, graph)
    site_residue = graph["site_residue"]; site_type = graph["site_type"]
    tokens = (residue[site_residue] + params["groups"][site_type]) * graph["site_mask"][:, None]
    tokens, local_trace = attend_with_trace(
        params["query"], tokens, residue,
        graph["neighbors"][site_residue], graph["edge"][site_residue],
        graph["edge_mask"][site_residue], graph["switch"][site_residue],
    )
    tokens = tokens * graph["site_mask"][:, None]
    tokens, site_trace = _attend_sites_trace(params["site"], tokens, graph)
    all_shifts = 8 * jnp.tanh(linear(params["head"], tokens)[:, 0])
    return {"predicted_shift": all_shifts[graph["query_site"]],
            "local_attention": local_trace, "site_attention": site_trace,
            "encoder_attention": tuple(encoder_trace)}
