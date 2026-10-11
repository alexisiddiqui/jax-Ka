"""Residue-encoder depth and weight-tied recurrence for production oGQT (2026-10-11; scaling plan steps 4 and 5).

oGQT has two untied residue-encoder blocks (pkanet.model.initialize: blocks = [block(), block()]), then the site query
block and the site-site block. pkanet.model.encode_indexed already loops over params["blocks"], so:
- depth D (--depth): D untied residue blocks. The first two are oGQT's own draws for the seed; block i >= 2 is the
  first block of pkanet.model.initialize(fold_in(PRNGKey(seed), DEPTH_SALT + i)), the same distribution. The forward
  pass is pkanet.site_model's unchanged.
- loops r (--loops): the two residue blocks are tied as one block group and applied r times, with full backprop through
  every loop. Before loop t >= 1 the encoded input e = embed(nodes) is re-injected, h <- h + e W_inject + b_inject,
  W_inject drawn at INJECT_SCALE x the default linear scale (small injection weights), b_inject zero; W_inject is
  weight-decayed like the other matrices. r = 1 is the plain oGQT forward pass (no injection), so step 2's runs serve
  as r = 1. Dropout draws one key per block application.
The site query and site-site blocks are unchanged in both. The code-hashed pkanet modules are imported, never edited.
"""
from __future__ import annotations

DEPTH_SALT = 4099
INJECT_SALT = 7919
INJECT_SCALE = 0.1


def deepen(params, seed, depth, width, ff, node_dim):
    """params with `depth` untied residue blocks (depth >= 2)."""
    import jax
    from pkanet.model import initialize
    if depth < 2: raise ValueError("depth counts residue blocks; oGQT has 2")
    extra = [initialize(jax.random.fold_in(jax.random.PRNGKey(seed), DEPTH_SALT + i), width=width, ff=ff, node_dim=node_dim)["blocks"][0]
             for i in range(2, depth)]
    return {**params, "blocks": list(params["blocks"]) + extra}


def add_injection(params, seed):
    import jax
    import jax.numpy as jnp
    width = params["head"]["w"].shape[0]; key = jax.random.fold_in(jax.random.PRNGKey(seed), INJECT_SALT)
    return {**params, "inject": {"w": INJECT_SCALE * jax.random.normal(key, (width, width), jnp.float32) / jnp.sqrt(float(width)),
                                 "b": jnp.zeros(width, jnp.float32)}}


def encode_looped(params, graph, loops, *, key=None, dropout_rate=0.0):
    import jax
    from pkanet.model import attend_indexed, linear
    encoded = linear(params["embed"], graph["nodes"]); mask = graph["node_mask"][:, None]; h = encoded
    blocks = params["blocks"]; keys = jax.random.split(key, loops * len(blocks)) if dropout_rate else [None] * (loops * len(blocks))
    for t in range(loops):
        if t: h = (h + linear(params["inject"], encoded)) * mask
        for b, block in enumerate(blocks):
            h = attend_indexed(block, h, h, graph["neighbors"], graph["edge"], graph["edge_mask"], graph["switch"],
                               key=keys[t * len(blocks) + b], dropout_rate=dropout_rate) * mask
    return h


def site_embeddings_looped(params, graph, loops, *, key=None, dropout_rate=0.0):
    """pkanet.site_model.site_embeddings_indexed with encode_looped as the residue encoder."""
    import jax
    from pkanet.site_model import _site_tokens, attend_sites_indexed
    keys = jax.random.split(key, 3) if dropout_rate else (None, None, None)
    residue = encode_looped(params, graph, loops, key=keys[0], dropout_rate=dropout_rate)
    tokens = _site_tokens(params, graph, residue, key=keys[1], dropout_rate=dropout_rate)
    tokens = attend_sites_indexed(params["site"], tokens, graph, key=keys[2], dropout_rate=dropout_rate)
    return tokens, graph["query_site"]


def model_functions(loops=1):
    """(site_embeddings, predict_shift) with pkanet.site_model's call signatures; loops = 1 returns its own functions."""
    import jax.numpy as jnp
    from pkanet.model import linear
    from pkanet.site_model import predict_site_shift_indexed, site_embeddings_indexed
    if loops == 1: return site_embeddings_indexed, predict_site_shift_indexed

    def embeddings(params, graph, *, key=None, dropout_rate=0.0):
        return site_embeddings_looped(params, graph, loops, key=key, dropout_rate=dropout_rate)

    def predict_shift(params, graph, *, key=None, dropout_rate=0.0):
        tokens, query_site = embeddings(params, graph, key=key, dropout_rate=dropout_rate)
        return (8 * jnp.tanh(linear(params["head"], tokens)[:, 0]))[query_site]
    return embeddings, predict_shift
