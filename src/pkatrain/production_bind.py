"""Paired binding auxiliary on the final site tokens of the PINDER bound (AB) and free branches (2026-10-12, --bind-weight).

Per query site, from the AB token h_AB and the free token h_free (both LayerNorm'd with one shared norm: the tokens are
the un-normalised residual stream, and the product would otherwise scale quadratically):

    a = norm(h_AB), b = norm(h_free)
    w = W_2 gelu(W_1 [a; b]) + w_0              per-channel weights from a small block on the two site tokens
    logit = sum_k w_k x_k + c,   x = a * b       (operand "product": weighted Hadamard product, a context-dependent
                                                   diagonal bilinear form; with W_2 = 0 it is the static DistMult form)
                                 x = a - b       (operand "difference": the same block weighting the bound - free change)

W_2, w_0 and c start at zero, so every site starts at logit 0. (Unit static weights are not neutral for the product:
most sites barely change on binding, a ~ b, so sum_k a_k b_k / sqrt(width) ~ sqrt(width) = 8 and the head would start
confidently predicting interface everywhere; a smoke run started at cross-entropy 3.6 vs 0.69.) The block's gradient
reaches the trunk once W_2 / w_0 move off zero, after the first update.
Target: the interface soft target log(1 + min(c, 8)) / log 9 of the bound-state contact score (production_aux), sigmoid
cross-entropy, averaged per site then per structure over the pKa-labelled sites (AuxEngine's interface reduction).
Unlike the interface head on h_AB alone, the head sees the pair, so its gradient asks the trunk to make interface sites'
bound and free tokens interact (product) or differ (difference) in a way the weights can read.
"""
from __future__ import annotations

OPERANDS = ("product", "difference")
SALT = 2_4_6_8  # fold_in salt for the head's key (initialisation independent of the other heads)


def initialize(params, seed):
    import jax
    import jax.numpy as jnp
    width = params["head"]["w"].shape[0]; k1 = jax.random.fold_in(jax.random.PRNGKey(seed), SALT)
    return {**params, "aux_bind": {"norm": jnp.stack((jnp.ones(width), jnp.zeros(width))),
                                   "w1": {"w": jax.random.normal(k1, (2 * width, width), jnp.float32) / jnp.sqrt(2.0 * width), "b": jnp.zeros(width, jnp.float32)},
                                   "w2": {"w": jnp.zeros((width, width), jnp.float32), "b": jnp.zeros(width, jnp.float32)},
                                   "c": jnp.zeros((), jnp.float32)}}


def logits(p, bound, free, operand):
    """(..., width) AB and free site tokens -> (...) binding logits."""
    import jax
    import jax.numpy as jnp
    from pkanet.model import linear, norm
    if operand not in OPERANDS: raise ValueError(operand)
    a = norm(p["norm"], bound); b = norm(p["norm"], free)
    weights = linear(p["w2"], jax.nn.gelu(linear(p["w1"], jnp.concatenate((a, b), axis=-1))))
    return jnp.sum(weights * (a * b if operand == "product" else a - b), axis=-1) + p["c"]
