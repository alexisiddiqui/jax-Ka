"""Differentiable structural adapter for the pinned relational GQT backbone.

Parameters are fixed; input derivatives include the attention switch weights.
The free state isolates chains in residue, query and site attention.
"""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from .model import HEADS, linear, norm
from .site_model import _site_tokens, _attend_sites_trace


@dataclass(frozen=True)
class RelationalStructureFeatures:
    base: object
    state: str = "bound"

    def __post_init__(self):
        if self.state not in ("bound", "free"):
            raise ValueError("state must be bound or free")
        if self.base.sidechains:
            raise ValueError("The default relational checkpoint requires backbone-only features")

    def __call__(self, sequence, backbone, sidechains):
        g = self.base(sequence, backbone, sidechains)
        chain = jnp.asarray(self.base.chain)
        table = jnp.concatenate((backbone.reshape((-1, 12)), chain[:, None],
                                 jnp.asarray(self.base.sequence_index)[:, None]), axis=1)
        g = {**g, "nodes": jnp.concatenate((g["nodes"], table), axis=1)}
        if self.state == "free":
            same = chain[:, None] == chain[g["neighbors"]]
            sites = chain[g["site_residue"]]
            site_same = sites[:, None] == sites[g["site_neighbors"]]
            g = {**g, "edge_mask": g["edge_mask"] & same,
                 "switch": g["switch"] * same,
                 "site_edge_mask": g["site_edge_mask"] & site_same,
                 "site_switch": g["site_switch"] * site_same}
        return g


def _relation_block(block, x, graph):
    from pkatrain.production_position import relation_index
    width = x.shape[-1]
    c = norm(block["norm1"], x)
    q = linear(block["q"], c).reshape((-1, HEADS, width // HEADS))
    proj = block["relation_projection"]
    mapped = jnp.einsum("nw,rwv->rnv", c, proj["w"]) + proj["b"][:, None, :]
    k = linear(block["k"], mapped).reshape((-1, HEADS, width // HEADS))
    v = linear(block["v"], mapped).reshape(k.shape)
    nb = graph["neighbors"] + relation_index(graph) * x.shape[0]
    logits = jnp.einsum("nhd,nkhd->nkh", q, k[nb]) / jnp.sqrt(float(width // HEADS))
    logits = logits + linear(block["edge"], graph["edge"])
    mask = graph["edge_mask"][..., None]
    weights = jax.nn.softmax(jnp.where(mask, logits, -1e9), axis=1) * mask
    weights = weights * graph["switch"][..., None]
    weights = weights / jnp.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    message = jnp.einsum("nkh,nkhd->nhd", weights, v[nb]).reshape(x.shape)
    x = x + linear(block["o"], message)
    return x + linear(block["down"], jax.nn.gelu(linear(block["up"], norm(block["norm2"], x))))


def relational_functions():
    """Portable production forward, retaining all feature/input derivatives."""
    from pkatrain.production_position import annotate
    from pkatrain.production_inputs import input_functions

    def embeddings(params, graph, *, key=None, dropout_rate=0.):
        if dropout_rate:
            raise ValueError("The frozen structural endpoint is inference-only")
        mask = graph["node_mask"][:, None]
        h = linear(params["embed"], graph["nodes"]) * mask
        for block in params["blocks"]:
            h = _relation_block(block, h, graph) * mask
        tokens = _site_tokens(params, graph, h)
        tokens, _ = _attend_sites_trace(params["site"], tokens, graph)
        return tokens, graph["query_site"]

    wrapped = input_functions((embeddings, None), orientation=True, table=True, dihedrals=True)
    def wrap(fn):
        def call(params, graph):
            return fn(params, annotate(graph, "relations"))
        return call
    return tuple(wrap(fn) for fn in wrapped)
