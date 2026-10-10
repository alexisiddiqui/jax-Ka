"""Opt-in separate local-query and residue-context LayerNorm affine parameters."""
import jax
import jax.numpy as jnp
from .model import attend, encode_indexed, linear
from .site_model import attend_sites_indexed


def predict_multi(params, graph):
    residue=encode_indexed(params,graph)
    sr=graph["site_residue"]; mask=graph["site_mask"][:,None]
    tokens=(residue[sr]+params["groups"][graph["site_type"]])*mask
    tokens=attend(params["query"],tokens,residue,graph["neighbors"][sr],graph["edge"][sr],
        graph["edge_mask"][sr],graph["switch"][sr],context_norm=params["query_context_norm"])*mask
    tokens=attend_sites_indexed(params["site"],tokens,graph)
    query=graph["query_site"]
    return {"shift":(8*jnp.tanh(linear(params["head"],tokens)[:,0]))[query],
        "burial":jax.nn.sigmoid(linear(params["auxiliary"]["burial"],tokens)[:,0])[query],
        "interface":jax.nn.sigmoid(linear(params["auxiliary"]["interface"],tokens)[:,0])[query]}
