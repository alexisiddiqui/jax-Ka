"""Branch-specific APO geometry without modifying native production loaders.

The caller supplies canonical native graph order and an explicit partner map.
Geometry-derived labels are deliberately absent from the returned graph inputs.
"""
from __future__ import annotations

import numpy as np

from pkanet.graph import geometry, sidechain_features
from .site_graph_data import build_site_graph, candidate_sites
from .production_graphs import GRAPH_FIELDS, SITE_FIELDS

INPUT_FIELDS = GRAPH_FIELDS + SITE_FIELDS
NODE_FIELDS = ("nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch")
TOKEN_FIELDS = ("site_residue", "site_type", "site_mask", "site_neighbors", "site_edge", "site_edge_mask", "site_switch")


def backbone_table(backbone, chain, sequence):
    return np.concatenate((np.asarray(backbone).reshape(len(backbone), 12),
                           np.asarray(chain)[:, None], np.asarray(sequence)[:, None]), axis=1).astype(np.float32)


def align_residues(source_backbone, source_identity, canonical_backbone, canonical_identity, *, allowed=None):
    """Injective coordinate+identity mapping; ambiguity is a hard error.

    Author-chain aliases are insufficient for assemblies. Native source coordinates
    establish node order; augmentation displacements are never used for matching.
    allowed constrains a partner to the native parent nodes belonging to that partner.
    """
    from scipy.spatial import cKDTree
    source = np.asarray(source_backbone, float); canonical = np.asarray(canonical_backbone, float)
    allowed = np.arange(len(canonical)) if allowed is None else np.asarray(allowed, np.int32)
    tree = cKDTree(canonical[allowed, 1]); result = []
    for bb, identity in zip(source, source_identity):
        nearby = tree.query_ball_point(bb[1], 0.002)
        choices = [int(allowed[j]) for j in nearby if canonical_identity[allowed[j]] == identity and
                   np.max(np.abs(canonical[allowed[j]]-bb)) <= 0.002]
        if len(choices) != 1: raise ValueError("ambiguous or missing canonical residue mapping")
        result.append(choices[0])
    if len(set(result)) != len(result): raise ValueError("non-injective canonical residue mapping")
    return np.asarray(result, np.int32)


def free_graph(native, backbone, chain, partners):
    """Build each partner independently, then remap residue and site indices."""
    backbone = np.asarray(backbone); chain = np.asarray(chain); partners = np.asarray(partners)
    n = len(native["nodes"])
    if backbone.shape != (n, 4, 3) or chain.shape != (n,) or partners.shape != (n,):
        raise ValueError("canonical backbone/chain/partner shape mismatch")
    if not np.isfinite(backbone).all(): raise ValueError("nonfinite backbone")
    sites = candidate_sites(native["nodes"])
    if not np.array_equal(sites[:, 0], native["site_residue"]) or not np.array_equal(sites[:, 1], native["site_type"]):
        raise ValueError("native site ordering differs from canonical candidate ordering")
    pieces = []
    for partner in np.unique(partners):
        residues = np.flatnonzero(partners == partner); local_of = {int(r): i for i, r in enumerate(residues)}
        nodes = native["nodes"][residues].copy()
        graph, valid = geometry(backbone[residues], chain[residues]); nodes[:, 23] = valid
        graph.update(nodes=nodes, node_mask=native["node_mask"][residues].copy())
        tokens = np.flatnonzero(np.isin(native["site_residue"], residues))
        if len(tokens):
            query = np.flatnonzero(np.isin(native["query_residue"], residues) & (native["query_site"] >= 0))
            qr = np.asarray([local_of[int(r)] for r in native["query_residue"][query]], np.int32)
            site = build_site_graph(backbone[residues], chain[residues], nodes, qr, native["query_group"][query])
            if not np.array_equal(residues[site["site_residue"]], native["site_residue"][tokens]) or \
                    not np.array_equal(site["site_type"], native["site_type"][tokens]):
                raise ValueError("partner site remapping changed identity")
            if not np.array_equal(tokens[site["query_site"]], native["query_site"][query]):
                raise ValueError("partner query remapping changed identity")
        else: site = None
        pieces.append((residues, tokens, graph, site))
    k = max(g["neighbors"].shape[1] for _, _, g, _ in pieces)
    sk = max((g["site_neighbors"].shape[1] for _, _, _, g in pieces if g is not None), default=0)
    s = len(sites)
    out = {name: np.array(native[name], copy=True) for name in INPUT_FIELDS if name not in
           ("neighbors", "edge", "edge_mask", "switch", "site_neighbors", "site_edge", "site_edge_mask", "site_switch")}
    out.update(neighbors=np.zeros((n, k), np.int32), edge=np.zeros((n, k, 20), np.float32),
               edge_mask=np.zeros((n, k), bool), switch=np.zeros((n, k), np.float32),
               site_neighbors=np.zeros((s, sk), np.int32), site_edge=np.zeros((s, sk, 34), np.float32),
               site_edge_mask=np.zeros((s, sk), bool), site_switch=np.zeros((s, sk), np.float32))
    for residues, tokens, graph, site in pieces:
        width = graph["neighbors"].shape[1]; out["nodes"][residues] = graph["nodes"]
        out["neighbors"][residues, :width] = residues[graph["neighbors"]]
        for field in ("edge", "edge_mask", "switch"): out[field][residues, :width] = graph[field]
        if site is not None:
            width = site["site_neighbors"].shape[1]
            out["site_neighbors"][tokens, :width] = tokens[site["site_neighbors"]]
            for field in ("site_edge", "site_edge_mask", "site_switch"): out[field][tokens, :width] = site[field]
    if np.any(partners[out["neighbors"]][out["edge_mask"]] != np.broadcast_to(partners[:, None], (n, k))[out["edge_mask"]]):
        raise ValueError("free graph has interpartner residue edges")
    sp = partners[out["site_residue"]]
    if np.any(sp[out["site_neighbors"]][out["site_edge_mask"]] != np.broadcast_to(sp[:, None], (s, sk))[out["site_edge_mask"]]):
        raise ValueError("free graph has interpartner site edges")
    return out


def pair_inputs(bound, free, bound_backbone, free_backbone, chain, sequence, bound_atoms=None, free_atoms=None):
    """Pad branch-specific graph inputs; bound values retain their native bytes."""
    for key in ("query_residue", "query_group", "site_residue", "site_type", "query_site"):
        if not np.array_equal(bound[key], free[key]): raise ValueError(f"branch identity differs: {key}")
    result = {}
    for name in INPUT_FIELDS:
        a = bound[name]; b = free[name]
        shape = tuple(max(x, y) for x, y in zip(a.shape, b.shape))
        padded = np.zeros((2, *shape), a.dtype)
        for branch, value in enumerate((a, b)):
            padded[(branch, *(slice(0, n) for n in value.shape))] = value
        result[name] = padded
    tables = np.stack([backbone_table(bb, chain, sequence) for bb in (bound_backbone, free_backbone)])
    # The model's existing input wrapper strips these 14 columns before embedding.
    result["nodes"] = np.concatenate((result["nodes"], tables), axis=-1)
    if bound_atoms is not None or free_atoms is not None:
        if bound_atoms is None or free_atoms is None: raise ValueError("both branch sidechains are required")
        # Store separately; a caller enabling sidechains inserts them before the backbone table.
        result["sidechains"] = np.stack([sidechain_features(bb, aa) for bb, aa in
                                         ((bound_backbone, bound_atoms), (free_backbone, free_atoms))]).astype(np.float16)
    return result


def capacities(records, bounds=(128, 256, 384, 512, 640, 768, 1024, 1280, 1536)):
    result = {}
    for arrays in records:
        n = arrays["nodes"].shape[-2]
        bound = next((b for b in bounds if n <= b), None)
        if bound is None: raise ValueError("pilot exceeds supported production buckets")
        dims = [bound, arrays["neighbors"].shape[-1], arrays["query_residue"].shape[-1],
                arrays["site_residue"].shape[-1], arrays["site_neighbors"].shape[-1]]
        result[str(bound)] = [max(a, b) for a, b in zip(result.get(str(bound), [0]*5), dims)]
    return result
