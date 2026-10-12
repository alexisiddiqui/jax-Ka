"""Standalone CPU oGQT features, matching production backbone graph rules.

The site helpers below are copied unchanged from pkatrain.site_graph_data to
keep inference independent of dataset/training dependencies and historical code.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os
import tempfile

import numpy as np
from scipy.spatial import cKDTree
from jaxpropka.parameters import GROUP_AA, GROUPS
from jaxpropka.topology import load_topology
from .graph import geometry

SITE_EDGE_DIM = 34
RADIUS = 20.0
SCHEMA = "ogqt-backbone-inference-v1"

def frames(backbone):
    ca = backbone[:, 1]; x = backbone[:, 2] - ca; y = backbone[:, 0] - ca
    xn = np.linalg.norm(x, axis=-1, keepdims=True); x = x / np.maximum(xn, 1e-6)
    y = y - (y * x).sum(-1, keepdims=True) * x
    yn = np.linalg.norm(y, axis=-1, keepdims=True); y = y / np.maximum(yn, 1e-6)
    valid = (xn[:, 0] > 1e-5) & (yn[:, 0] > 1e-5)
    return ca, np.stack((x, y, np.cross(x, y)), axis=-1), valid

def candidate_sites(nodes):
    native = np.argmax(nodes[:, :20], axis=1); sites = []
    aa_to_group = {int(aa): group for group, aa in enumerate(GROUP_AA)}
    for residue in range(len(nodes)):
        if int(native[residue]) in aa_to_group: sites.append((residue, aa_to_group[int(native[residue])]))
        if bool(nodes[residue, 20]): sites.append((residue, 7))
        if bool(nodes[residue, 21]): sites.append((residue, 8))
    return np.asarray(sites, np.int32)

def build_site_graph(backbone, chain, nodes, query_residue, query_group):
    sites = candidate_sites(nodes); site_residue = sites[:, 0]; site_type = sites[:, 1]
    lookup = {(int(residue), int(group)): index for index, (residue, group) in enumerate(sites)}
    if len(lookup) != len(sites): raise AssertionError("Duplicate candidate site")
    if np.any(np.asarray(query_group) == 6): raise AssertionError("pKPDB unexpectedly contains ARG targets")
    query_site = np.asarray([lookup[int(r), int(g)] for r, g in zip(query_residue, query_group)], np.int32)
    ca, frame, valid = frames(backbone); site_ca = ca[site_residue]
    rows = cKDTree(site_ca).query_ball_point(site_ca, RADIUS)
    width = max(map(len, rows)); neighbors = np.zeros((len(sites), width), np.int32)
    mask = np.zeros((len(sites), width), bool)
    for index, row in enumerate(rows):
        row = sorted(row); neighbors[index, :len(row)] = row; mask[index, :len(row)] = True
    ri = site_residue; rj = site_residue[neighbors]
    delta = ca[rj] - ca[ri, None]; distance = np.sqrt((delta * delta).sum(-1) + 1e-8)
    fi = frame[ri]; fj = frame[rj]
    direction_i = np.einsum("skc,scl->skl", delta, fi) / distance[..., None]
    direction_j = np.einsum("skc,skcl->skl", -delta, fj) / distance[..., None]
    orientation = np.matmul(np.swapaxes(fi[:, None], -1, -2), fj).reshape((len(sites), width, 9))
    pair_valid = valid[ri, None] & valid[rj]
    direction_i *= valid[ri, None, None]; direction_j *= valid[rj][..., None]
    orientation *= pair_valid[..., None]
    rbf = np.exp(-((distance[..., None] - np.linspace(0, RADIUS, 16)) / 1.5) ** 2)
    same_chain = chain[ri, None] == chain[rj]; same_residue = ri[:, None] == rj
    sequence = np.zeros(len(chain), np.int32); counters = {}
    for index, name in enumerate(chain):
        sequence[index] = counters.get(int(name), 0); counters[int(name)] = sequence[index] + 1
    separation = np.where(same_chain, np.clip((sequence[rj] - sequence[ri, None]) / 32.0, -1, 1), 0.0)
    edge = np.concatenate((rbf, direction_i, direction_j, orientation,
        same_chain[..., None], same_residue[..., None], separation[..., None]), axis=-1).astype(np.float32)
    if edge.shape[-1] != SITE_EDGE_DIM: raise AssertionError(edge.shape)
    switch = np.where(distance < RADIUS - 2, 1.0,
        0.5 * (1 + np.cos(np.pi * np.clip((distance - RADIUS + 2) / 2, 0, 1)))).astype(np.float32)
    return {"site_residue": site_residue.astype(np.int32), "site_type": site_type.astype(np.int32),
            "site_mask": np.ones(len(sites), bool), "site_neighbors": neighbors,
            "site_edge": edge, "site_edge_mask": mask, "site_switch": switch,
            "query_site": query_site}


def featurise(path, *, gap_policy="cap", ignore_nonprotein=False):
    """Prepare all native labelled-type sites; ARG stays a context token only."""
    top = load_topology(path, gap_policy=gap_policy, freeze_disulfides=True,
                        ignore_nonprotein=ignore_nonprotein)
    graph, valid = geometry(top.backbone, top.chain_index)
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[top.native_index],
        np.stack((top.nterm, top.cterm, np.zeros(top.n_residues, bool), valid), -1)), -1).astype(np.float32)
    graph.update(nodes=nodes, node_mask=np.ones(top.n_residues, bool))
    candidates = candidate_sites(nodes).reshape(-1, 2)
    eligible = (candidates[:, 1] != 6) & ~((candidates[:, 1] == 3) & top.disulfide[candidates[:, 0]])
    queries = candidates[eligible]
    if not len(queries):
        raise ValueError(f"No supported titratable query sites in {path}")
    graph.update(query_residue=queries[:, 0], query_group=queries[:, 1])
    graph.update(build_site_graph(top.backbone, top.chain_index, nodes, queries[:, 0], queries[:, 1]))
    metadata = dict(schema=SCHEMA, topology=top.metadata, source=str(Path(path).resolve()),
        sites=[dict(**asdict(top.keys[int(r)]), group=GROUPS[int(g)]) for r, g in queries],
        dimensions=[len(nodes), graph['neighbors'].shape[1], len(queries),
                    len(graph['site_residue']), graph['site_neighbors'].shape[1]])
    return graph, metadata


def prepare_file(path, directory, **options):
    """Content-addressed feature cache; atomic writes allow concurrent readers."""
    path, directory = Path(path), Path(directory)
    identity = dict(schema=SCHEMA, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    options=options, biotite=importlib.metadata.version('biotite'))
    # Invalidate when either graph feature implementation or topology code changes.
    from . import graph as graph_module
    from jaxpropka import topology
    identity['code'] = [hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in (__file__, graph_module.__file__, topology.__file__)]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    dest = directory / (key + '.npz')
    if dest.exists():
        with np.load(dest, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
        if metadata['identity'] != identity:
            raise ValueError(f"Feature cache identity mismatch: {dest}")
        return dest, metadata
    graph, metadata = featurise(path, **options)
    metadata['identity'] = identity
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **graph, metadata=np.asarray(json.dumps(metadata)))
        os.replace(temporary, dest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dest, metadata


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['schema'] != SCHEMA:
            raise ValueError(f"Unsupported feature schema: {metadata['schema']}")
        return {k: data[k] for k in data.files if k != 'metadata'}, metadata
