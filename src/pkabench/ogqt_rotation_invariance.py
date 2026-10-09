"""Coordinate-level rotation invariance audit for paired oGQT auxiliaries."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import jax
import numpy as np
from biotite.structure.io import pdbx

from jaxpropka.parameters import GROUPS
from jaxpropka.topology import load_topology
from pkanet.graph import geometry
from pkanet.ogqt import initialize_auxiliary
from pkabench.runtime import atomic_json, require_compute
from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine
from pkatrain.gqt_paired_pinder import RAW_FIELDS, _bucket_n, _load_one
from pkatrain.site_graph_data import build_site_graph
from pkatrain.trainer import load_checkpoint

SEED = 17
N_COMPLEXES = 50
REPEATS = 3


def read(path):
    with Path(path).open() as stream: return json.load(stream)


def _read_cif(path):
    with gzip.open(path, "rt") as stream: file = pdbx.CIFFile.read(stream)
    return pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True, include_bonds=True)


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0: q[:, 0] *= -1
    return q


def _transform(atoms, chain_partner, mode, rng):
    moved = atoms.copy()
    if mode == "joint":
        center = moved.coord.mean(0); moved.coord = (moved.coord-center) @ _rotation(rng) + center
    elif mode == "independent":
        for partner in ("A", "B"):
            chains = [chain for chain, value in chain_partner.items() if value == partner]
            mask = np.isin(moved.chain_id, chains); center = moved.coord[mask].mean(0)
            moved.coord[mask] = (moved.coord[mask]-center) @ _rotation(rng) + center
    else: raise ValueError(mode)
    return moved


def _intrachain_only(graph, prefix=""):
    edge_name = prefix + "edge"; neighbor_name = prefix + "neighbors"
    mask_name = prefix + "edge_mask"; switch_name = prefix + "switch"
    same_column = 31 if prefix else -1
    keep = graph[mask_name] & (graph[edge_name][..., same_column] > .5)
    width = int(np.max(keep.sum(1))); rows = len(keep)
    neighbors = np.zeros((rows, width), graph[neighbor_name].dtype)
    edge = np.zeros((rows, width, graph[edge_name].shape[-1]), graph[edge_name].dtype)
    mask = np.zeros((rows, width), bool); switch = np.zeros((rows, width), graph[switch_name].dtype)
    for row in range(rows):
        slots = np.flatnonzero(keep[row]); count = len(slots)
        neighbors[row, :count] = graph[neighbor_name][row, slots]
        edge[row, :count] = graph[edge_name][row, slots]
        mask[row, :count] = True; switch[row, :count] = graph[switch_name][row, slots]
    graph.update({neighbor_name: neighbors, edge_name: edge, mask_name: mask, switch_name: switch})


def _pack(atoms, record, raw, manifest, free_only=False):
    topology = load_topology(atoms, gap_policy="cap", freeze_disulfides=True)
    graph, frames_valid = geometry(topology.backbone, topology.chain_index, radius=20.0)
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[topology.native_index],
        np.stack((topology.nterm, topology.cterm, np.zeros(topology.n_residues, bool), frames_valid), -1)), -1)
    graph.update(nodes=nodes.astype(np.float32), node_mask=np.ones(topology.n_residues, bool))
    lookup = {(key.chain, key.number, key.insertion): i for i, key in enumerate(topology.keys)}
    query = np.asarray([(lookup[tuple(key[:3])], GROUPS.index(key[3])) for key in record["keys"]], np.int32)
    graph.update(query_residue=query[:, 0], query_group=query[:, 1])
    site = build_site_graph(topology.backbone, topology.chain_index, nodes, query[:, 0], query[:, 1])
    if free_only:
        _intrachain_only(graph); _intrachain_only(site, "site_")
    payload = {**{name: raw[name] for name in RAW_FIELDS}, **graph, **site,
        "branch_edge_mask": np.stack((graph["edge_mask"], graph["edge_mask"] & (graph["edge"][..., -1] > .5))),
        "branch_site_edge_mask": np.stack((site["site_edge_mask"], site["site_edge_mask"] & (site["site_edge"][..., 31] > .5)))}
    local = {**record, "n": topology.n_residues, "k": graph["neighbors"].shape[1],
        "s": len(site["site_type"]), "sk": site["site_neighbors"].shape[1]}
    capacity = manifest["capacities"][_bucket_n(local["n"])]
    if any(local[name] > limit for name, limit in zip(("n","k","q","s","sk"), capacity)):
        raise ValueError((record["id"], "transformed graph exceeds capacity", local, capacity))
    return _load_one(payload, local, capacity, manifest["normalization"])[0]


def run(runtime):
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    runtime = Path(runtime); base = runtime / "training/ogqt-auxiliary-pilot-v1"
    source = runtime / "pretraining/pinder-pkai-v1"; manifest = read(base / "manifest.json")
    validation = [x for x in manifest["records"] if x["split"] == "val"]
    rank = lambda row: hashlib.sha256(f"ogqt-rotation-invariance-v1|{row['id']}".encode()).hexdigest()
    selected = sorted(validation, key=rank)[:N_COMPLEXES]
    params = initialize_auxiliary(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = AuxiliaryEngine(params, 0, {"burial": 1, "interface": 1}); state = engine.optimizer.init(params)
    verification = read(base / "standard/seed-17/verification.json")
    params, _, _ = load_checkpoint(base / "standard/seed-17/checkpoints" /
        f"epoch-{verification['selected_epoch']:03d}", (params, state))
    deltas = {"joint": {head: [] for head in ("interface_bound","interface_free","burial_bound","burial_free")},
              "independent_free": {head: [] for head in ("interface_free","burial_free")}}
    records = []
    for number, record in enumerate(selected, 1):
        folder = source / "entries" / record["id"]
        with np.load(base / "graphs" / record["id"] / "graph.npz", allow_pickle=False) as handle:
            raw = {name: handle[name] for name in RAW_FIELDS}
        sites = read(folder / "sites.json"); chain_partner = {}
        for site in sites: chain_partner[str(site["chain"])] = str(site["partner"])
        atoms = _read_cif(folder / "AB.cif.gz")
        native_graph = _pack(atoms, record, raw, manifest)
        native = {name: np.asarray(value)[0] for name, value in engine.predictions(params,
            {name: jax.numpy.asarray(value[None]) for name, value in native_graph.items()}).items()}
        row = {"complex_id": record["id"], "queries": record["q"], "repeats": []}
        for repeat in range(REPEATS):
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
                f"{SEED}|{record['id']}|{repeat}".encode()).digest()[:8], "little"))
            item = {"repeat": repeat}
            for mode in ("joint", "independent"):
                graph = _pack(_transform(atoms, chain_partner, mode, rng), record, raw, manifest,
                    free_only=(mode == "independent"))
                prediction = {name: np.asarray(value)[0] for name, value in engine.predictions(params,
                    {name: jax.numpy.asarray(value[None]) for name, value in graph.items()}).items()}
                heads = (("interface_bound","interface",0),("interface_free","interface",1),
                         ("burial_bound","burial",0),("burial_free","burial",1)) if mode == "joint" else (
                         ("interface_free","interface",1),("burial_free","burial",1))
                item[mode] = {}
                for label, head, branch in heads:
                    value = np.abs(prediction[head][branch, :record["q"]] - native[head][branch, :record["q"]])
                    deltas["joint" if mode == "joint" else "independent_free"][label].extend(value.tolist())
                    item[mode][label] = float(value.max())
            row["repeats"].append(item)
        records.append(row)
        if number % 10 == 0: print(json.dumps({"complexes": number, "total": len(selected)}), flush=True)
    summary = {"version": "ogqt-rotation-invariance-v1", "complexes": len(selected), "repeats": REPEATS,
        "checkpoint_epoch": verification["selected_epoch"], "test_data_included": False, "results": {}, "records": records}
    for mode, heads in deltas.items():
        summary["results"][mode] = {head: {"observations": len(values), "mean_abs_difference": float(np.mean(values)),
            "p99_abs_difference": float(np.quantile(values, .99)), "max_abs_difference": float(np.max(values))}
            for head, values in heads.items()}
    summary["passed_1e-5"] = all(value["max_abs_difference"] <= 1e-5
        for mode in summary["results"].values() for value in mode.values())
    output = base / "rotation-invariance-v1"; output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__": run(Path(os.environ["PKABENCH_RUNTIME"]))
