"""Canonical source adapters for the 500-unit infrastructure pilot.

ATLAS reads exact archived members in memory, never extracts loose structures.
Its eligibility rule is the frozen atlas release-v1 rule, not a new label policy.
"""
import hashlib
import gzip
import io
import json
import tarfile

import numpy as np

from .augmentation_graphs import align_residues


def _read_cif_gz(path):
    from biotite.structure.io import pdbx
    with gzip.open(path, "rt") as stream: cif = pdbx.CIFFile.read(stream)
    return pdbx.get_structure(cif, model=1, altloc="occupancy", use_author_fields=True, include_bonds=True)


def _pinder_native(src, split):
    """The frozen production query rule, without importing the training runtime."""
    from jaxpropka.topology import load_topology
    from jaxpropka.parameters import GROUPS
    from pkanet.graph import geometry
    from .site_graph_data import build_site_graph
    topology = load_topology(_read_cif_gz(src/"AB.cif.gz"), gap_policy="cap", freeze_disulfides=True)
    labels = json.loads((src/"labels.json").read_text())["pkai"]
    alias = {"NTR": "NTERM", "CTR": "CTERM"}
    maps = {state: {(str(c), int(n), str(ins), alias.get(g, g)): float(v)
                   for c, n, ins, g, v in labels.get(state, []) if v is not None and np.isfinite(v)}
            for state in ("AB", "A", "B")}
    lookup = {(k.chain, k.number, k.insertion): i for i, k in enumerate(topology.keys)}
    queries = []; interface = []
    for site in json.loads((src/"sites.json").read_text()):
        key = tuple(site[k] for k in ("chain", "resnum", "icode", "group"))
        if not site["train_mask" if split == "train" else "eval_mask"]: continue
        if key not in maps["AB"] or key not in maps[site["partner"]]: continue
        if site.get("w_burial") is None or site.get("w_interface") is None or key[:3] not in lookup: continue
        queries.append((lookup[key[:3]], GROUPS.index(key[3]))); interface.append(site["interface"])
    if not queries or not any(interface): raise ValueError("PINDER parent has no eligible native query")
    graph, valid = geometry(topology.backbone, topology.chain_index)
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[topology.native_index],
        np.stack((topology.nterm, topology.cterm, np.zeros(topology.n_residues, bool), valid), axis=-1)), axis=1).astype(np.float32)
    q = np.asarray(queries, np.int32)
    graph.update(nodes=nodes, node_mask=np.ones(topology.n_residues, bool), query_residue=q[:, 0], query_group=q[:, 1])
    graph.update(build_site_graph(topology.backbone, topology.chain_index, nodes, q[:, 0], q[:, 1]))
    graph.update(branch_edge_mask=np.stack((graph["edge_mask"], graph["edge_mask"] & (graph["edge"][..., -1] > 0.5))),
                 branch_site_edge_mask=np.stack((graph["site_edge_mask"], graph["site_edge_mask"] & (graph["site_edge"][..., 31] > 0.5))))
    return graph, topology


def _atlas(row, archive):
    from biotite.structure.io import pdbx
    from jaxpropka.topology import load_topology
    from jaxpropka.parameters import GROUPS
    from pkanet.graph import geometry
    from .site_graph_data import build_site_graph
    member = archive.extractfile(row["id"]+".tar.gz").read()
    if hashlib.sha256(member).hexdigest() != row["archive"]["sha256"]:
        raise ValueError("ATLAS source archive hash changed")
    with tarfile.open(fileobj=io.BytesIO(member), mode="r:gz") as tf:
        def read(name): return json.load(tf.extractfile(name))
        def cif(name): return pdbx.CIFFile.read(io.StringIO(tf.extractfile(name).read().decode()))
        source = pdbx.get_structure(cif("source.cif"), model=1, extra_fields=["b_factor"])
        topology = load_topology(pdbx.get_structure(cif("AB.cif"), model=1, altloc="occupancy",
                                                  use_author_fields=True, include_bonds=True),
                                 gap_policy="cap", freeze_disulfides=True)
        labels = read("labels.json"); sites = read("sites.json"); evidence = read("input_atom_mask.json")
    if labels["errors"]: raise ValueError("ATLAS source label errors")
    ca = source[source.atom_name == "CA"]
    conf = {(str(a.chain_id), int(a.res_id), str(a.ins_code).strip()): float(a.b_factor) for a in ca}
    if len(conf) != len(ca): raise ValueError("ambiguous ATLAS confidence mapping")
    lookup = {(k.chain, k.number, k.insertion): i for i, k in enumerate(topology.keys)}
    ev = {tuple(r["key"]) for r in evidence["atoms"]}
    alias = {"NTR": "NTERM", "CTR": "CTERM"}
    maps = {state: {(c, int(n), ins, alias.get(g, g)): v for c, n, ins, g, v in labels["pkai"][state]}
            for state in ("AB", "A", "B")}
    queries = []; flags = []
    for site in sites:
        key = tuple(site[k] for k in ("chain", "resnum", "icode", "group")); residue = key[:3]
        if not site["train_mask"] or not site["functional_atoms_complete"] or site["was_completed"] or site["is_break_terminus"]:
            continue
        if residue not in ev or residue not in lookup or conf.get(residue, -1) < 70: continue
        if site["group"] == "ARG" or key not in maps["AB"] or key not in maps[site["partner"]]: continue
        if not np.isfinite(maps["AB"][key]) or not np.isfinite(maps[site["partner"]][key]):
            raise ValueError("nonfinite native ATLAS label")
        queries.append((lookup[residue], GROUPS.index(site["group"])))
        flags.append(site["interface"])
    if not queries or not any(flags): raise ValueError("ATLAS parent has no eligible native interface query")
    graph, valid = geometry(topology.backbone, topology.chain_index)
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[topology.native_index],
        np.stack((topology.nterm, topology.cterm, np.zeros(topology.n_residues, bool), valid), axis=-1)), axis=1).astype(np.float32)
    q = np.asarray(queries, np.int32)
    graph.update(nodes=nodes, node_mask=np.ones(topology.n_residues, bool), query_residue=q[:, 0], query_group=q[:, 1])
    graph.update(build_site_graph(topology.backbone, topology.chain_index, nodes, q[:, 0], q[:, 1]))
    graph.update(branch_edge_mask=np.stack((graph["edge_mask"], graph["edge_mask"] & (graph["edge"][..., -1] > 0.5))),
                 branch_site_edge_mask=np.stack((graph["site_edge_mask"], graph["site_edge_mask"] & (graph["site_edge"][..., 31] > 0.5))))
    keys = [k.chain for k in topology.keys]
    if not set(keys) <= {"A", "B"}: raise ValueError("unexpected ATLAS partner IDs")
    return graph, topology.backbone, topology.chain_index, np.asarray(keys), \
        [{str(a.atom_name): a.coord for a in topology.residue(i)} for i in range(topology.n_residues)]


def canonical_source(root, record, *, atlas_archive=None):
    from jaxpropka.parameters import THREE_TO_INDEX
    from .production_inputs import _residues
    from .production_sidechains import _pkpdb_atoms
    from .production_graphs import _pkpdb_one, PINDER, source
    row = record["metadata"]["source"]; dataset = row["dataset"]; cid = row["id"]
    if dataset == "pkpdb":
        graph, _ = _pkpdb_one(root, cid)
        bb, chain, identity = _residues(dataset, root, cid)
        _, atoms, _ = _pkpdb_atoms(root, cid)
        partners = np.zeros(len(bb), np.int32); allowed = None
    elif dataset == "pinder":
        src = source(root, f"{PINDER}/entries/{cid}")
        graph, topology = _pinder_native(src, row["split"])
        bb = topology.backbone; chain = topology.chain_index; identity = topology.native_index
        lookup = {(k.chain, k.number, k.insertion): i for i, k in enumerate(topology.keys)}
        partners = np.full(len(bb), -1, np.int32)
        for i, state in enumerate(("A", "B")):
            mapping = json.loads((src/f"map_{state}.json").read_text())
            for _, _, c, n, ins in mapping:
                if (c, int(n), ins) in lookup:
                    j = lookup[c, int(n), ins]
                    if partners[j] != -1: raise ValueError("overlapping PINDER partners")
                    partners[j] = i
        if np.any(partners < 0): raise ValueError("unmapped PINDER parent residue")
        allowed = np.flatnonzero(partners == (0 if row["state"] == "A" else 1))
        atoms = [{str(a.atom_name): a.coord for a in topology.residue(i)} for i in range(len(bb))]
    elif dataset == "atlas":
        if atlas_archive is None: raise ValueError("ATLAS packed source archive required")
        graph, bb, chain, partners, atoms = _atlas(row, atlas_archive)
        identity = np.argmax(graph["nodes"][:, :20], axis=1)
        allowed = np.flatnonzero(partners == row["state"])
    else: raise ValueError(dataset)
    mapping = align_residues(record["arrays"]["source_backbone"],
        [THREE_TO_INDEX[n] for n in record["metadata"]["residue_names"]], bb, identity, allowed=allowed)
    if allowed is None and len(mapping) != len(bb) or allowed is not None and set(mapping) != set(allowed):
        raise ValueError("pilot source does not cover every canonical partner residue")
    if not np.array_equal(identity, graph["nodes"][:, :20].argmax(1)):
        raise ValueError("canonical graph residue identity mismatch")
    return graph, np.asarray(bb), np.asarray(chain), partners, atoms, mapping
