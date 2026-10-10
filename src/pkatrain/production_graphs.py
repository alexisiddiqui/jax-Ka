"""Production GQT graph stores for the pool-v3 datasets, built with the existing graph rules (2026-10-10).

Nothing here changes how a graph is made; each builder repeats an existing, verified construction:
- pKPDB (pretraining/pkpdb-full-v1): the residue graph of pkabench.pkpdb_mask_all.clean (same conformer resolution,
  residue selection, node features and 20 A geometry). Queries are every mapped site in entries/<id>/sites.json order,
  exactly as in the build's graph.npz, plus per-site train/eval masks and environment (rsa, w_burial, chain distance).
  Every structure is checked against its recorded receipt (n, k, q) and conformer selection.
- PINDER (pretraining/pinder-pkai-v1): gqt_paired_pinder._prepare_one (load_topology, 20 A geometry, paired rows,
  partner-removed branch masks) for every pool-v3 complex (training masks) and the 400 validation complexes
  (evaluation masks). Complexes without a graph-mapped paired interface site are recorded and skipped, as in prepare.
- benchmark-val (training/shared-v4-float32 'val', the 142-complex PypKa validation set both pools were screened
  against): graph_data.prepare (AB state, load_topology, 20 A geometry, supervision_eligible finite PypKa midpoints),
  node column 22 zeroed (strict backbone); eval_mask all True. Record paths are mapped from the source cluster's runtime.
All add the site graph of site_graph_data.build_site_graph.

Versions (PKATRAIN_GQT_VERSION, default gqt-production-v2):
- gqt-production-v1: pool-v3 training structures; validation = the 400 PINDER complexes of the factorial cohort.
- gqt-production-v2 (2026-10-10): pool-v4 (training/pool-v4, pkabench.training_pools build_v4): training pools plus
  the new validation sets, PINDER pinder-val.tsv (1,600 clusters, evaluation masks) and pKPDB pkpdb-val.tsv (1,600
  groups; split 'val', scored on train_mask by the loader). Sources not in the dataset images are read from
  <runtime>/overlay/<same relative path> (entries transferred after the images were made).

Layout under <runtime>/training/<version>/<dataset>/:
  ids.json                      the deterministic structure list (pool order, then validation)
  shards/<t>-of-<T>/            written by `build` tasks (task t takes ids[t::T]): records.bin (one zstd record per
                                structure) + receipt.json; re-runs skip finished shards
  store-v2/                     `pack`: records.bin (zstd records, ids.json order) + index.npz (dims, byte offsets) +
                                records.json + metadata.json + verification.json (every record decompressed and checked
                                against its build digest; sha256 per file). ProductionStore(path).raw(id) reads one.
  store-v1/                     superseded uncompressed layout (one flat .npy per field); `compress` converts it
  compare.json                  `compare`: rebuilt arrays vs reference graphs from the source cluster

Run under scripts/sqfs_run.sh (the sources are squashfs images).
  python -m pkatrain.production_graphs ids {pkpdb,pinder,benchmark-val}
  python -m pkatrain.production_graphs build {pkpdb,pinder} TASK TASKS
  python -m pkatrain.production_graphs pack {pkpdb,pinder}
  python -m pkatrain.production_graphs compress {pkpdb,pinder}      (store-v1 -> store-v2)
  python -m pkatrain.production_graphs compare {pkpdb,pinder} REFERENCE_DIR
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest

VERSION = os.environ.get("PKATRAIN_GQT_VERSION", "gqt-production-v2")
VERSIONS = ("gqt-production-v1", "gqt-production-v2")
if VERSION not in VERSIONS: raise ValueError(VERSION)
POOL_V4 = "training/pool-v4"
OVERLAY = "overlay"
DATASETS = ("pkpdb", "pinder", "benchmark-val")
BENCHMARK_SOURCE = "training/shared-v4-float32"   # benchmark records; its 'val' split is the 142-complex PypKa validation set
SOURCE_RUNTIME = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench"  # absolute prefix inside the benchmark records
PKPDB = "pretraining/pkpdb-full-v1"
PKPDB_STRUCTURES = "pretraining/pkpdb-v1/structures"
PINDER = "pretraining/pinder-pkai-v1"
PINDER_COHORT = "training/ogqt-pinder-factorial-v1/cohort.json"
GRAPH_FIELDS = ("nodes", "node_mask", "neighbors", "edge", "edge_mask", "switch", "query_residue", "query_group")
SITE_FIELDS = ("site_residue", "site_type", "site_mask", "site_neighbors", "site_edge", "site_edge_mask", "site_switch",
               "query_site")
FIELDS = {
    "pkpdb": GRAPH_FIELDS + ("labels",) + SITE_FIELDS + ("train_mask", "eval_mask", "rsa", "w_burial", "chain_distance_A"),
    "pinder": GRAPH_FIELDS + SITE_FIELDS + ("branch_edge_mask", "branch_site_edge_mask", "targets", "w_burial",
                                            "w_interface", "interface", "partner_distance_A", "rsa_free"),
    "benchmark-val": GRAPH_FIELDS + ("labels",) + SITE_FIELDS + ("eval_mask",),
}
DIMS = ("n", "k", "q", "s", "sk")


def shape(name, n, k, q, s, sk):
    return {"nodes": (n, 24), "node_mask": (n,), "neighbors": (n, k), "edge": (n, k, 20), "edge_mask": (n, k),
            "switch": (n, k), "query_residue": (q,), "query_group": (q,), "labels": (q,),
            "site_residue": (s,), "site_type": (s,), "site_mask": (s,), "site_neighbors": (s, sk),
            "site_edge": (s, sk, 34), "site_edge_mask": (s, sk), "site_switch": (s, sk), "query_site": (q,),
            "branch_edge_mask": (2, n, k), "branch_site_edge_mask": (2, s, sk), "targets": (2, q)}.get(name, (q,))


def read(path):
    return json.loads(Path(path).read_text())


def output(root, dataset):
    return Path(root) / "training" / VERSION / dataset


def _tsv(path):
    with open(path) as handle: return list(csv.DictReader(handle, delimiter="\t"))


def source(root, rel):
    """root/rel, or root/overlay/rel when the dataset image lacks it."""
    path = Path(root) / rel
    if path.exists(): return path
    overlay = Path(root) / OVERLAY / rel
    return overlay if overlay.exists() else path


# ---------------------------------------------------------------- structure lists
def ids(root, dataset):
    """[(id, split)]: v1: pool-v3 order (training), then for PINDER the fixed 400 validation complexes; v2: pool-v4
    order, then the dataset's pool-v4 validation set."""
    root = Path(root)
    if VERSION == "gqt-production-v2" and dataset in ("pkpdb", "pinder"):
        items = [(row["id"], "train") for row in _tsv(root / POOL_V4 / f"{dataset}.tsv")]
        items += [(row["id"], "val") for row in _tsv(root / POOL_V4 / f"{dataset}-val.tsv")]
    elif dataset == "pkpdb":
        items = [(row["id"], "train") for row in _tsv(root / PKPDB / "pool-v3.tsv")]
    elif dataset == "pinder":
        items = [(row["id"], "train") for row in _tsv(root / PINDER / "pool-v3.tsv")]
        items += [(row["id"], "val") for row in read(root / PINDER_COHORT)["records"] if row["split"] == "val"]
    elif dataset == "benchmark-val":
        items = [(cid, "val") for cid in read(root / BENCHMARK_SOURCE / "manifest.json")["val"]]
    else: raise ValueError(dataset)
    if len({cid for cid, _ in items}) != len(items): raise AssertionError("duplicate structure id")
    return items


def write_ids(root, dataset):
    root = Path(root); out = output(root, dataset); out.mkdir(parents=True, exist_ok=True)
    items = ids(root, dataset); path = out / "ids.json"
    sources = {"pkpdb": [f"{PKPDB}/pool-v3.tsv"], "pinder": [f"{PINDER}/pool-v3.tsv", PINDER_COHORT],
               "benchmark-val": [f"{BENCHMARK_SOURCE}/manifest.json"]}[dataset]
    if VERSION == "gqt-production-v2" and dataset != "benchmark-val":
        sources = [f"{POOL_V4}/{dataset}.tsv", f"{POOL_V4}/{dataset}-val.tsv", f"{POOL_V4}/pool-v4.json"]
    value = {"dataset": dataset, "version": VERSION, "ids": items, "sources": {p: digest(root / p) for p in sources}}
    if path.exists():
        if read(path)["ids"] != [list(x) for x in items]: raise AssertionError(f"{path} exists with a different list")
        return read(path)
    atomic_json(path, value); return value


# ---------------------------------------------------------------- pKPDB (pkpdb_mask_all.clean graph path)
def _pkpdb_one(root, cid):
    import biotite.structure as struc
    from biotite.structure.io import pdbx
    from jaxpropka.parameters import GROUPS, THREE_TO_INDEX
    from pkabench.conformers import resolve
    from pkabench.pkpdb_pilot_refs import cif
    from pkabench.prep import CANONICAL
    from pkanet.graph import geometry
    from .site_graph_data import build_site_graph, candidate_sites
    entry = source(root, f"{PKPDB}/entries/{cid}")
    receipt = read(entry / "receipt.json"); defects = read(entry / "defects.json")
    if receipt.get("status") != "accepted": raise AssertionError((cid, "receipt not accepted"))
    path = source(root, f"{PKPDB_STRUCTURES}/{cid[1:3]}/{cid}.cif.gz")
    if digest(path) != receipt["source_sha256"]: raise AssertionError((cid, "source structure hash"))
    selected = [row["chain"] for row in defects["sequences"]]
    polylen = {row["chain"]: len(row["sequence"]) for row in defects["sequences"]}
    file, conformers = resolve(cif(path), selected)
    if json.loads(json.dumps(conformers)) != read(entry / "conformers.json"): raise AssertionError((cid, "conformer selection differs"))
    label = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=False)
    author = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True)
    if not (len(label) == len(author) and np.array_equal(label.coord, author.coord)): raise AssertionError((cid, "label/author"))
    working = label.copy(); working.res_id = author.res_id.copy(); working.ins_code = author.ins_code.copy()
    keep = np.isin(working.res_name, list(CANONICAL)) & np.isin(working.chain_id, selected) & ~np.isin(np.char.upper(working.element), ["H", "D"])
    starts = struc.get_residue_starts(working, add_exclusive_stop=True)
    residues = set(); nodes = []; backbone = []; chainindex = []; lookup = defaultdict(list)
    for s, e in zip(starts[:-1], starts[1:]):
        if not keep[s]: continue
        a = working[s:e]; k = (str(a.chain_id[0]), int(a.res_id[0]), str(a.ins_code[0]).strip())
        if k in residues: raise AssertionError((cid, "ambiguous residue key", k))
        residues.add(k); names = {str(n): i for i, n in enumerate(a.atom_name)}
        if not {"N", "CA", "C"} <= names.keys(): raise AssertionError((cid, "incomplete backbone"))
        seqpos = int(label.res_id[s]); authchain = str(author.chain_id[s]); n = len(nodes)
        backbone.append(np.stack([a.coord[names[name]] if name in names else a.coord[names["C"]] for name in ("N", "CA", "C", "O")]))
        chainindex.append(selected.index(k[0]))
        aa = np.eye(20, dtype=np.float32)[THREE_TO_INDEX[str(a.res_name[0])]]
        nodes.append(np.concatenate((aa, [seqpos == 1, seqpos == polylen[k[0]], False, True])))
        lookup[authchain, k[1], k[2]].append(n)
    backbone = np.asarray(backbone); chainindex = np.asarray(chainindex)
    graph, valid = geometry(backbone, chainindex); nodes = np.asarray(nodes, np.float32); nodes[:, 23] = valid
    graph.update(nodes=nodes, node_mask=np.ones(len(nodes), bool))
    sites = read(entry / "sites.json")
    environment = {(r["chain"], r["resnum"], r["icode"], r["group"]): r for r in read(entry / "environment.json")["sites"]}
    queries = []; per = defaultdict(list)
    for site in sites:
        choices = lookup.get((site["chain"], site["resnum"], site["icode"]), [])
        if len(choices) != 1: raise AssertionError((cid, "site does not map to one residue", site["chain"], site["resnum"]))
        queries.append((choices[0], GROUPS.index(site["group"])))
        env = environment[site["chain"], site["resnum"], site["icode"], site["group"]]
        per["labels"].append(site["pka"]); per["train_mask"].append(site["train_mask"]); per["eval_mask"].append(site["eval_mask"])
        per["rsa"].append(env["rsa"]); per["w_burial"].append(env["w_burial"])
        per["chain_distance_A"].append(np.nan if env["chain_distance_A"] is None else env["chain_distance_A"])
    q = np.asarray(queries, np.int32); graph.update(query_residue=q[:, 0], query_group=q[:, 1])
    found = (len(nodes), int(graph["neighbors"].shape[1]), len(sites))
    if found != (receipt["n"], receipt["k"], receipt["q"]): raise AssertionError((cid, "n/k/q differ from receipt", found))
    # Site tokens exist only for candidate sites (site_graph_data.candidate_sites). A terminal label on a residue that
    # is not the chain's first/last sequence position has no token; the build never trains or evaluates such sites
    # (pkpdb_mask_all terminal rule), so they keep query_site = -1 and must be masked.
    candidates = {(int(r), int(g)) for r, g in candidate_sites(nodes)}
    token = np.asarray([(int(r), int(g)) in candidates for r, g in q], bool)
    masked = np.asarray(per["train_mask"], bool) | np.asarray(per["eval_mask"], bool)
    if np.any(~token & masked): raise AssertionError((cid, "supervised site without a site token"))
    site = build_site_graph(backbone, chainindex, nodes, q[token, 0], q[token, 1])
    query_site = np.full(len(q), -1, np.int32); query_site[token] = site["query_site"]; site["query_site"] = query_site
    graph.update(site)
    graph.update(labels=np.asarray(per["labels"], np.float32), train_mask=np.asarray(per["train_mask"], bool),
                 eval_mask=np.asarray(per["eval_mask"], bool),
                 **{name: np.asarray(per[name], np.float32) for name in ("rsa", "w_burial", "chain_distance_A")})
    return graph, {"train_sites": int(np.sum(per["train_mask"])), "eval_sites": int(np.sum(per["eval_mask"])),
                   "untokenised_sites": int(np.sum(~token))}


# ---------------------------------------------------------------- PINDER (gqt_paired_pinder._prepare_one)
def _pinder_one(root, cid, split):
    from jaxpropka.parameters import GROUPS
    from jaxpropka.topology import load_topology
    from pkanet.graph import geometry
    from .gqt_paired_pinder import _paired_rows, _read_cif_gz
    from .site_graph_data import build_site_graph
    src = source(root, f"{PINDER}/entries/{cid}")
    topology = load_topology(_read_cif_gz(src / "AB.cif.gz"), gap_policy="cap", freeze_disulfides=True)
    graph, frames_valid = geometry(topology.backbone, topology.chain_index, radius=20.0)
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[topology.native_index],
        np.stack((topology.nterm, topology.cterm, np.zeros(topology.n_residues, bool), frames_valid), axis=-1)), axis=-1)
    graph.update(nodes=nodes.astype(np.float32), node_mask=np.ones(topology.n_residues, bool))
    paired = _paired_rows(src, split); lookup = {(key.chain, key.number, key.insertion): i for i, key in enumerate(topology.keys)}
    queries = []; targets = []; burial = []; interface_weight = []; interface = []; distance = []; rsa = []
    for row in paired:
        residue_key = row["key"][:3]
        if residue_key not in lookup: continue
        queries.append((lookup[residue_key], GROUPS.index(row["key"][3])))
        targets.append((row["target_ab"], row["target_free"])); burial.append(row["w_burial"])
        interface_weight.append(row["w_interface"]); interface.append(row["interface"])
        distance.append(row["partner_distance_A"]); rsa.append(row["rsa_free"])
    if not queries or not any(interface): return None, {"reason": "no graph-mapped paired interface sites"}
    query = np.asarray(queries, np.int32); graph.update(query_residue=query[:, 0], query_group=query[:, 1])
    site = build_site_graph(topology.backbone, topology.chain_index, nodes, query[:, 0], query[:, 1])
    residue_free = graph["edge_mask"] & (graph["edge"][..., -1] > 0.5)
    site_free = site["site_edge_mask"] & (site["site_edge"][..., 31] > 0.5)
    graph.update(site, branch_edge_mask=np.stack((graph["edge_mask"], residue_free)),
        branch_site_edge_mask=np.stack((site["site_edge_mask"], site_free)),
        targets=np.asarray(targets, np.float32).T, w_burial=np.asarray(burial, np.float32),
        w_interface=np.asarray(interface_weight, np.float32), interface=np.asarray(interface, bool),
        partner_distance_A=np.asarray(distance, np.float32), rsa_free=np.asarray(rsa, np.float32))
    return graph, {"q_interface": int(np.sum(interface))}


# ---------------------------------------------------------------- benchmark validation (graph_data.prepare)
def _local(root, path):
    """Benchmark records hold absolute source-cluster paths; map them under this runtime."""
    path = str(path)
    return Path(root) / Path(path).relative_to(SOURCE_RUNTIME) if path.startswith(SOURCE_RUNTIME + "/") else Path(path)


def _benchmark_one(root, cid):
    """graph_data.prepare for one benchmark validation complex: AB structure, load_topology (gap cap), 20 A geometry,
    current PypKa midpoints that are supervision_eligible and finite, plus the site graph. Node column 22 (disulfide,
    not a backbone observable) is zeroed here, as graph_data.mask_features does at load under strict_backbone."""
    from jaxpropka.parameters import GROUP_AA, GROUPS
    from jaxpropka.topology import load_topology
    from pkabench.prep import read_cif
    from pkanet.graph import geometry
    from .records import KEY
    from .site_graph_data import build_site_graph
    source = root / BENCHMARK_SOURCE; parent = read(source / "manifest.json")
    record_path = source / "records" / f"{cid}.json"
    if digest(record_path) != parent["records_sha256"][cid]: raise AssertionError((cid, "record hash"))
    r = read(record_path); state = r["states"]["AB"]; raw = _local(root, state["source"])
    cif = _local(root, r["structures"]["AB"])
    if digest(cif) != r["structure_sha256"]["AB"]: raise AssertionError((cid, "structure hash"))
    for name in ("mapping.json", "result.json"):
        if digest(raw / name) != state["source_hashes"][name]: raise AssertionError((cid, name))
    sitepath = _local(root, state["export"]) / "sites.json"
    if digest(sitepath) != state["sites_sha256"]: raise AssertionError((cid, "sites hash"))
    masks = {tuple(s[k] for k in KEY): s for s in read(sitepath)}
    mapping = {(m["chain"], m["resnum"]): tuple(m["original"]) for m in read(raw / "mapping.json")}
    topology = load_topology(read_cif(cif), gap_policy="cap", freeze_disulfides=True)
    lookup = {(k.chain, k.number, k.insertion): i for i, k in enumerate(topology.keys)}
    graph, frames_valid = geometry(topology.backbone, topology.chain_index)
    nodes = np.concatenate((np.eye(20, dtype=np.float32)[topology.native_index],
        np.stack((topology.nterm, topology.cterm, topology.disulfide, frames_valid), axis=-1)), axis=-1).astype(np.float32)
    nodes[:, 22] = 0.0
    graph.update(nodes=nodes, node_mask=np.ones(topology.n_residues, bool))
    queries = []; labels = []; seen = set()
    for s in read(raw / "result.json")["rows"]:
        original = mapping[s["chain"], s["resnum"]]; key = (cid, *original, s["group"])
        if key in seen: raise AssertionError((cid, "duplicate site"))
        seen.add(key)
        if not masks[key]["supervision_eligible"]: continue
        if s["pka"] is None or not np.isfinite(s["pka"]): continue
        i = lookup[original]; g = GROUPS.index(s["group"])
        if g < 7 and topology.native_index[i] != GROUP_AA[g]: raise AssertionError((cid, "residue type", key))
        queries.append((i, g)); labels.append(s["pka"])
    if not labels: raise AssertionError((cid, "no eligible scalar sites"))
    q = np.asarray(queries, np.int32); graph.update(query_residue=q[:, 0], query_group=q[:, 1])
    graph.update(build_site_graph(topology.backbone, topology.chain_index, nodes, q[:, 0], q[:, 1]))
    graph.update(labels=np.asarray(labels, np.float32), eval_mask=np.ones(len(labels), bool))
    return graph, {"component_id": r["component_id"], "role": r["role"]}


def build_one(root, dataset, cid, split):
    """(arrays, record) or (None, skip record)."""
    root = Path(root)
    if dataset == "pkpdb": arrays, extra = _pkpdb_one(root, cid)
    elif dataset == "benchmark-val": arrays, extra = _benchmark_one(root, cid)
    else: arrays, extra = _pinder_one(root, cid, split)
    if arrays is None: return None, {"id": cid, "split": split, **extra}
    dims = {"n": arrays["nodes"].shape[0], "k": arrays["neighbors"].shape[1], "q": arrays["query_residue"].shape[0],
            "s": arrays["site_type"].shape[0], "sk": arrays["site_neighbors"].shape[1]}
    for name in FIELDS[dataset]:
        if arrays[name].shape != shape(name, *(dims[d] for d in DIMS)): raise AssertionError((cid, name, arrays[name].shape))
    return {name: arrays[name] for name in FIELDS[dataset]}, {"id": cid, "split": split, **{d: int(v) for d, v in dims.items()},
                                                              "arrays_sha256": arrays_digest(arrays, FIELDS[dataset]), **extra}


# ---------------------------------------------------------------- compressed records
# A record is every field's raw bytes, concatenated in FIELDS order, as one zstd frame (level ZSTD_LEVEL). Records
# are individually addressable (offset, length) in one records.bin per store. Measured on 4,000 PINDER structures
# (Isambard, 2026-10-10): 3.2x smaller than the raw arrays and 416 structures/s at 8 threads, against 138/s for the
# uncompressed per-field mmap store (and 172/s for one file per structure in a squashfs image).
ZSTD_LEVEL = 3


def encode(arrays, fields):
    import zstandard
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(b"".join(np.ascontiguousarray(arrays[f]).tobytes() for f in fields))


def decode(blob, fields, dtypes, dims):
    import zstandard
    raw = zstandard.ZstdDecompressor().decompress(blob); out = {}; position = 0
    for f in fields:
        dtype = np.dtype(dtypes[f]); size = int(np.prod(shape(f, *dims)))
        out[f] = np.frombuffer(raw, dtype, count=size, offset=position).reshape(shape(f, *dims)); position += size * dtype.itemsize
    if position != len(raw): raise ValueError("record length does not match its dims")
    return out


def arrays_digest(arrays, fields):
    h = hashlib.sha256()
    for f in fields: h.update(f.encode()); h.update(np.ascontiguousarray(arrays[f]).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- shards
def build_shard(root, dataset, task, tasks):
    """Task t of T builds ids[t::T] one structure at a time and appends each as a compressed record to
    shards/<name>/records.bin (bounded memory, no uncompressed intermediate); the directory appears when complete."""
    root = Path(root); out = output(root, dataset); shards = out / "shards"; shards.mkdir(parents=True, exist_ok=True)
    listing = read(out / "ids.json")["ids"]; mine = listing[task::tasks]; name = f"{task:04d}-of-{tasks:04d}"
    folder = shards / name
    if (folder / "receipt.json").exists(): return read(folder / "receipt.json")
    pending = shards / f".{name}.pending-{os.getpid()}"; pending.mkdir()
    began = time.time(); records = []; skipped = []; failures = []; dtypes = {}; fields = FIELDS[dataset]; position = 0
    with open(pending / "records.bin", "wb") as handle:
        for number, (cid, split) in enumerate(mine, 1):
            try:
                arrays, record = build_one(root, dataset, cid, split)
            except Exception as exc:
                failures.append({"id": cid, "split": split, "error": repr(exc)}); continue
            if arrays is None: skipped.append(record); continue
            for f in fields:
                if dtypes.setdefault(f, arrays[f].dtype.str) != arrays[f].dtype.str: raise AssertionError((cid, f, arrays[f].dtype))
            blob = encode(arrays, fields); handle.write(blob)
            records.append({**record, "blob": [position, len(blob)]}); position += len(blob)
            if number % 25 == 0: print(json.dumps({"dataset": dataset, "task": task, "done": number, "of": len(mine)}), flush=True)
    receipt = {"dataset": dataset, "task": task, "tasks": tasks, "assigned": len(mine), "records": records, "skipped": skipped,
               "failures": failures, "dtypes": dtypes, "seconds": round(time.time() - began, 1)}
    atomic_json(pending / "receipt.json", receipt)
    if failures:
        atomic_json(shards / f"{name}.failures.json", receipt)
        raise RuntimeError(f"{len(failures)} failures in shard {name}; see {shards / name}.failures.json")
    os.replace(pending, folder)
    return receipt


# ---------------------------------------------------------------- packed store
def _write_store(out, dataset, records, dtypes, blobs, skipped, workers):
    """records in final order; blobs yields each record's compressed bytes in that order. Writes store-v2 into a
    pending directory, verifies every record (decompress + arrays_sha256) in parallel, then installs it."""
    from concurrent.futures import ThreadPoolExecutor
    fields = FIELDS[dataset]; destination = out / "store-v2"
    pending = out / f".store-v2.pending-{os.getpid()}"; pending.mkdir()
    offsets = np.zeros(len(records) + 1, np.int64)
    with open(pending / "records.bin", "wb") as handle:
        for i, blob in enumerate(blobs):
            handle.write(blob); offsets[i + 1] = offsets[i] + len(blob)
            if (i + 1) % 5000 == 0: print(json.dumps({"written": i + 1, "of": len(records)}), flush=True)
    if len(offsets) - 1 != len(records) or offsets[-1] != (pending / "records.bin").stat().st_size: raise AssertionError("record count")
    np.savez(pending / "index.npz", ids=np.asarray([r["id"] for r in records]),
             dims=np.asarray([[r[d] for d in DIMS] for r in records], np.int32), offsets=offsets)
    atomic_json(pending / "records.json", [{k: v for k, v in r.items() if k != "blob"} for r in records])
    atomic_json(pending / "metadata.json", {"version": VERSION, "format": "zstd-records-v1", "zstd_level": ZSTD_LEVEL,
        "dataset": dataset, "fields": list(fields), "dtypes": dtypes, "dims": list(DIMS), "structures": len(records),
        "skipped": sorted(skipped.values(), key=lambda r: r["id"]), "ids_sha256": digest(out / "ids.json"),
        "sites": int(sum(r["q"] for r in records)), "raw_bytes": int(sum(sum(int(np.prod(shape(f, *(r[d] for d in DIMS)))) * np.dtype(dtypes[f]).itemsize
                                                                            for f in fields) for r in records)),
        "compressed_bytes": int(offsets[-1])})
    store = ProductionStore(pending, verify=False)
    def check(record):
        return None if arrays_digest(store.raw(record["id"]), fields) == record["arrays_sha256"] else record["id"]
    with ThreadPoolExecutor(workers) as pool: bad = [cid for cid in pool.map(check, records) if cid]
    store.close()
    if bad: raise AssertionError(f"{len(bad)} records differ after packing, e.g. {bad[:5]}")
    files = {p.name: digest(p) for p in sorted(pending.iterdir())}
    atomic_json(pending / "verification.json", {"passed": True, "records_checked": len(records), "files": files,
                "bytes": sum(p.stat().st_size for p in pending.iterdir())})
    os.replace(pending, destination)
    return read(destination / "verification.json")


def pack(root, dataset, workers=8):
    """Concatenate the shards' compressed records (ids.json order) into store-v2/records.bin."""
    root = Path(root); out = output(root, dataset)
    if (out / "store-v2" / "verification.json").exists(): return read(out / "store-v2" / "verification.json")
    listing = [tuple(x) for x in read(out / "ids.json")["ids"]]
    receipts = [read(p) for p in sorted((out / "shards").glob("*/receipt.json"))]
    tasks = {r["tasks"] for r in receipts}
    if len(tasks) != 1 or len(receipts) != next(iter(tasks)): raise AssertionError(f"incomplete shards: {len(receipts)} of {tasks}")
    by_id = {}; folder_of = {}
    for receipt in receipts:
        for r in receipt["records"]: by_id[r["id"]] = r; folder_of[r["id"]] = out / "shards" / f"{receipt['task']:04d}-of-{receipt['tasks']:04d}"
    skipped = {r["id"]: r for receipt in receipts for r in receipt["skipped"]}
    if set(by_id) | set(skipped) != {cid for cid, _ in listing}: raise AssertionError("shards do not cover ids.json")
    records = [by_id[cid] for cid, _ in listing if cid in by_id]
    dtypes = next(r["dtypes"] for r in receipts if r["records"])
    if any(r["dtypes"] != dtypes for r in receipts if r["records"]): raise AssertionError("shard dtypes differ")
    handles = {}
    def blobs():
        for r in records:
            folder = folder_of[r["id"]]
            if folder not in handles: handles[folder] = os.open(folder / "records.bin", os.O_RDONLY)
            yield os.pread(handles[folder], r["blob"][1], r["blob"][0])
    try: return _write_store(out, dataset, records, dtypes, blobs(), skipped, workers)
    finally:
        for fd in handles.values(): os.close(fd)


def compress_store(root, dataset, workers=8):
    """Convert an uncompressed store-v1 (one flat .npy per field) into store-v2, record by record."""
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root); out = output(root, dataset)
    if (out / "store-v2" / "verification.json").exists(): return read(out / "store-v2" / "verification.json")
    old = ProductionStore(out / "store-v1"); records = read(out / "store-v1" / "records.json"); fields = FIELDS[dataset]
    skipped = {r["id"]: r for r in old.metadata["skipped"]}
    def blob(record): return encode(old.raw(record["id"]), fields)
    def blobs():
        with ThreadPoolExecutor(workers) as pool:
            for start in range(0, len(records), 4 * workers):  # bounded look-ahead keeps memory flat
                yield from pool.map(blob, records[start:start + 4 * workers])
    try: return _write_store(out, dataset, records, old.metadata["dtypes"], blobs(), skipped, workers)
    finally: old.close()


class ProductionStore:
    """Read-only access to a packed store: raw(id) -> {field: array} with the recorded shapes. store-v2 (zstd records):
    one pread + decompress per structure, safe from many threads; store-v1 (flat per-field .npy): mmap views."""

    def __init__(self, path, verify=True):
        self.path = Path(path); self.metadata = read(self.path / "metadata.json")
        if verify and not read(self.path / "verification.json")["passed"]: raise AssertionError(self.path)
        with np.load(self.path / "index.npz") as index:
            self.ids = index["ids"].astype(str); self.dims = index["dims"]; self.offsets = index["offsets"]
        self.by_id = {cid: i for i, cid in enumerate(self.ids)}; self.fields = self.metadata["fields"]
        self.compressed = self.metadata.get("format") == "zstd-records-v1"
        if self.compressed: self.fd = os.open(self.path / "records.bin", os.O_RDONLY); self.arrays = {}
        else: self.fd = None; self.arrays = {f: np.load(self.path / f"{f}.npy", mmap_mode="r") for f in self.fields}

    def raw(self, cid):
        i = self.by_id[cid]; dims = [int(x) for x in self.dims[i]]
        if self.compressed:
            start, stop = int(self.offsets[i]), int(self.offsets[i + 1])
            return decode(os.pread(self.fd, stop - start, start), self.fields, self.metadata["dtypes"], dims)
        return {f: self.arrays[f][self.offsets[i, j]:self.offsets[i + 1, j]].reshape(shape(f, *dims)) for j, f in enumerate(self.fields)}

    def close(self):
        self.arrays.clear()
        if self.fd is not None: os.close(self.fd); self.fd = None


# ---------------------------------------------------------------- comparison with the source cluster's graphs
# Edge and site-edge features may differ in the last float32 bit across numpy versions/CPUs (einsum summation order
# in the direction/orientation terms; measured max 2.4e-7, numpy 2.5.3 aarch64 vs 2.2.6 x86). Everything else is exact.
FLOAT_TOLERANT = ("edge", "site_edge")
EDGE_TOLERANCE = 1e-6
GEOMETRY_FIELDS = GRAPH_FIELDS[:6] + SITE_FIELDS[:7] + ("branch_edge_mask", "branch_site_edge_mask")


def compare(root, dataset, reference):
    """Rebuild the structures whose reference graph is under REFERENCE/<id>/graph.npz and compare arrays exactly.
    pKPDB references are pkpdb-full-v1 graph.npz files (residue graph, queries and labels identical; edge features to
    EDGE_TOLERANCE).
    PINDER references are ogqt-pinder-factorial-v1 graphs: the geometry (residue graph, site graph, branch masks) must
    be identical; per-site fields may differ only where the complex's sites.json/labels.json/meta.json changed since
    the factorial registration (recorded source hashes), because the rebuild uses the current sources."""
    root = Path(root); reference = Path(reference); results = Counter(); mismatches = []; max_difference = {}
    split_of = dict((cid, split) for cid, split in read(output(root, dataset) / "ids.json")["ids"])
    recorded = {r["id"]: r["source_hashes"] for r in read(root / PINDER_COHORT)["records"]} if dataset == "pinder" else {}
    for path in sorted(reference.glob("*/graph.npz")):
        cid = path.parent.name
        if cid not in split_of: results["not_in_ids"] += 1; continue
        arrays, _ = build_one(root, dataset, cid, split_of[cid])
        with np.load(path, allow_pickle=False) as ref:
            ref = {n: ref[n] for n in ref.files}
            if dataset == "benchmark-val": ref["nodes"] = ref["nodes"].copy(); ref["nodes"][:, 22] = 0.0
            names = [n for n in ref if n in arrays] if dataset == "pinder" else list(GRAPH_FIELDS) + ["labels"]
            bad = []
            for n in names:
                same_shape = ref[n].shape == arrays[n].shape
                if same_shape and n in FLOAT_TOLERANT:  # einsum direction terms: float32 rounding differs by numpy/CPU
                    difference = float(np.max(np.abs(ref[n].astype(np.float64) - arrays[n]), initial=0.0))
                    max_difference[n] = max(max_difference.get(n, 0.0), difference)
                    if difference > EDGE_TOLERANCE: bad.append(n)
                elif not (same_shape and np.array_equal(ref[n], arrays[n])): bad.append(n)
        if not bad: results["identical"] += 1; continue
        if dataset == "pinder" and not set(bad) & set(GEOMETRY_FIELDS):
            changed = [n for n, h in recorded[cid].items() if digest(root / PINDER / "entries" / cid / n) != h]
            if changed: results["site_fields_differ_sources_revised"] += 1; continue
        results["different"] += 1; mismatches.append({"id": cid, "fields": bad})
    report = {"dataset": dataset, "reference": str(reference), "counts": dict(results), "mismatches": mismatches[:50],
              "edge_tolerance": EDGE_TOLERANCE, "max_abs_difference": max_difference,
              "passed": sum(results.values()) - results["not_in_ids"] > 0 and not mismatches}
    atomic_json(output(root, dataset) / "compare.json", report); return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(os.environ["PKABENCH_RUNTIME"]); action, dataset = argv[0], argv[1]
    if dataset not in DATASETS: raise ValueError(dataset)
    if action == "ids": print(len(write_ids(root, dataset)["ids"]))
    elif action == "build":
        receipt = build_shard(root, dataset, int(argv[2]), int(argv[3]))
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in receipt.items()}))
    elif action == "pack":
        print(json.dumps({k: v for k, v in pack(root, dataset, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))).items() if k != "files"}))
    elif action == "compress":
        print(json.dumps({k: v for k, v in compress_store(root, dataset, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))).items() if k != "files"}))
    elif action == "compare":
        report = compare(root, dataset, argv[2]); print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}))
        if not report["passed"]: raise SystemExit(1)
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
