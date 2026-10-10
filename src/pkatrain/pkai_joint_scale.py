"""Joint pKPDB state and PINDER Siamese training for native pKAI.

The initial smoke gate uses the frozen 5k pilot only.  Production pool loading
is deliberately separate so no run can start before the revised pool manifests
and environment weights are complete.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
import csv
import math
import time
from pathlib import Path

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io import pdbx

from pkabench.runtime import atomic_json, digest, require_compute
from pkabench.pkai_backbone_pinder_eval import _features as backbone_features
from .pkai_scratch import CUTOFF, GEOMETRY, SLOTS, architecture_gate, feature_matrix, feature_width, model_class, native


SEED = 17
MODES = ("backbone", "full")
SIDECHAIN_GROUPS = frozenset(("ASP", "CYS", "TYR", "GLU", "HIS", "LYS"))
GROUP_ALIAS = {"NTR": "NTERM", "CTR": "CTERM"}
KEY_FIELDS = ("chain", "resnum", "icode", "group")
# pool-v3 fraction (nested 0.1/0.5/0.75/1.0 subsets); PKAI_FRACTION selects another one, with its own output directory
FRACTION = float(os.environ.get("PKAI_FRACTION", "0.1"))
# neighbour-slot encoding (pkai_scratch): native "atom16" (4,008 inputs) or "aa20" (5,008); PKAI_ENCODING=aa20 uses its
# own output directory and pKPDB validation package
ENCODING = os.environ.get("PKAI_ENCODING", "atom16")
WIDTH = feature_width(ENCODING)
BATCH_SIZE = 256
REFERENCE_BATCH = 64
REFERENCE_LR = 1e-6
LEARNING_RATE = REFERENCE_LR * math.sqrt(BATCH_SIZE / REFERENCE_BATCH)
OBJECTIVES = ("pkpdb", "joint")
# Feature failures that are properties of the record, not of the code: the record is excluded and listed in the feature
# store's metadata. Any other error still fails the shard and blocks packing.
EXCLUDABLE = ("no mapped pKPDB sites", "no paired sites", "PDB capacity", "Coincident pKAI environment/reference atoms",
              "non-canonical residue in pKAI environment")


def _excludable(failure):
    return any(reason in failure["error"] for reason in EXCLUDABLE)


def read(path):
    return json.loads(Path(path).read_text())


def output(root):
    suffix = "" if FRACTION == 0.1 else f"-f{round(FRACTION * 100)}"  # the registered 10% runs keep their path
    if ENCODING != "atom16": suffix += f"-{ENCODING}"
    suffix += GEOMETRY  # non-native cutoff/slots (pkai_scratch)
    return Path(root) / f"training/pkai-joint-scale-v1{suffix}"


def _pool_rows(path, fraction=FRACTION):
    with open(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if float(row["min_fraction"]) <= fraction + 1e-12]


def register(root):
    """Freeze the four 10% arms after both leakage-screened pools exist."""
    root = Path(root); out = output(root); out.mkdir(parents=True, exist_ok=True)
    pk_pool = root / "pretraining/pkpdb-full-v1/pool-v3.tsv"
    pi_pool = root / "pretraining/pinder-pkai-v1/pool-v3.tsv"
    pk_manifest = pk_pool.with_suffix(".json"); pi_manifest = pi_pool.with_suffix(".json")
    for path in (pk_pool, pi_pool, pk_manifest, pi_manifest):
        if not path.exists(): raise FileNotFoundError(path)
    pk = _pool_rows(pk_pool); pi = _pool_rows(pi_pool)
    fixed = read(cohort(root))
    val = [row for row in fixed["records"] if row["split"] == "val"]
    if len(val) != 400 or len({row["cluster_id"] for row in val}) != 400:
        raise AssertionError("expected fixed 400-cluster PINDER validation cohort")
    records = {
        "pkpdb_train": [row["id"] for row in pk],
        "pinder_train": [row["id"] for row in pi],
        "pinder_val": [row["id"] for row in val],
    }
    atomic_json(out / "records.json", records)
    manifest = {
        "version": f"pkai-joint-scale-{round(FRACTION * 100)}pct-v1", "fraction": FRACTION,
        "modes": list(MODES), "objectives": list(OBJECTIVES), "seed": SEED,
        "batch_size": BATCH_SIZE, "reference_batch_size": REFERENCE_BATCH,
        "reference_learning_rate": REFERENCE_LR, "learning_rate": LEARNING_RATE,
        "learning_rate_rule": "eta = 1e-6 * sqrt(batch / 64)",
        "initialization": "scratch; PyTorch Linear defaults", "precision": "float32",
        "losses": {"pkpdb_state": "burial-weighted MSE", "pinder_state": "burial-weighted MSE",
                   "pinder_siamese": "interface-distance-weighted MSE"},
        "validation": "unweighted fixed pKPDB validation plus fixed 400-cluster PINDER validation; no test data",
        "cpus_per_gpu": 2,
        "inputs": {"pkpdb_pool": str(pk_pool), "pkpdb_pool_sha256": digest(pk_pool),
                   "pkpdb_pool_manifest_sha256": digest(pk_manifest),
                   "pinder_pool": str(pi_pool), "pinder_pool_sha256": digest(pi_pool),
                   "pinder_pool_manifest_sha256": digest(pi_manifest),
                   "pinder_validation_sha256": digest(root / "training/ogqt-pinder-factorial-v1/cohort.json")},
        "counts": {"pkpdb_structures": len(pk), "pkpdb_sites_declared": sum(int(r["labelled_sites"]) for r in pk),
                   "pinder_complexes": len(pi), "pinder_sites_declared": sum(int(r["labelled_sites"]) for r in pi),
                   "pinder_validation_complexes": len(val)},
        "records_sha256": digest(out / "records.json"), "test_data_included": False,
    }
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], sort_keys=True), flush=True)
    return manifest


def pinder_source(root):
    return Path(root) / "pretraining/pinder-pkai-v1"


def cohort(root):
    return Path(root) / "training/ogqt-pinder-factorial-v1/cohort.json"


def _paired_rows(folder, split):
    labels = read(folder / "labels.json")["pkai"]
    maps = {}
    for state in ("AB", "A", "B"):
        maps[state] = {(str(chain), int(number), str(insertion), GROUP_ALIAS.get(group, group)): float(value)
                       for chain, number, insertion, group, value in labels.get(state, [])
                       if value is not None and np.isfinite(value)}
    mask_name = "train_mask" if split == "train" else "eval_mask"
    result = []
    for site in read(folder / "sites.json"):
        key = tuple(site[name] for name in KEY_FIELDS)
        free_state = site["partner"]
        if (not site[mask_name] or key not in maps["AB"] or key not in maps[free_state]
                or site.get("w_burial") is None or site.get("w_interface") is None):
            continue
        result.append({**site, "key": key, "target_ab": maps["AB"][key],
                       "target_free": maps[free_state][key]})
    return result


def _read_cif(path):
    with gzip.open(path, "rt") as stream:
        file = pdbx.CIFFile.read(stream)
    return pdbx.get_structure(file, model=1, altloc="occupancy",
                              use_author_fields=True, include_bonds=False)


def _export_pdb(atoms, path):
    """Local copy of the benchmark's author-key-preserving heavy-atom export."""
    import string
    out = atoms.copy(); mapping = {}; number = 0
    codes = string.ascii_uppercase + string.ascii_lowercase + string.digits
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    segment = -1; previous = None
    for start, stop in zip(starts[:-1], starts[1:]):
        number += 1
        residue = atoms[start:stop]
        new_segment = previous is None or str(previous.chain_id[0]) != str(residue.chain_id[0])
        if not new_segment:
            carbon = previous.coord[previous.atom_name == "C"]
            nitrogen = residue.coord[residue.atom_name == "N"]
            new_segment = (len(carbon) != 1 or len(nitrogen) != 1 or
                           float(np.linalg.norm(carbon[0] - nitrogen[0])) > 2.0)
        if new_segment: segment += 1
        if segment >= len(codes) or number > 9999: raise ValueError("PDB capacity")
        chain = codes[segment]; previous = residue
        mapping[(chain, number)] = (str(atoms.chain_id[start]), int(atoms.res_id[start]),
                                    str(atoms.ins_code[start]).strip())
        out.chain_id[start:stop] = chain; out.res_id[start:stop] = number
        out.ins_code[start:stop] = ""
    file = PDBFile(); file.set_structure(out); file.write(path)
    check = PDBFile.read(path).get_structure(model=1)
    if len(check) != len(out) or not np.array_equal(check.atom_name, out.atom_name):
        raise ValueError("PDB export changed atoms")
    return mapping


def _array_digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        h.update(str(value.shape).encode())
        h.update(value.dtype.str.encode())
        h.update(value.view(np.uint8))
    return h.hexdigest()


def _full_features(root, cid, state, keys):
    """Generate native pKAI inputs and restore author residue identifiers."""
    atoms = _read_cif(pinder_source(root) / "entries" / cid / f"{state}.cif.gz")
    with tempfile.TemporaryDirectory(prefix="pkai-joint-smoke-") as directory:
        path = Path(directory) / "input.pdb"
        mapping = _export_pdb(atoms, path)
        torch, package = native()
        import sys
        sys.path.insert(0, str(package))
        from protein import Protein
        residues, matrix = feature_matrix(Protein(path), ENCODING)
    lookup = {}
    for index, residue in enumerate(residues):
        original = mapping[(str(residue.chain), int(residue.resnumb))]
        key = (*original, str(residue.resname))
        if key in lookup:
            raise AssertionError((cid, state, key, "duplicate pKAI site"))
        lookup[key] = index
    retained = np.asarray([key in lookup for key in keys], bool)
    result = np.zeros((len(keys), WIDTH), np.float32)
    for index, key in enumerate(keys):
        if retained[index]:
            result[index] = matrix[lookup[key]]
    return result, retained


def _state_features(root, cid, state, keys, mode):
    atoms = _read_cif(pinder_source(root) / "entries" / cid / f"{state}.cif.gz")
    if mode == "backbone":
        return backbone_features(atoms, keys, ENCODING)
    return _full_features(root, cid, state, keys)


def _pinder_batch(root, mode, minimum=16):
    records = [row for row in read(cohort(root))["records"] if row["split"] == "train"]
    for record in records:
        cid = record["id"]
        folder = pinder_source(root) / "entries" / cid
        paired = [row for row in _paired_rows(folder, "train")
                  if row["key"][3] in SIDECHAIN_GROUPS]
        if len(paired) < minimum:
            continue
        keys = [tuple(row["key"]) for row in paired]
        ab, keep_ab = _state_features(root, cid, "AB", keys, mode)
        free = np.zeros_like(ab)
        keep_free = np.zeros(len(keys), bool)
        for state in ("A", "B"):
            ids = np.asarray([i for i, row in enumerate(paired) if row["partner"] == state])
            if not len(ids):
                continue
            values, retained = _state_features(root, cid, state, [keys[i] for i in ids], mode)
            free[ids] = values
            keep_free[ids] = retained
        keep = keep_ab & keep_free
        ids = np.flatnonzero(keep)[:64]
        if len(ids) < minimum:
            continue
        targets = np.asarray([[paired[i]["target_ab"], paired[i]["target_free"]]
                              for i in ids], np.float32)
        from protein import PK_MODS
        base = np.asarray([PK_MODS[keys[i][3]] for i in ids], np.float32)
        return {"complex_id": cid, "ab": ab[ids], "free": free[ids],
                "target_ab": targets[:, 0] - base,
                "target_free": targets[:, 1] - base,
                "keys": [keys[i] for i in ids]}
    raise RuntimeError(f"no PINDER smoke record with {minimum} matched {mode} sites")


def _pkpdb_atoms(root, cid):
    """Read the deposited model with the same first-positive alternate policy."""
    path = Path(root) / "pretraining/pkpdb-v1/structures" / cid[1:3] / f"{cid}.cif.gz"
    with gzip.open(path, "rt") as stream: file = pdbx.CIFFile.read(stream)
    cat = file.block["atom_site"]; n = cat.row_count
    def field(name, default):
        return cat[name].as_array(str) if name in cat else np.full(n, default)
    chain = field("label_asym_id", ""); seq = field("label_seq_id", "?")
    auth = field("auth_seq_id", "?"); ins = field("pdbx_PDB_ins_code", "?")
    alt = field("label_alt_id", "."); occ = field("occupancy", "0").astype(float)
    # defects.json lists the polymer chains by label_asym_id, while the structure below carries author chain IDs, so the
    # chains are selected here on the label IDs (selecting on author IDs kept nothing when they differ, e.g. 5ma7 A -> E,
    # and could keep the wrong chain when the letters overlap)
    chains = {row["chain"] for row in read(Path(root) / "pretraining/pkpdb-full-v1/entries" / cid / "defects.json")["sequences"]}
    groups = {}
    for i, key in enumerate(zip(chain, seq, auth, ins)): groups.setdefault(key, []).append(i)
    keep = np.zeros(n, bool); blank = {"", " ", "?", "."}
    for ids in groups.values():
        ids = np.asarray(ids); positive = ids[occ[ids] > 0]
        shared = positive[np.isin(alt[positive], list(blank))]
        alternatives = list(dict.fromkeys(a for a in alt[positive] if a not in blank))
        selected = alternatives[0] if alternatives else None
        keep[shared] = True
        if selected is not None: keep[positive[alt[positive] == selected]] = True
    keep &= np.isin(chain, list(chains))
    new = pdbx.CIFCategory()
    for name in cat: new[name] = pdbx.CIFColumn(cat[name].as_array(str)[keep])
    new["label_alt_id"] = pdbx.CIFColumn(np.full(int(keep.sum()), "."))
    file.block["atom_site"] = new
    atoms = pdbx.get_structure(file, model=1, altloc="occupancy", use_author_fields=True, include_bonds=False)
    canonical = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
    mask = np.isin(atoms.res_name, list(canonical)) & ~np.isin(np.char.upper(atoms.element), ["H", "D"])
    return atoms[mask]


def _native_features_for_atoms(atoms, keys):
    with tempfile.TemporaryDirectory(prefix="pkai-scale-features-") as directory:
        path = Path(directory) / "input.pdb"; mapping = _export_pdb(atoms, path)
        _, package = native(); import sys
        sys.path.insert(0, str(package)); from protein import Protein
        residues, matrix = feature_matrix(Protein(path), ENCODING)
    lookup = {}
    for index, residue in enumerate(residues):
        original = mapping[(str(residue.chain), int(residue.resnumb))]
        key = (*original, GROUP_ALIAS.get(str(residue.resname), str(residue.resname)))
        if key in lookup: raise AssertionError((key, "duplicate pKAI site"))
        lookup[key] = index
    retained = np.asarray([key in lookup for key in keys], bool)
    result = np.zeros((len(keys), WIDTH), np.float32)
    for i, key in enumerate(keys):
        if retained[i]: result[i] = matrix[lookup[key]]
    return result, retained


def _pkpdb_feature_record(root, cid):
    folder = Path(root) / "pretraining/pkpdb-full-v1/entries" / cid
    sites = [s for s in read(folder / "sites.json") if s["train_mask"] and s["group"] in SIDECHAIN_GROUPS]
    env = {tuple(row[k] for k in KEY_FIELDS): row for row in read(folder / "environment.json")["sites"]}
    keys = [tuple(s[k] for k in KEY_FIELDS) for s in sites]
    atoms = _pkpdb_atoms(root, cid)
    full, kf = _native_features_for_atoms(atoms, keys); bb, kb = backbone_features(atoms, keys, ENCODING)
    keep = kf & kb & np.asarray([key in env for key in keys], bool)  # bool also when there are no keys
    if not keep.any(): raise ValueError((cid, "no mapped pKPDB sites"))
    from protein import PK_MODS
    target = np.asarray([s["pka"] - PK_MODS[s["group"]] for s in sites], np.float32)
    weight = np.asarray([env[key]["w_burial"] for key in keys], np.float32)
    return {"full": full[keep], "backbone": bb[keep], "target": target[keep], "weight": weight[keep]}


def _pinder_feature_record(root, cid, split):
    folder = pinder_source(root) / "entries" / cid
    rows = [r for r in _paired_rows(folder, split) if r["key"][3] in SIDECHAIN_GROUPS]
    if not rows: raise ValueError((cid, "no paired sites"))
    keys = [tuple(r["key"]) for r in rows]
    arrays = {}; retained = np.ones(len(keys), bool)
    for mode in MODES:
        ab, ka = _state_features(root, cid, "AB", keys, mode)
        free = np.zeros_like(ab); kfree = np.zeros(len(keys), bool)
        for state in ("A", "B"):
            ids = np.asarray([i for i, row in enumerate(rows) if row["partner"] == state])
            if len(ids):
                value, ok = _state_features(root, cid, state, [keys[i] for i in ids], mode)
                free[ids] = value; kfree[ids] = ok
        arrays[f"{mode}_ab"] = ab; arrays[f"{mode}_free"] = free
        retained &= ka & kfree
    from protein import PK_MODS
    base = np.asarray([PK_MODS[r["key"][3]] for r in rows], np.float32)
    arrays.update(target_ab=np.asarray([r["target_ab"] for r in rows], np.float32)-base,
                  target_free=np.asarray([r["target_free"] for r in rows], np.float32)-base,
                  w_burial=np.asarray([r["w_burial"] for r in rows], np.float32),
                  w_interface=np.asarray([r["w_interface"] for r in rows], np.float32),
                  interface=np.asarray([bool(r.get("interface")) for r in rows], bool))  # not stored; see build_interface
    if not retained.any(): raise ValueError((cid, "no sites mapped in both representations"))
    return {key: value[retained] for key, value in arrays.items()}


# ---------------------------------------------------------------- feature store
# One store per encoding (2026-10-10), over the 100% pool-v3 records plus the PINDER validation cohort, shared by every
# fraction: a run selects its registered records from it at training time (no per-run copies). Each record holds one
# structure's sites in compact slot form (pkai_scratch.compact: per site 250 values + atom class/residue type + site
# class, checked to expand back to the dense features exactly) and its targets/weights, as one zstd frame (an
# uncompressed .npz inside). Layout under training/pkai-features-v2/<encoding>/<dataset>/:
#   ids.json                     every (id, split) of the 100% pool (+ PINDER validation)
#   shards/<name>/               records.bin (zstd records) + receipt.json (records, excludable failures); written by
#                                `prepare` tasks (<t>-of-<T>) or `import` (import-<run>: an earlier run's per-record files)
#   store/                       `pack-store`: records.bin in file-name order (the order the per-record packing used),
#                                index.npz (names, sites, byte offsets), records.json, metadata.json, verification.json
#                                (every record decompressed and checked against its digest); the shards are then removed
STORE_VERSION = "pkai-features-v2"
ZSTD_LEVEL = 3
KINDS = {"pkpdb": ("full", "backbone"), "pinder": ("full_ab", "full_free", "backbone_ab", "backbone_free")}
SCALARS = {"pkpdb": ("target", "weight"), "pinder": ("target_ab", "target_free", "w_burial", "w_interface")}


def feature_root(root, dataset):
    return Path(root) / "training" / STORE_VERSION / f"{ENCODING}{GEOMETRY}" / dataset


def _name(cid, split): return f"{split}-{cid}"


def _encode(arrays):
    import io, zstandard
    buffer = io.BytesIO(); np.savez(buffer, **arrays)
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(buffer.getvalue())


def _decode(blob):
    import io, zstandard
    with np.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(blob))) as handle: return {k: handle[k] for k in handle.files}


def _record_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays): h.update(name.encode()); h.update(_array_digest(arrays[name]).encode())
    return h.hexdigest()


def _compact_record(dataset, dense):
    from .pkai_scratch import compact
    if any(dense[kind].shape[1:] != (WIDTH,) for kind in KINDS[dataset]): raise AssertionError(("feature width", WIDTH))
    out = {f"{kind}.{field}": value for kind in KINDS[dataset] for field, value in compact(dense[kind], ENCODING).items()}
    out.update({name: np.asarray(dense[name], np.float32) for name in SCALARS[dataset]})
    return out


def store_ids(root):
    """ids.json of both datasets: the 100% pool-v3 lists (every fraction is a subset) and the PINDER validation cohort."""
    root = Path(root)
    pk = [[row["id"], "train"] for row in _pool_rows(root / "pretraining/pkpdb-full-v1/pool-v3.tsv", 1.0)]
    pi = ([[row["id"], "train"] for row in _pool_rows(pinder_source(root) / "pool-v3.tsv", 1.0)] +
          [[row["id"], "val"] for row in read(cohort(root))["records"] if row["split"] == "val"])
    for dataset, ids in (("pkpdb", pk), ("pinder", pi)):
        base = feature_root(root, dataset); base.mkdir(parents=True, exist_ok=True)
        if len({_name(*x) for x in ids}) != len(ids): raise AssertionError((dataset, "duplicate ids"))
        listing = {"encoding": ENCODING, "ids": ids}
        if (base / "ids.json").exists() and read(base / "ids.json") != listing: raise AssertionError((dataset, "ids.json changed"))
        atomic_json(base / "ids.json", listing)
        print(json.dumps({"dataset": dataset, "encoding": ENCODING, "ids": len(ids)}), flush=True)


def _write_shard(base, name, items, produce, total):
    """For each (cid, split, payload) of items, produce(cid, split, payload) -> compact arrays (or raises); each result is
    appended as one zstd record to shards/<name>/records.bin. Excludable failures are recorded; the shard directory is
    installed only when no other failure occurred."""
    shards = base / "shards"; shards.mkdir(parents=True, exist_ok=True); folder = shards / name
    if (folder / "receipt.json").exists(): return read(folder / "receipt.json")
    pending = shards / f".{name}.pending-{os.getpid()}"; pending.mkdir()
    began = time.time(); records = []; failures = []; position = 0
    with open(pending / "records.bin", "wb") as handle:
        for number, (cid, split, payload) in enumerate(items, 1):
            try:
                arrays = produce(cid, split, payload)
            except Exception as exc:
                failures.append({"id": cid, "split": split, "error": repr(exc)}); continue
            blob = _encode(arrays); handle.write(blob); sites = {len(v) for k, v in arrays.items() if k.endswith(".value")}
            if len(sites) != 1: raise AssertionError((cid, "site counts differ between feature kinds"))
            records.append({"name": _name(cid, split), "sites": int(sites.pop()), "sha256": _record_digest(arrays),
                            "blob": [position, len(blob)]}); position += len(blob)
            if number % 100 == 0: print(json.dumps({"shard": name, "done": number, "of": total, "failures": len(failures)}), flush=True)
    receipt = {"name": name, "encoding": ENCODING, "assigned": total, "records": records, "failures": failures,
               "seconds": round(time.time() - began, 1)}
    atomic_json(pending / "receipt.json", receipt)
    blocking = [f for f in failures if not _excludable(f)]
    if blocking:
        atomic_json(shards / f"{name}.failures.json", receipt); shutil.rmtree(pending)
        raise RuntimeError(f"{len(blocking)} feature failures in shard {name} ({len(failures) - len(blocking)} excludable); see {name}.failures.json")
    os.replace(pending, folder)
    return receipt


def _shard_names(base, pattern="*"):
    receipts = [read(p) for p in sorted((base / "shards").glob(f"{pattern}/receipt.json")) if not p.parent.name.startswith(".")]
    return {r["name"] for x in receipts for r in x["records"]} | {_name(f["id"], f["split"]) for x in receipts for f in x["failures"]}


def prepare_shard(root, dataset, task, tasks):
    """Task t of T computes ids[t::T], skipping records that an import shard already holds."""
    root = Path(root); base = feature_root(root, dataset); imported = _shard_names(base, "import-*")
    mine = [tuple(x) for x in read(base / "ids.json")["ids"][task::tasks] if _name(*x) not in imported]
    def produce(cid, split, _):
        return _compact_record(dataset, _pkpdb_feature_record(root, cid) if dataset == "pkpdb" else _pinder_feature_record(root, cid, split))
    receipt = _write_shard(base, f"{task:04d}-of-{tasks:04d}", ((cid, split, None) for cid, split in mine), produce, len(mine))
    print(json.dumps({"dataset": dataset, "task": task, "records": len(receipt["records"]), "failures": len(receipt["failures"])}), flush=True)


def import_run(root, run, dataset):
    """Import an earlier run's per-record feature files (features/<dataset>.tar from the archive job, else the loose
    features/<dataset>/ directory) as shards/import-<run>/, with the run's excludable failures (prepare-<dataset>-*.json).
    Every record must be in ids.json and is converted by compact(), whose exact round trip is checked."""
    import io, tarfile
    root = Path(root); run = Path(run); base = feature_root(root, dataset)
    wanted = {_name(*x) for x in read(base / "ids.json")["ids"]}
    failures = [f for p in sorted(run.glob(f"prepare-{dataset}-*.json")) for f in read(p)["failures"]]
    if any(not _excludable(f) for f in failures): raise AssertionError((run.name, dataset, "blocking failures in the run"))
    def members():
        tar = run / "features" / f"{dataset}.tar"
        if tar.exists():
            with tarfile.open(tar, "r|") as stream:  # one sequential pass
                for member in stream:
                    if member.isfile() and member.name.endswith(".npz"):
                        with np.load(io.BytesIO(stream.extractfile(member).read())) as handle: yield Path(member.name).stem, {k: handle[k] for k in handle.files}
        else:
            for path in sorted((run / "features" / dataset).glob("*.npz")):
                with np.load(path) as handle: yield path.stem, {k: handle[k] for k in handle.files}
    def produce(cid, split, dense):
        if _name(cid, split) not in wanted: raise AssertionError((_name(cid, split), "not in ids.json"))
        return _compact_record(dataset, dense)
    items = ((*reversed(stem.split("-", 1)), dense) for stem, dense in members())  # "<split>-<id>" -> (id, split, dense)
    receipt = _write_shard(base, f"import-{run.name}", items, produce, None)
    if receipt["failures"]: raise AssertionError((run.name, dataset, receipt["failures"][:3], "import failures"))
    receipt["failures"] = [f for f in failures if _name(f["id"], f["split"]) in wanted]
    atomic_json(base / "shards" / f"import-{run.name}" / "receipt.json", receipt)
    print(json.dumps({"run": run.name, "dataset": dataset, "records": len(receipt["records"]), "failures": len(receipt["failures"])}), flush=True)


def _order(names): return sorted(names, key=lambda n: n + ".npz")  # the file-name order of the per-record packing


def pack_store(root, dataset, workers=8):
    """Concatenate the shards' records (each name once; a name in several shards must have the same digest) into
    store/records.bin in file-name order, verify every record, install, then remove the shards."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root); base = feature_root(root, dataset); destination = base / "store"
    if (destination / "verification.json").exists(): return read(destination / "verification.json")
    listing = [_name(*x) for x in read(base / "ids.json")["ids"]]
    receipts = [read(p) for p in sorted((base / "shards").glob("*/receipt.json")) if not p.parent.name.startswith(".")]
    found = {}; excluded = {}
    for receipt in receipts:
        if receipt["encoding"] != ENCODING: raise AssertionError((receipt["name"], "encoding"))
        for r in receipt["records"]:
            if r["name"] in found and found[r["name"]][0]["sha256"] != r["sha256"]: raise AssertionError((r["name"], "records differ between shards"))
            found.setdefault(r["name"], (r, base / "shards" / receipt["name"]))
        for f in receipt["failures"]: excluded.setdefault(_name(f["id"], f["split"]), f)
    if set(found) & set(excluded): raise AssertionError(("both built and excluded", sorted(set(found) & set(excluded))[:5]))
    missing = [n for n in listing if n not in found and n not in excluded]
    if missing: raise AssertionError(f"{len(missing)} records not built yet, e.g. {missing[:5]}")
    if set(found) - set(listing): raise AssertionError("shard records outside ids.json")
    names = _order(found); pending = base / f".store.pending-{os.getpid()}"; pending.mkdir()
    offsets = np.zeros(len(names) + 1, np.int64); handles = {}; dtypes = None
    with open(pending / "records.bin", "wb") as out:
        for i, name in enumerate(names):
            record, folder = found[name]
            if folder not in handles: handles[folder] = open(folder / "records.bin", "rb")
            start, length = record["blob"]; blob = os.pread(handles[folder].fileno(), length, start)
            if len(blob) != length: raise AssertionError((name, "short read"))
            if dtypes is None: dtypes = {k: [v.dtype.str, list(v.shape[1:])] for k, v in _decode(blob).items()}
            out.write(blob); offsets[i + 1] = offsets[i] + length
    for handle in handles.values(): handle.close()
    np.savez(pending / "index.npz", names=np.asarray(names), sites=np.asarray([found[n][0]["sites"] for n in names], np.int64), offsets=offsets)
    atomic_json(pending / "records.json", [{k: v for k, v in found[n][0].items() if k != "blob"} for n in names])
    sites = int(sum(found[n][0]["sites"] for n in names))
    atomic_json(pending / "metadata.json", {"version": STORE_VERSION, "format": "zstd-npz-records-v1", "zstd_level": ZSTD_LEVEL,
        "encoding": ENCODING, "width": WIDTH, "dataset": dataset, "fields": dtypes, "records": len(names), "sites": sites,
        "excluded": [excluded[n] for n in _order(excluded)], "ids_sha256": digest(base / "ids.json"),
        "dense_float32_bytes": sites * WIDTH * 4 * len(KINDS[dataset]), "compressed_bytes": int(offsets[-1])})
    store = FeatureStore(pending, verify=False)
    with ThreadPoolExecutor(workers) as pool:
        bad = [n for n, ok in zip(names, pool.map(lambda n: _record_digest(store.record(n)) == found[n][0]["sha256"], names)) if not ok]
    store.close()
    if bad: raise AssertionError(f"{len(bad)} records differ after packing, e.g. {bad[:5]}")
    atomic_json(pending / "verification.json", {"passed": True, "records_checked": len(names),
                "files": {p.name: digest(p) for p in sorted(pending.iterdir())}})
    os.replace(pending, destination)
    shutil.rmtree(base / "shards")
    result = read(destination / "verification.json"); meta = read(destination / "metadata.json")
    print(json.dumps({"dataset": dataset, "encoding": ENCODING, "records": meta["records"], "sites": sites, "excluded": len(meta["excluded"]),
                      "compressed_gb": round(meta["compressed_bytes"] / 1e9, 2), "dense_gb": round(meta["dense_float32_bytes"] / 1e9, 2)}), flush=True)
    return result


class FeatureStore:
    """Read access to a packed feature store: record(name) -> {field: array}; select(names, fields) -> the named
    records' arrays concatenated in the given order (threaded decompression)."""
    def __init__(self, path, verify=True):
        self.path = Path(path); self.metadata = read(self.path / "metadata.json")
        if verify and not read(self.path / "verification.json")["passed"]: raise AssertionError((str(path), "unverified store"))
        index = np.load(self.path / "index.npz"); self.names = [str(n) for n in index["names"]]
        self.position = {n: i for i, n in enumerate(self.names)}; self.sites = index["sites"]; self.offsets = index["offsets"]
        self.fd = os.open(self.path / "records.bin", os.O_RDONLY)

    def record(self, name):
        i = self.position[name]; start, stop = int(self.offsets[i]), int(self.offsets[i + 1])
        return _decode(os.pread(self.fd, stop - start, start))

    def select(self, names, fields, workers=16):
        from concurrent.futures import ThreadPoolExecutor
        counts = np.asarray([self.sites[self.position[n]] for n in names], np.int64); starts = np.concatenate(([0], np.cumsum(counts)))
        spec = self.metadata["fields"]
        out = {f: np.empty((int(starts[-1]), *spec[f][1]), np.dtype(spec[f][0])) for f in fields}
        def fill(i):
            arrays = self.record(names[i])
            for f in fields: out[f][starts[i]:starts[i + 1]] = arrays[f]
        with ThreadPoolExecutor(workers) as pool: list(pool.map(fill, range(len(names))))
        return out

    def close(self): os.close(self.fd)


def run_selection(root):
    """The run's registered records per training/validation group, in file-name order, against the two stores; records
    a store excluded are listed, any other absent record is an error."""
    records = read(output(root) / "records.json"); groups = {}
    for group, dataset, split, key in (("pkpdb-train", "pkpdb", "train", "pkpdb_train"), ("pinder-train", "pinder", "train", "pinder_train"),
                                       ("pinder-val", "pinder", "val", "pinder_val")):
        path = feature_root(root, dataset) / "store"; meta = read(path / "metadata.json")
        index = set(str(n) for n in np.load(path / "index.npz")["names"]); excluded = {_name(f["id"], f["split"]) for f in meta["excluded"]}
        names = _order(_name(cid, split) for cid in records[key])
        absent = [n for n in names if n not in index]
        if set(absent) - excluded: raise AssertionError((group, "records missing from the store", sorted(set(absent) - excluded)[:5]))
        groups[group] = {"dataset": dataset, "names": [n for n in names if n in index], "excluded": absent,
                         "store_verification_sha256": digest(path / "verification.json")}
    return groups


def load_group(root, selection, group, kinds, scalars):
    """Compact arrays of one group: '<kind>.<field>' for each kind plus the scalar fields."""
    from .pkai_scratch import compact_fields
    item = selection[group]; store = FeatureStore(feature_root(root, item["dataset"]) / "store")
    try: return store.select(item["names"], [f"{k}.{f}" for k in kinds for f in compact_fields(ENCODING)] + list(scalars))
    finally: store.close()


def dense(torch, batch, prefix):
    """Batch features: the dense array if the source holds one, else expanded on the device from '<prefix>.<field>'."""
    from .pkai_scratch import compact_fields, expand_torch
    if prefix in batch: return batch[prefix]
    return expand_torch(torch, {f: batch[f"{prefix}.{f}"] for f in compact_fields(ENCODING)}, ENCODING)


def compare_packed(root, workers=16):
    """Before an earlier run's dense packed/ arrays are removed: the store's selection, expanded, must equal them
    bit for bit (same rows, same order)."""
    from .pkai_scratch import compact_fields, expand
    root = Path(root); out = output(root); selection = run_selection(root); report = {}
    for group, kinds, scalars in (("pkpdb-train", KINDS["pkpdb"], SCALARS["pkpdb"]), ("pinder-train", KINDS["pinder"], SCALARS["pinder"]),
                                  ("pinder-val", KINDS["pinder"], SCALARS["pinder"])):
        arrays = load_group(root, selection, group, kinds, scalars); packed = out / "packed" / group; rows = len(arrays[scalars[0]])
        for name in scalars:
            if not np.array_equal(np.load(packed / f"{name}.npy"), arrays[name]): raise AssertionError((group, name))
        for kind in kinds:
            reference = np.load(packed / f"{kind}.npy", mmap_mode="r")
            if reference.shape != (rows, WIDTH): raise AssertionError((group, kind, reference.shape, rows))
            for start in range(0, rows, 65536):
                stop = min(start + 65536, rows)
                block = expand({f: arrays[f"{kind}.{f}"][start:stop] for f in compact_fields(ENCODING)}, ENCODING)
                if not np.array_equal(block, reference[start:stop]): raise AssertionError((group, kind, start))
        report[group] = {"rows": rows, "identical": True}
    atomic_json(out / "packed-comparison.json", {"passed": True, "groups": report, "selection": {g: {k: v for k, v in s.items() if k != "names"} for g, s in selection.items()}})
    print(json.dumps(report), flush=True)


def feature_smoke(root):
    root=Path(root);require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK","1")))
    cid="6dms";pinder_id="3j47__A1_P43588--3j47__D1_Q12250"
    pk=_pkpdb_feature_record(root,cid);pi=_pinder_feature_record(root,pinder_id,"train")
    checks={"pkpdb":{k:list(v.shape) for k,v in pk.items()},"pinder":{k:list(v.shape) for k,v in pi.items()}}
    for arrays in (pk,pi):
        if not all(np.isfinite(v).all() for v in arrays.values()):raise FloatingPointError("nonfinite feature smoke")
    if not (len(pk["target"]) and len(pi["target_ab"])):raise AssertionError("empty feature smoke")
    atomic_json(output(root)/"feature-smoke.json",{"passed":True,"records":{"pkpdb":cid,"pinder":pinder_id},"shapes":checks,"test_data_included":False})
    print(json.dumps(checks,sort_keys=True),flush=True)


def _arrays(path, fields): return {field: np.load(path / f"{field}.npy", mmap_mode="r") for field in fields}


def _weighted_mse(torch, prediction, target, weight):
    return (weight * (prediction-target).square()).sum() / weight.sum().clamp_min(1e-8)


def _squared_sums(torch, model, source, length, terms, config, forward):
    """Mean squared error per term over contiguous BATCH_SIZE slices. Per-batch float32 sums are added as Python floats
    in slice order (the same arithmetic as reading each batch sum back with float()), but read back in one transfer.
    forward(batch) runs the model once per batch; each term maps (batch, outputs) to per-row squared errors."""
    from .loading import DeferredScalars, Prefetcher
    sums = {name: DeferredScalars(every=1 << 30) for name in terms}
    specs = [np.arange(start, min(start + BATCH_SIZE, length)) for start in range(0, length, BATCH_SIZE)]
    to_device, on_consume = source.hooks(); model.eval()
    with torch.no_grad():
        for _, batch in Prefetcher(source, specs, config, to_device, on_consume):
            outputs = forward(batch)
            for name, term in terms.items(): sums[name].add(term(batch, outputs).sum())
    return {name: sum(values.flush()) / length for name, values in sums.items()}


def _pkpdb_validation_arrays(root, mode):
    """Frozen 5k-pilot validation rows: the compact package (pkabench.dataset_transfer build-validation) when present,
    checked against the pilot row selection, else the pilot arrays themselves."""
    pilot = Path(root) / "pretraining/pkpdb-5k-comparison-v1/pkai-packed"; package = Path(root) / "pretraining/pkpdb-val-pkai-v1"
    if ENCODING != "atom16" or GEOMETRY:
        package = validation_package(root, ENCODING)
        return {"x": np.load(package / ("full.npy" if mode == "full" else "backbone.npy"), mmap_mode="r"), "y": np.load(package / "target.npy")}
    if package.exists():
        if (pilot / "rows.json").exists():
            rows = read(pilot / "rows.json"); ids = np.asarray([i for i, r in enumerate(rows) if r["split"] == "val" and r["group"] in SIDECHAIN_GROUPS])
            if not np.array_equal(ids, np.load(package / "source_rows.npy")): raise AssertionError("validation package rows differ from the pilot")
        return {"x": np.load(package / ("full.npy" if mode == "full" else "backbone.npy"), mmap_mode="r"), "y": np.load(package / "target.npy")}
    rows = read(pilot / "rows.json"); ids = np.asarray([i for i, r in enumerate(rows) if r["split"] == "val" and r["group"] in SIDECHAIN_GROUPS])
    feature_path = pilot / "features.npy" if mode == "full" else Path(root) / "pretraining/pkai-backbone-ablation-v1/backbone-features.npy"
    return {"x": np.load(feature_path, mmap_mode="r")[ids], "y": np.asarray([rows[i]["pka"] - rows[i]["model_pka"] for i in ids], np.float32)}


def interface_path(root): return Path(root) / "training" / STORE_VERSION / "pinder-val-interface.npz"


def build_interface(root, workers=8):
    """Per-row interface flags (sites.json 'interface') of the PINDER validation records, in each record's row order:
    the same rows as the stores (paired, side-chain, mapped in both representations and states; this does not depend on
    the encoding or geometry), with w_interface and targets kept to check the alignment against a store at load."""
    from concurrent.futures import ProcessPoolExecutor
    root = Path(root); cids = sorted(row["id"] for row in read(cohort(root))["records"] if row["split"] == "val")
    with ProcessPoolExecutor(workers) as pool: done = list(pool.map(_interface_one, [(str(root), cid) for cid in cids]))
    names = []; parts = []; failed = []
    for cid, arrays in done:
        if arrays is None: failed.append(cid); continue
        names.append(_name(cid, "val")); parts.append(arrays)
    offsets = np.concatenate(([0], np.cumsum([len(a["interface"]) for a in parts])))
    path = interface_path(root); pending = path.with_suffix(f".pending-{os.getpid()}.npz")
    np.savez(pending, names=np.asarray(names), offsets=offsets, **{k: np.concatenate([a[k] for a in parts]) for k in ("interface", "w_interface", "target_ab", "target_free")})
    os.replace(pending, path)
    print(json.dumps({"records": len(names), "failed": failed, "sites": int(offsets[-1]), "interface_sites": int(sum(a["interface"].sum() for a in parts))}), flush=True)


def _interface_one(task):
    root, cid = task
    try:
        record = _pinder_feature_record(Path(root), cid, "val")
    except Exception as exc:
        if not _excludable({"error": repr(exc)}): raise
        return cid, None
    return cid, {k: record[k] for k in ("interface", "w_interface", "target_ab", "target_free")}


def _interface_rows(root, names, arrays):
    """Interface flags for the selected PINDER validation rows (names in order), checked row by row against the
    store's w_interface and targets."""
    side = np.load(interface_path(root)); position = {str(n): i for i, n in enumerate(side["names"])}; offsets = side["offsets"]
    rows = np.concatenate([np.arange(offsets[position[n]], offsets[position[n] + 1]) for n in names])
    for field in ("w_interface", "target_ab", "target_free"):
        if not np.array_equal(side[field][rows], arrays[field]): raise AssertionError(("interface flags misaligned", field))
    return side["interface"][rows]


def _validation_sources(torch, root, mode, objective, config, selection):
    from .loading_torch import PackedSiteSource
    sources = {"pk": PackedSiteSource(_pkpdb_validation_arrays(root, mode), config=config)}
    if objective in ("joint", "pkpdb"):  # PINDER validation is reported for pKPDB-only arms too (not used for their selection)
        raw = load_group(root, selection, "pinder-val", (f"{mode}_ab", f"{mode}_free"), ("target_ab", "target_free", "w_interface"))
        flag = _interface_rows(root, selection["pinder-val"]["names"], raw).astype(np.float32)
        pi = _source_arrays(raw, {f"{mode}_ab": "xa", f"{mode}_free": "xf", "target_ab": "ya", "target_free": "yf", "w_interface": "wi"})
        pi["if"] = flag
        sources["pi"] = PackedSiteSource(pi, config=config)
        sources["pi"].weights = {"wi": float(raw["w_interface"].mean()), "if": float(flag.mean())}  # per-row means (normalisers)
    return sources


def _validation(torch, model, sources, config, objective="joint"):
    # The established clean 5k validation is independent of the new 10% pool.
    pk = sources["pk"]
    result = {"pkpdb_mse": _squared_sums(torch, model, pk, pk.length, {"pk": lambda b, o: (o - b["y"]).square()}, config,
                                         lambda b: model(dense(torch, b, "x")))["pk"]}
    if "pi" in sources:
        pi = sources["pi"]
        terms = {"a": lambda b, o: (o[0] - b["ya"]).square(), "f": lambda b, o: (o[1] - b["yf"]).square(),
                 "pair": lambda b, o: ((o[0] - o[1]) - (b["ya"] - b["yf"])).square()}
        if hasattr(pi, "weights"):  # interface paired metrics (2026-10-10): w_interface-weighted (the training paired loss) and interface sites only
            terms.update(pair_w=lambda b, o: b["wi"] * ((o[0] - o[1]) - (b["ya"] - b["yf"])).square(),
                         pair_if=lambda b, o: b["if"] * ((o[0] - o[1]) - (b["ya"] - b["yf"])).square())
        mse = _squared_sums(torch, model, pi, pi.length, terms, config, lambda b: (model(dense(torch, b, "xa")), model(dense(torch, b, "xf"))))
        result.update(pinder_state_mse=(mse["a"] + mse["f"]) / 2, pinder_paired_mse=mse["pair"])
        if hasattr(pi, "weights"):
            result.update(pinder_paired_weighted_mse=mse["pair_w"] / pi.weights["wi"], pinder_paired_interface_mse=mse["pair_if"] / pi.weights["if"])
    # selection keeps the registered metrics (the interface metrics are reported only)
    selected = ["pkpdb_mse"] if objective == "pkpdb" else [k for k in ("pkpdb_mse", "pinder_state_mse", "pinder_paired_mse") if k in result]
    result["selection_mse"] = sum(result[k] for k in selected) / len(selected)
    return result


def _source_arrays(arrays, rename):
    """Rename '<kind>.<field>' / scalar fields to the loader's batch keys ('<key>.<field>' / key)."""
    out = {}
    for name, value in arrays.items():
        kind, _, field = name.partition(".")
        out[f"{rename[kind]}.{field}" if field else rename[kind]] = value
    return out


def _epoch_specs(order_pk, order_pi):
    """The same index slices as before: BATCH_SIZE rows per step, wrapping to the start of the permutation."""
    steps = max(math.ceil(len(order_pk) / BATCH_SIZE), math.ceil(len(order_pi) / BATCH_SIZE) if order_pi is not None else 0)
    specs = []
    for step in range(steps):
        ip = order_pk[(step * BATCH_SIZE) % len(order_pk):((step * BATCH_SIZE) % len(order_pk)) + BATCH_SIZE]
        if len(ip) < BATCH_SIZE: ip = np.concatenate((ip, order_pk[:BATCH_SIZE - len(ip)]))
        spec = {"pk": ip}
        if order_pi is not None:
            ii = order_pi[(step * BATCH_SIZE) % len(order_pi):((step * BATCH_SIZE) % len(order_pi)) + BATCH_SIZE]
            if len(ii) < BATCH_SIZE: ii = np.concatenate((ii, order_pi[:BATCH_SIZE - len(ii)]))
            spec["pi"] = ii
        specs.append(spec)
    return specs


def train_scale(root, mode, objective, batch_size=None, max_epochs=100, learning_rate=None, schedule="constant"):
    """batch_size/max_epochs other than the registered 256/100 (batch-size sweep, 2026-10-10) use the sqrt learning-rate
    rule and write to runs/<mode>-<objective>-b<batch>-e<epochs>/; the defaults keep the registered run paths."""
    global BATCH_SIZE, LEARNING_RATE
    tag = ""
    if (batch_size or BATCH_SIZE) != BATCH_SIZE or max_epochs != 100:
        BATCH_SIZE = int(batch_size or BATCH_SIZE); LEARNING_RATE = REFERENCE_LR * math.sqrt(BATCH_SIZE / REFERENCE_BATCH)
        tag = f"-b{BATCH_SIZE}-e{max_epochs}"
    if learning_rate is not None:  # explicit rate (learning-rate sweep, 2026-10-10) instead of the sqrt rule
        LEARNING_RATE = float(learning_rate); tag += f"-lr{LEARNING_RATE:g}"
    if schedule not in ("constant", "cosine"): raise ValueError(schedule)
    if schedule == "cosine": tag += "-cos"  # per-epoch cosine decay from LEARNING_RATE to 0 over max_epochs
    from .loading import DeferredScalars, LoaderConfig, Prefetcher
    from .loading_torch import CombinedSource, PackedSiteSource
    if mode not in MODES or objective not in OBJECTIVES: raise ValueError((mode,objective))
    root=Path(root);out=output(root); selection=run_selection(root)
    torch,_=native();require_compute(threads=2,gpu_benchmark=True,allow_comp1400=True)
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    model=model_class(torch,inputs=WIDTH)().cuda().train();opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    config=LoaderConfig()
    # compact features (pkai_scratch.compact) resident on the GPU when they fit, expanded per batch by dense()
    pk=_source_arrays(load_group(root,selection,"pkpdb-train",(mode,),("target","weight")),{mode:"x","target":"y","weight":"w"}); pi=None
    sources={"pk":PackedSiteSource(pk,config=config)}
    if objective=="joint":
        pi=_source_arrays(load_group(root,selection,"pinder-train",(f"{mode}_ab",f"{mode}_free"),("target_ab","target_free","w_burial","w_interface")),
                          {f"{mode}_ab":"xa",f"{mode}_free":"xf","target_ab":"ya","target_free":"yf","w_burial":"wb","w_interface":"wi"})
        sources["pi"]=PackedSiteSource(pi,config=config)
    train_source=CombinedSource(sources); validation_sources=_validation_sources(torch,root,mode,objective,config,selection)
    dest=out/"runs"/f"{mode}-{objective}{tag}"/f"seed-{SEED}";dest.mkdir(parents=True,exist_ok=True)
    provenance={"mode":mode,"objective":objective,"seed":SEED,"batch_size":BATCH_SIZE,"max_epochs":max_epochs,"learning_rate":LEARNING_RATE,"lr_rule":"1e-6*sqrt(batch/64)" if learning_rate is None else "explicit","lr_schedule":schedule,"precision":"float32","cpus":2,"initialization":"scratch","features":{g:{k:(len(v) if k in ("names","excluded") else v) for k,v in s.items()} for g,s in selection.items()},"feature_encoding":ENCODING,"manifest_sha256":digest(out/"manifest.json"),"test_data_included":False,
                "loader":{"workers":config.workers,"prefetch":config.prefetch,"train":train_source.provenance(),"validation":{k:v.provenance() for k,v in validation_sources.items()}}}
    atomic_json(dest/"manifest.json",provenance)
    params=list(model.parameters())
    rng=np.random.default_rng(SEED);best=float("inf");anchor=float("inf");stall=0;history=[];began=time.monotonic();patience=8
    for epoch in range(1,max_epochs+1):
        if stall>=patience:break
        if schedule=="cosine":
            for group in opt.param_groups: group["lr"]=LEARNING_RATE*0.5*(1+math.cos(math.pi*(epoch-1)/max_epochs))
        model.train();order_pk=rng.permutation(len(pk["y"]));order_pi=rng.permutation(len(pi["ya"])) if pi else None
        losses=DeferredScalars(every=50)
        prefetcher=Prefetcher(train_source,_epoch_specs(order_pk,order_pi),config,train_source.to_device,train_source.on_consume)
        for _,batch in prefetcher:
            b=batch["pk"];lpk=_weighted_mse(torch,model(dense(torch,b,"x")),b["y"],b["w"]);components=[lpk]
            if pi:
                b=batch["pi"];pa=model(dense(torch,b,"xa"));pf=model(dense(torch,b,"xf"))
                components += [(_weighted_mse(torch,pa,b["ya"],b["wb"])+_weighted_mse(torch,pf,b["yf"],b["wb"]))/2,_weighted_mse(torch,pa-pf,b["ya"]-b["yf"],b["wi"])]
            loss=sum(components)/len(components);opt.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is None for p in params):raise FloatingPointError("missing gradient")
            finite=torch.isfinite(loss)&torch.stack([torch.isfinite(p.grad).all() for p in params]).all()
            if not bool(finite):raise FloatingPointError("nonfinite training step")  # one synchronisation per step
            opt.step();losses.add(loss.detach())
        loader=prefetcher.telemetry.summary()
        metrics=_validation(torch,model,validation_sources,config,objective);score=metrics["selection_mse"]
        if score<best:best=score;torch.save(model.state_dict(),dest/"best.pending.pt");os.replace(dest/"best.pending.pt",dest/"best.pt")
        if score<anchor-.001:anchor=score;stall=0
        else:stall+=1
        row={"epoch":epoch,"train_mse":float(np.mean(losses.flush())),**metrics,"stall":stall,"seconds":time.monotonic()-began,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
             "loader_wait_fraction":loader["wait_fraction"],"loader_max_wait_seconds":loader["max_wait_seconds"]};history.append(row);atomic_json(dest/"history.json",history);atomic_json(dest/"progress.json",row);print(json.dumps(row),flush=True)
    train_source.close()
    for source in validation_sources.values(): source.close()
    atomic_json(dest/"final.json",{"complete":True,"epochs":len(history),"best_selection_mse":best,"history_sha256":digest(dest/"history.json"),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"test_data_included":False})


def rescore(root, runs):
    """Re-run the default validation (with the interface paired metrics) on saved best.pt checkpoints of this
    encoding/geometry: runs are seed directories (runs/<arm>/seed-17), or "reference" for the released pKAI model
    (atom16, 15 A, all-atom features). Writes validation-v2.json beside each checkpoint (reference:
    training/pkai-features-v2/reference-pkai-validation.json)."""
    from .loading import LoaderConfig
    torch, package = native(); require_compute(threads=2, gpu_benchmark=True, allow_comp1400=True); torch.set_num_threads(2)
    root = Path(root); config = LoaderConfig(); store = FeatureStore(feature_root(root, "pinder") / "store")
    selection = {"pinder-val": {"dataset": "pinder", "names": [n for n in store.names if n.startswith("val-")]}}; store.close()
    for run in runs:
        if run == "reference":
            if ENCODING != "atom16" or GEOMETRY: raise ValueError("the released model takes atom16 15 A features")
            mode = "full"; released = torch.jit.load(str(package / "models/pKAI_model.pt"), map_location="cuda").eval()
            class Released(torch.nn.Module):
                def __init__(self): super().__init__(); self.inner = released
                def forward(self, x): return self.inner(x).reshape(-1)
            forward = Released(); dest = feature_root(root, "pinder").parent / "reference-pkai-validation.json"
        else:
            run = Path(run); manifest = read(run / "manifest.json"); mode = manifest["mode"]
            net = model_class(torch, inputs=WIDTH)().cuda(); net.load_state_dict(torch.load(run / "best.pt", map_location="cuda")); net.eval()
            forward = net; dest = run / "validation-v2.json"
        sources = _validation_sources(torch, root, mode, "joint", config, selection)
        metrics = _validation(torch, forward, sources, config, "pkpdb" if run != "reference" and manifest["objective"] == "pkpdb" else "joint")
        for source in sources.values(): source.close()
        atomic_json(dest, {"run": str(run), "mode": mode, "encoding": ENCODING, "geometry": GEOMETRY or "r15s250", **metrics})
        print(json.dumps({"run": str(run), **{k: round(v, 4) for k, v in metrics.items()}}), flush=True)


def _pkpdb_batch(root, mode, size=64):
    packed = Path(root) / "pretraining/pkpdb-5k-comparison-v1/pkai-packed"
    rows = read(packed / "rows.json")
    ids = np.asarray([i for i, row in enumerate(rows)
                      if row["split"] == "train" and row["train_mask"]
                      and row["group"] in SIDECHAIN_GROUPS])[:size]
    if len(ids) != size:
        raise AssertionError((mode, len(ids)))
    feature_path = (packed / "features.npy" if mode == "full" else
                    Path(root) / "pretraining/pkai-backbone-ablation-v1/backbone-features.npy")
    features = np.load(feature_path, mmap_mode="r")
    values = np.asarray(features[ids]).copy()
    targets = np.asarray([rows[i]["pka"] - rows[i]["model_pka"] for i in ids], np.float32)
    if mode == "full":
        recorded = read(packed / "verification.json")["features_sha256"]
    else:
        recorded = read(Path(root) / "pretraining/pkai-backbone-ablation-v1/preparation.json")[
            "backbone_features_sha256"]
    return values, targets, {"features": str(feature_path), "features_sha256": recorded,
                             "rows_sha256": read(packed / "verification.json")["rows_sha256"]}


def _one_mode(root, mode):
    torch, _ = native()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model_class(torch)().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
    pk_x, pk_y, provenance = _pkpdb_batch(root, mode)
    paired = _pinder_batch(root, mode)
    before_inputs = _array_digest(pk_x, pk_y, paired["ab"], paired["free"],
                                  paired["target_ab"], paired["target_free"])
    x_pk = torch.tensor(pk_x, device="cuda")
    y_pk = torch.tensor(pk_y, device="cuda")
    x_pair = torch.tensor(np.concatenate((paired["ab"], paired["free"])), device="cuda")
    y_ab = torch.tensor(paired["target_ab"], device="cuda")
    y_free = torch.tensor(paired["target_free"], device="cuda")
    pred_pk = model(x_pk)
    pred_pair = model(x_pair)
    pred_ab, pred_free = pred_pair.split(len(y_ab))
    loss_pk = (pred_pk - y_pk).square().mean()
    loss_state = torch.cat((pred_ab - y_ab, pred_free - y_free)).square().mean()
    teacher_delta = y_ab - y_free
    model_delta = pred_ab - pred_free
    loss_paired = (model_delta - teacher_delta).square().mean()
    swapped = ((pred_free - pred_ab) - (y_free - y_ab)).square().mean()
    if not torch.equal(loss_paired, swapped):
        raise AssertionError((mode, "branch-swap loss changed"))
    component_gradient_norms = {}
    for name, loss in (("pkpdb", loss_pk), ("pinder_state", loss_state),
                       ("pinder_paired", loss_paired)):
        gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
        if not all(torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError((mode, name, "nonfinite component gradient"))
        norm = torch.sqrt(sum(torch.sum(value * value) for value in gradients))
        component_gradient_norms[name] = float(norm.detach().cpu())
        if not component_gradient_norms[name] > 0:
            raise AssertionError((mode, name, "zero component gradient"))
    loss = (loss_pk + loss_state + loss_paired) / 3.0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
        raise FloatingPointError((mode, "nonfinite joint gradient"))
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    parameter_changed = any(not torch.equal(before, after.detach())
                            for before, after in zip(parameters_before, model.parameters()))
    model.eval()
    with torch.no_grad():
        identical = model(torch.tensor(paired["ab"], device="cuda"))
        identical_again = model(torch.tensor(paired["ab"], device="cuda"))
    if not torch.equal(identical, identical_again):
        raise AssertionError((mode, "deterministic eval failed"))
    after_inputs = _array_digest(pk_x, pk_y, paired["ab"], paired["free"],
                                 paired["target_ab"], paired["target_free"])
    result = {"passed": bool(parameter_changed and before_inputs == after_inputs),
        "mode": mode, "seed": SEED, "pinder_complex": paired["complex_id"],
        "pkpdb_sites": len(pk_y), "pinder_sites": len(y_ab),
        "losses": {"pkpdb": float(loss_pk.detach().cpu()),
                   "pinder_state": float(loss_state.detach().cpu()),
                   "pinder_paired": float(loss_paired.detach().cpu()),
                   "joint": float(loss.detach().cpu())},
        "component_gradient_norms": component_gradient_norms,
        "branch_swap_invariant": True, "shared_model_instance": True,
        "parameter_changed": parameter_changed, "inputs_unchanged": before_inputs == after_inputs,
        "input_sha256": before_inputs, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        **provenance}
    if not result["passed"]:
        raise AssertionError(result)
    return result


def smoke(root):
    root = Path(root)
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    gpu_benchmark=True, allow_comp1400=True)
    torch, _ = native()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    gate = architecture_gate()
    results = {mode: _one_mode(root, mode) for mode in MODES}
    out = output(root)
    out.mkdir(parents=True, exist_ok=True)
    report = {"version": "pkai-joint-scale-smoke-v1", "passed": all(
        row["passed"] for row in results.values()), "architecture_gate": gate,
        "results": results, "production_pools_used": False,
        "production_gate": "wait for final pKPDB pool-v2 and environment manifests",
        "test_data_included": False, "code_sha256": digest(Path(__file__))}
    atomic_json(out / "smoke.json", report)
    print(json.dumps(report, indent=2))


def validation_package(root, encoding):
    return Path(root) / f"pretraining/pkpdb-val-pkai-{encoding}{GEOMETRY}-v1"


def _validation_component(task):
    """Full and backbone features of one frozen 5k-pilot validation component, from its pKAI input.pdb (internal
    numbering, request.json mapping to author keys), in the given encoding. The backbone path mirrors
    pkai_backbone_ablation.backbone_features: N/O atoms of other residues within 15 A of the query C-alpha."""
    record, encoding, wanted = task
    from .pkai_backbone_ablation import BACKBONE_ATOMS
    from .pkai_scratch import SLOT_WIDTH, aa20_index
    _, package = native(); import sys
    sys.path.insert(0, str(package)); from protein import Protein
    from residue import ATOM_OHE, RES_OHE
    path = Path(record["path"])
    if digest(path / "input.pdb") != record["pdb_sha256"] or digest(path / "rows.json") != record["rows_sha256"]:
        raise AssertionError((record["complex_id"], "validation input changed"))
    request = read(path / "request.json"); protein = Protein(path / "input.pdb")
    residues, full = feature_matrix(protein, encoding)
    lookup = {(*request["mapping"][str(r.resnumb)], r.resname): i for i, r in enumerate(residues)}
    if len(lookup) != len(residues): raise AssertionError((record["complex_id"], "ambiguous residue map"))
    ca = {}
    with open(path / "input.pdb") as handle:
        for line in handle:
            if line.startswith("ATOM ") and line[12:16].strip() == "CA" and line[16] in (" ", "A"):
                ca[(line[21], int(line[22:26]))] = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    atoms = [a for a in protein.iter_atoms() if a.aname in BACKBONE_ATOMS]
    coords = np.asarray([a.coords for a in atoms], np.float64); slot = SLOT_WIDTH[encoding]
    out = {}
    for key in wanted:
        index = lookup[key]; residue = residues[index]; origin = ca[(residue.chain, residue.resnumb)]
        distance = np.sqrt(((coords - origin) ** 2).sum(-1))
        ids = np.flatnonzero(np.asarray([a.residue is not residue for a in atoms]) & (distance < CUTOFF))
        if np.any(distance[ids] == 0): raise ValueError((record["complex_id"], key, "coincident backbone atom"))
        bb = np.zeros(feature_width(encoding), np.float32)
        if encoding in ("atom16", "atom16aa20"):
            residue.env_anames = [atoms[j].aname for j in ids]; residue.env_resnames = [atoms[j].residue.resname for j in ids]
            residue.env_oheclasses = []; residue.encode_atoms()
            aa = [aa20_index(atoms[j].residue.resname) for j in ids] if encoding == "atom16aa20" else [None] * len(ids)
            ordered = [(d, ATOM_OHE.index(c), a) for d, c, a in sorted(zip(distance[ids], residue.env_oheclasses, aa), key=lambda v: (v[0], v[1]))[:SLOTS]]
        else:
            ordered = [(d, a, None) for d, a in sorted(zip(distance[ids], [aa20_index(atoms[j].residue.resname) for j in ids]))[:SLOTS]]
        for position, (value, cls, a) in enumerate(ordered):
            bb[position * slot + cls] = 1 / float(value) ** 2
            if a is not None: bb[position * slot + 16 + a] = 1 / float(value) ** 2
        bb[SLOTS * slot + RES_OHE.index(residue.resname)] = 1.0
        out[key] = (full[index], bb)
    return record["complex_id"], out


def build_validation(root, encoding, workers=1):
    """Re-encode the frozen 5k-pilot pKPDB validation rows (the same 7,778 rows, order and targets as
    pkpdb-val-pkai-v1) from their pilot input.pdb files. With encoding="atom16" the arrays must reproduce the existing
    package exactly; that check is recorded, and is what validates the builder for other encodings."""
    from concurrent.futures import ProcessPoolExecutor
    root = Path(root); base = Path(root) / "pretraining/pkpdb-val-pkai-v1"; rows = read(base / "rows.json")
    pilot = Path(root) / "pretraining/pkpdb-5k-comparison-v1"
    records = {r["complex_id"]: r for r in read(pilot / "pkai-features.json")["records"] if r["split"] == "val"}
    for record in records.values():  # paths were recorded on coulson; resolve them under this runtime
        record["path"] = str(pilot / "pkai-data/val" / Path(record["path"]).name)
    wanted = {}
    for r in rows: wanted.setdefault(r["complex_id"], []).append((r["chain"], r["resnum"], r["icode"], r["group"]))
    with ProcessPoolExecutor(workers) as pool:
        computed = dict(pool.map(_validation_component, [(records[c], encoding, keys) for c, keys in sorted(wanted.items())]))
    full = np.zeros((len(rows), feature_width(encoding)), np.float32); bb = np.zeros_like(full)
    for i, r in enumerate(rows):
        f, b = computed[r["complex_id"]][(r["chain"], r["resnum"], r["icode"], r["group"])]; full[i] = f; bb[i] = b
    dest = validation_package(root, encoding); pending = dest.parent / f".{dest.name}.pending-{os.getpid()}"
    pending.mkdir(parents=True, exist_ok=True)
    np.save(pending / "full.npy", full); np.save(pending / "backbone.npy", bb)
    for name in ("target.npy", "source_rows.npy", "rows.json"): (pending / name).write_bytes((base / name).read_bytes())
    check = None
    if encoding == "atom16" and not GEOMETRY:
        check = {"full_identical": bool(np.array_equal(full, np.load(base / "full.npy"))),
                 "backbone_identical": bool(np.array_equal(bb, np.load(base / "backbone.npy")))}
        if not all(check.values()): raise AssertionError(("atom16 re-encoding differs from pkpdb-val-pkai-v1", check))
    atomic_json(pending / "manifest.json", {"encoding": encoding, "rows": len(rows), "width": feature_width(encoding),
        "source": "pkpdb-5k-comparison-v1 pilot validation input.pdb files; rows, order and targets of pkpdb-val-pkai-v1",
        "atom16_reproduces_existing_package": check,
        "files": {p.name: digest(p) for p in sorted(pending.iterdir()) if p.name != "manifest.json"}})
    if dest.exists():
        import shutil; shutil.rmtree(dest)
    os.replace(pending, dest)
    print(json.dumps({"encoding": encoding, "rows": len(rows), "check": check}), flush=True)
    return dest


def main():
    import sys
    root = Path(os.environ["PKABENCH_RUNTIME"])
    action = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if action == "smoke": smoke(root)
    elif action == "register": register(root)
    elif action == "store-ids": store_ids(root)
    elif action == "prepare": prepare_shard(root, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif action == "import": import_run(root, sys.argv[2], sys.argv[3])
    elif action == "pack-store": pack_store(root, sys.argv[2], int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    elif action == "compare-packed": compare_packed(root)
    elif action == "build-interface": build_interface(root, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    elif action == "rescore": rescore(root, sys.argv[2:])
    elif action == "feature-smoke": feature_smoke(root)
    elif action == "build-validation":
        build_validation(root, sys.argv[2] if len(sys.argv) > 2 else ENCODING, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    elif action == "train":
        train_scale(root, sys.argv[2], sys.argv[3], *(int(v) for v in sys.argv[4:6]), *(float(v) for v in sys.argv[6:7]), *sys.argv[7:8])
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
