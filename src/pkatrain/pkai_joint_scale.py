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
from .pkai_scratch import architecture_gate, feature_matrix, model_class, native


SEED = 17
MODES = ("backbone", "full")
SIDECHAIN_GROUPS = frozenset(("ASP", "CYS", "TYR", "GLU", "HIS", "LYS"))
GROUP_ALIAS = {"NTR": "NTERM", "CTR": "CTERM"}
KEY_FIELDS = ("chain", "resnum", "icode", "group")
# pool-v3 fraction (nested 0.1/0.5/0.75/1.0 subsets); PKAI_FRACTION selects another one, with its own output directory
FRACTION = float(os.environ.get("PKAI_FRACTION", "0.1"))
BATCH_SIZE = 256
REFERENCE_BATCH = 64
REFERENCE_LR = 1e-6
LEARNING_RATE = REFERENCE_LR * math.sqrt(BATCH_SIZE / REFERENCE_BATCH)
OBJECTIVES = ("pkpdb", "joint")
# Feature failures that are properties of the record, not of the code: the record is excluded and listed in the packed
# verification. Any other error still fails the shard and blocks packing.
EXCLUDABLE = ("no mapped pKPDB sites", "no paired sites", "PDB capacity", "Coincident pKAI environment/reference atoms")


def _excludable(failure):
    return any(reason in failure["error"] for reason in EXCLUDABLE)


def read(path):
    return json.loads(Path(path).read_text())


def output(root):
    suffix = "" if FRACTION == 0.1 else f"-f{round(FRACTION * 100)}"  # the registered 10% runs keep their path
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
        residues, matrix = feature_matrix(Protein(path))
    lookup = {}
    for index, residue in enumerate(residues):
        original = mapping[(str(residue.chain), int(residue.resnumb))]
        key = (*original, str(residue.resname))
        if key in lookup:
            raise AssertionError((cid, state, key, "duplicate pKAI site"))
        lookup[key] = index
    retained = np.asarray([key in lookup for key in keys], bool)
    result = np.zeros((len(keys), 4008), np.float32)
    for index, key in enumerate(keys):
        if retained[index]:
            result[index] = matrix[lookup[key]]
    return result, retained


def _state_features(root, cid, state, keys, mode):
    atoms = _read_cif(pinder_source(root) / "entries" / cid / f"{state}.cif.gz")
    if mode == "backbone":
        return backbone_features(atoms, keys)
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
        residues, matrix = feature_matrix(Protein(path))
    lookup = {}
    for index, residue in enumerate(residues):
        original = mapping[(str(residue.chain), int(residue.resnumb))]
        key = (*original, GROUP_ALIAS.get(str(residue.resname), str(residue.resname)))
        if key in lookup: raise AssertionError((key, "duplicate pKAI site"))
        lookup[key] = index
    retained = np.asarray([key in lookup for key in keys], bool)
    result = np.zeros((len(keys), 4008), np.float32)
    for i, key in enumerate(keys):
        if retained[i]: result[i] = matrix[lookup[key]]
    return result, retained


def _pkpdb_feature_record(root, cid):
    folder = Path(root) / "pretraining/pkpdb-full-v1/entries" / cid
    sites = [s for s in read(folder / "sites.json") if s["train_mask"] and s["group"] in SIDECHAIN_GROUPS]
    env = {tuple(row[k] for k in KEY_FIELDS): row for row in read(folder / "environment.json")["sites"]}
    keys = [tuple(s[k] for k in KEY_FIELDS) for s in sites]
    atoms = _pkpdb_atoms(root, cid)
    full, kf = _native_features_for_atoms(atoms, keys); bb, kb = backbone_features(atoms, keys)
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
                  w_interface=np.asarray([r["w_interface"] for r in rows], np.float32))
    if not retained.any(): raise ValueError((cid, "no sites mapped in both representations"))
    return {key: value[retained] for key, value in arrays.items()}


def prepare_shard(root, dataset, task, tasks):
    root = Path(root); out = output(root); manifest = read(out / "manifest.json")
    if digest(out / "records.json") != manifest["records_sha256"]: raise AssertionError("records changed")
    records = read(out / "records.json")
    if dataset == "pkpdb": ids = records["pkpdb_train"]
    elif dataset == "pinder": ids = [(cid, "train") for cid in records["pinder_train"]] + [(cid, "val") for cid in records["pinder_val"]]
    else: raise ValueError(dataset)
    selected = ids[task::tasks]; dest = out / "features" / dataset; dest.mkdir(parents=True, exist_ok=True)
    done = 0; sites = 0; failures = []
    for item in selected:
        cid, split = (item, "train") if dataset == "pkpdb" else item
        path = dest / f"{split}-{cid}.npz"
        try:
            if path.exists():
                with np.load(path) as handle: n = len(handle["target"] if dataset == "pkpdb" else handle["target_ab"])
            else:
                values = _pkpdb_feature_record(root, cid) if dataset == "pkpdb" else _pinder_feature_record(root, cid, split)
                pending = path.with_suffix(f".pending-{os.getpid()}.npz")
                np.savez_compressed(pending, **values); os.replace(pending, path); n = len(next(iter(values.values())))
            done += 1; sites += n
        except Exception as exc:
            failures.append({"id": cid, "split": split, "error": repr(exc)})
        if (done + len(failures)) % 10 == 0: print(json.dumps({"dataset":dataset,"task":task,"done":done,"failures":len(failures)}), flush=True)
    receipt = {"dataset":dataset,"task":task,"tasks":tasks,"assigned":len(selected),"completed":done,"sites":sites,"failures":failures}
    atomic_json(out / f"prepare-{dataset}-{task:03d}.json", receipt)
    blocking = [f for f in failures if not _excludable(f)]
    if blocking: raise RuntimeError(f"{len(blocking)} feature failures ({len(failures) - len(blocking)} excludable); see receipt")


def _pack_group(paths, destination, fields):
    counts = []
    for path in paths:
        with np.load(path) as handle: counts.append(len(handle[fields[0]]))
    total = sum(counts); destination.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for field in fields:
        shape = (total, 4008) if field.startswith(("full", "backbone")) else (total,)
        arrays[field] = np.lib.format.open_memmap(destination / f"{field}.npy", mode="w+", dtype=np.float32, shape=shape)
    offsets = []; start = 0
    for path, count in zip(paths, counts):
        with np.load(path) as handle:
            for field in fields: arrays[field][start:start+count] = handle[field]
        offsets.append({"file": path.name, "start": start, "stop": start+count}); start += count
    for value in arrays.values(): value.flush()
    del arrays
    atomic_json(destination / "offsets.json", offsets)
    return {"sites": total, "files": len(paths), "offsets_sha256": digest(destination / "offsets.json"),
            "arrays": {field: digest(destination / f"{field}.npy") for field in fields}}


def pack_features(root):
    root = Path(root); out = output(root); records = read(out / "records.json")
    expected = {"pkpdb": len(records["pkpdb_train"]),
                "pinder": len(records["pinder_train"]) + len(records["pinder_val"])}
    receipts = {}
    for dataset in ("pkpdb", "pinder"):
        rr = [read(p) for p in sorted(out.glob(f"prepare-{dataset}-*.json"))]
        failures = [f for r in rr for f in r["failures"]]
        if any(not _excludable(f) for f in failures): raise AssertionError((dataset, "feature failures"))
        if not rr or sum(r["completed"] for r in rr) + len(failures) != expected[dataset]:
            raise AssertionError((dataset, len(rr), sum(r["completed"] for r in rr), len(failures), expected[dataset]))
        receipts[f"{dataset}_excluded"] = sorted(failures, key=lambda f: (f["split"], f["id"]))
    pkpaths = sorted((out / "features/pkpdb").glob("train-*.npz"))
    receipts["pkpdb_train"] = _pack_group(pkpaths, out / "packed/pkpdb-train",
        ("full","backbone","target","weight"))
    for split in ("train", "val"):
        paths = sorted((out / "features/pinder").glob(f"{split}-*.npz"))
        receipts[f"pinder_{split}"] = _pack_group(paths, out / f"packed/pinder-{split}",
            ("full_ab","full_free","backbone_ab","backbone_free","target_ab","target_free","w_burial","w_interface"))
    excluded = {k: receipts.pop(k) for k in [k for k in receipts if k.endswith("_excluded")]}
    atomic_json(out / "packed/verification.json", {"passed":True,"groups":receipts,"excluded":excluded,"test_data_included":False})
    print(json.dumps({**{k:v["sites"] for k,v in receipts.items()}, **{k:len(v) for k,v in excluded.items()}}, sort_keys=True), flush=True)


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
    if package.exists():
        if (pilot / "rows.json").exists():
            rows = read(pilot / "rows.json"); ids = np.asarray([i for i, r in enumerate(rows) if r["split"] == "val" and r["group"] in SIDECHAIN_GROUPS])
            if not np.array_equal(ids, np.load(package / "source_rows.npy")): raise AssertionError("validation package rows differ from the pilot")
        return {"x": np.load(package / ("full.npy" if mode == "full" else "backbone.npy"), mmap_mode="r"), "y": np.load(package / "target.npy")}
    rows = read(pilot / "rows.json"); ids = np.asarray([i for i, r in enumerate(rows) if r["split"] == "val" and r["group"] in SIDECHAIN_GROUPS])
    feature_path = pilot / "features.npy" if mode == "full" else Path(root) / "pretraining/pkai-backbone-ablation-v1/backbone-features.npy"
    return {"x": np.load(feature_path, mmap_mode="r")[ids], "y": np.asarray([rows[i]["pka"] - rows[i]["model_pka"] for i in ids], np.float32)}


def _validation_sources(torch, root, mode, objective, config):
    from .loading_torch import PackedSiteSource
    sources = {"pk": PackedSiteSource(_pkpdb_validation_arrays(root, mode), config=config)}
    if objective == "joint":
        pi = _arrays(output(root) / "packed/pinder-val", (f"{mode}_ab", f"{mode}_free", "target_ab", "target_free"))
        sources["pi"] = PackedSiteSource({"xa": pi[f"{mode}_ab"], "xf": pi[f"{mode}_free"], "ya": pi["target_ab"], "yf": pi["target_free"]}, config=config)
    return sources


def _validation(torch, model, sources, config):
    # The established clean 5k validation is independent of the new 10% pool.
    pk = sources["pk"]
    result = {"pkpdb_mse": _squared_sums(torch, model, pk, pk.length, {"pk": lambda b, o: (o - b["y"]).square()}, config,
                                         lambda b: model(b["x"]))["pk"]}
    if "pi" in sources:
        pi = sources["pi"]
        terms = {"a": lambda b, o: (o[0] - b["ya"]).square(), "f": lambda b, o: (o[1] - b["yf"]).square(),
                 "pair": lambda b, o: ((o[0] - o[1]) - (b["ya"] - b["yf"])).square()}
        mse = _squared_sums(torch, model, pi, pi.length, terms, config, lambda b: (model(b["xa"]), model(b["xf"])))
        result.update(pinder_state_mse=(mse["a"] + mse["f"]) / 2, pinder_paired_mse=mse["pair"])
    result["selection_mse"] = sum(result.values()) / len(result)
    return result


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


def train_scale(root, mode, objective, batch_size=None, max_epochs=100):
    """batch_size/max_epochs other than the registered 256/100 (batch-size sweep, 2026-10-10) use the sqrt learning-rate
    rule and write to runs/<mode>-<objective>-b<batch>-e<epochs>/; the defaults keep the registered run paths."""
    global BATCH_SIZE, LEARNING_RATE
    tag = ""
    if (batch_size or BATCH_SIZE) != BATCH_SIZE or max_epochs != 100:
        BATCH_SIZE = int(batch_size or BATCH_SIZE); LEARNING_RATE = REFERENCE_LR * math.sqrt(BATCH_SIZE / REFERENCE_BATCH)
        tag = f"-b{BATCH_SIZE}-e{max_epochs}"
    from .loading import DeferredScalars, LoaderConfig, Prefetcher
    from .loading_torch import CombinedSource, PackedSiteSource
    if mode not in MODES or objective not in OBJECTIVES: raise ValueError((mode,objective))
    root=Path(root);out=output(root); packed=read(out/"packed/verification.json")
    if not packed["passed"]: raise AssertionError("unverified packed inputs")
    torch,_=native();require_compute(threads=2,gpu_benchmark=True,allow_comp1400=True)
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
    model=model_class(torch)().cuda().train();opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
    config=LoaderConfig()
    pk=_arrays(out/"packed/pkpdb-train",(mode,"target","weight")); pi=None
    sources={"pk":PackedSiteSource({"x":pk[mode],"y":pk["target"],"w":pk["weight"]},config=config)}
    if objective=="joint":
        pi=_arrays(out/"packed/pinder-train",(f"{mode}_ab",f"{mode}_free","target_ab","target_free","w_burial","w_interface"))
        sources["pi"]=PackedSiteSource({"xa":pi[f"{mode}_ab"],"xf":pi[f"{mode}_free"],"ya":pi["target_ab"],"yf":pi["target_free"],"wb":pi["w_burial"],"wi":pi["w_interface"]},config=config)
    train_source=CombinedSource(sources); validation_sources=_validation_sources(torch,root,mode,objective,config)
    dest=out/"runs"/f"{mode}-{objective}{tag}"/f"seed-{SEED}";dest.mkdir(parents=True,exist_ok=True)
    provenance={"mode":mode,"objective":objective,"seed":SEED,"batch_size":BATCH_SIZE,"max_epochs":max_epochs,"learning_rate":LEARNING_RATE,"lr_rule":"1e-6*sqrt(batch/64)","precision":"float32","cpus":2,"initialization":"scratch","packed_verification_sha256":digest(out/"packed/verification.json"),"manifest_sha256":digest(out/"manifest.json"),"test_data_included":False,
                "loader":{"workers":config.workers,"prefetch":config.prefetch,"train":train_source.provenance(),"validation":{k:v.provenance() for k,v in validation_sources.items()}}}
    atomic_json(dest/"manifest.json",provenance)
    params=list(model.parameters())
    rng=np.random.default_rng(SEED);best=float("inf");anchor=float("inf");stall=0;history=[];began=time.monotonic();patience=8
    for epoch in range(1,max_epochs+1):
        if stall>=patience:break
        model.train();order_pk=rng.permutation(len(pk["target"]));order_pi=rng.permutation(len(pi["target_ab"])) if pi else None
        losses=DeferredScalars(every=50)
        prefetcher=Prefetcher(train_source,_epoch_specs(order_pk,order_pi),config,train_source.to_device,train_source.on_consume)
        for _,batch in prefetcher:
            b=batch["pk"];lpk=_weighted_mse(torch,model(b["x"]),b["y"],b["w"]);components=[lpk]
            if pi:
                b=batch["pi"];pa=model(b["xa"]);pf=model(b["xf"])
                components += [(_weighted_mse(torch,pa,b["ya"],b["wb"])+_weighted_mse(torch,pf,b["yf"],b["wb"]))/2,_weighted_mse(torch,pa-pf,b["ya"]-b["yf"],b["wi"])]
            loss=sum(components)/len(components);opt.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is None for p in params):raise FloatingPointError("missing gradient")
            finite=torch.isfinite(loss)&torch.stack([torch.isfinite(p.grad).all() for p in params]).all()
            if not bool(finite):raise FloatingPointError("nonfinite training step")  # one synchronisation per step
            opt.step();losses.add(loss.detach())
        loader=prefetcher.telemetry.summary()
        metrics=_validation(torch,model,validation_sources,config);score=metrics["selection_mse"]
        if score<best:best=score;torch.save(model.state_dict(),dest/"best.pending.pt");os.replace(dest/"best.pending.pt",dest/"best.pt")
        if score<anchor-.001:anchor=score;stall=0
        else:stall+=1
        row={"epoch":epoch,"train_mse":float(np.mean(losses.flush())),**metrics,"stall":stall,"seconds":time.monotonic()-began,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
             "loader_wait_fraction":loader["wait_fraction"],"loader_max_wait_seconds":loader["max_wait_seconds"]};history.append(row);atomic_json(dest/"history.json",history);atomic_json(dest/"progress.json",row);print(json.dumps(row),flush=True)
    train_source.close()
    for source in validation_sources.values(): source.close()
    atomic_json(dest/"final.json",{"complete":True,"epochs":len(history),"best_selection_mse":best,"history_sha256":digest(dest/"history.json"),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"test_data_included":False})


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


def main():
    import sys
    root = Path(os.environ["PKABENCH_RUNTIME"])
    action = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if action == "smoke": smoke(root)
    elif action == "register": register(root)
    elif action == "prepare": prepare_shard(root, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif action == "pack": pack_features(root)
    elif action == "feature-smoke": feature_smoke(root)
    elif action == "train":
        train_scale(root, sys.argv[2], sys.argv[3], *(int(v) for v in sys.argv[4:6]))
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
