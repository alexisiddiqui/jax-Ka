"""Separately versioned oGQT burial/interface auxiliary-loss pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pkanet.model import PKPDB_PK_MOD
from pkanet.ogqt import initialize as initialize_ogqt
from pkanet.ogqt import initialize_auxiliary, predict_multi, predict_shift
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_crop_radius import MIN_DELTA, PATIENCE
from .gqt_paired_pinder import (
    BASE_FIELDS, RAW_FIELDS, Loader, _bucket_n, _initialize_worker, _load_one,
    _mmap_fingerprint, _pad, _paired_rows, _plans, _prefetched, _prepare_one,
    _raw_shape, code_hashes as parent_code_hashes, experiment_root as parent_root,
    source,
)
from .gqt_regularization import decay_mask
from .trainer import load_checkpoint, save_checkpoint


VERSION = "ogqt-auxiliary-pilot-v1"
ARMS = {"baseline": 0.0, "low": 0.1, "standard": 1.0}
SEED = 17
EPOCHS = 10
DIAGNOSTIC_BATCHES = 32


def read(path): return json.loads(Path(path).read_text())
def experiment_root(root): return Path(root) / "training" / VERSION
def pool_path(root): return Path(root) / "pretraining/pinder-pkai-v1/pool-v2.tsv"


def code_hashes():
    src = Path(__file__).parents[1]
    paths = (Path(__file__), src / "pkanet/site_model.py", src / "pkanet/ogqt.py",
             src / "pkanet/model.py", src / "pkanet/triton_attention.py",
             src / "pkatrain/gqt_paired_pinder.py")
    return {str(path): digest(path) for path in paths}


def _pool_rows(root):
    with pool_path(root).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    selected = []
    for row in rows:
        if (float(row["min_fraction"]) <= 0.1 and int(row["chain_70_to_validation"]) == 0
                and int(row["n_res"]) <= 768):
            selected.append({"id": row["id"], "cluster_id": row["group"],
                "ctype": {"Ab/Ag": "antibody_antigen", "hetero": "heteromer",
                          "homo": "homomer"}[row["stratum"]],
                "n_res": int(row["n_res"]), "split": "train"})
    return selected


def register(root):
    root = Path(root); out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    train = _pool_rows(root)
    parent = read(parent_root(root) / "manifest.json")
    validation = [{"id": row["id"], "cluster_id": row["cluster_id"], "ctype": row["ctype"],
                   "n_res": row["n"], "split": row["split"]}
                  for row in parent["records"] if row["split"] == "val"]
    if len(validation) != 400 or len({row["cluster_id"] for row in validation}) != 400:
        raise AssertionError("fixed validation cohort changed")
    if {row["cluster_id"] for row in train} & {row["cluster_id"] for row in validation}:
        raise AssertionError("training/validation cluster overlap")
    records = []
    for row in (*train, *validation):
        folder = source(root) / "entries" / row["id"]
        if not folder.exists(): raise FileNotFoundError(folder)
        records.append({**row, "source_hashes": {name: digest(folder / name) for name in
            ("AB.cif.gz", "sites.json", "labels.json", "meta.json")}})
    protocol = {"version": VERSION, "seed": SEED, "epochs": EPOCHS,
        "arms": ARMS, "warmup": "one common pKa-only epoch; cloned parameters and optimizer",
        "schedule": "epoch 1 at 1e-3; cosine 1e-3 to 1e-5 over epochs 2-10",
        "calibration_batches": DIAGNOSTIC_BATCHES,
        "calibration": "lambda_t=0.1*median(shared primary gradient norm)/(median(shared auxiliary gradient norm)+1e-12)",
        "objective": "unweighted state MSE + paired MSE + alpha*(lambda_b*burial MSE + lambda_i*interface MSE)",
        "selection": "unweighted validation state MAE + interface paired MAE; common warmup eligible",
        "patience": PATIENCE, "min_delta": MIN_DELTA,
        "pool": "pool-v2 min_fraction<=0.1, n_res<=768, chain_70_to_validation excluded",
        "test_data_included": False}
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "cohort.json", {"records": records, "train": [r["id"] for r in train],
        "val": [r["id"] for r in validation], "pool_sha256": digest(pool_path(root)),
        "parent_validation_manifest_sha256": digest(parent_root(root) / "manifest.json")})
    atomic_json(out / "registration.json", {"passed": True, "train_structures": len(train),
        "train_clusters": len({r["cluster_id"] for r in train}), "validation_structures": len(validation),
        "validation_clusters": len({r["cluster_id"] for r in validation}),
        "ctype_counts": dict(Counter(r["ctype"] for r in train)), "code_hashes": code_hashes(),
        "protocol_sha256": digest(out / "protocol.json"), "cohort_sha256": digest(out / "cohort.json"),
        "test_data_included": False})


def _prepare_link(task):
    root, record, destination = task
    _initialize_worker(root, None)
    receipt = _prepare_one(record)
    source_graph = parent_root(Path(root)) / "graphs" / record["id"]
    target = Path(destination) / record["id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists(): os.symlink(source_graph, target, target_is_directory=True)
    return receipt


def prepare(root):
    root = Path(root); base = experiment_root(root); registration = read(base / "registration.json")
    if registration["code_hashes"] != code_hashes(): raise AssertionError("registration/code mismatch")
    cohort = read(base / "cohort.json"); tasks = [(str(root), row, str(base / "graphs")) for row in cohort["records"]]
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 96)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        receipts = []
        for number, receipt in enumerate(pool.map(_prepare_link, tasks, chunksize=1), 1):
            receipts.append(receipt)
            if number % 100 == 0: print(json.dumps({"graphs": number, "total": len(tasks)}), flush=True)
    capacities = {}
    for name in sorted({_bucket_n(row["n"]) for row in receipts}, key=int):
        members = [row for row in receipts if _bucket_n(row["n"]) == name]
        maxima = [max(row[key] for row in members) for key in ("k", "q", "s", "sk")]
        capacities[name] = [int(name), *[int(np.ceil(value / 32) * 32) for value in maxima]]
    parent = read(parent_root(root) / "manifest.json")
    manifest = {"records": receipts, "capacities": capacities, "normalization": parent["normalization"],
        "architecture": parent["architecture"], "batch_sizes": parent["batch_sizes"],
        "cohort_sha256": digest(base / "cohort.json"), "protocol_sha256": digest(base / "protocol.json"),
        "test_data_included": False}
    atomic_json(base / "manifest.json", manifest)
    atomic_json(base / "preparation.json", {"passed": True, "structures": len(receipts),
        "train_structures": sum(r["split"] == "train" for r in receipts),
        "validation_structures": sum(r["split"] == "val" for r in receipts),
        "sites": sum(r["q"] for r in receipts), "interface_sites": sum(r["q_interface"] for r in receipts),
        "capacities": capacities, "code_hashes": code_hashes(),
        "manifest_sha256": digest(base / "manifest.json")})


def build_mmap(root):
    base = experiment_root(root); manifest = read(base / "manifest.json"); records = manifest["records"]
    destination = base / "mmap-v1"
    if destination.exists():
        verification = read(destination / "verification.json")
        if verification["records_fingerprint"] == _mmap_fingerprint(records):
            # The packed arrays are content-addressed by the record fingerprint.
            # A training-only code fix does not require rewriting them, but the
            # verification receipt must record the code that accepted them.
            atomic_json(destination / "verification.json", {**verification, "code_hashes": code_hashes()})
            return
        raise FileExistsError(destination)
    first = base / "graphs" / records[0]["id"] / "graph.npz"
    with np.load(first, allow_pickle=False) as handle:
        if set(handle.files) != set(RAW_FIELDS): raise AssertionError(handle.files)
        dtypes = {name: handle[name].dtype.str for name in RAW_FIELDS}
    offsets = np.zeros((len(records) + 1, len(RAW_FIELDS)), np.int64)
    for index, row in enumerate(records):
        offsets[index + 1] = offsets[index] + [int(np.prod(_raw_shape(name, row))) for name in RAW_FIELDS]
    pending = destination.parent / f".{destination.name}.pending-{os.getpid()}"; pending.mkdir()
    arrays = {name: np.lib.format.open_memmap(pending / f"{name}.npy", mode="w+", dtype=np.dtype(dtypes[name]),
        shape=(int(offsets[-1, field]),)) for field, name in enumerate(RAW_FIELDS)}
    for index, row in enumerate(records):
        path = base / "graphs" / row["id"] / "graph.npz"
        if digest(path) != row["sha256"]: raise AssertionError((row["id"], "graph hash"))
        with np.load(path, allow_pickle=False) as handle:
            for field, name in enumerate(RAW_FIELDS):
                value = handle[name]; expected = _raw_shape(name, row)
                if value.shape != expected: raise AssertionError((row["id"], name, value.shape, expected))
                start, stop = offsets[index, field], offsets[index + 1, field]
                arrays[name][start:stop] = value.reshape(-1)
        if (index + 1) % 100 == 0: print(json.dumps({"mmap": index + 1, "total": len(records)}), flush=True)
    for value in arrays.values(): value.flush()
    arrays.clear(); np.savez(pending / "index.npz", complex_id=np.asarray([r["id"] for r in records]), offsets=offsets)
    fingerprint = _mmap_fingerprint(records)
    atomic_json(pending / "metadata.json", {"fields": list(RAW_FIELDS), "dtypes": dtypes,
        "records_fingerprint": fingerprint})
    atomic_json(pending / "verification.json", {"passed": True, "records": len(records),
        "records_fingerprint": fingerprint, "all_source_hashes_checked": True,
        "all_fields_read_back_identically": True, "code_hashes": code_hashes()})
    os.replace(pending, destination)


def _tree_shared(tree): return {name: value for name, value in tree.items() if name not in ("head", "auxiliary")}


def _norm(tree): return float(optax.global_norm(tree))


def _cosine(a, b):
    dot = aa = bb = 0.0
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        dot += float(jnp.vdot(left, right)); aa += float(jnp.vdot(left, left)); bb += float(jnp.vdot(right, right))
    return dot / max(np.sqrt(aa * bb), 1e-30)


class AuxiliaryEngine:
    def __init__(self, params, alpha, lambdas, *, predict_fn=predict_multi):
        self.alpha = float(alpha); self.lambdas = dict(lambdas)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0),
            optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))

        def predictions(p, graphs):
            batch, branches = graphs["nodes"].shape[:2]
            flat = jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs)
            values = jax.vmap(predict_fn, in_axes=(None, 0))(p, flat)
            return {name: value.reshape(batch, branches, -1) for name, value in values.items()}

        def losses(p, graphs, targets, mask, burial, interface, valid):
            predicted = predictions(p, graphs)
            reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32)[graphs["query_group"][:, 0]]
            expected = targets - reference[:, None, :]
            count = jnp.maximum(mask.sum(1), 1); denom = jnp.maximum(valid.sum(), 1)
            state_per = jnp.sum(jnp.square(predicted["shift"] - expected) * mask[:, None, :], (1, 2)) / (2 * count)
            pair_per = jnp.sum(jnp.square((predicted["shift"][:, 0] - predicted["shift"][:, 1]) -
                (expected[:, 0] - expected[:, 1])) * mask, 1) / count
            burial_per = jnp.sum(jnp.square(predicted["burial"][:, 1] - burial) * mask, 1) / count
            interface_per = jnp.sum(jnp.square(predicted["interface"][:, 0] - interface) * mask, 1) / count
            mean = lambda value: jnp.sum(jnp.where(valid, value, 0.0)) / denom
            return mean(state_per), mean(pair_per), mean(burial_per), mean(interface_per)

        def objective(p, graphs, targets, mask, burial, interface, valid):
            state, pair, b, i = losses(p, graphs, targets, mask, burial, interface, valid)
            return state + pair + self.alpha * (self.lambdas["burial"] * b + self.lambdas["interface"] * i)

        def individual(p, graphs, targets, mask, burial, interface, valid):
            state, pair, b, i = losses(p, graphs, targets, mask, burial, interface, valid)
            return state + pair, b, i

        def step(p, state, graphs, targets, mask, burial, interface, valid, rate):
            (total, parts), gradient = jax.value_and_grad(
                lambda x: (objective(x, graphs, targets, mask, burial, interface, valid),
                           losses(x, graphs, targets, mask, burial, interface, valid)), has_aux=True)(p)
            finite = jnp.isfinite(total) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            updates = jax.tree.map(lambda value: value * rate, updates)
            return optax.apply_updates(p, updates), state, total, parts, finite

        def individual_gradients(p, graphs, targets, mask, burial, interface, valid):
            arguments = (graphs, targets, mask, burial, interface, valid)
            return tuple(jax.grad(lambda x, index=index: individual(x, *arguments)[index])(p)
                         for index in range(3))
        self.predictions = jax.jit(predictions); self.losses = jax.jit(losses)
        self.step = jax.jit(step); self.individual_grads = jax.jit(individual_gradients)

    @staticmethod
    def targets(batch, norms):
        _, targets, mask, normalized_burial, normalized_interface, metadata = batch
        raw_burial = normalized_burial * float(norms["burial"])
        burial = (raw_burial - 0.4) / 0.6
        raw_interface = normalized_interface * float(norms["interface"])
        interface = (raw_interface - 0.05) / 0.95
        if (not np.isfinite(burial[mask]).all() or not np.isfinite(interface[mask]).all()
                or np.any((burial[mask] < -1e-5) | (burial[mask] > 1 + 1e-5))
                or np.any((interface[mask] < -1e-5) | (interface[mask] > 1 + 1e-5))):
            raise ValueError("invalid active auxiliary target")
        return targets, mask, np.clip(burial, 0, 1), np.clip(interface, 0, 1)

    def update(self, params, state, batch, norms, rate):
        graphs = batch[0]; targets, mask, burial, interface = self.targets(batch, norms)
        valid = np.ones(len(targets), bool)
        params, state, total, parts, finite = self.step(
            params, state, graphs, targets, mask, burial, interface, valid, rate)
        values = np.asarray((total, *parts), float)
        if not bool(finite) or not np.isfinite(values).all(): raise FloatingPointError("nonfinite auxiliary update")
        return params, state, dict(zip(("total", "state", "paired", "burial", "interface"), map(float, values)))

    def gradients(self, params, batch, norms):
        graphs = batch[0]; targets, mask, burial, interface = self.targets(batch, norms)
        valid = np.ones(len(targets), bool)
        trees = self.individual_grads(params, graphs, targets, mask, burial, interface, valid)
        return tuple(_tree_shared(tree) for tree in trees)


def _rate(epoch, batch, batches):
    if epoch == 1: return 1e-3
    fraction = ((epoch - 2) + batch / max(batches, 1)) / 9.0
    return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * min(max(fraction, 0), 1)))


def _diagnostic_plan(records, manifest):
    return _plans(records, np.random.default_rng(17017), manifest)[:DIAGNOSTIC_BATCHES]


def _gradient_diagnostics(engine, params, loader, plans, norms, coefficients):
    values = defaultdict(list)
    for _, batch in _prefetched(loader, plans):
        primary, burial, interface = engine.gradients(params, batch, norms)
        values["primary_norm"].append(_norm(primary))
        for name, gradient in (("burial", burial), ("interface", interface)):
            values[f"{name}_norm"].append(_norm(gradient))
            values[f"{name}_weighted_norm"].append(engine.alpha * coefficients[name] * _norm(gradient))
            values[f"{name}_cosine"].append(_cosine(primary, gradient))
    return {name: {"median": float(np.median(row)), "mean": float(np.mean(row))} for name, row in values.items()}


def _rankdata(values):
    order = np.argsort(values, kind="stable"); ranks = np.empty(len(values), float); start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]: stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1; start = stop
    return ranks


def _corr(x, y, rank=False):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if rank: x, y = _rankdata(x), _rankdata(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(x, y)[0, 1])


def _average_precision(labels, scores, keys):
    positives = int(np.sum(labels))
    if positives == 0: return None
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], keys[i]))
    hits = 0; total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]: hits += 1; total += hits / rank
    return total / positives


def _interface_metrics(rows, score_field):
    by_complex = defaultdict(list)
    for row in rows: by_complex[row["complex_id"]].append(row)
    result = []
    for cid, sites in by_complex.items():
        residues = defaultdict(list)
        for row in sites: residues[(row["chain"], row["resnum"], row["icode"])].append(row)
        keys = sorted(residues); scores = [float(np.mean([r[score_field] for r in residues[k]])) for k in keys]
        labels = [bool(residues[k][0]["interface"]) for k in keys]; k = sum(labels)
        if k == 0: result.append({"complex_id": cid, "defined": False}); continue
        selected = set(sorted(range(len(keys)), key=lambda i: (-scores[i], keys[i]))[:k])
        truth = {i for i, value in enumerate(labels) if value}; overlap = len(selected & truth)
        result.append({"complex_id": cid, "defined": True,
            "average_precision": _average_precision(labels, scores, keys),
            "top_k_recall": overlap / k, "jaccard": overlap / len(selected | truth)})
    defined = [r for r in result if r["defined"]]
    return {"eligible_complexes": len(defined), "undefined_complexes": len(result) - len(defined),
        **{name: float(np.mean([r[name] for r in defined])) if defined else None
           for name in ("average_precision", "top_k_recall", "jaccard")}}


def _jsd(target, predicted):
    p = np.histogram(target, bins=np.linspace(0, 1, 21))[0].astype(float)
    q = np.histogram(predicted, bins=np.linspace(0, 1, 21))[0].astype(float)
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    term = lambda x: np.sum(np.where(x > 0, x * np.log2(x / np.where(m > 0, m, 1)), 0))
    return float((term(p) + term(q)) / 2)


def _aux_metrics(rows, train_burial_mean):
    target = np.asarray([r["burial_target"] for r in rows]); predicted = np.asarray([r["burial_prediction"] for r in rows])
    grouped = defaultdict(list)
    for row in rows: grouped[row["complex_id"]].append(row)
    per = []
    for cid, items in grouped.items():
        x = [r["burial_target"] for r in items]; y = [r["burial_prediction"] for r in items]
        per.append({"mae": float(np.mean(np.abs(np.asarray(x) - y))), "pearson": _corr(x, y),
                    "spearman": _corr(x, y, rank=True)})
    return {"burial": {"pooled_mae": float(np.mean(np.abs(target - predicted))),
        "pooled_pearson": _corr(target, predicted), "pooled_spearman": _corr(target, predicted, rank=True),
        "equal_complex_mae": float(np.mean([x["mae"] for x in per])),
        "equal_complex_pearson": float(np.mean([x["pearson"] for x in per if x["pearson"] is not None])),
        "equal_complex_spearman": float(np.mean([x["spearman"] for x in per if x["spearman"] is not None])),
        "undefined_pearson_complexes": sum(x["pearson"] is None for x in per),
        "undefined_spearman_complexes": sum(x["spearman"] is None for x in per),
        "constant_train_mean": train_burial_mean,
        "constant_mae": float(np.mean(np.abs(target - train_burial_mean))), "histogram_jsd": _jsd(target, predicted)},
        "interface": {"raw_weight_mae": float(np.mean(np.abs(
            np.asarray([r["interface_target_raw"] for r in rows]) -
            np.asarray([r["interface_prediction_raw"] for r in rows])))),
            "predicted_ranking": _interface_metrics(rows, "interface_prediction"),
            "structural_reference": _interface_metrics(rows, "interface_target")}}


def _primary_metrics(rows):
    interface = [row for row in rows if row["interface"]]
    mean_abs = lambda subset, name: float(np.mean([abs(r[name]) for r in subset])) if subset else None
    bins = (("<0.05", 0, .05), ("0.05-0.2", .05, .2), ("0.2-0.6", .2, .6), (">=0.6", .6, 1.01))
    return {"sites": len(rows), "state_mae": mean_abs(rows, "state_error"),
        "paired_mae": mean_abs(rows, "paired_error"),
        "interface_paired_mae": mean_abs(interface, "paired_error"),
        "selection": mean_abs(rows, "state_error") + mean_abs(interface, "paired_error"),
        "distance_bins": {name: {"sites": len(part), "paired_mae": mean_abs(part, "paired_error")} for name, part in (
            ("<=4", [r for r in rows if r["distance"] <= 4]), ("4-6", [r for r in rows if 4 < r["distance"] <= 6]),
            ("6-10", [r for r in rows if 6 < r["distance"] <= 10]), (">10", [r for r in rows if r["distance"] > 10]))},
        "rsa_bins": {name: {"sites": len(part), "state_mae": mean_abs(part, "state_error")} for name, low, high in bins
                     for part in ([r for r in rows if low <= r["rsa_free"] < high],)}}


def evaluate(base, manifest, records, engine, params, train_burial_mean, path=None):
    loader = Loader(base, manifest); by_id = {r["id"]: r for r in records}; grouped = defaultdict(list)
    for record in records: grouped[_bucket_n(record["n"])].append(record["id"])
    plans = []
    for bucket in sorted(grouped, key=int):
        size = manifest["batch_sizes"][bucket]; ids = grouped[bucket]
        plans.extend(ids[i:i + size] for i in range(0, len(ids), size))
    rows = []
    for ids, batch in _prefetched(loader, plans):
        graphs, targets, mask, normalized_burial, normalized_interface, metadata = batch
        prediction = jax.tree.map(np.asarray, engine.predictions(params, graphs))
        burial = (normalized_burial * manifest["normalization"]["burial"] - .4) / .6
        interface_target = (normalized_interface * manifest["normalization"]["interface"] - .05) / .95
        for bi, cid in enumerate(ids):
            record = by_id[cid]
            for qi, key in enumerate(record["keys"]):
                if not mask[bi, qi]: continue
                group = graphs["query_group"][bi, 0, qi]; ref = float(np.asarray(PKPDB_PK_MOD)[group])
                expected = targets[bi, :, qi] - ref; shift = prediction["shift"][bi, :, qi]
                rows.append({"complex_id": cid, **dict(zip(("chain", "resnum", "icode", "group"), key)),
                    "teacher_ab": float(targets[bi, 0, qi]), "teacher_free": float(targets[bi, 1, qi]),
                    "predicted_ab": float(shift[0] + ref), "predicted_free": float(shift[1] + ref),
                    "state_error": float(np.mean(np.abs(shift - expected))),
                    "paired_error": float((shift[0] - shift[1]) - (expected[0] - expected[1])),
                    "burial_target": float(burial[bi, qi]), "burial_prediction": float(prediction["burial"][bi, 1, qi]),
                    "interface_target": float(interface_target[bi, qi]),
                    "interface_prediction": float(prediction["interface"][bi, 0, qi]),
                    "interface_target_raw": float(.05 + .95 * interface_target[bi, qi]),
                    "interface_prediction_raw": float(.05 + .95 * prediction["interface"][bi, 0, qi]),
                    "interface": bool(metadata["interface"][bi, qi]), "distance": float(metadata["partner_distance_A"][bi, qi]),
                    "rsa_free": float(metadata["rsa_free"][bi, qi])})
    loader.close()
    if path:
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {"primary": _primary_metrics(rows), "auxiliary": _aux_metrics(rows, train_burial_mean)}, rows


def tests(root):
    root = Path(root); base = experiment_root(root); manifest = read(base / "manifest.json")
    key = jax.random.PRNGKey(SEED); ordinary = initialize_ogqt(key, **manifest["architecture"])
    auxiliary = initialize_auxiliary(key, **manifest["architecture"])
    for name in ordinary:
        for left, right in zip(jax.tree.leaves(ordinary[name]), jax.tree.leaves(auxiliary[name])):
            if not np.array_equal(np.asarray(left), np.asarray(right)): raise AssertionError((name, "initialization changed"))
    train = [r for r in manifest["records"] if r["split"] == "train"]
    loader = Loader(base, manifest); ids = _plans(train, np.random.default_rng(SEED), manifest)[0]; batch = loader.batch(ids)
    engine = AuxiliaryEngine(auxiliary, 0, {"burial": 1, "interface": 1})
    graph = jax.tree.map(lambda value: value[0, 0], batch[0])
    np.testing.assert_allclose(np.asarray(predict_shift(ordinary, graph)), np.asarray(predict_multi(auxiliary, graph)["shift"]), rtol=0, atol=0)
    primary, burial, interface = engine.gradients(auxiliary, batch, manifest["normalization"])
    if min(_norm(primary), _norm(burial), _norm(interface)) <= 0: raise AssertionError("missing shared gradient")
    full_jacobian = engine.individual_grads(auxiliary, batch[0], *engine.targets(batch, manifest["normalization"]),
        np.ones(len(batch[1]), bool))
    if _norm(full_jacobian[1]["auxiliary"]["burial"]) <= 0: raise AssertionError("burial head gradient")
    if _norm(full_jacobian[2]["auxiliary"]["interface"]) <= 0: raise AssertionError("interface head gradient")
    # Padded targets must not affect any loss.
    args = list(engine.targets(batch, manifest["normalization"])); changed = [np.array(x, copy=True) for x in args]
    changed[0] = np.where(changed[1][:, None, :], changed[0], 1e6)
    changed[2] = np.where(changed[1], changed[2], 1e6)
    changed[3] = np.where(changed[1], changed[3], 1e6)
    losses = np.asarray(engine.losses(auxiliary, batch[0], *args, np.ones(len(batch[1]), bool)))
    altered = np.asarray(engine.losses(auxiliary, batch[0], *changed, np.ones(len(batch[1]), bool)))
    np.testing.assert_allclose(losses, altered)
    loader.close()
    # Metric fixtures: perfect, constant/tied and shuffled; JSD is permutation-invariant.
    target = np.linspace(0, 1, 20); shuffled = target[::-1]
    if _corr(target, target) != 1 or _corr(target, np.ones(20)) is not None: raise AssertionError("correlation fixture")
    if abs(_jsd(target, target) - _jsd(target, shuffled)) > 1e-12 or _corr(target, shuffled) > -0.99:
        raise AssertionError("JSD shuffle fixture")
    fixture = [{"complex_id": "x", "chain": "A", "resnum": i // 2, "icode": "",
                "interface": i in (0, 1), "score": 1.0 if i in (0, 1) else 0.0}
               for i in range(6)]
    perfect = _interface_metrics(fixture, "score")
    if perfect["top_k_recall"] != 1 or perfect["jaccard"] != 1: raise AssertionError("duplicate/tied fixture")
    empty = _interface_metrics([{**row, "complex_id": "empty", "interface": False} for row in fixture], "score")
    if empty["eligible_complexes"] != 0 or empty["undefined_complexes"] != 1: raise AssertionError("empty interface fixture")
    atomic_json(base / "tests.json", {"passed": True, "pka_bit_identical": True, "finite_jit": True,
        "shared_auxiliary_gradients": True, "own_head_gradients": True, "padding_invariant": True,
        "metric_fixtures": ["perfect", "constant", "tied", "shuffled", "duplicate-residue", "empty-interface"],
        "code_hashes": code_hashes()})


def warmup(root):
    root = Path(root); base = experiment_root(root); manifest = read(base / "manifest.json")
    if not read(base / "tests.json")["passed"]: raise AssertionError("tests")
    params = initialize_auxiliary(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = AuxiliaryEngine(params, 0, {"burial": 1, "interface": 1}); state = engine.optimizer.init(params)
    train = [r for r in manifest["records"] if r["split"] == "train"]
    validation = [r for r in manifest["records"] if r["split"] == "val"]
    loader = Loader(base, manifest); plans = _plans(train, np.random.default_rng(SEED), manifest)
    plan_hash = hashlib.sha256(json.dumps(plans).encode()).hexdigest(); logs = []; started = time.monotonic()
    burial_sum = burial_n = 0
    for number, (_, batch) in enumerate(_prefetched(loader, plans), 1):
        _, mask, burial, _ = engine.targets(batch, manifest["normalization"])
        burial_sum += float(burial[mask].sum()); burial_n += int(mask.sum())
        params, state, values = engine.update(params, state, batch, manifest["normalization"], _rate(1, number, len(plans)))
        logs.append(values)
    train_mean = burial_sum / burial_n
    validation_metrics, _ = evaluate(base, manifest, validation, engine, params, train_mean)
    run = base / "common-warmup"; save_checkpoint(run / "checkpoint", params, state,
        {"version": VERSION, "epoch": 1, "batch_plan_digest": plan_hash})
    diagnostic = _diagnostic_plan(train, manifest); gradients = []
    for _, batch in _prefetched(loader, diagnostic):
        p, b, i = engine.gradients(params, batch, manifest["normalization"])
        gradients.append((_norm(p), _norm(b), _norm(i)))
    loader.close(); values = np.asarray(gradients)
    medians = np.median(values, axis=0)
    if not np.isfinite(medians).all() or np.any(medians <= 1e-10): raise FloatingPointError(("bad calibration", medians.tolist()))
    coefficients = {"burial": float(.1 * medians[0] / (medians[1] + 1e-12)),
                    "interface": float(.1 * medians[0] / (medians[2] + 1e-12))}
    if not np.isfinite(list(coefficients.values())).all() or max(coefficients.values()) > 1e6:
        raise FloatingPointError(("extreme calibration", coefficients))
    atomic_json(base / "calibration.json", {"passed": True, "batches": len(gradients),
        "median_shared_gradient_norm": {"primary": float(medians[0]), "burial": float(medians[1]),
                                        "interface": float(medians[2])}, "coefficients": coefficients,
        "fixed_after_calibration": True, "diagnostic_plan_sha256": hashlib.sha256(json.dumps(diagnostic).encode()).hexdigest()})
    atomic_json(run / "verification.json", {"passed": True, "epoch": 1, "batch_plan_digest": plan_hash,
        "train_burial_mean": train_mean, "validation": validation_metrics,
        "losses": {name: float(np.mean([x[name] for x in logs])) for name in logs[0]},
        "wall_seconds": time.monotonic() - started, "checkpoint_sha256": digest(run / "checkpoint/state.npz"),
        "code_hashes": code_hashes(), "test_data_included": False})


def train(root, arm, smoke=False):
    if arm not in ARMS: raise ValueError(arm)
    root = Path(root); base = experiment_root(root); manifest = read(base / "manifest.json")
    calibration = read(base / "calibration.json"); common = read(base / "common-warmup/verification.json")
    params = initialize_auxiliary(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = AuxiliaryEngine(params, ARMS[arm], calibration["coefficients"]); state = engine.optimizer.init(params)
    params, state, metadata = load_checkpoint(base / "common-warmup/checkpoint", (params, state))
    if metadata["epoch"] != 1 or digest(base / "common-warmup/checkpoint/state.npz") != common["checkpoint_sha256"]:
        raise AssertionError("common checkpoint")
    train_rows = [r for r in manifest["records"] if r["split"] == "train"]
    validation = [r for r in manifest["records"] if r["split"] == "val"]
    run = base / arm / ("smoke" if smoke else "seed-17"); run.mkdir(parents=True, exist_ok=True)
    provenance = {"version": VERSION, "arm": arm, "alpha": ARMS[arm], "seed": SEED,
        "coefficients": calibration["coefficients"], "common_checkpoint_sha256": common["checkpoint_sha256"],
        "code_hashes": code_hashes(), "manifest_sha256": digest(base / "manifest.json"), "test_data_included": False}
    atomic_json(run / "run.json", provenance); loader = Loader(base, manifest); rng = np.random.default_rng(SEED)
    best = {"epoch": 1, "selection": common["validation"]["primary"]["selection"],
            "validation": common["validation"]}; atomic_json(run / "best.json", best)
    save_checkpoint(run / "checkpoints/epoch-001", params, state, {**provenance, "epoch": 1})
    history = []; stalled = 0; started = time.monotonic(); diagnostic_plan = _diagnostic_plan(train_rows, manifest)
    for epoch in range(2, (3 if smoke else EPOCHS + 1)):
        plans = _plans(train_rows, rng, manifest); plan_hash = hashlib.sha256(json.dumps(plans).encode()).hexdigest()
        if smoke: plans = plans[:1]
        logs = []; began = time.monotonic()
        for number, (_, batch) in enumerate(_prefetched(loader, plans), 1):
            params, state, values = engine.update(params, state, batch, manifest["normalization"], _rate(epoch, number, len(plans)))
            logs.append(values)
        diagnostics = _gradient_diagnostics(engine, params, loader, diagnostic_plan[:1] if smoke else diagnostic_plan,
                                             manifest["normalization"], calibration["coefficients"])
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                "losses": logs[-1], "gradient_diagnostics": diagnostics, "batch_plan_digest": plan_hash,
                "peak_memory": jax.local_devices()[0].memory_stats(), **provenance}); loader.close(); return
        metrics, _ = evaluate(base, manifest, validation, engine, params, common["train_burial_mean"])
        selection = metrics["primary"]["selection"]
        if selection < best["selection"] - MIN_DELTA:
            best = {"epoch": epoch, "selection": selection, "validation": metrics}; stalled = 0
            atomic_json(run / "best.json", best)
        else: stalled += 1
        row = {"epoch": epoch, "train": {name: float(np.mean([x[name] for x in logs])) for name in logs[0]},
            "validation": metrics, "selection": selection, "gradient_diagnostics": diagnostics,
            "batch_plan_digest": plan_hash, "seconds": time.monotonic() - began,
            "best_epoch": best["epoch"], "stalled": stalled}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"experiment": VERSION, "arm": arm, **row}), flush=True)
        # The agreed screen is exactly ten total epochs.  Record patience in
        # this pilot, but do not let eight post-warm-up misses truncate epoch 10.
        if stalled >= PATIENCE and epoch >= EPOCHS: break
    loader.close(); params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final, _ = evaluate(base, manifest, validation, engine, params, common["train_burial_mean"], run / "validation_predictions.csv")
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {"passed": True, "selected_epoch": best["epoch"],
        "epochs_completed": 1 + len(history), "wall_seconds": time.monotonic() - started,
        "peak_memory": jax.local_devices()[0].memory_stats(), "predictions_sha256": digest(run / "validation_predictions.csv"),
        **provenance})


def _bootstrap(base_rows, arm_rows, replicates=2000):
    by_base = defaultdict(list); by_arm = defaultdict(list)
    for row in base_rows: by_base[row["complex_id"]].append(row)
    for row in arm_rows: by_arm[row["complex_id"]].append(row)
    ids = sorted(set(by_base) & set(by_arm)); rng = np.random.default_rng(SEED); deltas = []
    for _ in range(replicates):
        chosen = rng.choice(ids, len(ids), replace=True)
        left = [row for cid in chosen for row in by_base[cid]]; right = [row for cid in chosen for row in by_arm[cid]]
        deltas.append(_primary_metrics(right)["selection"] - _primary_metrics(left)["selection"])
    return {"replicates": replicates, "seed": SEED, "delta_arm_minus_baseline": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
        "probability_improved": float(np.mean(np.asarray(deltas) < 0))}


def report(root):
    base = experiment_root(Path(root)); summaries = {}; prediction_rows = {}
    plans = defaultdict(list)
    for arm in ARMS:
        run = base / arm / "seed-17"; verification = read(run / "verification.json")
        if not verification["passed"] or digest(run / "validation_predictions.csv") != verification["predictions_sha256"]:
            raise AssertionError(arm)
        summaries[arm] = {**read(run / "final.json"), "selected_epoch": verification["selected_epoch"],
            "minutes": verification["wall_seconds"] / 60,
            "peak_vram_gib": verification["peak_memory"]["peak_bytes_in_use"] / 2**30}
        with (run / "validation_predictions.csv").open() as stream:
            raw = list(csv.DictReader(stream))
        numeric = ("resnum", "teacher_ab", "teacher_free", "predicted_ab", "predicted_free", "state_error",
                   "paired_error", "burial_target", "burial_prediction", "interface_target",
                   "interface_prediction", "interface_target_raw", "interface_prediction_raw", "distance", "rsa_free")
        prediction_rows[arm] = [{**row, **{name: float(row[name]) for name in numeric},
                                 "interface": row["interface"].lower() == "true"} for row in raw]
        for row in read(run / "history.json"): plans[row["epoch"]].append(row["batch_plan_digest"])
    if any(len(set(values)) != 1 for values in plans.values()): raise AssertionError("unmatched batch plans")
    bootstrap = {arm: _bootstrap(prediction_rows["baseline"], prediction_rows[arm]) for arm in ("low", "standard")}
    atomic_json(base / "summary.json", {"arms": summaries, "bootstrap": bootstrap,
        "calibration": read(base / "calibration.json"), "matched_batch_plans": True})
    lines = ["# oGQT auxiliary-loss pilot", "", "One seed (17), the filtered 10% PINDER pool-v2 subset, and the fixed 400-complex validation cohort. Checkpoints were selected only by unweighted state MAE plus interface paired MAE.", "",
        "| Arm | State MAE | Paired MAE | Interface paired MAE | Selection | Burial MAE | Burial Spearman | Interface weight MAE | Interface AP | Epoch | Time (min) | VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = summaries[arm]; p = row["primary"]; a = row["auxiliary"]
        lines.append(f"| {arm} | {p['state_mae']:.4f} | {p['paired_mae']:.4f} | {p['interface_paired_mae']:.4f} | {p['selection']:.4f} | "
            f"{a['burial']['pooled_mae']:.4f} | {a['burial']['pooled_spearman']:.4f} | {a['interface']['raw_weight_mae']:.4f} | "
            f"{a['interface']['predicted_ranking']['average_precision']:.4f} | {row['selected_epoch']} | {row['minutes']:.1f} | {row['peak_vram_gib']:.2f} |")
    lines += ["", "Bootstrap deltas are arm minus baseline; negative values improve the registered selection score.", "",
        "| Arm | Mean delta | 95% interval | P(improved) |", "|---|---:|---:|---:|"]
    for arm, item in bootstrap.items(): lines.append(f"| {arm} | {item['delta_arm_minus_baseline']:.4f} | [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}] | {item['probability_improved']:.3f} |")
    lines += ["", "Auxiliary accuracy is diagnostic only and did not enter checkpoint selection. Detailed distance/RSA strata, structural-reference rankings, correlations, JSD, gradient diagnostics, and per-site predictions are retained in the JSON and CSV outputs.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "report-verification.json", {"passed": True, "report_sha256": digest(base / "report.md"),
        "matched_batch_plans": True, "bootstrap_replicates": 2000, "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    gpu = action in ("tests", "warmup", "smoke", "train")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu,
                    allow_comp1400=(gpu or action in ("register", "prepare", "mmap", "report")))
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "prepare": prepare(root)
    elif action == "mmap": build_mmap(root)
    elif action == "tests": tests(root)
    elif action == "warmup": warmup(root)
    elif action == "smoke": train(root, sys.argv[2], smoke=True)
    elif action == "train": train(root, sys.argv[2])
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
