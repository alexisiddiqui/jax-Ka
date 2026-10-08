"""Matched full-atom versus strictly backbone-only pKAI training."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.frozen_score import aggregate, measures, write_csv
from pkabench.runtime import atomic_json, digest, require_compute
from .pkai_scratch import architecture_gate, model_class, native


BACKBONE_ATOMS = frozenset(("N", "O"))
SEEDS = (17, 29, 43)
MODES = ("full", "backbone")
INITIALIZATIONS = ("scratch", "pretrained")
ARMS = tuple(f"{mode}-{initialization}" for mode in MODES for initialization in INITIALIZATIONS)


def experiment_root(root):
    return root / "pretraining/pkai-backbone-ablation-v1"


def code_hashes():
    here = Path(__file__)
    parent = here.with_name("pkai_scratch.py")
    return {str(here): digest(here), str(parent): digest(parent)}


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy()
        h.update(name.encode()); h.update(str(array.shape).encode()); h.update(str(array.dtype).encode()); h.update(array.tobytes())
    return h.hexdigest()


def _install_pkai(runtime):
    package = Path(runtime) / "envs/pkai/lib/python3.11/site-packages"
    sys.path.append(str(package)); sys.path.insert(0, str(package / "pkai"))


def backbone_features(task):
    """Use only N/CA/C/O context and the query residue C-alpha as origin."""
    record, runtime = task
    _install_pkai(runtime)
    from protein import Protein
    from residue import ATOM_OHE, RES_OHE

    path = Path(record["path"])
    if digest(path / "input.pdb") != record["pdb_sha256"] or digest(path / "rows.json") != record["rows_sha256"]:
        raise AssertionError(record["complex_id"])
    request = json.loads((path / "request.json").read_text())
    rows = json.loads((path / "rows.json").read_text())
    protein = Protein(path / "input.pdb")
    residues = list(protein.iter_residues(titrable_only=True))
    lookup = {(*request["mapping"][str(residue.resnumb)], residue.resname): residue for residue in residues}
    if len(lookup) != len(residues):
        raise AssertionError((record["complex_id"], "ambiguous residue map"))
    # Native pKAI deliberately discards carbon atoms and has no C/CA input
    # class. Preserve its 4008-wide schema by using the representable backbone
    # N/O atoms. Recover C-alpha separately only as the query origin.
    ca_by_residue = {}
    with open(path / "input.pdb") as handle:
        for line in handle:
            if not line.startswith("ATOM ") or line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
                continue
            internal = (line[21], int(line[22:26]))
            coordinate = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
            if internal in ca_by_residue:
                raise AssertionError((record["complex_id"], internal, "duplicate CA"))
            ca_by_residue[internal] = coordinate
    atoms = [atom for atom in protein.iter_atoms() if atom.aname in BACKBONE_ATOMS]
    coordinates = np.asarray([atom.coords for atom in atoms], dtype=np.float64)
    result = np.zeros((len(rows), 4008), np.float32)
    for index, row in enumerate(rows):
        key = (row["chain"], row["resnum"], row["icode"], row["group"])
        residue = lookup[key]
        internal = (residue.chain, residue.resnumb)
        if internal not in ca_by_residue:
            raise AssertionError((record["complex_id"], key, "missing CA"))
        distance = np.sqrt(((coordinates - ca_by_residue[internal]) ** 2).sum(-1))
        ids = np.flatnonzero(np.asarray([atom.residue is not residue for atom in atoms]) & (distance < 15.0))
        if np.any(distance[ids] == 0):
            raise ValueError((record["complex_id"], key, "coincident backbone atom"))
        residue.env_anames = [atoms[j].aname for j in ids]
        residue.env_resnames = [atoms[j].residue.resname for j in ids]
        residue.encode_atoms()
        ordered = sorted(zip(distance[ids], residue.env_oheclasses), key=lambda value: (value[0], value[1]))[:250]
        for slot, (value, cls) in enumerate(ordered):
            result[index, slot * 16 + ATOM_OHE.index(cls)] = 1 / float(value) ** 2
        result[index, 4000 + RES_OHE.index(residue.resname)] = 1.0
    return record["complex_id"], result


def test_features(root):
    source = root / "pretraining/pkpdb-5k-comparison-v1"
    info = json.loads((source / "pkai-features.json").read_text())
    record = info["records"][0]
    cid, values = backbone_features((record, str(root)))
    if cid != record["complex_id"] or values.shape != (record["sites"], 4008):
        raise AssertionError((cid, values.shape, record))
    if not np.isfinite(values).all() or not np.all(values[:, 4000:].sum(axis=1) == 1):
        raise AssertionError("Invalid backbone feature tensor")
    if not np.any(values[:, :4000] > 0):
        raise AssertionError("Backbone environments are empty")
    out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "tests.json", {
        "passed": True, "complex_id": cid, "sites": len(values),
        "finite": True, "one_residue_identity_per_site": True,
        "nonempty_backbone_environments": True, "code_hashes": code_hashes(),
    })


def prepare(root):
    out = experiment_root(root)
    out.mkdir(parents=True, exist_ok=True)
    tests = json.loads((out / "tests.json").read_text())
    if not tests["passed"] or tests["code_hashes"] != code_hashes():
        raise AssertionError("Feature gate/code provenance mismatch")
    source = root / "pretraining/pkpdb-5k-comparison-v1"
    info = json.loads((source / "pkai-features.json").read_text())
    packed = source / "pkai-packed"
    verification = json.loads((packed / "verification.json").read_text())
    if not info["passed"] or not verification["passed"]:
        raise AssertionError("Parent pKAI features did not pass")
    if digest(packed / "features.npy") != verification["features_sha256"] or digest(packed / "rows.json") != verification["rows_sha256"]:
        raise AssertionError("Parent packed feature hashes changed")
    total = sum(int(record["sites"]) for record in info["records"])
    features = np.lib.format.open_memmap(out / "backbone-features.npy", mode="w+", dtype=np.float32, shape=(total, 4008))
    # Two allocated CPU cores per worker leaves room for parser/library threads.
    workers = min(16, max(1, len(os.sched_getaffinity(0)) // 2))
    offset = 0
    receipts = []
    tasks = [(record, str(root)) for record in info["records"]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for number, (record, output) in enumerate(zip(info["records"], pool.map(backbone_features, tasks, chunksize=1)), 1):
            cid, array = output
            if cid != record["complex_id"] or array.shape != (record["sites"], 4008):
                raise AssertionError((cid, array.shape, record))
            features[offset:offset + len(array)] = array
            receipts.append({"complex_id": cid, "start": offset, "stop": offset + len(array), "sites": len(array)})
            offset += len(array)
            if number % 100 == 0:
                features.flush()
                atomic_json(out / "preparation-progress.json", {"structures": number, "total": len(tasks), "sites": offset})
                print(json.dumps({"structures": number, "total": len(tasks), "sites": offset}), flush=True)
    features.flush(); del features
    if offset != total:
        raise AssertionError((offset, total))
    # Identity columns and row order must match the existing full-atom tensor.
    full = np.load(packed / "features.npy", mmap_mode="r")
    backbone = np.load(out / "backbone-features.npy", mmap_mode="r")
    np.testing.assert_array_equal(backbone[:, 4000:], full[:, 4000:])
    atomic_json(out / "feature-receipts.json", receipts)
    protocol = {
        "arms": list(ARMS), "seeds": list(SEEDS), "batch_size": 64,
        "architecture": "native pKAI 4008-800-400-200-1 (3,608,001 parameters)",
        "backbone_definition": "query origin is residue CA; environment is backbone N/O atoms of other residues within 15 A; at most 250 ordered atoms; residue identity retained",
        "native_schema_constraint": "pKAI discards carbon atoms and has no C/CA input class, so CA is used only as query origin and backbone C/CA atoms cannot be represented without changing the architecture",
        "full_definition": "unaltered native pKAI functional-atom-centred full-heavy-atom features",
        "objective": "unweighted MSE on signed pKa minus native pKAI PK_MOD",
        "optimizer": "Adam 1e-6, weight decay 1e-4", "dropout": [0.5, 0.125, 0.03125],
        "early_stopping": "minimum clean validation shift MSE; min_delta 0.001; patience 5; cap 200 epochs",
        "pretrained_epoch_zero": "released pKAI checkpoint is selectable for pretrained arms",
        "test_data_included": False,
    }
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "preparation.json", {
        "passed": True, "structures": len(tasks), "sites": total,
        "backbone_features_sha256": digest(out / "backbone-features.npy"),
        "full_features_sha256": verification["features_sha256"],
        "rows_sha256": digest(packed / "rows.json"), "parent_verification_sha256": digest(packed / "verification.json"),
        "identity_columns_match_full": True, "protocol_sha256": digest(out / "protocol.json"), "code_hashes": code_hashes(),
    })


def group_macro(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["complex_id"]].append(row)
    scores = []
    for cid, subset in grouped.items():
        target = np.asarray([row["teacher_shift"] for row in subset])
        predicted = np.asarray([row["predicted_shift"] for row in subset])
        scores.append({"complex_id": cid, "component_id": subset[0]["component_id"], "n": len(subset), **measures(target, predicted)})
    return aggregate(scores, replicates=2000)[0]


def predict(torch, model, rows, features, ids, batch_size=64):
    model.eval(); values = []
    with torch.no_grad():
        for batch in np.array_split(ids, range(batch_size, len(ids), batch_size)):
            values.append(model(torch.tensor(np.asarray(features[batch]), device="cuda")).cpu().numpy())
    output = []
    for index, shift in zip(ids, np.concatenate(values)):
        row = rows[int(index)]
        output.append({**row, "teacher_pka": row["pka"], "predicted_pka": float(row["model_pka"] + shift),
                       "teacher_shift": float(row["pka"] - row["model_pka"]), "predicted_shift": float(shift)})
    return output


def train(root, arm, seed, *, smoke=False):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError((arm, seed))
    out = experiment_root(root)
    prep = json.loads((out / "preparation.json").read_text())
    if not prep["passed"] or prep["code_hashes"] != code_hashes():
        raise AssertionError("Preparation/code provenance mismatch")
    protocol = json.loads((out / "protocol.json").read_text())
    torch, package = native()
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed); np.random.seed(seed)
    gate = architecture_gate(); torch.manual_seed(seed)
    mode, initialization = arm.split("-")
    model = model_class(torch)()
    if initialization == "pretrained":
        reference = torch.jit.load(str(package / "models/pKAI_model.pt"), map_location="cpu").eval()
        model.load_state_dict(dict(reference.named_parameters()))
    model = model.cuda()
    initial_digest = parameter_digest(model)
    source = root / "pretraining/pkpdb-5k-comparison-v1/pkai-packed"
    rows = json.loads((source / "rows.json").read_text())
    feature_path = source / "features.npy" if mode == "full" else out / "backbone-features.npy"
    feature_sha256 = prep["full_features_sha256"] if mode == "full" else prep["backbone_features_sha256"]
    features = np.load(feature_path, mmap_mode="r")
    train_ids = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train" and row["train_mask"]])
    valid = np.asarray([i for i, row in enumerate(rows) if row["split"] == "val"])
    targets = torch.tensor([row["pka"] - row["model_pka"] for row in rows], dtype=torch.float32, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
    run = out / arm / (f"smoke-seed-{seed}" if smoke else f"seed-{seed}")
    run.mkdir(parents=True, exist_ok=True)
    provenance = {
        "arm": arm, "mode": mode, "initialization": initialization, "seed": seed, "batch_size": 64,
        "feature_sha256": feature_sha256, "rows_sha256": prep["rows_sha256"],
        "protocol_sha256": digest(out / "protocol.json"), "code_hashes": code_hashes(), "gate": gate,
        "initial_parameter_digest": initial_digest, "train_sites": len(train_ids), "validation_sites": len(valid),
        "test_data_included": False,
    }
    atomic_json(run / "manifest.json", provenance)
    torch.cuda.reset_peak_memory_stats()
    run_started = time.monotonic()
    initial_rows = predict(torch, model, rows, features, valid)
    initial_mse = float(np.mean([(row["predicted_shift"] - row["teacher_shift"]) ** 2 for row in initial_rows]))
    best = anchor = initial_mse; best_epoch = 0; stall = 0
    torch.save(model.state_dict(), run / "best.pt")
    history = []
    rng = np.random.default_rng(seed)
    max_epochs = 1 if smoke else int(protocol.get("max_epochs", 200))
    for epoch in range(1, max_epochs + 1):
        if stall >= 5:
            break
        started = time.monotonic(); model.train(); losses, counts = [], []
        for ids in np.array_split(rng.permutation(train_ids), range(64, len(train_ids), 64)):
            tensor_ids = torch.tensor(ids, device="cuda")
            xb = torch.tensor(np.asarray(features[ids]), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = (model(xb) - targets[tensor_ids]).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite pKAI loss")
            loss.backward()
            if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("Nonfinite pKAI gradient")
            optimizer.step(); losses.append(float(loss.detach())); counts.append(len(ids))
        current = predict(torch, model, rows, features, valid)
        mse = float(np.mean([(row["predicted_shift"] - row["teacher_shift"]) ** 2 for row in current]))
        if mse < best:
            best, best_epoch = mse, epoch
            torch.save(model.state_dict(), run / "best.pending.pt"); os.replace(run / "best.pending.pt", run / "best.pt")
        if mse < anchor - 0.001: anchor, stall = mse, 0
        else: stall += 1
        record = {"epoch": epoch, "train_shift_mse": float(np.average(losses, weights=counts)),
                  "validation_shift_mse": mse, "seconds": time.monotonic() - started,
                  "stall": stall, "best_epoch": best_epoch}
        history.append(record); atomic_json(run / "history.json", history); atomic_json(run / "progress.json", record)
        print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
    if smoke:
        atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **provenance})
        return
    model.load_state_dict(torch.load(run / "best.pt")); final_rows = predict(torch, model, rows, features, valid)
    write_csv(run / "validation_predictions.csv", final_rows)
    metrics = group_macro(final_rows)
    atomic_json(run / "final.json", {"metrics": metrics, "initial_validation_shift_mse": initial_mse,
                                      "best_epoch": best_epoch, "epochs": len(history)})
    atomic_json(run / "verification.json", {
        "passed": True, "complete": True, "finite_gradients": True, "wall_seconds": time.monotonic() - run_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "predictions_sha256": digest(run / "validation_predictions.csv"), **provenance,
    })


def report(root):
    out = experiment_root(root); rows = []
    for arm in ARMS:
        for seed in SEEDS:
            run = out / arm / f"seed-{seed}"
            verify = json.loads((run / "verification.json").read_text())
            final = json.loads((run / "final.json").read_text())
            metric = final["metrics"]
            rows.append({"arm": arm, "seed": seed, "mae": metric["mae"], "mae_ci95": metric["mae_ci95"],
                         "best_epoch": final["best_epoch"], "epochs": final["epochs"], "wall_seconds": verify["wall_seconds"],
                         "peak_allocated_bytes": verify["peak_allocated_bytes"], "peak_reserved_bytes": verify["peak_reserved_bytes"]})
    summary = []
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]; values = np.asarray([row["mae"] for row in subset])
        summary.append({"arm": arm, "mean_mae": float(values.mean()), "sd_mae": float(values.std(ddof=1)),
                        "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in subset])),
                        "max_peak_allocated_gib": float(max(row["peak_allocated_bytes"] for row in subset) / 2**30),
                        "max_peak_reserved_gib": float(max(row["peak_reserved_bytes"] for row in subset) / 2**30),
                        "best_epochs": [row["best_epoch"] for row in subset]})
    comparisons = {}
    for initialization in INITIALIZATIONS:
        full = {row["seed"]: row["mae"] for row in rows if row["arm"] == f"full-{initialization}"}
        backbone = {row["seed"]: row["mae"] for row in rows if row["arm"] == f"backbone-{initialization}"}
        comparisons[initialization] = {"mean_paired_delta_backbone_minus_full": float(np.mean([backbone[s] - full[s] for s in SEEDS]))}
    atomic_json(out / "results.json", rows); atomic_json(out / "summary.json", {"arms": summary, "comparisons": comparisons})
    lines = ["# Backbone-only pKAI ablation", "",
             "Matched full-atom and strict-backbone inputs, scratch and released-checkpoint initialization, three seeds. Full float32; batch 64; native dropout and optimizer recipe; no test data.", "",
             "| Input | Initialization | Validation MAE | Seed SD | Best epochs | Mean time/run (min) | Peak allocated/reserved VRAM (GiB) |",
             "|---|---|---:|---:|---|---:|---:|"]
    for row in summary:
        mode, initialization = row["arm"].split("-")
        lines.append(f"| {mode} | {initialization} | {row['mean_mae']:.4f} | {row['sd_mae']:.4f} | {', '.join(map(str,row['best_epochs']))} | {row['mean_wall_seconds']/60:.1f} | {row['max_peak_allocated_gib']:.2f} / {row['max_peak_reserved_gib']:.2f} |")
    lines += ["", f"Backbone minus full MAE, scratch: **{comparisons['scratch']['mean_paired_delta_backbone_minus_full']:+.4f}**.",
              f"Backbone minus full MAE, pretrained: **{comparisons['pretrained']['mean_paired_delta_backbone_minus_full']:+.4f}**.", ""]
    (out / "report.md").write_text("\n".join(lines))
    atomic_json(out / "verification.json", {"passed": True, "runs": len(rows), "report_sha256": digest(out / "report.md"), "test_data_included": False})


def main():
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    gpu = action in ("smoke", "train")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu, allow_comp1400=gpu)
    if action == "test": test_features(root)
    elif action == "prepare": prepare(root)
    elif action == "smoke": train(root, "backbone-scratch", 17, smoke=True)
    elif action == "train": train(root, sys.argv[2], int(sys.argv[3]))
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__":
    main()
