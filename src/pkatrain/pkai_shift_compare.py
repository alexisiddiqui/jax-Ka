"""Matched pKAI scratch/pretrained and shift-weighted/unweighted experiment."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from pkabench.frozen_score import aggregate, measures, write_csv
from pkabench.runtime import atomic_json, digest, require_compute
from .pkai_scratch import architecture_gate, model_class, native


BINS = (0.5, 1.0, 2.0)
ARMS = (
    "scratch-unweighted", "scratch-weighted",
    "pretrained-unweighted", "pretrained-weighted",
)


def code_hashes():
    here = Path(__file__)
    return {
        str(here): digest(here),
        str(here.with_name("pkai_scratch.py")): digest(here.with_name("pkai_scratch.py")),
    }


def shift_bin(value):
    return int(np.searchsorted(np.asarray(BINS), abs(float(value)), side="right"))


def inverse_frequency_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (4,) or np.any(counts <= 0):
        raise ValueError(f"All four pKAI shift bins require positive train counts: {counts}")
    return counts.sum() / (len(counts) * counts)


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy()
        h.update(name.encode())
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def register(root):
    source = root / "pretraining/pkpdb-5k-comparison-v1"
    packed = source / "pkai-packed"
    verification = json.loads((packed / "verification.json").read_text())
    assert verification["passed"]
    assert digest(packed / "features.npy") == verification["features_sha256"]
    assert digest(packed / "rows.json") == verification["rows_sha256"]
    rows = json.loads((packed / "rows.json").read_text())
    counts = np.zeros(4, dtype=np.int64)
    group_constants = {}
    for row in rows:
        group_constants.setdefault(row["group"], row["model_pka"])
        assert group_constants[row["group"]] == row["model_pka"]
        if row["split"] == "train" and row["train_mask"]:
            counts[shift_bin(row["pka"] - row["model_pka"])] += 1
    weights = inverse_frequency_weights(counts)
    destination = root / "pretraining/pkai-5k-shift-weight-v1"
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {
        "source": str(source), "packed_verification_sha256": digest(packed / "verification.json"),
        "target": "signed pKa - native pKAI PK_MOD", "pk_mod": group_constants,
        "train_filter": "split=train and train_mask=true", "validation_filter": "split=val",
        "train_sites": int(counts.sum()), "shift_bin_edges": list(BINS),
        "shift_bin_counts": counts.tolist(), "weighted_arm_weights": weights.tolist(),
        "arms": list(ARMS), "seed": 17, "batch_size": 256,
        "optimizer": "Adam", "learning_rate": 1e-6, "weight_decay": 1e-4,
        "dropout": [0.5, 0.125, 0.03125], "max_epochs": 200,
        "early_stopping": "minimum unweighted validation shift MSE; min_delta 0.001; patience 5",
        "pretrained_epoch_zero": "released pKAI_model.pt is eligible for selection before fine-tuning",
        "comparison": "within each initialization, only train loss weighting differs",
        "test_data_included": False,
    }
    path = destination / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        atomic_json(path, protocol)
    return destination


def group_macro(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["complex_id"]].append(row)
    complexes = []
    for cid, subset in grouped.items():
        target = np.asarray([row["teacher_shift"] for row in subset])
        predicted = np.asarray([row["predicted_shift"] for row in subset])
        complexes.append({
            "complex_id": cid, "component_id": subset[0]["component_id"], "n": len(subset),
            **measures(target, predicted),
        })
    return aggregate(complexes, replicates=2000)[0]


def summarize_bins(rows):
    output = []
    for index, name in enumerate(("0-0.5", "0.5-1", "1-2", "2-inf")):
        subset = [row for row in rows if row["shift_bin"] == index]
        target = np.asarray([row["teacher_shift"] for row in subset])
        predicted = np.asarray([row["predicted_shift"] for row in subset])
        slope = float(np.cov(target, predicted, ddof=0)[0, 1] / np.var(target)) if np.var(target) else None
        macro = group_macro(subset)
        output.append({
            "bin": name, "sites": len(subset),
            "site_mae": float(np.mean(abs(predicted - target))),
            "shift_slope": slope, "group_macro_mae": macro["mae"],
            "group_macro_mae_ci95": macro["mae_ci95"],
            "complexes": macro["complexes"], "groups": macro["groups"],
        })
    return output


def validation_predictions(torch, model, rows, features, valid, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for ids in np.array_split(valid, range(batch_size, len(valid), batch_size)):
            batch = torch.tensor(np.asarray(features[ids]), device="cuda")
            predictions.append(model(batch).cpu().numpy())
    shifts = np.concatenate(predictions)
    output = []
    for index, predicted_shift in zip(valid, shifts):
        row = rows[int(index)]
        teacher_shift = float(row["pka"] - row["model_pka"])
        output.append({
            **row, "teacher_pka": row["pka"],
            "predicted_pka": float(row["model_pka"] + predicted_shift),
            "teacher_shift": teacher_shift, "predicted_shift": float(predicted_shift),
            "shift_bin": shift_bin(teacher_shift),
        })
    return output


def train(root, experiment, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    torch, package = native()
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads((experiment / "protocol.json").read_text())
    assert json.loads((experiment / "tests.json").read_text())["code_hashes"] == code_hashes()
    gate = architecture_gate()
    seed = protocol["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_class(torch)()
    initialization, weighting = arm.split("-")
    if initialization == "pretrained":
        reference = torch.jit.load(str(package / "models/pKAI_model.pt"), map_location="cpu").eval()
        model.load_state_dict(dict(reference.named_parameters()))
    model = model.cuda()
    initial_digest = parameter_digest(model)
    destination = experiment / arm
    destination.mkdir(exist_ok=True)
    source = root / "pretraining/pkpdb-5k-comparison-v1/pkai-packed"
    packed_verification = json.loads((source / "verification.json").read_text())
    assert digest(source / "features.npy") == packed_verification["features_sha256"]
    assert digest(source / "rows.json") == packed_verification["rows_sha256"]
    rows = json.loads((source / "rows.json").read_text())
    features = np.load(source / "features.npy", mmap_mode="r")
    train_ids = np.asarray([
        index for index, row in enumerate(rows) if row["split"] == "train" and row["train_mask"]
    ])
    valid = np.asarray([index for index, row in enumerate(rows) if row["split"] == "val"])
    targets = torch.tensor(
        [row["pka"] - row["model_pka"] for row in rows], dtype=torch.float32, device="cuda"
    )
    bin_weights = np.ones(4) if weighting == "unweighted" else np.asarray(protocol["weighted_arm_weights"])
    site_weights = torch.tensor(
        [bin_weights[shift_bin(row["pka"] - row["model_pka"])] for row in rows],
        dtype=torch.float32, device="cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    config = {
        "arm": arm, "initialization": initialization, "weighting": weighting,
        "seed": seed, "batch_size": protocol["batch_size"], "precision": "float32",
        "objective": "weighted MSE on explicit signed pKAI PK_MOD shift",
        "selection": protocol["early_stopping"], "max_epochs": protocol["max_epochs"],
        "initial_parameter_digest": initial_digest,
    }
    provenance = {
        "config": config, "gate": gate, "protocol_sha256": digest(experiment / "protocol.json"),
        "packed_verification_sha256": digest(source / "verification.json"),
        "code_hashes": code_hashes(), "train_sites": len(train_ids),
        "validation_sites": len(valid), "test_data_included": False,
    }
    atomic_json(destination / "manifest.json", provenance)

    initial_rows = validation_predictions(torch, model, rows, features, valid, protocol["batch_size"])
    initial_mse = float(np.mean([
        (row["predicted_shift"] - row["teacher_shift"]) ** 2 for row in initial_rows
    ]))
    best = initial_mse
    anchor = initial_mse
    best_epoch = 0
    stall = 0
    torch.save(model.state_dict(), destination / "best.pending.pt")
    os.replace(destination / "best.pending.pt", destination / "best.pt")
    atomic_json(destination / "epoch-zero.json", {
        "validation_shift_mse": initial_mse, "initial_parameter_digest": initial_digest,
        "released_checkpoint": initialization == "pretrained",
    })
    history = []
    for epoch in range(protocol["max_epochs"]):
        if stall >= 5:
            break
        model.train()
        started = time.monotonic()
        losses, counts = [], []
        for ids in np.array_split(np.random.permutation(train_ids), range(protocol["batch_size"], len(train_ids), protocol["batch_size"])):
            tensor_ids = torch.tensor(ids, device="cuda")
            x = torch.tensor(np.asarray(features[ids]), device="cuda")
            y = targets[tensor_ids]
            weight = site_weights[tensor_ids]
            optimizer.zero_grad(set_to_none=True)
            error = model(x) - y
            loss = (weight * error.square()).sum() / weight.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pKAI shift loss")
            loss.backward()
            if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("Nonfinite pKAI shift gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            counts.append(len(ids))
        current_rows = validation_predictions(torch, model, rows, features, valid, protocol["batch_size"])
        mse = float(np.mean([
            (row["predicted_shift"] - row["teacher_shift"]) ** 2 for row in current_rows
        ]))
        if mse < best:
            best = mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), destination / "best.pending.pt")
            os.replace(destination / "best.pending.pt", destination / "best.pt")
        if mse < anchor - 0.001:
            anchor = mse
            stall = 0
        else:
            stall += 1
        record = {
            "epoch": epoch + 1, "train_shift_mse": float(np.average(losses, weights=counts)),
            "validation_shift_mse": mse, "seconds": time.monotonic() - started,
            "stall": stall, "best_epoch": best_epoch,
        }
        history.append(record)
        atomic_json(destination / "history.json", history)
        atomic_json(destination / "progress.json", record)
        print(json.dumps({"arm": arm, **record}), flush=True)

    model.load_state_dict(torch.load(destination / "best.pt"))
    final_rows = validation_predictions(torch, model, rows, features, valid, protocol["batch_size"])
    write_csv(destination / "validation_predictions.csv", final_rows)
    overall = group_macro(final_rows)
    bins = summarize_bins(final_rows)
    atomic_json(destination / "shift_bins.json", bins)
    atomic_json(destination / "final.json", {
        "metrics": overall, "bins": bins, "initial_validation_shift_mse": initial_mse,
        "best_epoch": best_epoch, "epochs": len(history), "validation_selected": True,
    })
    atomic_json(destination / "verification.json", {
        "passed": True, "finite_gradients": True, "complete": True,
        "initial_parameter_digest": initial_digest,
        "predictions_sha256": digest(destination / "validation_predictions.csv"),
        "manifest_sha256": digest(destination / "manifest.json"),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "test_data_included": False,
    })


def report(experiment):
    results = {arm: json.loads((experiment / arm / "final.json").read_text()) for arm in ARMS}
    verifications = {arm: json.loads((experiment / arm / "verification.json").read_text()) for arm in ARMS}
    assert all(value["passed"] and not value["test_data_included"] for value in verifications.values())
    assert verifications["scratch-unweighted"]["initial_parameter_digest"] == verifications["scratch-weighted"]["initial_parameter_digest"]
    assert verifications["pretrained-unweighted"]["initial_parameter_digest"] == verifications["pretrained-weighted"]["initial_parameter_digest"]
    protocol = json.loads((experiment / "protocol.json").read_text())
    lines = [
        "# pKAI explicit-shift weighting comparison", "",
        "All arms predict signed shift relative to pKAI's native pKPDB `PK_MOD` constants. Within each initialization, the arms have identical data and hyperparameters and differ only in training-loss weights. The released pretrained checkpoint is eligible as epoch 0.", "",
        "| Initialization | Weighting | Best epoch | Validation group-macro MAE | 95% CI |", "|---|---|---:|---:|---|",
    ]
    for arm in ARMS:
        result = results[arm]
        initialization, weighting = arm.split("-")
        metric = result["metrics"]
        lines.append(f"| {initialization} | {weighting} | {result['best_epoch']} | {metric['mae']:.4f} | {metric['mae_ci95']} |")
    lines += ["", "| Shift bin | Train sites | Unweighted weight | Weighted weight |", "|---|---:|---:|---:|"]
    for name, count, weight in zip(("0-0.5", "0.5-1", "1-2", "2-inf"), protocol["shift_bin_counts"], protocol["weighted_arm_weights"]):
        lines.append(f"| {name} | {count:,} | 1.0000 | {weight:.4f} |")
    for initialization in ("scratch", "pretrained"):
        lines += ["", f"## {initialization.capitalize()} validation by teacher-shift magnitude", "",
                  "| Shift bin | Sites | Unweighted site MAE | Weighted site MAE | Unweighted group-macro MAE (95% CI) | Weighted group-macro MAE (95% CI) |", "|---|---:|---:|---:|---|---|"]
        left = {row["bin"]: row for row in results[f"{initialization}-unweighted"]["bins"]}
        right = {row["bin"]: row for row in results[f"{initialization}-weighted"]["bins"]}
        for name in ("0-0.5", "0.5-1", "1-2", "2-inf"):
            a, b = left[name], right[name]
            lines.append(
                f"| {name} | {a['sites']:,} | {a['site_mae']:.4f} | {b['site_mae']:.4f} | "
                f"{a['group_macro_mae']:.4f} {a['group_macro_mae_ci95']} | {b['group_macro_mae']:.4f} {b['group_macro_mae_ci95']} |"
            )
    lines += ["", "Validation determines early stopping under pKAI's native recipe. No test data were read."]
    (experiment / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(experiment / "verification.json", {
        "passed": True, "matched_initializations": True,
        "report_sha256": digest(experiment / "report.md"), "test_data_included": False,
    })


if __name__ == "__main__":
    import sys
    runtime = Path(os.environ["PKABENCH_RUNTIME"])
    action = sys.argv[1]
    require_compute(threads=8 if action == "train" else 1, gpu_benchmark=action == "train", allow_comp1400=True)
    experiment = runtime / "pretraining/pkai-5k-shift-weight-v1"
    if action == "register":
        register(runtime)
    elif action == "tests":
        register(runtime)
        counts = np.asarray(json.loads((experiment / "protocol.json").read_text())["shift_bin_counts"])
        weights = inverse_frequency_weights(counts)
        np.testing.assert_allclose(counts * weights, np.repeat(counts.sum() / 4, 4))
        atomic_json(experiment / "tests.json", {"passed": True, "code_hashes": code_hashes()})
    elif action == "train":
        train(runtime, experiment, sys.argv[2])
    elif action == "report":
        report(experiment)
    else:
        raise ValueError(action)
