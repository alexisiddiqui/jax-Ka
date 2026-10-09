"""Five-epoch backbone pKAI batch scaling for scratch and released initializations."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from .pkai_backbone_ablation import (
    architecture_gate, group_macro, model_class, native, parameter_digest, predict,
)


SEED = 17
EPOCHS = 5
BASE_BATCH = 64
BASE_LR = 1e-6
SCALES = (0.5, 1.0, 2.0, 4.0)
INITIALIZATIONS = ("scratch", "pretrained")
NOISE_BATCH = 32
NOISE_BATCHES = 32


def read(path): return json.loads(Path(path).read_text())
def root_path(root): return Path(root) / "pretraining/pkai-backbone-batch-scale-v1"
def source_path(root): return Path(root) / "pretraining/pkai-backbone-ablation-v1"
def arm_name(initialization, scale): return f"{initialization}-b{str(scale).replace('.', 'p')}"


def register(root):
    root = Path(root); out = root_path(root); out.mkdir(parents=True, exist_ok=True)
    source = source_path(root); prep = read(source / "preparation.json")
    if not prep["passed"]: raise AssertionError("backbone feature preparation")
    protocol = {"version": "pkai-backbone-batch-scale-v1", "epochs": EPOCHS,
        "initializations": list(INITIALIZATIONS), "scales": list(SCALES),
        "batches": [int(BASE_BATCH * x) for x in SCALES], "base_learning_rate": BASE_LR,
        "learning_rate_rule": "eta(scale) = 1e-6 * sqrt(scale); Adam beta/epsilon unchanged",
        "optimizer": "Adam, weight_decay=1e-4", "dropout": [0.5, 0.125, 0.03125],
        "objective": "unweighted MSE on signed pKa minus native pKAI PK_MOD",
        "input": "strict backbone pKAI tensor: N/O environment around query CA; residue identity retained",
        "comparison": "fixed epoch 5; identical features, site order, seed and initialization within each family",
        "noise": {"microbatch": NOISE_BATCH, "batches": NOISE_BATCHES,
            "estimator": "B_noise = microbatch * tr(Cov(batch gradients)) / corrected |G|^2; native dropout active"},
        "source_preparation_sha256": digest(source / "preparation.json"), "test_data_included": False}
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "registration.json", {"passed": True, "protocol_sha256": digest(out / "protocol.json"),
        "test_data_included": False})


def setup(root, initialization):
    if initialization not in INITIALIZATIONS: raise ValueError(initialization)
    torch, package = native(); torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED); np.random.seed(SEED); gate = architecture_gate(); torch.manual_seed(SEED)
    model = model_class(torch)()
    if initialization == "pretrained":
        reference = torch.jit.load(str(package / "models/pKAI_model.pt"), map_location="cpu").eval()
        model.load_state_dict(dict(reference.named_parameters()))
    initial_digest = parameter_digest(model); model = model.cuda()
    source = source_path(root); prep = read(source / "preparation.json")
    packed = Path(root) / "pretraining/pkpdb-5k-comparison-v1/pkai-packed"
    rows = read(packed / "rows.json")
    features = np.load(source / "backbone-features.npy", mmap_mode="r")
    train_ids = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train" and row["train_mask"]])
    valid_ids = np.asarray([i for i, row in enumerate(rows) if row["split"] == "val"])
    targets = torch.tensor([row["pka"] - row["model_pka"] for row in rows], dtype=torch.float32, device="cuda")
    provenance = {"initialization": initialization, "seed": SEED, "initial_parameter_digest": initial_digest,
        "gate": gate, "feature_sha256": prep["backbone_features_sha256"], "rows_sha256": prep["rows_sha256"],
        "train_sites": len(train_ids), "validation_sites": len(valid_ids), "test_data_included": False}
    return torch, model, rows, features, train_ids, valid_ids, targets, provenance


def train(root, initialization, scale, *, smoke=False):
    root = Path(root); register(root)
    if scale not in SCALES: raise ValueError(scale)
    torch, model, rows, features, train_ids, valid_ids, targets, provenance = setup(root, initialization)
    batch = int(BASE_BATCH * scale); rate = BASE_LR * math.sqrt(scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=1e-4)
    run = root_path(root) / arm_name(initialization, scale) / ("smoke" if smoke else "seed-17")
    run.mkdir(parents=True, exist_ok=True)
    provenance.update({"scale": scale, "batch_size": batch, "learning_rate": rate,
        "protocol_sha256": digest(root_path(root) / "protocol.json")})
    atomic_json(run / "run.json", provenance); torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(SEED); history = []; began_all = time.monotonic()
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        started = time.monotonic(); model.train(); losses, counts = [], []
        for number, ids in enumerate(np.array_split(rng.permutation(train_ids), range(batch, len(train_ids), batch)), 1):
            tensor_ids = torch.tensor(ids, device="cuda")
            inputs = torch.tensor(np.asarray(features[ids]), device="cuda")
            optimizer.zero_grad(set_to_none=True); loss = (model(inputs) - targets[tensor_ids]).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite pKAI loss")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite pKAI gradient")
            optimizer.step(); losses.append(float(loss.detach())); counts.append(len(ids))
            if smoke: break
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                "loss": losses[-1], "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **provenance}); return
        validation_rows = predict(torch, model, rows, features, valid_ids, batch_size=max(64, batch))
        validation_mse = float(np.mean([(r["predicted_shift"] - r["teacher_shift"]) ** 2 for r in validation_rows]))
        row = {"epoch": epoch, "train_shift_mse": float(np.average(losses, weights=counts)),
            "validation_shift_mse": validation_mse, "updates": len(losses), "seconds": time.monotonic() - started}
        history.append(row); atomic_json(run / "history.json", history); atomic_json(run / "progress.json", row)
        print(json.dumps({"experiment": "pkai-backbone-batch-scale-v1", "arm": arm_name(initialization, scale), **row}), flush=True)
    state = run / "epoch-005.pt"; torch.save(model.state_dict(), state)
    final_rows = predict(torch, model, rows, features, valid_ids, batch_size=max(64, batch))
    from pkabench.frozen_score import write_csv
    write_csv(run / "validation_predictions.csv", final_rows)
    metrics = group_macro(final_rows)
    atomic_json(run / "final.json", {"epoch": EPOCHS, "site_shift_mse": history[-1]["validation_shift_mse"],
        "metrics": metrics, "model_sha256": digest(state)})
    atomic_json(run / "verification.json", {"passed": True, "epochs_completed": EPOCHS,
        "wall_seconds": time.monotonic() - began_all, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "predictions_sha256": digest(run / "validation_predictions.csv"),
        **provenance})


def _noise_one(root, initialization):
    torch, model, _, features, train_ids, _, targets, provenance = setup(root, initialization)
    rng = np.random.default_rng(20261009); selected = rng.choice(train_ids, NOISE_BATCH * NOISE_BATCHES, replace=False)
    sums = [torch.zeros_like(p, dtype=torch.float64) for p in model.parameters()]
    sum_norm = torch.zeros((), dtype=torch.float64, device="cuda"); model.train()
    for number, ids in enumerate(selected.reshape(NOISE_BATCHES, NOISE_BATCH), 1):
        torch.manual_seed(SEED + number); optimizer_ids = torch.tensor(ids, device="cuda")
        inputs = torch.tensor(np.asarray(features[ids]), device="cuda")
        model.zero_grad(set_to_none=True); loss = (model(inputs) - targets[optimizer_ids]).square().mean(); loss.backward()
        norm = torch.zeros((), dtype=torch.float64, device="cuda")
        for total, parameter in zip(sums, model.parameters()):
            gradient = parameter.grad.to(torch.float64); total.add_(gradient); norm.add_(torch.sum(gradient * gradient))
        sum_norm.add_(norm)
    n = NOISE_BATCHES
    mean_norm = sum(float(torch.sum((value / n) ** 2).cpu()) for value in sums)
    sample_trace = max((float(sum_norm.cpu()) - n * mean_norm) / (n - 1), 0.0)
    covariance_trace = NOISE_BATCH * sample_trace
    corrected_g2 = max(mean_norm - sample_trace / n, np.finfo(float).tiny)
    return {"initialization": initialization, "samples": n, "microbatch": NOISE_BATCH,
        "gradient_norm_squared": corrected_g2, "covariance_trace": covariance_trace,
        "noise_scale_sites": covariance_trace / corrected_g2, "native_dropout_active": True,
        "initial_parameter_digest": provenance["initial_parameter_digest"]}


def noise(root):
    root = Path(root); register(root)
    result = {name: _noise_one(root, name) for name in INITIALIZATIONS}
    atomic_json(root_path(root) / "gradient-noise.json", {"version": "pkai-backbone-gradient-noise-v1",
        "results": result, "test_data_included": False})


def report(root):
    root = Path(root); out = root_path(root); rows = []
    for initialization in INITIALIZATIONS:
        for scale in SCALES:
            run = out / arm_name(initialization, scale) / "seed-17"
            verification, final = read(run / "verification.json"), read(run / "final.json")
            if not verification["passed"]: raise AssertionError(run)
            rows.append({"initialization": initialization, "scale": scale, "batch_size": verification["batch_size"],
                "learning_rate": verification["learning_rate"], "wall_seconds": verification["wall_seconds"],
                "peak_allocated_bytes": verification["peak_allocated_bytes"], "site_shift_mse": final["site_shift_mse"],
                "group_macro_mae": final["metrics"]["mae"], "group_macro_mae_ci95": final["metrics"]["mae_ci95"]})
    result = {"runs": rows, "gradient_noise": read(out / "gradient-noise.json"), "test_data_included": False}
    atomic_json(out / "report.json", result)
    lines = ["# Backbone pKAI batch-size and gradient-noise audit", "",
        "Five fixed epochs on identical strict-backbone features. Learning rate is 1e-6*sqrt(batch/64).", "",
        "| Initialization | Batch | LR | Wall min | Peak GB | Site shift MSE | Group-macro MAE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['initialization']} | {row['batch_size']} | {row['learning_rate']:.2g} | {row['wall_seconds']/60:.2f} | {row['peak_allocated_bytes']/1e9:.2f} | {row['site_shift_mse']:.4f} | {row['group_macro_mae']:.4f} |")
    lines += [""]
    for name, value in result["gradient_noise"]["results"].items():
        lines.append(f"- {name} gradient noise scale: **{value['noise_scale_sites']:.1f} sites**.")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(out / "report-verification.json", {"passed": True, "json_sha256": digest(out / "report.json"),
        "markdown_sha256": digest(out / "report.md"), "test_data_included": False})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    gpu_benchmark=action in ("noise", "smoke", "train"), allow_comp1400=True)
    if action == "register": register(root)
    elif action == "noise": noise(root)
    elif action in ("smoke", "train"): train(root, sys.argv[2], float(sys.argv[3]), smoke=action == "smoke")
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
