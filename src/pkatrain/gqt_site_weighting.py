"""Three-seed target-range weighting study for the orientation site GQT."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxpropka.parameters import GROUPS
from pkanet.model import PKPDB_PK_MOD
from pkanet.site_model import initialize_site, predict_site_pkpdb_indexed, predict_site_shift_indexed
from pkabench.frozen_score import aggregate, measures, write_csv
from pkabench.runtime import atomic_json, digest, require_compute
from .gqt_crop_radius import EPOCHS, MIN_DELTA, PATIENCE, learning_rate
from .gqt_regularization import decay_mask
from .graph_batches import epoch_batches
from .graph_mmap import GraphMMap, default_bundle
from .graph_pkmod_compare import digest_array_tree
from .records import read
from .site_graph_data import SiteBatchLoader
from .trainer import load_checkpoint, sample_epoch, save_checkpoint


ARMS = ("baseline", "balanced", "mild")
SEEDS = (17, 29, 43)
BIN_NAMES = ("<0.5", "0.5-1", "1-2", ">=2")
BIN_INTERVALS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf))


def experiment_root(root): return root / "pretraining/gqt-site-weighting-v1"
def site_source(root): return root / "pretraining/gqt-site-tokens-v1/site-orientation"
def control_source(root): return root / "pretraining/gqt-neighbor-cutoff-v1-triton/backbone/cutoff-20A"


def code_hashes():
    here = Path(__file__); src = here.parents[1]
    paths = (here, src / "pkanet/site_model.py", src / "pkanet/model.py",
             src / "pkanet/triton_attention.py", src / "pkatrain/site_graph_data.py",
             src / "pkatrain/graph_batches.py", src / "pkatrain/trainer.py")
    return {str(path): digest(path) for path in paths}


def bin_index(shift):
    value = np.abs(np.asarray(shift, float))
    return np.searchsorted(np.asarray((0.5, 1.0, 2.0)), value, side="right")


def training_distribution(root, manifest):
    store = GraphMMap(default_bundle(site_source(root)), manifest["records"])
    counts = np.zeros(4, np.int64); mass = np.zeros(4, np.float64)
    by_id = {row["complex_id"]: row for row in manifest["records"]}
    for cid in manifest["train"]:
        graph, labels = store.raw(cid)
        groups = np.asarray(graph["query_group"], int)
        bins = bin_index(np.asarray(labels) - np.asarray(PKPDB_PK_MOD)[groups])
        counts += np.bincount(bins, minlength=4)
        mass += np.bincount(bins, weights=np.full(len(bins), 1.0 / len(bins)), minlength=4)
        if len(labels) != by_id[cid]["q"]: raise AssertionError(cid)
    store.close()
    probability = mass / mass.sum()
    balanced = 1.0 / (4.0 * probability)
    mild = 1.0 / np.sqrt(probability)
    mild /= np.sum(probability * mild)
    mild = np.minimum(mild, 3.0)
    return {"site_counts": counts.tolist(), "effective_structure_balanced_mass": mass.tolist(),
            "effective_frequency": probability.tolist(),
            "weights": {"baseline": np.ones(4).tolist(), "balanced": balanced.tolist(),
                        "mild": mild.tolist()}, "mild_absolute_cap": 3.0}


def register(root):
    out = experiment_root(root); out.mkdir(parents=True, exist_ok=True)
    parent = read(site_source(root) / "manifest.json")
    distribution = training_distribution(root, parent)
    protocol = {"arms": list(ARMS), "seeds": list(SEEDS), "architecture": "site-orientation GQT",
        "parameter_count": parent["config"]["parameter_count"], "edge_cutoff_A": 20.0,
        "loss": "per-labelled-site signed pKPDB shift MSE; ARG remains context-only",
        "frequency_measure": "sum of 1/q per site, matching equal-structure training loss",
        "balanced": "inverse effective bin frequency; each bin has equal expected loss mass",
        "mild": "inverse square root effective bin frequency, normalized then capped at 3",
        "selection": "mean of the four validation-bin group-macro MAEs; not overall MAE",
        "optimizer": "AdamW weight decay 1e-4; gradient norm clip 1",
        "schedule": "1e-3 through epoch 10, cosine to 1e-5", "batch_size": 8,
        "current_gqt_control": str(control_source(root)), "distribution": distribution,
        "test_data_included": False}
    atomic_json(out / "protocol.json", protocol)
    for arm in ARMS:
        destination = out / arm; destination.mkdir(exist_ok=True)
        manifest = copy.deepcopy(parent)
        manifest["parent"] = {"path": str(site_source(root)), "manifest_sha256": digest(site_source(root) / "manifest.json")}
        manifest["weighting_protocol_sha256"] = digest(out / "protocol.json")
        manifest["config"].update(model_arm=f"site-orientation-{arm}", seed=None,
            loss_weights=distribution["weights"][arm], selection=protocol["selection"])
        data = destination / "data"
        if not data.exists(): data.symlink_to((site_source(root) / "data").resolve(), target_is_directory=True)
        for name in ("preparation.json", "train_type_means.json"):
            if (site_source(root) / name).exists() and not (destination / name).exists():
                shutil.copy2(site_source(root) / name, destination / name)
        atomic_json(destination / "manifest.json", manifest)
    plan = [{"arm": arm, "seed": seed} for arm in ARMS for seed in SEEDS]
    atomic_json(out / "plan.json", {"passed": True, "runs": plan, "code_hashes": code_hashes(),
                "protocol_sha256": digest(out / "protocol.json")})
    atomic_json(out / "registration.json", {"passed": True, "runs": len(plan),
                "code_hashes": code_hashes(), "protocol_sha256": digest(out / "protocol.json")})


class WeightedSiteEngine:
    def __init__(self, params, weights):
        self.weights = jnp.asarray(weights, jnp.float32)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0),
            optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        self.forward = jax.jit(predict_site_pkpdb_indexed)
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32)
        thresholds = jnp.asarray((0.5, 1.0, 2.0), jnp.float32)
        def one_loss(p, graph, target, eligible):
            expected = target - reference[graph["query_group"]]
            predicted = predict_site_shift_indexed(p, graph)
            bins = jnp.sum(jnp.abs(expected)[:, None] >= thresholds[None], axis=1)
            weighted = self.weights[bins] * jnp.square(predicted - expected)
            return jnp.sum(jnp.where(eligible, weighted, 0.0)) / jnp.maximum(jnp.sum(eligible), 1)
        def batch_loss(p, graphs, targets, eligible, valid):
            losses = jax.vmap(one_loss, in_axes=(None, 0, 0, 0))(p, graphs, targets, eligible)
            return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
        def step(p, state, graphs, targets, eligible, valid, rate):
            loss, gradient = jax.value_and_grad(batch_loss)(p, graphs, targets, eligible, valid)
            finite = jnp.isfinite(loss) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            updates = jax.tree.map(lambda value: value * rate, updates)
            return optax.apply_updates(p, updates), state, loss, finite
        self.step = jax.jit(step)
    def update(self, params, state, batch, rate):
        params, state, loss, finite = self.step(params, state, *batch, rate)
        if not bool(finite): raise FloatingPointError("Nonfinite weighted site-token update")
        return params, state, float(loss)


def macro_metric(rows):
    by_complex = defaultdict(list)
    for row in rows: by_complex[row["complex_id"]].append(row)
    scores = []
    for cid, subset in by_complex.items():
        target = np.asarray([row["teacher_shift"] for row in subset])
        predicted = np.asarray([row["predicted_shift"] for row in subset])
        scores.append({"complex_id": cid, "component_id": subset[0]["component_id"], "n": len(subset),
                       **measures(target, predicted)})
    return aggregate(scores, replicates=1)[0]


def metrics_from_rows(rows):
    overall = macro_metric(rows)
    bins = []
    for (low, high), name in zip(BIN_INTERVALS, BIN_NAMES):
        subset = [row for row in rows if low <= abs(row["teacher_shift"]) < high]
        bins.append({"bin": name, "sites": len(subset), "mae": macro_metric(subset)["mae"]})
    return overall, bins, float(np.mean([row["mae"] for row in bins]))


def evaluate(out, manifest, arm, engine, params, *, predictions=None, bootstrap=False):
    loader = SiteBatchLoader(out, manifest, manifest["config"]["batch_size"], "site-orientation")
    rows = []
    for record in manifest["records"]:
        if record["split"] != "val": continue
        graph, target, eligible = loader.one(record["complex_id"])
        predicted = np.asarray(engine.forward(params, graph))[eligible]; observed = target[eligible]
        groups = graph["query_group"][eligible]; reference = np.asarray(PKPDB_PK_MOD)[groups]
        for key, y, value, ref in zip(record["keys"], observed, predicted, reference):
            rows.append(dict(zip(("complex_id", "chain", "resnum", "icode", "group"), key),
                component_id=record["component_id"], teacher_pka=float(y), predicted_pka=float(value),
                teacher_shift=float(y - ref), predicted_shift=float(value - ref)))
    loader.close(); overall, bins, selection = metrics_from_rows(rows)
    if bootstrap:
        # Replace only the overall point/CI with the preregistered 2,000 group bootstrap.
        by_complex = defaultdict(list)
        for row in rows: by_complex[row["complex_id"]].append(row)
        scores = []
        for cid, subset in by_complex.items():
            scores.append({"complex_id": cid, "component_id": subset[0]["component_id"], "n": len(subset),
                           **measures([x["teacher_shift"] for x in subset], [x["predicted_shift"] for x in subset])})
        overall = aggregate(scores, replicates=2000)[0]
    if predictions is not None: write_csv(predictions, rows)
    return {"overall": overall, "bins": bins, "selection_equal_bin_mae": selection}


def train(root, arm, seed, *, smoke=False):
    if arm not in ARMS or seed not in SEEDS: raise ValueError((arm, seed))
    base = experiment_root(root); plan = read(base / "plan.json")
    if not plan["passed"] or plan["code_hashes"] != code_hashes(): raise AssertionError("plan/code mismatch")
    out = base / arm; manifest_path = out / "manifest.json"; manifest = read(manifest_path)
    cfg = manifest["config"]; params = initialize_site(jax.random.PRNGKey(seed), **cfg["architecture"])
    engine = WeightedSiteEngine(params, cfg["loss_weights"]); state = engine.optimizer.init(params)
    rng = np.random.default_rng(seed); run = out / (f"smoke-seed-{seed}" if smoke else f"seed-{seed}")
    run.mkdir(parents=True, exist_ok=True)
    by_id = {row["complex_id"]: row for row in manifest["records"]}; records = [by_id[cid] for cid in manifest["train"]]
    loader = SiteBatchLoader(out, manifest, cfg["batch_size"], "site-orientation")
    provenance = {"arm": arm, "seed": seed, "manifest_sha256": digest(manifest_path),
                  "code_hashes": code_hashes(), "parameter_count": cfg["parameter_count"],
                  "loss_weights": cfg["loss_weights"]}
    atomic_json(run / "run.json", {**provenance, "config": cfg, "loader": loader.provenance(),
                "initial_parameter_digest": digest_array_tree(params)})
    best = None; stalled = 0; history = []; began_run = time.monotonic()
    for epoch in range(1, (1 if smoke else EPOCHS) + 1):
        plans = epoch_batches(sample_epoch(records, rng), by_id, rng, cfg["batch_size"])
        plan_digest = hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()
        if smoke: plans = plans[:1]
        losses = []; began = time.monotonic()
        for number, (_, batch) in enumerate(loader.iterate(plans), 1):
            rate = learning_rate(epoch, number, len(plans))
            params, state, loss = engine.update(params, state, batch, rate); losses.append(loss)
        if smoke:
            atomic_json(run / "verification.json", {"passed": True, "finite_update": True,
                "loss": float(np.mean(losses)), "batch_plan_digest": plan_digest,
                "device_memory": jax.local_devices()[0].memory_stats(), **provenance})
            loader.close(); return
        validation = evaluate(out, manifest, arm, engine, params); score = validation["selection_equal_bin_mae"]
        if best is None or score < best["selection_equal_bin_mae"] - MIN_DELTA:
            best = {"epoch": epoch, "selection_equal_bin_mae": score,
                    "overall_mae": validation["overall"]["mae"]}; stalled = 0
            atomic_json(run / "best.json", best)
        else: stalled += 1
        row = {"epoch": epoch, "train_weighted_shift_mse": float(np.mean(losses)), "validation": validation,
               "seconds": time.monotonic() - began, "batch_plan_digest": plan_digest,
               "best_epoch": best["epoch"], "stalled_epochs": stalled}
        history.append(row); atomic_json(run / "history.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch-{epoch:03d}", params, state, {**provenance, "epoch": epoch})
        print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch,
            "train_loss": row["train_weighted_shift_mse"], "overall_mae": validation["overall"]["mae"],
            "equal_bin_mae": score, "large_shift_mae": validation["bins"][3]["mae"],
            "best_epoch": best["epoch"], "seconds": row["seconds"]}), flush=True)
        if stalled >= PATIENCE: break
    loader.close(); params, _, _ = load_checkpoint(run / "checkpoints" / f"epoch-{best['epoch']:03d}", (params, state))
    final = evaluate(out, manifest, arm, engine, params, predictions=run / "validation_predictions.csv", bootstrap=True)
    atomic_json(run / "final.json", final)
    atomic_json(run / "verification.json", {"passed": True, "epochs_completed": len(history),
        "selected_epoch": best["epoch"], "selection_equal_bin_mae": best["selection_equal_bin_mae"],
        "wall_seconds": time.monotonic() - began_run, "device_memory": jax.local_devices()[0].memory_stats(),
        "predictions_sha256": digest(run / "validation_predictions.csv"), "test_data_included": False, **provenance})


def read_predictions(path):
    with Path(path).open(newline="") as stream: rows = list(csv.DictReader(stream))
    for row in rows:
        group = GROUPS.index(row["group"]); ref = float(PKPDB_PK_MOD[group])
        row["teacher_shift"] = float(row.get("teacher_shift") or (float(row["teacher_pka"]) - ref))
        row["predicted_shift"] = float(row.get("predicted_shift") or (float(row["predicted_pka"]) - ref))
    return rows


def summarize(path):
    rows = read_predictions(path); overall, bins, selection = metrics_from_rows(rows)
    target = np.asarray([row["teacher_shift"] for row in rows]); predicted = np.asarray([row["predicted_shift"] for row in rows])
    slope = float(np.cov(target, predicted, ddof=0)[0, 1] / np.var(target))
    residue = {}
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        if subset: residue[group] = {"sites": len(subset), "mae": macro_metric(subset)["mae"]}
    return {"overall_mae": overall["mae"], "bins": bins, "selection_equal_bin_mae": selection,
            "shift_slope": slope, "prediction_std": float(np.std(predicted)), "target_std": float(np.std(target)),
            "residue": residue, "sites": len(rows)}


def mean_sd(values):
    value = np.asarray(values, float)
    return {"mean": float(value.mean()), "sd": float(value.std(ddof=1))}


def report(root):
    base = experiment_root(root); per_seed = []
    sources = [("current-gqt", seed, control_source(root) / f"seed-{seed}" / "validation_predictions.csv") for seed in SEEDS]
    sources += [(arm, seed, base / arm / f"seed-{seed}" / "validation_predictions.csv") for arm in ARMS for seed in SEEDS]
    for arm, seed, path in sources:
        row = {"arm": arm, "seed": seed, **summarize(path)}; per_seed.append(row)
    # Prove matched batch membership/order across all orientation losses for each seed.
    for seed in SEEDS:
        histories = [read(base / arm / f"seed-{seed}" / "history.json") for arm in ARMS]
        shortest = min(map(len, histories))
        for epoch in range(shortest):
            if len({history[epoch]["batch_plan_digest"] for history in histories}) != 1:
                raise AssertionError((seed, epoch + 1, "batch plan mismatch"))
    summaries = []
    for arm in ("current-gqt", *ARMS):
        subset = [row for row in per_seed if row["arm"] == arm]
        item = {"arm": arm, "seeds": list(SEEDS), "overall_mae": mean_sd([row["overall_mae"] for row in subset]),
                "selection_equal_bin_mae": mean_sd([row["selection_equal_bin_mae"] for row in subset]),
                "shift_slope": mean_sd([row["shift_slope"] for row in subset]),
                "prediction_std": mean_sd([row["prediction_std"] for row in subset]), "bins": {}, "residue": {}}
        for name in BIN_NAMES:
            item["bins"][name] = mean_sd([next(x["mae"] for x in row["bins"] if x["bin"] == name) for row in subset])
        for group in sorted(set.intersection(*(set(row["residue"]) for row in subset))):
            item["residue"][group] = mean_sd([row["residue"][group]["mae"] for row in subset])
        summaries.append(item)
    atomic_json(base / "results-per-seed.json", per_seed); atomic_json(base / "summary.json", summaries)
    first = per_seed[0]; counts = {row["bin"]: row["sites"] for row in first["bins"]}
    fmt = lambda value: f"{value['mean']:.4f} +/- {value['sd']:.4f}"
    lines = ["# Site-orientation target-range weighting", "",
        "Three seeds (17, 29, 43), frozen backbone-only 20 A graphs. Orientation arms have 67,725 parameters; ARG remains an unsupervised context token. Checkpoints use equal-bin validation MAE.", "",
        f"| Arm | Overall MAE | Equal-bin MAE | <0.5 (n={counts['<0.5']:,}) | 0.5-1 (n={counts['0.5-1']:,}) | 1-2 (n={counts['1-2']:,}) | **>=2 (n={counts['>=2']:,})** | Shift slope | Prediction SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['arm']} | {fmt(row['overall_mae'])} | {fmt(row['selection_equal_bin_mae'])} | "
            f"{fmt(row['bins']['<0.5'])} | {fmt(row['bins']['0.5-1'])} | {fmt(row['bins']['1-2'])} | "
            f"**{fmt(row['bins']['>=2'])}** | {fmt(row['shift_slope'])} | {fmt(row['prediction_std'])} |")
    lines += ["", "## MAE by residue type", "", "| Arm | " + " | ".join(GROUPS) + " |",
              "|---|" + "---:|" * len(GROUPS)]
    for row in summaries:
        lines.append("| " + row["arm"] + " | " + " | ".join(fmt(row["residue"][group]) if group in row["residue"] else "n/a" for group in GROUPS) + " |")
    lines += ["", "Mean +/- sample SD across three seeds. The current-GQT row reuses the completed matched 20 A control runs; the three orientation arms used identical batch plans within each seed. No test data were read.", ""]
    (base / "report.md").write_text("\n".join(lines))
    atomic_json(base / "verification.json", {"passed": True, "runs": len(per_seed), "seeds": list(SEEDS),
                "matched_orientation_batch_plans": True, "report_sha256": digest(base / "report.md"),
                "test_data_included": False})


def verify_smokes(root):
    base = experiment_root(root); rows = []
    for arm in ARMS:
        value = read(base / arm / "smoke-seed-17/verification.json")
        if not value["passed"] or not value["finite_update"] or value["code_hashes"] != code_hashes(): raise AssertionError(arm)
        rows.append({"arm": arm, "loss": value["loss"], "batch_plan_digest": value["batch_plan_digest"]})
    if len({row["batch_plan_digest"] for row in rows}) != 1: raise AssertionError("smoke batch mismatch")
    atomic_json(base / "smoke-verification.json", {"passed": True, "rows": rows, "code_hashes": code_hashes()})


def main():
    import sys
    action = sys.argv[1]; root = Path(os.environ["PKABENCH_RUNTIME"])
    gpu = action in ("smoke", "train")
    require_compute(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), gpu_benchmark=gpu, allow_comp1400=gpu)
    jax.config.update("jax_enable_x64", False)
    if action == "register": register(root)
    elif action == "smoke": train(root, sys.argv[2], int(sys.argv[3]), smoke=True)
    elif action == "verify-smokes": verify_smokes(root)
    elif action == "train": train(root, sys.argv[2], int(sys.argv[3]))
    elif action == "report": report(root)
    else: raise ValueError(action)


if __name__ == "__main__": main()
