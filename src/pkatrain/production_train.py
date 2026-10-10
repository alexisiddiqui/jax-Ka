"""Production joint GQT training on the pool-v3 stores (pkatrain.production_loading), 2026-10-10.

User decisions (2026-10-10): constant batch of 16 structures in every size bucket (GH200 sweep: memory never binds;
near experiment 45's gradient-noise scale of about 18); scratch initialisation; the experiment 43 joint objective with
the 67,725-parameter backbone oGQT (width 44, ff 88); first run on the 10% pool.

- Engine: gqt_multitask_replay.JointEngine, unchanged. Each update sums one PINDER gradient (AB/free state-shift MSE +
  binding-shift MSE) and one pKPDB state-shift gradient (train_mask sites), then one AdamW step (weight decay 1e-4,
  global clip 1). Equal coefficients, no site weights.
- Sampling: each epoch is one pass over the fraction's PINDER training complexes (BucketPolicy.plans); one pKPDB batch
  per PINDER batch from a continuing stream of shuffled pKPDB epochs. Plans are a deterministic function of the seed
  and epoch, so a run resumes from its last checkpoint.
- Schedule: gqt_crop_radius.learning_rate (hold through epoch 10, cosine to epoch 20) at 1e-3 * sqrt(16 / 8) ->
  1e-5 * sqrt(16 / 8) (experiment 45's sqrt batch-scale rule relative to the factorial's 8); 20-epoch cap; stop after 8
  epochs without a MIN_DELTA improvement.
- Validation every epoch: PINDER 400 (state MAE, interface paired MAE; gqt_paired_pinder._metrics) and the 142-complex
  benchmark PypKa set (benchmark-val store; group-macro MAE, gqt_site_weighting.metrics_from_rows). Selection: lowest
  PINDER state MAE + interface paired MAE (experiments 41-43). No test data.

Layout: <runtime>/training/<version>/runs/<run>/ (PKATRAIN_GQT_VERSION, default gqt-production-v2: pool-v4 with
the new PINDER and pKPDB validation sets, the latter scored on pKPDB's own train_mask labels; protocol.json, history.jsonl, checkpoints/epoch-NNN/,
selection.json, predictions-{pinder,benchmark}-epoch-NNN.csv.

Batch-size sweep (2026-10-10): --batch B trains with a constant B structures per batch in every bucket and the
learning rate scaled by sqrt(B / 8) (the same rule); validation always runs at 16 per batch (per-structure predictions,
so the batch size only changes padding). The default 16 reproduces the pilot's protocol. Batches above
MICRO_RESIDUES / n (n = the bucket's padded residue count) are split on the host into chunks of the largest divisor of
the batch within that budget (98,304 residue slots = 64 x 1,536, measured at 17.8 GB peak; batch 256 in the 1,536-residue
bucket needs a single 45 GiB allocation, and 128-chunks failed with prefetched batches resident on the device), so small
buckets run in large chunks and only the large buckets in chunks of 64. Both objectives are means over the batch's
valid structures, so the full-batch value and gradient are the chunk values and gradients weighted by (valid
structures in chunk / valid in batch); the chunk size changes only float summation order. Chunks that
are pure padding are skipped. One optimizer step per batch, as before.

  python -m pkatrain.production_train train RUN [--fraction 0.1] [--batch 16] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest
from .loading import PRODUCTION_BOUNDS, BucketPolicy, DeferredScalars, LoaderConfig, Prefetcher
from .production_graphs import VERSION, output, read
from .production_loading import MANIFEST, PinderSource, PkpdbSource, normalization, select

SEED = 17
BATCH = 16
ARCHITECTURE = {"width": 44, "ff": 88}


def lr_scale(batch): return math.sqrt(batch / 8)


MICRO_RESIDUES = 64 * 1536


def micro_size(batch, residues):
    """Largest divisor of `batch` whose chunk stays within MICRO_RESIDUES residue slots (at least 1)."""
    return max([d for d in range(1, batch + 1) if batch % d == 0 and d * residues <= MICRO_RESIDUES] or [1])


def chunked(batch):
    """Whether some bucket needs chunks: batches up to 64 run whole (the pilot's path), larger ones per-bucket chunks."""
    return batch * max(PRODUCTION_BOUNDS) > MICRO_RESIDUES


def run_dir(root, run): return Path(root) / "training" / VERSION / "runs" / run


def code_hashes():
    src = Path(__file__).parents[1]
    paths = [Path(__file__), src / "pkatrain/production_loading.py", src / "pkatrain/production_graphs.py",
             src / "pkatrain/loading.py", src / "pkatrain/gqt_multitask_replay.py", src / "pkatrain/gqt_paired_pinder.py",
             src / "pkatrain/site_graph_data.py", src / "pkanet/ogqt.py", src / "pkanet/model.py", src / "pkanet/triton_attention.py"]
    return {str(p.relative_to(src)): digest(p) for p in paths}


def _policy(manifest, batch=BATCH):
    bounds = BucketPolicy.from_json(manifest["bucket_policy"]).bounds
    return BucketPolicy(bounds, (batch,) * len(bounds))


def _with_policy(manifest, batch=BATCH):
    return {**manifest, "bucket_policy": _policy(manifest, batch).to_json()}


def epoch_plans(manifests, fraction, epoch, batch=BATCH):
    """(PINDER plans, pKPDB plans) for one epoch, deterministic in (SEED, epoch); the pKPDB stream continues across epochs."""
    pinder, pkpdb = manifests["pinder"], manifests["pkpdb"]
    pair = _policy(pinder, batch).plans(select(pinder, "train", fraction), np.random.default_rng((SEED, epoch, 1)))
    per_epoch = len(pair); start = (epoch - 1) * per_epoch; stream = []; cycle = 0
    pk_records = select(pkpdb, "train", fraction)
    while len(stream) < start + per_epoch:
        stream.extend(_policy(pkpdb, batch).plans(pk_records, np.random.default_rng((SEED, cycle, 2)))); cycle += 1
    return pair, stream[start:start + per_epoch]


class JointSource:
    """BatchSource over (PINDER ids, pKPDB ids) specs."""

    def __init__(self, pinder, pkpdb): self.pinder = pinder; self.pkpdb = pkpdb

    def load(self, spec): return self.pinder.load(spec[0]), self.pkpdb.load(spec[1])

    def close(self): self.pinder.close(); self.pkpdb.close()

    def provenance(self): return {"pinder": self.pinder.provenance(), "pkpdb": self.pkpdb.provenance()}


def to_device(batch):
    import jax
    pair, pk = batch; graphs, targets, mask, wb, wi, metadata, valid = pair
    return (*jax.device_put((graphs, targets, mask, wb, wi)), metadata, jax.device_put(valid)), jax.device_put(pk)


def host_chunks(batch, micro=None):
    """Host batch -> ([(PINDER chunk, valid count)], [(pKPDB chunk, valid count)]); padding-only chunks dropped. Chunk
    size per dataset: `micro` if given, else micro_size(batch, padded residues). Chunks stay on the host and transfer
    when their step is dispatched: prefetched batches resident on the device ran a batch-256 run out of memory."""
    import jax
    pair, pk = batch; graphs, targets, mask, wb, wi, _, valid = pair; pair = (graphs, targets, mask, wb, wi, valid)

    def chunks(arrays, valid):
        size = micro or micro_size(len(valid), arrays[0]["nodes"].shape[-2]); out = []
        for start in range(0, len(valid), size):
            count = int(valid[start:start + size].sum())
            if count: out.append((jax.tree.map(lambda x: x[start:start + size], arrays), count))
        return out
    return chunks(pair, valid), chunks(pk, pk[-1])


def _accumulate(value_grad, chunks):
    """Full-batch (value, gradient) of a mean over valid structures: sum over chunks of (count / total) x chunk's."""
    import jax
    import jax.numpy as jnp
    total = sum(count for _, count in chunks); value = gradient = None
    for arrays, count in chunks:
        v, g = jax.tree.map(lambda x: (count / total) * x, value_grad(*arrays))
        value, gradient = (v, g) if value is None else (jax.tree.map(jnp.add, value, v), jax.tree.map(jnp.add, gradient, g))
    return value, gradient


def _pinder_rows(engine, params, source, records, config):
    """Per-site validation rows with gqt_paired_pinder.evaluate's definitions."""
    from pkanet.model import PKPDB_PK_MOD
    reference_table = np.asarray(PKPDB_PK_MOD); rows = []
    plans = source.policy.plans(records, np.random.default_rng(0))
    for ids, batch in Prefetcher(source, plans, config):
        graphs, targets, mask, _, _, metadata, valid = batch
        predicted_all = np.asarray(engine.predictions(params, graphs))
        for slot, cid in enumerate(ids):
            active = mask[slot]; predicted = predicted_all[slot][:, active]
            reference = reference_table[graphs["query_group"][slot, 0, active]]
            expected = targets[slot][:, active] - reference[None]
            for i in range(predicted.shape[1]):
                rows.append({"complex_id": cid, "site": i, "teacher_ab": float(targets[slot, 0, active][i]),
                    "teacher_free": float(targets[slot, 1, active][i]),
                    "predicted_ab": float(predicted[0, i] + reference[i]), "predicted_free": float(predicted[1, i] + reference[i]),
                    "state_error": float(np.mean(np.abs(predicted[:, i] - expected[:, i]))),
                    "paired_error": float((predicted[0, i] - predicted[1, i]) - (expected[0, i] - expected[1, i])),
                    "interface": bool(metadata["interface"][slot][active][i]),
                    "distance": float(metadata["partner_distance_A"][slot][active][i]),
                    "rsa_free": float(metadata["rsa_free"][slot][active][i])})
    return rows


def _pkpdb_rows(predict, params, source, records, config):
    """pKPDB validation rows (gqt-production-v2): train_mask sites of the held-out pKPDB structures, shift relative to
    PKPDB_PK_MOD, the training loss's definition."""
    from pkanet.model import PKPDB_PK_MOD
    reference_table = np.asarray(PKPDB_PK_MOD); rows = []
    plans = source.policy.plans(records, np.random.default_rng(0))
    for ids, batch in Prefetcher(source, plans, config):
        graphs, targets, eligible, valid = batch
        predicted = np.asarray(predict(params, graphs))
        for slot, cid in enumerate(ids):
            active = eligible[slot]; groups = graphs["query_group"][slot][active]; reference = reference_table[groups]
            for site, (y, shift, ref) in enumerate(zip(targets[slot][active], predicted[slot][active], reference)):
                rows.append({"structure_id": cid, "site": site, "group": int(groups[site]), "teacher_shift": float(y - ref),
                             "predicted_shift": float(shift)})
    return rows


def pkpdb_metrics(rows):
    """Site-level MAE/MSE and the structure-macro MAE (mean over structures of each structure's site MAE)."""
    from collections import defaultdict
    error = np.asarray([r["predicted_shift"] - r["teacher_shift"] for r in rows]); by = defaultdict(list)
    for r, e in zip(rows, error): by[r["structure_id"]].append(abs(e))
    return {"structures": len(by), "sites": len(rows), "site_mae": float(np.mean(np.abs(error))), "site_mse": float(np.mean(error ** 2)),
            "structure_macro_mae": float(np.mean([np.mean(v) for v in by.values()]))}


def _benchmark_rows(predict, params, source, records, config):
    """Per-site rows with gqt_site_weighting.evaluate's definitions (shift relative to PKPDB_PK_MOD)."""
    from pkanet.model import PKPDB_PK_MOD
    reference_table = np.asarray(PKPDB_PK_MOD); rows = []; component = {r["id"]: r.get("component_id") for r in records}
    plans = source.policy.plans(records, np.random.default_rng(0))
    for ids, batch in Prefetcher(source, plans, config):
        graphs, targets, eligible, valid = batch
        predicted = np.asarray(predict(params, graphs))
        for slot, cid in enumerate(ids):
            active = eligible[slot]; groups = graphs["query_group"][slot][active]; reference = reference_table[groups]
            for site, (y, shift, ref) in enumerate(zip(targets[slot][active], predicted[slot][active], reference)):
                rows.append({"complex_id": cid, "site": site, "group": int(groups[site]), "component_id": component[cid],
                             "teacher_pka": float(y), "predicted_pka": float(shift + ref),
                             "teacher_shift": float(y - ref), "predicted_shift": float(shift)})
    return rows


def _write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def squared_errors(pair_rows, bench_rows):
    """Validation MSEs on the training losses' scales (2026-10-10): PINDER state (mean over AB/free of the squared error),
    paired (squared error of the AB - free shift), interface paired; benchmark shift MSE per site and macro over
    component groups. Site-level means; the training losses average per structure first."""
    from collections import defaultdict
    state = [((r["predicted_ab"] - r["teacher_ab"]) ** 2 + (r["predicted_free"] - r["teacher_free"]) ** 2) / 2 for r in pair_rows]
    paired = [r["paired_error"] ** 2 for r in pair_rows]; interface = [r["paired_error"] ** 2 for r in pair_rows if r["interface"]]
    groups = defaultdict(list)
    for r in bench_rows: groups[r["component_id"]].append((r["predicted_shift"] - r["teacher_shift"]) ** 2)
    return ({"state_mse": float(np.mean(state)), "paired_mse": float(np.mean(paired)), "interface_paired_mse": float(np.mean(interface))},
            {"site_mse": float(np.mean([v for g in groups.values() for v in g])), "group_macro_mse": float(np.mean([np.mean(g) for g in groups.values()])),
             "site_mae": float(np.mean([abs(r["predicted_shift"] - r["teacher_shift"]) for r in bench_rows]))})


def validate(engine, predict, params, sources, manifests, config, out=None, epoch=None):
    from .gqt_paired_pinder import _metrics
    from .gqt_site_weighting import metrics_from_rows
    pair_rows = _pinder_rows(engine, params, sources["pinder-val"], select(manifests["pinder"], "val"), config)
    bench_rows = _benchmark_rows(predict, params, sources["benchmark"], select(manifests["benchmark-val"], "val"), config)
    pinder = _metrics(pair_rows); overall, bins, equal_bin = metrics_from_rows(bench_rows)
    pinder_squared, bench_squared = squared_errors(pair_rows, bench_rows); pinder.update(pinder_squared)
    pkpdb = None
    if "pkpdb-val" in sources:
        pk_rows = _pkpdb_rows(predict, params, sources["pkpdb-val"], select(manifests["pkpdb"], "val"), config); pkpdb = pkpdb_metrics(pk_rows)
        if out is not None: _write(out / f"predictions-pkpdb-epoch-{epoch:03d}.csv", pk_rows)
    if out is not None:
        _write(out / f"predictions-pinder-epoch-{epoch:03d}.csv", pair_rows); _write(out / f"predictions-benchmark-epoch-{epoch:03d}.csv", bench_rows)
    return {"pinder": pinder, **({"pkpdb": pkpdb} if pkpdb else {}), "benchmark": {"overall": overall, "bins": bins, "equal_bin_mae": equal_bin, **bench_squared},
            "selection": pinder["state_mae"] + pinder["interface_paired_mae"]}


def train(root, run, fraction=0.1, smoke=False, batch=BATCH):
    import jax
    import jax.numpy as jnp
    from pkanet.ogqt import initialize as initialize_ogqt, predict_shift
    from .gqt_crop_radius import EPOCHS, MIN_DELTA, PATIENCE, learning_rate
    from .gqt_multitask_replay import JointEngine
    from .trainer import load_checkpoint, save_checkpoint
    root = Path(root); out = run_dir(root, run); out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(out.parent / "compilation-cache"))
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb", "benchmark-val")}
    norms = normalization(manifests["pinder"], fraction); scale = lr_scale(batch)
    protocol = {"version": "gqt-production-joint-v1", "run": run, "fraction": fraction, "seed": SEED, "batch": batch,
        "architecture": ARCHITECTURE, "initialization": "scratch (pkanet.ogqt.initialize, PRNGKey(17))",
        "objective": "gqt_multitask_replay.JointEngine: pKPDB state-shift MSE (train_mask) + PINDER AB/free state-shift MSE + binding-shift MSE; equal coefficients; no site weights",
        "optimizer": "AdamW weight decay 1e-4, global clip 1 after gradient summation",
        "schedule": f"gqt_crop_radius.learning_rate x {scale:.4f} (1e-3 hold to epoch 10, cosine to 1e-5 by epoch {EPOCHS}); patience {PATIENCE}, min delta {MIN_DELTA}",
        "sampling": "full PINDER fraction per epoch; one pKPDB batch per PINDER batch from a continuing shuffled stream",
        "selection": "min PINDER validation state MAE + interface paired MAE", "validation": (["PINDER 400 (pKAI)", "benchmark-val 142 (PypKa)"] if VERSION == "gqt-production-v1" else
            ["PINDER pool-v4 validation (pKAI, eval_mask)", "pKPDB pool-v4 validation (pKPDB labels, train_mask)", "benchmark-val 142 (PypKa)"]),
        "counts": {"pinder_train": len(select(manifests["pinder"], "train", fraction)), "pkpdb_train": len(select(manifests["pkpdb"], "train", fraction)),
                   "pinder_val": len(select(manifests["pinder"], "val")), "benchmark_val": len(select(manifests["benchmark-val"], "val")),
                   **({"pkpdb_val": len(select(manifests["pkpdb"], "val"))} if select(manifests["pkpdb"], "val") else {})},
        **({"micro_residues": MICRO_RESIDUES, "accumulation": "exact: per-bucket chunks (largest divisor of the batch within the residue budget), values/gradients weighted by valid structures"} if chunked(batch) else {}),
        "pinder_weight_normalization": norms, "smoke": smoke, "test_data_included": False,
        "manifests": {d: digest(output(root, d) / MANIFEST) for d in manifests}, "code": code_hashes()}
    if (out / "protocol.json").exists():
        stored = read(out / "protocol.json")
        if {k: v for k, v in stored.items() if k != "code"} != json.loads(json.dumps({k: v for k, v in protocol.items() if k != "code"})):
            raise AssertionError(f"{out} was registered with a different protocol")
    else: atomic_json(out / "protocol.json", protocol)
    config = LoaderConfig()
    sources = {"train": JointSource(PinderSource(_with_policy(manifests["pinder"], batch), config=config, norms=norms),
                                    PkpdbSource(_with_policy(manifests["pkpdb"], batch), config=config)),
               "pinder-val": PinderSource(_with_policy(manifests["pinder"]), config=config, norms=norms),
               "benchmark": PkpdbSource(_with_policy(manifests["benchmark-val"]), config=config, mask="eval_mask"),
               **({"pkpdb-val": PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask")} if select(manifests["pkpdb"], "val") else {})}
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **ARCHITECTURE); engine = JointEngine(params); state = engine.optimizer.init(params)
    predict = jax.jit(jax.vmap(predict_shift, in_axes=(None, 0)))
    history = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()] if (out / "history.jsonl").exists() else []
    if history:
        last = history[-1]["epoch"]; params, state, _ = load_checkpoint(out / "checkpoints" / f"epoch-{last:03d}", (params, state))
    best = min(history, key=lambda h: h["validation"]["selection"]) if history else None; stale = 0
    if history:
        stale = max(0, history[-1]["epoch"] - best["epoch"])
    epochs = 2 if smoke else EPOCHS
    for epoch in range(len(history) + 1, epochs + 1):
        if stale >= PATIENCE: break
        pair, pk = epoch_plans(manifests, fraction, epoch, batch)
        if smoke: pair, pk = pair[:5], pk[:5]
        specs = list(zip(pair, pk)); deferred = DeferredScalars(every=50); began = time.time()
        prefetcher = Prefetcher(sources["train"], specs, config, to_device if not chunked(batch) else host_chunks)
        for number, (_, (pair_batch, pk_batch)) in enumerate(prefetcher, 1):
            rate = jnp.asarray(scale * learning_rate(epoch, number, len(specs)), jnp.float32)
            if not chunked(batch):
                graphs, targets, mask, wb, wi, _, valid = pair_batch
                (pair_total, (state_loss, pair_loss)), pair_gradient = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, valid)
                pk_loss, pk_gradient = engine.pkpdb_value_grad(params, *pk_batch)
            else:
                (pair_total, (state_loss, pair_loss)), pair_gradient = _accumulate(lambda *a: engine.paired_value_grad(params, *a), pair_batch)
                pk_loss, pk_gradient = _accumulate(lambda *a: engine.pkpdb_value_grad(params, *a), pk_batch)
            params, state, finite = engine.apply(params, state, pair_gradient, pk_gradient, rate)
            losses = jnp.stack((pair_total + pk_loss, pk_loss, state_loss, pair_loss))
            if not bool(finite & jnp.all(jnp.isfinite(losses))): raise FloatingPointError(f"nonfinite update at epoch {epoch} batch {number}")
            deferred.add(losses)
        deferred.flush(); train_seconds = time.time() - began; values = np.asarray(deferred.values).reshape(-1, 4)
        telemetry = prefetcher.telemetry.summary()
        began = time.time(); validation = validate(engine, predict, params, sources, manifests, config); val_seconds = time.time() - began
        save_checkpoint(out / "checkpoints" / f"epoch-{epoch:03d}", params, state, {"epoch": epoch, "run": run})
        row = {"epoch": epoch, "updates": len(specs), "learning_rate_end": scale * learning_rate(epoch, len(specs), len(specs)),
               "train_loss": {name: float(values[:, i].mean()) for i, name in enumerate(("total", "pkpdb", "pinder_state", "pinder_paired"))},
               "validation": validation, "train_seconds": round(train_seconds, 1), "validation_seconds": round(val_seconds, 1),
               "loader_wait_fraction": telemetry.get("wait_fraction")}
        history.append(row)
        with open(out / "history.jsonl", "a") as handle: handle.write(json.dumps(row) + "\n")
        improved = best is None or validation["selection"] < best["validation"]["selection"] - MIN_DELTA
        if best is None or validation["selection"] < best["validation"]["selection"]: best = row
        stale = 0 if improved else stale + 1
        print(json.dumps({"epoch": epoch, "loss": row["train_loss"]["total"], "pinder_state_mae": validation["pinder"]["state_mae"],
                          "interface_paired_mae": validation["pinder"]["interface_paired_mae"],
                          "benchmark_mae": validation["benchmark"]["overall"]["mae"], "pkpdb_val_mae": validation.get("pkpdb", {}).get("site_mae"),
                          "selection": validation["selection"],
                          "train_s": row["train_seconds"], "val_s": row["validation_seconds"], "wait": row["loader_wait_fraction"]}), flush=True)
    selected = best["epoch"]
    params, state, _ = load_checkpoint(out / "checkpoints" / f"epoch-{selected:03d}", (params, state))
    final = validate(engine, predict, params, sources, manifests, config, out=out, epoch=selected)
    for source in sources.values(): source.close()
    atomic_json(out / "selection.json", {"selected_epoch": selected, "epochs_run": len(history), "validation": final,
                "checkpoint_sha256": digest(out / "checkpoints" / f"epoch-{selected:03d}" / "state.npz"), "test_data_included": False})
    return read(out / "selection.json")


def rescore(root, run, epochs=None):
    """Re-run validation of saved checkpoints with the current manifests (e.g. after the benchmark manifest gained
    component_id for group-macro metrics); writes rescore.json, leaves history.jsonl and selection.json untouched."""
    import jax
    from pkanet.ogqt import initialize as initialize_ogqt, predict_shift
    from .gqt_multitask_replay import JointEngine
    from .trainer import load_checkpoint
    root = Path(root); out = run_dir(root, run); protocol = read(out / "protocol.json")
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb", "benchmark-val")}
    config = LoaderConfig(); norms = protocol["pinder_weight_normalization"]
    sources = {"pinder-val": PinderSource(_with_policy(manifests["pinder"]), config=config, norms=norms),
               "benchmark": PkpdbSource(_with_policy(manifests["benchmark-val"]), config=config, mask="eval_mask"),
               **({"pkpdb-val": PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask")} if select(manifests["pkpdb"], "val") else {})}
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **ARCHITECTURE); engine = JointEngine(params); state = engine.optimizer.init(params)
    predict = jax.jit(jax.vmap(predict_shift, in_axes=(None, 0)))
    folders = sorted((out / "checkpoints").glob("epoch-*")); rows = []
    for folder in folders:
        epoch = int(folder.name.split("-")[1])
        if epochs and epoch not in epochs: continue
        params, state, _ = load_checkpoint(folder, (params, state))
        rows.append({"epoch": epoch, "validation": validate(engine, predict, params, sources, manifests, config)})
        print(json.dumps({"epoch": epoch, "benchmark_mae": rows[-1]["validation"]["benchmark"]["overall"]["mae"],
                          "benchmark_groups": rows[-1]["validation"]["benchmark"]["overall"].get("mae_groups"),
                          "selection": rows[-1]["validation"]["selection"]}), flush=True)
    for source in sources.values(): source.close()
    atomic_json(out / "rescore.json", {"manifests": {d: digest(output(root, d) / MANIFEST) for d in manifests}, "epochs": rows})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pkatrain.production_train"); sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("train"); p.add_argument("run"); p.add_argument("--fraction", type=float, default=0.1); p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("rescore"); p.add_argument("runs", nargs="+")
    args = parser.parse_args(argv); root = Path(os.environ["PKABENCH_RUNTIME"])
    if args.action == "rescore":
        for run in args.runs: rescore(root, run)
        return
    if args.action == "train":
        result = train(root, args.run, args.fraction, args.smoke, args.batch)
        print(json.dumps({"selected_epoch": result["selected_epoch"], "selection": result["validation"]["selection"]}))


if __name__ == "__main__":
    main()
