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
from .production_aux import BURIAL_RSA_CUTS, INTERFACE_CAP, INTERFACE_THRESHOLDS, RSA_BOUND_TABLE, ContactTable
from .production_aux import TABLE as CONTACTS_TABLE
from .production_loading import MANIFEST, AuxPinderSource, PinderSource, PkpdbSource, normalization, select

SEED = 17
BATCH = 16
ARCHITECTURE = {"width": 44, "ff": 88}


def lr_scale(batch): return math.sqrt(batch / 8)


def predict_shift_separate(params, graph):
    """oGQT shift with separate query/context LayerNorm affine parameters in the site-token query attention
    (pkanet.site_query_norm's path, shift head only): the residue context uses params["query_context_norm"]."""
    import jax.numpy as jnp
    from pkanet.model import attend, encode_indexed, linear
    from pkanet.site_model import attend_sites_indexed
    residue = encode_indexed(params, graph); sr = graph["site_residue"]; mask = graph["site_mask"][:, None]
    tokens = (residue[sr] + params["groups"][graph["site_type"]]) * mask
    tokens = attend(params["query"], tokens, residue, graph["neighbors"][sr], graph["edge"][sr], graph["edge_mask"][sr],
                    graph["switch"][sr], context_norm=params["query_context_norm"]) * mask
    tokens = attend_sites_indexed(params["site"], tokens, graph)
    return (8 * jnp.tanh(linear(params["head"], tokens)[:, 0]))[graph["query_site"]]


class VariantEngine:
    """gqt_multitask_replay.JointEngine with the model function and the paired weighting as options (same optimizer,
    objectives and apply); used for --query-norm separate. The default configuration keeps JointEngine itself."""

    def __init__(self, params, predict, paired_weight="none", reduction="structure"):
        import jax
        import jax.numpy as jnp
        import optax
        from pkanet.model import PKPDB_PK_MOD
        from .gqt_regularization import decay_mask
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32); weighted = paired_weight == "interface"
        site_level = reduction in ("site", "pkai"); pkai = reduction == "pkai"
        # reduction "site": every loss is a mean over the batch's supervised sites; "pkai": experiment 48's weighted
        # MSEs, sum(w * err^2) / sum(w) over the batch's sites, w_burial on both state losses (pKPDB and PINDER AB/free)
        # and w_interface on the paired (Siamese) loss (pkatrain.pkai_joint_scale._weighted_mse).
        def site_mean(err, weight, mask):
            w = weight * mask; return jnp.sum(w * err) / jnp.maximum(jnp.sum(w), 1e-8)

        def paired_predictions(p, graphs):
            batch, branches = graphs["nodes"].shape[:2]
            flat = jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs)
            return jax.vmap(predict, in_axes=(None, 0))(p, flat).reshape(batch, branches, -1)

        def paired_objective(p, graphs, targets, mask, wb, wi, valid):
            predicted = paired_predictions(p, graphs)
            expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
            if site_level:
                m = mask * valid[:, None]; ws = wb if pkai else jnp.ones_like(wb); wp = wi if pkai else jnp.ones_like(wi)
                err = jnp.square(predicted - expected)
                state_loss = (site_mean(err[:, 0], ws, m) + site_mean(err[:, 1], ws, m)) / 2
                pair_loss = site_mean(jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1])), wp, m)
                return state_loss + pair_loss, (state_loss, pair_loss)
            state_error = jnp.square(predicted - expected) * mask[:, None, :]
            pair_error = jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1])) * mask
            if weighted: pair_error = pair_error * wi
            state_loss = jnp.sum(state_error, axis=(1, 2)) / jnp.maximum(2 * jnp.sum(mask, axis=1), 1)
            pair_loss = jnp.sum(pair_error, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
            denominator = jnp.maximum(jnp.sum(valid), 1)
            state_loss = jnp.sum(jnp.where(valid, state_loss, 0.0)) / denominator
            pair_loss = jnp.sum(jnp.where(valid, pair_loss, 0.0)) / denominator
            return state_loss + pair_loss, (state_loss, pair_loss)

        def pkpdb_objective(p, graphs, targets, eligible, *rest):
            *weight, valid = rest
            expected = targets - reference[graphs["query_group"]]
            predicted = jax.vmap(predict, in_axes=(None, 0))(p, graphs)
            if site_level:
                w = weight[0] if pkai else jnp.ones_like(targets)
                return site_mean(jnp.square(predicted - expected), w, eligible * valid[:, None])
            per_structure = jnp.sum(jnp.square(predicted - expected) * eligible, axis=1) / jnp.maximum(jnp.sum(eligible, axis=1), 1)
            return jnp.sum(jnp.where(valid, per_structure, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        def apply(p, state, pair_gradient, pk_gradient, rate):
            gradient = jax.tree.map(lambda a, b: a + b, pair_gradient, pk_gradient)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            return optax.apply_updates(p, jax.tree.map(lambda value: value * rate, updates)), state, finite

        self.predictions = jax.jit(paired_predictions)
        self.paired_value_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
        self.pkpdb_value_grad = jax.jit(jax.value_and_grad(pkpdb_objective))
        self.apply = jax.jit(apply)


def weight_table(values, points=512):
    """Mid-rank empirical CDF of site weights as (x, u) for jnp.interp: u = (#below + #equal / 2) / N, so ties (the
    0.05 floor of w_interface) share one percentile; at most `points` knots, endpoints kept."""
    values = np.sort(np.asarray(values, np.float64)); x, counts = np.unique(values, return_counts=True)
    u = (np.cumsum(counts) - counts / 2) / len(values)
    if len(x) > points:
        keep = np.unique(np.r_[np.linspace(0, len(x) - 1, points).round().astype(int), 0, len(x) - 1]); x, u = x[keep], u[keep]
    return x.astype(np.float32), u.astype(np.float32)


def site_weight_tables(manifests, norms, fraction, sample=3000, seed=0):
    """Percentile tables of the training sites' weights in batch units: PINDER w_burial / norm, w_interface / norm
    (PinderSource divides by the fraction's normalisation); pKPDB w_burial at train_mask sites (raw)."""
    from .production_loading import ProductionStore
    rng = np.random.default_rng(seed); out = {}
    for dataset in ("pinder", "pkpdb"):
        records = select(manifests[dataset], "train", fraction); pick = rng.choice(len(records), min(sample, len(records)), replace=False)
        store = ProductionStore(Path(manifests[dataset]["store"])); burial = []; interface = []
        for i in pick:
            raw = store.raw(records[i]["id"])
            if dataset == "pinder": burial.append(raw["w_burial"] / norms["burial"]); interface.append(raw["w_interface"] / norms["interface"])
            else: burial.append(np.nan_to_num(raw["w_burial"][raw["train_mask"].astype(bool)], nan=0.0))
        store.close()
        out[f"{dataset}_burial"] = weight_table(np.concatenate(burial))
        if interface: out["pinder_interface"] = weight_table(np.concatenate(interface))
    return out


class MaskedEngine:
    """--site-mask-pmin P (2026-10-10): instead of weighting the losses, sites are dropped at random each step with
    keep probability P + (1 - P) * u, u the site weight's training-pool percentile (site_weight_tables), so low-weight
    sites are masked most often and the mean keep fraction is (1 + P) / 2 whatever the weight's scale. w_burial
    percentiles mask the state losses (PINDER AB and free share one draw; pKPDB its own), w_interface percentiles the
    paired loss (an independent draw). Losses keep JointEngine's per-structure means over the kept sites; a structure
    with no kept site leaves that step's mean. Shared-norm oGQT; same optimizer and apply as JointEngine."""

    def __init__(self, params, tables, pmin):
        import jax
        import jax.numpy as jnp
        import optax
        from pkanet.model import PKPDB_PK_MOD
        from pkanet.ogqt import predict_shift
        from .gqt_regularization import decay_mask
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32); tables = {k: (jnp.asarray(x), jnp.asarray(u)) for k, (x, u) in tables.items()}

        def keep(key, weight, table):
            x, u = tables[table]; probability = pmin + (1 - pmin) * jnp.interp(weight, x, u)
            return jax.random.bernoulli(key, probability).astype(jnp.float32)

        def structure_mean(err, kept, valid):
            n = jnp.sum(kept, axis=1); per = jnp.sum(err * kept, axis=1) / jnp.maximum(n, 1)
            counted = valid & (n > 0); return jnp.sum(jnp.where(counted, per, 0.0)) / jnp.maximum(jnp.sum(counted), 1)

        def paired_predictions(p, graphs):
            batch, branches = graphs["nodes"].shape[:2]
            flat = jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs)
            return jax.vmap(predict_shift, in_axes=(None, 0))(p, flat).reshape(batch, branches, -1)

        def paired_objective(p, graphs, targets, mask, wb, wi, valid, key):
            predicted = paired_predictions(p, graphs)
            expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
            k1, k2 = jax.random.split(key)
            state_kept = mask * keep(k1, wb, "pinder_burial"); pair_kept = mask * keep(k2, wi, "pinder_interface")
            state_err = jnp.mean(jnp.square(predicted - expected), axis=1)
            pair_err = jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1]))
            state_loss = structure_mean(state_err, state_kept, valid); pair_loss = structure_mean(pair_err, pair_kept, valid)
            return state_loss + pair_loss, (state_loss, pair_loss)

        def pkpdb_objective(p, graphs, targets, eligible, wb, valid, key):
            expected = targets - reference[graphs["query_group"]]
            predicted = jax.vmap(predict_shift, in_axes=(None, 0))(p, graphs)
            return structure_mean(jnp.square(predicted - expected), eligible * keep(key, wb, "pkpdb_burial"), valid)

        def apply(p, state, pair_gradient, pk_gradient, rate):
            gradient = jax.tree.map(lambda a, b: a + b, pair_gradient, pk_gradient)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            return optax.apply_updates(p, jax.tree.map(lambda value: value * rate, updates)), state, finite

        self.predictions = jax.jit(paired_predictions)
        self.paired_value_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
        self.pkpdb_value_grad = jax.jit(jax.value_and_grad(pkpdb_objective))
        self.apply = jax.jit(apply)
        self.keep_fraction = jax.jit(lambda key, weight, table: jnp.mean(keep(key, weight, table)), static_argnums=2)


class DropoutEngine:
    """--dropout R (2026-10-10): JointEngine's objectives with oGQT's built-in residual dropout (pkanet.model.dropout on
    every attention and feed-forward output: residue blocks, site query, site-site block) at rate R during training.
    One key per complex is shared by its AB and free branches (coordinated dropout, gqt_diagnostic_campaign's layout);
    pKPDB structures draw their own keys. Validation and predictions are deterministic (no dropout)."""

    def __init__(self, params, rate):
        import jax
        import jax.numpy as jnp
        import optax
        from pkanet.model import PKPDB_PK_MOD
        from pkanet.ogqt import predict_shift
        from .gqt_regularization import decay_mask
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32); rate = float(rate)

        def flatten(graphs):
            batch, branches = graphs["nodes"].shape[:2]
            return jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs), batch, branches

        def paired_predictions(p, graphs):
            flat, batch, branches = flatten(graphs)
            return jax.vmap(predict_shift, in_axes=(None, 0))(p, flat).reshape(batch, branches, -1)

        def paired_objective(p, graphs, targets, mask, wb, wi, valid, key):
            del wb, wi
            flat, batch, branches = flatten(graphs); keys = jnp.repeat(jax.random.split(key, batch), branches, axis=0)
            predicted = jax.vmap(lambda graph, k: predict_shift(p, graph, key=k, dropout_rate=rate))(flat, keys).reshape(batch, branches, -1)
            expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
            state_error = jnp.square(predicted - expected) * mask[:, None, :]
            pair_error = jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1])) * mask
            state_loss = jnp.sum(state_error, axis=(1, 2)) / jnp.maximum(2 * jnp.sum(mask, axis=1), 1)
            pair_loss = jnp.sum(pair_error, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
            denominator = jnp.maximum(jnp.sum(valid), 1)
            state_loss = jnp.sum(jnp.where(valid, state_loss, 0.0)) / denominator
            pair_loss = jnp.sum(jnp.where(valid, pair_loss, 0.0)) / denominator
            return state_loss + pair_loss, (state_loss, pair_loss)

        def pkpdb_objective(p, graphs, targets, eligible, valid, key):
            keys = jax.random.split(key, graphs["nodes"].shape[0])
            predicted = jax.vmap(lambda graph, k: predict_shift(p, graph, key=k, dropout_rate=rate))(graphs, keys)
            expected = targets - reference[graphs["query_group"]]
            per_structure = jnp.sum(jnp.square(predicted - expected) * eligible, axis=1) / jnp.maximum(jnp.sum(eligible, axis=1), 1)
            return jnp.sum(jnp.where(valid, per_structure, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        def apply(p, state, pair_gradient, pk_gradient, rate_):
            gradient = jax.tree.map(lambda a, b: a + b, pair_gradient, pk_gradient)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            return optax.apply_updates(p, jax.tree.map(lambda value: value * rate_, updates)), state, finite

        self.predictions = jax.jit(paired_predictions)
        self.paired_value_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
        self.pkpdb_value_grad = jax.jit(jax.value_and_grad(pkpdb_objective))
        self.apply = jax.jit(apply)


def initialize_aux(params, seed, mode="ce", heads="both"):
    """Auxiliary heads on the final site embedding (production_aux): aux_burial and aux_interface, one logit each
    ("ce") or one per cut point / threshold ("ordinal": 3 and 6). Drawn from their own key, so the shared parameters
    equal the pKa-only model's for the same seed."""
    import jax
    import jax.numpy as jnp
    from .production_aux import BURIAL_RSA_CUTS, INTERFACE_THRESHOLDS
    width = ARCHITECTURE["width"]; kb, ki = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 27183))
    nb, ni = (len(BURIAL_RSA_CUTS), len(INTERFACE_THRESHOLDS)) if mode == "ordinal" else (1, 1)
    head = lambda key, out: {"w": jax.random.normal(key, (width, out), jnp.float32) / jnp.sqrt(float(width)), "b": jnp.zeros(out, jnp.float32)}
    return {**params, "aux_burial": head(kb, nb), **({"aux_interface": head(ki, ni)} if heads == "both" else {})}


class AuxEngine:
    """--aux-weight W (2026-10-11): JointEngine's pKa objectives plus W x (burial + interface) auxiliary losses
    (production_aux). Burial on the PINDER free branch and pKPDB train_mask sites; interface trained on the PINDER bound
    branch (default) or, with interface_branch="free", the free branch. Validation scores the interface head on both
    branches (auxiliary_metrics), the free branch testing it without the partner in the graph.
    mode "ce" (--aux-loss ce, default): one logit per head, cross-entropy against the soft targets 1 - clip(RSA) and
    log(1 + min(c, 8)) / log 9. mode "ordinal": mean BCE over the cumulative targets [RSA < cut] / [c > t]. Each
    auxiliary is averaged per site, then per structure, then over structures with at least one target (JointEngine's
    reduction). Optional residual dropout as DropoutEngine (one key per complex shared by AB and free). Batches from
    AuxPinderSource / PkpdbSource(aux=True): slot 3 = RSA (< 0 masked), slot 4 = contact score.
    heads "burial" (--aux-heads burial): burial only. heads "burial-siamese" (2026-10-11): no interface head; the burial
    head learns each state on its own branch (bound 1 - clip(RSA_bound), free 1 - clip(RSA_free); cross-entropy, the
    two averaged) plus a paired squared error on the predicted bound - free burial (sigmoid outputs) against
    clip(RSA_free) - clip(RSA_bound), mirroring the pKa state + Siamese losses; slot 4 then holds RSA_bound
    (production_aux rsa-bound-v1, < 0 masked)."""

    def __init__(self, params, weight, rate=0.0, mode="ce", interface_branch="bound", heads="both"):
        import jax
        import jax.numpy as jnp
        import optax
        from pkanet.model import PKPDB_PK_MOD, linear
        from pkanet.site_model import site_embeddings_indexed
        from .gqt_regularization import decay_mask
        from .production_aux import BURIAL_RSA_CUTS, INTERFACE_CAP, INTERFACE_THRESHOLDS
        if mode not in ("ce", "ordinal"): raise ValueError(mode)
        if interface_branch not in ("free", "bound"): raise ValueError(interface_branch)
        if heads not in ("both", "burial", "burial-siamese"): raise ValueError(heads)
        if heads == "burial-siamese" and mode != "ce": raise ValueError("Siamese burial is implemented for mode ce")
        self.heads = heads; use_interface = heads == "both"; siamese = heads == "burial-siamese"
        self.mode = mode; self.interface_branch = branch = 1 if interface_branch == "free" else 0
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params)))
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32); rate = float(rate); weight = float(weight)
        cuts = jnp.asarray(BURIAL_RSA_CUTS, jnp.float32); thresholds = jnp.asarray(INTERFACE_THRESHOLDS, jnp.float32)

        def burial_labels(rsa):
            return (rsa[..., None] < cuts).astype(jnp.float32) if mode == "ordinal" else (1.0 - jnp.clip(rsa, 0.0, 1.0))[..., None]

        def interface_labels(score):
            return (score[..., None] > thresholds).astype(jnp.float32) if mode == "ordinal" else \
                (jnp.log1p(jnp.minimum(score, INTERFACE_CAP)) / jnp.log1p(INTERFACE_CAP))[..., None]

        def predict_all(p, graph, key=None):
            tokens, query_site = site_embeddings_indexed(p, graph, key=key, dropout_rate=rate if key is not None else 0.0)
            return (8 * jnp.tanh(linear(p["head"], tokens)[:, 0]))[query_site], linear(p["aux_burial"], tokens)[query_site], \
                (linear(p["aux_interface"], tokens)[query_site] if use_interface else jnp.zeros((query_site.shape[0], 1), jnp.float32))

        def flatten(graphs):
            batch, branches = graphs["nodes"].shape[:2]
            return jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs), batch, branches

        def paired_all(p, graphs, key=None):
            flat, batch, branches = flatten(graphs)
            if key is None: out = jax.vmap(predict_all, in_axes=(None, 0))(p, flat)
            else: out = jax.vmap(lambda graph, k: predict_all(p, graph, k))(flat, jnp.repeat(jax.random.split(key, batch), branches, axis=0))
            return jax.tree.map(lambda value: value.reshape((batch, branches) + value.shape[1:]), out)

        def structure_mean(per_site, sites, valid):
            n = jnp.sum(sites, axis=1); per = jnp.sum(per_site * sites, axis=1) / jnp.maximum(n, 1)
            counted = valid & (n > 0); return jnp.sum(jnp.where(counted, per, 0.0)) / jnp.maximum(jnp.sum(counted), 1)

        def cross_entropy(logits, labels): return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels), axis=-1)

        def burial_loss(logits, rsa, sites, valid):
            return structure_mean(cross_entropy(logits, burial_labels(rsa)), sites * (rsa >= 0), valid)

        def paired_objective(p, graphs, targets, mask, rsa, contacts, valid, key):
            predicted, burial_logits, interface_logits = paired_all(p, graphs, key)
            expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
            state_error = jnp.square(predicted - expected) * mask[:, None, :]
            pair_error = jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1])) * mask
            state_loss = jnp.sum(state_error, axis=(1, 2)) / jnp.maximum(2 * jnp.sum(mask, axis=1), 1)
            pair_loss = jnp.sum(pair_error, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
            denominator = jnp.maximum(jnp.sum(valid), 1)
            state_loss = jnp.sum(jnp.where(valid, state_loss, 0.0)) / denominator
            pair_loss = jnp.sum(jnp.where(valid, pair_loss, 0.0)) / denominator
            sites = mask.astype(jnp.float32)
            if siamese:
                rsa_bound = contacts; both = sites * (rsa >= 0) * (rsa_bound >= 0)
                states = (burial_loss(burial_logits[:, 0], rsa_bound, sites, valid) + burial_loss(burial_logits[:, 1], rsa, sites, valid)) / 2
                change = jax.nn.sigmoid(burial_logits[:, 0, :, 0]) - jax.nn.sigmoid(burial_logits[:, 1, :, 0])
                expected_change = jnp.clip(rsa, 0.0, 1.0) - jnp.clip(rsa_bound, 0.0, 1.0)
                burial = states + structure_mean(jnp.square(change - expected_change), both, valid)
            else: burial = burial_loss(burial_logits[:, 1], rsa, sites, valid)
            interface = structure_mean(cross_entropy(interface_logits[:, branch], interface_labels(contacts)), sites, valid) if use_interface else 0.0
            return state_loss + pair_loss + weight * (burial + interface), (state_loss, pair_loss)

        def pkpdb_objective(p, graphs, targets, eligible, rsa, valid, key):
            keys = jax.random.split(key, graphs["nodes"].shape[0])
            predicted, burial_logits, _ = jax.vmap(lambda graph, k: predict_all(p, graph, k if rate else None))(graphs, keys)
            expected = targets - reference[graphs["query_group"]]
            per_structure = jnp.sum(jnp.square(predicted - expected) * eligible, axis=1) / jnp.maximum(jnp.sum(eligible, axis=1), 1)
            primary = jnp.sum(jnp.where(valid, per_structure, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
            return primary + weight * burial_loss(burial_logits, rsa, eligible.astype(jnp.float32), valid)

        def apply(p, state, pair_gradient, pk_gradient, rate_):
            gradient = jax.tree.map(lambda a, b: a + b, pair_gradient, pk_gradient)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            return optax.apply_updates(p, jax.tree.map(lambda value: value * rate_, updates)), state, finite

        self.predictions = jax.jit(lambda p, graphs: paired_all(p, graphs)[0])
        self.paired_auxiliary = jax.jit(lambda p, graphs: paired_all(p, graphs)[1:])
        self.site_auxiliary = jax.jit(jax.vmap(lambda p, graph: predict_all(p, graph)[1], in_axes=(None, 0)))
        self.paired_value_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
        self.pkpdb_value_grad = jax.jit(jax.value_and_grad(pkpdb_objective))
        self.apply = jax.jit(apply)


def _auroc(score, label):
    from scipy.stats import rankdata
    label = np.asarray(label, bool); n1 = int(label.sum()); n0 = len(label) - n1
    return float((rankdata(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def _recall_at_fpr(score, label, fpr=0.1):
    """Fraction of positives scored above the (1 - fpr) quantile of the negatives: false-negative rejection at a fixed
    false-positive budget."""
    label = np.asarray(label, bool)
    if not label.any() or label.all(): return None
    return float(np.mean(score[label] > np.quantile(score[~label], 1 - fpr)))


def auxiliary_metrics(engine, params, pinder_source, pinder_records, pkpdb_source, pkpdb_records, config):
    """Validation discrimination of the auxiliary heads, pooled over sites: the interface head on the PINDER bound
    branch ("interface_bound", partner in the graph) and on the free branch ("interface_free", no partner: the test of
    interface prediction from the unbound structure), against the same bound-state contact score; burial
    (PINDER free branch; pKPDB val train_mask sites). For every head and each class boundary (contact score > 0 / 0.5 /
    1 / 2 / 4 / 8; RSA < 0.1 / 0.25 / 0.5): AUROC and recall at a 10% false-positive rate, scored by the head's output
    (mode "ce") or the boundary's own logit ("ordinal"). Mode "ce" adds the mean cross-entropy against the soft target
    and the Spearman correlation with it; "ordinal" the mean BCE."""
    from scipy.stats import spearmanr
    from .production_aux import BURIAL_RSA_CUTS, INTERFACE_THRESHOLDS, burial_soft, interface_soft
    collected = {"interface_bound": ([], []), "interface_free": ([], []), "pinder_burial": ([], []), "pkpdb_burial": ([], []),
                 "pinder_burial_bound": ([], [])}; change = ([], [])
    for ids, batch in Prefetcher(pinder_source, pinder_source.policy.plans(pinder_records, np.random.default_rng(0)), config):
        graphs, _, mask, rsa, contacts, _, valid = batch
        burial, interface = map(np.asarray, engine.paired_auxiliary(params, graphs)); keep = mask & valid[:, None]
        for branch, name in ((0, "interface_bound"), (1, "interface_free")) if engine.heads == "both" else ():
            collected[name][0].append(interface[:, branch][keep]); collected[name][1].append(contacts[keep])
        b = keep & (rsa >= 0); collected["pinder_burial"][0].append(burial[:, 1][b]); collected["pinder_burial"][1].append(rsa[b])
        if engine.heads == "burial-siamese":
            bb = keep & (contacts >= 0); collected["pinder_burial_bound"][0].append(burial[:, 0][bb]); collected["pinder_burial_bound"][1].append(contacts[bb])
            both = b & (contacts >= 0); sig = lambda x: 1 / (1 + np.exp(-x.astype(float)))
            change[0].append(sig(burial[:, 0, :, 0][both]) - sig(burial[:, 1, :, 0][both])); change[1].append(np.clip(rsa[both], 0, 1) - np.clip(contacts[both], 0, 1))
    if pkpdb_source is not None:
        for ids, batch in Prefetcher(pkpdb_source, pkpdb_source.policy.plans(pkpdb_records, np.random.default_rng(0)), config):
            graphs, _, eligible, rsa, valid = batch; burial = np.asarray(engine.site_auxiliary(params, graphs))
            b = eligible.astype(bool) & valid[:, None] & (rsa >= 0)
            collected["pkpdb_burial"][0].append(burial[b]); collected["pkpdb_burial"][1].append(rsa[b])
    out = {}
    for name, (logits, values) in collected.items():
        if not logits: continue
        logits = np.concatenate(logits).astype(float); values = np.concatenate(values).astype(float); interface = name.startswith("interface")
        boundaries = INTERFACE_THRESHOLDS if interface else BURIAL_RSA_CUTS
        labels = np.stack([values > c if interface else values < c for c in boundaries], axis=1)
        if engine.mode == "ce":
            soft = interface_soft(values) if interface else burial_soft(values); z = logits[:, 0]
            ce = np.logaddexp(0, z) - soft * z
            entry = {"mean_cross_entropy": float(ce.mean()), "spearman_vs_target": float(spearmanr(z, soft)[0])}
            scores = [z] * len(boundaries)
        else:
            p = np.clip(1 / (1 + np.exp(-logits)), 1e-7, 1 - 1e-7)
            entry = {"mean_bce": float(np.mean(-(labels * np.log(p) + (1 - labels) * np.log(1 - p))))}
            scores = [logits[:, j] for j in range(len(boundaries))]
        entry["boundaries"] = [{"boundary": c, "positive_fraction": float(labels[:, j].mean()), "auroc": _auroc(scores[j], labels[:, j]),
                                "recall_at_fpr_0.1": _recall_at_fpr(scores[j], labels[:, j])} for j, c in enumerate(boundaries)]
        out[name] = {"sites": int(len(logits)), **entry}
    if change[0]:
        predicted, expected = np.concatenate(change[0]), np.concatenate(change[1]); buried = expected > 0.1
        out["burial_change"] = {"sites": int(len(expected)), "mae": float(np.mean(np.abs(predicted - expected))),
            "spearman": float(spearmanr(predicted, expected)[0]), "fraction_change_gt_0.1": float(buried.mean()),
            "auroc_change_gt_0.1": _auroc(predicted, buried), "recall_at_fpr_0.1_change_gt_0.1": _recall_at_fpr(predicted, buried)}
    return out


def make_model(seed, query_norm="shared", paired_weight="none", reduction="structure"):
    """(params, engine, predict): scratch oGQT and its engine; separate query norm starts as a copy of query norm1."""
    import jax
    import jax.numpy as jnp
    from pkanet.ogqt import initialize as initialize_ogqt, predict_shift
    from .gqt_multitask_replay import JointEngine
    params = initialize_ogqt(jax.random.PRNGKey(seed), **ARCHITECTURE)
    if query_norm == "separate":
        params = {**params, "query_context_norm": jnp.array(params["query"]["norm1"])}
        return params, VariantEngine(params, predict_shift_separate, paired_weight, reduction), jax.jit(jax.vmap(predict_shift_separate, in_axes=(None, 0)))
    if reduction != "structure":
        return params, VariantEngine(params, predict_shift, paired_weight, reduction), jax.jit(jax.vmap(predict_shift, in_axes=(None, 0)))
    engine = JointEngine(params)
    if paired_weight == "interface": engine = interface_weighted(engine)
    return params, engine, jax.jit(jax.vmap(predict_shift, in_axes=(None, 0)))


def interface_weighted(engine):
    """--paired-weight interface (2026-10-10): the paired (AB - free) squared error times the normalised w_interface,
    experiment 41's 'interface' arm formula (gqt_paired_pinder.PairedEngine); state loss, pKPDB loss and optimizer as
    JointEngine. Replaces engine.paired_value_grad only."""
    import jax
    import jax.numpy as jnp
    from pkanet.model import PKPDB_PK_MOD
    reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32)
    predictions = engine.predictions

    def paired_objective(p, graphs, targets, mask, wb, wi, valid):
        predicted = predictions(p, graphs)
        expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
        state_error = jnp.square(predicted - expected) * mask[:, None, :]
        pair_error = jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1])) * wi * mask
        state_loss = jnp.sum(state_error, axis=(1, 2)) / jnp.maximum(2 * jnp.sum(mask, axis=1), 1)
        pair_loss = jnp.sum(pair_error, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1)
        denominator = jnp.maximum(jnp.sum(valid), 1)
        state_loss = jnp.sum(jnp.where(valid, state_loss, 0.0)) / denominator
        pair_loss = jnp.sum(jnp.where(valid, pair_loss, 0.0)) / denominator
        return state_loss + pair_loss, (state_loss, pair_loss)
    engine.paired_value_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
    return engine


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


def epoch_plans(manifests, fraction, epoch, batch=BATCH, seed=SEED):
    """(PINDER plans, pKPDB plans) for one epoch, deterministic in (SEED, epoch); the pKPDB stream continues across epochs."""
    pinder, pkpdb = manifests["pinder"], manifests["pkpdb"]
    pair = _policy(pinder, batch).plans(select(pinder, "train", fraction), np.random.default_rng((seed, epoch, 1)))
    per_epoch = len(pair); start = (epoch - 1) * per_epoch; stream = []; cycle = 0
    pk_records = select(pkpdb, "train", fraction)
    while len(stream) < start + per_epoch:
        stream.extend(_policy(pkpdb, batch).plans(pk_records, np.random.default_rng((seed, cycle, 2)))); cycle += 1
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


def train(root, run, fraction=0.1, smoke=False, batch=BATCH, paired_weight="none", seed=SEED, query_norm="shared", reduction="structure",
          mask_pmin=None, dropout=0.0, aux_weight=None, aux_loss="ce", aux_interface_branch="bound", aux_heads="both"):
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
    protocol = {"version": "gqt-production-joint-v1", "run": run, "fraction": fraction, "seed": seed, "batch": batch,
        "architecture": ARCHITECTURE, "initialization": f"scratch (pkanet.ogqt.initialize, PRNGKey({seed}))",
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
        **({"paired_weight": "w_interface (experiment 41 interface arm formula)"} if paired_weight == "interface" else {}),
        **({"loss_reduction": {"site": "mean over the batch's supervised sites, unweighted",
                                "pkai": "experiment 48: sum(w err^2)/sum(w) over the batch's sites; w_burial on pKPDB and PINDER state losses, w_interface on the paired loss"}[reduction]}
           if reduction != "structure" else {}),
        **({"site_mask": f"stochastic site masking, keep probability {mask_pmin} + (1 - {mask_pmin}) * weight percentile (MaskedEngine); "
                         "w_burial on state losses, w_interface on the paired loss"} if mask_pmin is not None else {}),
        **({"query_norm": "separate query/context LayerNorm affine in the site-token query attention (query_context_norm, initialised as query norm1)"}
           if query_norm == "separate" else {}),
        **({"dropout": f"oGQT residual dropout {dropout} on attention and feed-forward outputs during training (DropoutEngine); "
                       "one key per complex shared by its AB and free branches"} if dropout else {}),
        **({"auxiliary": (f"AuxEngine ce burial-siamese: + {aux_weight} x (mean of bound-branch cross-entropy vs 1 - clip(RSA_bound) and "
                          "free-branch cross-entropy vs 1 - clip(RSA_free), + squared error of predicted bound - free burial vs clip(RSA_free) - "
                          "clip(RSA_bound); PINDER, rsa-bound-v1) + burial cross-entropy on pKPDB train_mask sites; one-logit head aux_burial "
                          "(pkatrain.production_aux)") if aux_heads == "burial-siamese" else
                          (f"AuxEngine ce burial-only: + {aux_weight} x burial cross-entropy vs soft target 1 - clip(RSA, 0, 1), PINDER free "
                          "branch and pKPDB train_mask sites; one-logit head aux_burial on the final site embedding (pkatrain.production_aux)")
                         if aux_heads == "burial" else (f"AuxEngine: + {aux_weight} x (ordinal burial BCE, RSA cuts {list(BURIAL_RSA_CUTS)}, PINDER free branch and pKPDB "
                          f"train_mask sites; ordinal interface BCE, contact-score thresholds {list(INTERFACE_THRESHOLDS)}, PINDER {aux_interface_branch} branch); "
                          "heads aux_burial/aux_interface on the final site embedding (pkatrain.production_aux)") if aux_loss == "ordinal" else
                         (f"AuxEngine ce: + {aux_weight} x (burial cross-entropy vs soft target 1 - clip(RSA, 0, 1), PINDER free branch and pKPDB "
                          f"train_mask sites; interface cross-entropy vs soft target log(1 + min(c, {INTERFACE_CAP:g})) / log({INTERFACE_CAP + 1:g}), "
                          f"PINDER {aux_interface_branch} branch); one-logit heads aux_burial/aux_interface on the final site embedding (pkatrain.production_aux)"),
            ("rsa_bound_verification_sha256" if aux_heads == "burial-siamese" else "contacts_verification_sha256"):
                ContactTable(root, RSA_BOUND_TABLE if aux_heads == "burial-siamese" else CONTACTS_TABLE).verification_sha256} if aux_weight is not None else {}),
        "pinder_weight_normalization": norms, "smoke": smoke, "test_data_included": False,
        "manifests": {d: digest(output(root, d) / MANIFEST) for d in manifests}, "code": code_hashes()}
    if (out / "protocol.json").exists():
        stored = read(out / "protocol.json")
        if {k: v for k, v in stored.items() if k != "code"} != json.loads(json.dumps({k: v for k, v in protocol.items() if k != "code"})):
            raise AssertionError(f"{out} was registered with a different protocol")
    else: atomic_json(out / "protocol.json", protocol)
    config = LoaderConfig()
    contacts = ContactTable(root, RSA_BOUND_TABLE if aux_heads == "burial-siamese" else CONTACTS_TABLE) if aux_weight is not None else None
    train_pinder = AuxPinderSource(_with_policy(manifests["pinder"], batch), contacts, config=config, norms=norms) if contacts else \
        PinderSource(_with_policy(manifests["pinder"], batch), config=config, norms=norms)
    sources = {"train": JointSource(train_pinder, PkpdbSource(_with_policy(manifests["pkpdb"], batch), config=config,
                                    weights=reduction != "structure" or mask_pmin is not None, aux=contacts is not None)),
               "pinder-val": PinderSource(_with_policy(manifests["pinder"]), config=config, norms=norms),
               "benchmark": PkpdbSource(_with_policy(manifests["benchmark-val"]), config=config, mask="eval_mask"),
               **({"pkpdb-val": PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask")} if select(manifests["pkpdb"], "val") else {})}
    if (reduction != "structure" or mask_pmin is not None) and chunked(batch): raise ValueError("this loss is not chunk-additive; use batch <= 64")
    params, engine, predict = make_model(seed, query_norm, paired_weight, reduction); state = engine.optimizer.init(params)
    step_key = None
    if aux_weight is not None:
        if query_norm != "shared" or paired_weight != "none" or reduction != "structure" or mask_pmin is not None:
            raise ValueError("auxiliary heads combine with the defaults and dropout only")
        if chunked(batch): raise ValueError("the auxiliary objective runs the unchunked path; use batch <= 64")
        if aux_heads != "both" and aux_loss != "ce": raise ValueError("burial-only auxiliaries are implemented for --aux-loss ce")
        params = initialize_aux(params, seed, aux_loss, aux_heads)
        engine = AuxEngine(params, aux_weight, dropout, aux_loss, aux_interface_branch, aux_heads); state = engine.optimizer.init(params)
        step_key = jax.random.PRNGKey(seed + 200003)
    elif dropout:
        if query_norm != "shared" or paired_weight != "none" or reduction != "structure" or mask_pmin is not None:
            raise ValueError("dropout combines with the defaults only")
        if chunked(batch): raise ValueError("dropout runs the unchunked path; use batch <= 64")
        engine = DropoutEngine(params, dropout); state = engine.optimizer.init(params); step_key = jax.random.PRNGKey(seed + 200003)
    if mask_pmin is not None:
        if query_norm != "shared" or paired_weight != "none" or reduction != "structure": raise ValueError("site masking combines with the defaults only")
        tables = site_weight_tables(manifests, norms, fraction)
        engine = MaskedEngine(params, tables, mask_pmin); state = engine.optimizer.init(params); step_key = jax.random.PRNGKey(seed + 100003)
        atomic_json(out / "site-mask-tables.json", {k: {"x": x.tolist(), "u": u.tolist()} for k, (x, u) in tables.items()})
    history = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()] if (out / "history.jsonl").exists() else []
    if history:
        last = history[-1]["epoch"]; params, state, _ = load_checkpoint(out / "checkpoints" / f"epoch-{last:03d}", (params, state))
    best = min(history, key=lambda h: h["validation"]["selection"]) if history else None; stale = 0
    if history:
        stale = max(0, history[-1]["epoch"] - best["epoch"])
    epochs = 2 if smoke else EPOCHS
    for epoch in range(len(history) + 1, epochs + 1):
        if stale >= PATIENCE: break
        pair, pk = epoch_plans(manifests, fraction, epoch, batch, seed)
        if smoke: pair, pk = pair[:5], pk[:5]
        specs = list(zip(pair, pk)); deferred = DeferredScalars(every=50); began = time.time()
        prefetcher = Prefetcher(sources["train"], specs, config, to_device if not chunked(batch) else host_chunks)
        for number, (_, (pair_batch, pk_batch)) in enumerate(prefetcher, 1):
            rate = jnp.asarray(scale * learning_rate(epoch, number, len(specs)), jnp.float32)
            if step_key is not None:
                graphs, targets, mask, wb, wi, _, valid = pair_batch
                k_pair, k_pk = jax.random.split(jax.random.fold_in(jax.random.fold_in(step_key, epoch), number))
                (pair_total, (state_loss, pair_loss)), pair_gradient = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, valid, k_pair)
                pk_loss, pk_gradient = engine.pkpdb_value_grad(params, *pk_batch, k_pk)
            elif not chunked(batch):
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
    if contacts is not None:
        aux_sources = (AuxPinderSource(_with_policy(manifests["pinder"]), contacts, config=config, norms=norms),
                       PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask", aux=True) if "pkpdb-val" in sources else None)
        final["auxiliary"] = auxiliary_metrics(engine, params, aux_sources[0], select(manifests["pinder"], "val"), aux_sources[1],
                                               select(manifests["pkpdb"], "val"), config)
        for source in aux_sources:
            if source is not None: source.close()
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
    if "auxiliary" in protocol: raise NotImplementedError("rescore does not rebuild auxiliary-head runs")
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb", "benchmark-val")}
    config = LoaderConfig(); norms = protocol["pinder_weight_normalization"]
    sources = {"pinder-val": PinderSource(_with_policy(manifests["pinder"]), config=config, norms=norms),
               "benchmark": PkpdbSource(_with_policy(manifests["benchmark-val"]), config=config, mask="eval_mask"),
               **({"pkpdb-val": PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask")} if select(manifests["pkpdb"], "val") else {})}
    reduction = "structure" if "loss_reduction" not in protocol else ("pkai" if protocol["loss_reduction"].startswith("experiment 48") else "site")
    params, engine, predict = make_model(protocol.get("seed", SEED), "separate" if "query_norm" in protocol else "shared",
                                         "interface" if "paired_weight" in protocol else "none", reduction)
    state = engine.optimizer.init(params)
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


def aux_rescore(root, run):
    """Re-score an auxiliary-head run's selected checkpoint with the current auxiliary_metrics (both interface branches);
    writes aux-rescore.json, leaves selection.json untouched."""
    import jax
    from .trainer import load_checkpoint
    root = Path(root); out = run_dir(root, run); protocol = read(out / "protocol.json"); selection = read(out / "selection.json")
    text = protocol["auxiliary"]; mode = "ce" if text.startswith("AuxEngine ce") else "ordinal"; heads = "burial" if text.startswith("AuxEngine ce burial-only") else "burial-siamese" if text.startswith("AuxEngine ce burial-siamese") else "both"
    weight = float(text.split("+ ")[1].split(" x")[0]); branch = "free" if heads == "both" and "PINDER free branch); " in text.split("interface")[1] else "bound"
    dropout = float(protocol["dropout"].split("dropout ")[1].split(" ")[0]) if "dropout" in protocol else 0.0
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb")}
    config = LoaderConfig(); norms = protocol["pinder_weight_normalization"]
    contacts = ContactTable(root, RSA_BOUND_TABLE if heads == "burial-siamese" else CONTACTS_TABLE)
    params, _, _ = make_model(protocol["seed"]); params = initialize_aux(params, protocol["seed"], mode, heads)
    engine = AuxEngine(params, weight, dropout, mode, branch, heads); state = engine.optimizer.init(params)
    params, state, _ = load_checkpoint(out / "checkpoints" / f"epoch-{selection['selected_epoch']:03d}", (params, state))
    sources = (AuxPinderSource(_with_policy(manifests["pinder"]), contacts, config=config, norms=norms),
               PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask", aux=True))
    metrics = auxiliary_metrics(engine, params, sources[0], select(manifests["pinder"], "val"), sources[1], select(manifests["pkpdb"], "val"), config)
    for source in sources: source.close()
    result = {"run": run, "selected_epoch": selection["selected_epoch"], "mode": mode, "weight": weight, "interface_trained_branch": branch,
              "contacts_verification_sha256": contacts.verification_sha256, "auxiliary": metrics}
    atomic_json(out / "aux-rescore.json", result); return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pkatrain.production_train"); sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("train"); p.add_argument("run"); p.add_argument("--fraction", type=float, default=0.1); p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--paired-weight", choices=("none", "interface"), default="none")
    p.add_argument("--seed", type=int, default=SEED); p.add_argument("--query-norm", choices=("shared", "separate"), default="shared")
    p.add_argument("--loss-reduction", choices=("structure", "site", "pkai"), default="structure")
    p.add_argument("--site-mask-pmin", type=float, default=None); p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--aux-weight", type=float, default=None); p.add_argument("--aux-loss", choices=("ce", "ordinal"), default="ce")
    p.add_argument("--aux-interface-branch", choices=("bound", "free"), default="bound")
    p.add_argument("--aux-heads", choices=("both", "burial", "burial-siamese"), default="both")
    p = sub.add_parser("rescore"); p.add_argument("runs", nargs="+")
    p = sub.add_parser("aux-rescore"); p.add_argument("runs", nargs="+")
    args = parser.parse_args(argv); root = Path(os.environ["PKABENCH_RUNTIME"])
    if args.action == "rescore":
        for run in args.runs: rescore(root, run)
        return
    if args.action == "aux-rescore":
        for run in args.runs:
            r = aux_rescore(root, run)
            print(json.dumps({"run": run, **{n: [round(b["auroc"], 3) for b in v["boundaries"]] for n, v in r["auxiliary"].items()}}), flush=True)
        return
    if args.action == "train":
        result = train(root, args.run, args.fraction, args.smoke, args.batch, args.paired_weight, args.seed, args.query_norm, args.loss_reduction, args.site_mask_pmin,
                       args.dropout, args.aux_weight, args.aux_loss, args.aux_interface_branch,
                       args.aux_heads)
        print(json.dumps({"selected_epoch": result["selected_epoch"], "selection": result["validation"]["selection"]}))


if __name__ == "__main__":
    main()
