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
from .production_ph import THRESHOLDS as PH_THRESHOLDS
from .production_depth import INJECT_SCALE, add_injection, deepen, model_functions
from .production_inputs import BACKBONE_WIDTH, ORIENTATION_EDGE, BackboneTable, ConcatTable, add_chi_head, add_dihedral_inputs, add_geometry_heads, add_hbond_inputs, add_orientation, chi1_targets, geometry_targets, \
    input_functions, DIHEDRAL_FEATURES, HBOND_FEATURES
from .production_sidechains import WIDTH as SIDECHAIN_WIDTH, SidechainTable
from .production_packed import SOURCES as SIDECHAIN_SOURCES, PackedSidechainTable, sidechain_tables
from . import production_gate
from . import production_residue
from .production_loading import MANIFEST, AuxPinderSource, PinderSource, PkpdbSource, normalization, select

SEED = 17
BATCH = 16
ARCHITECTURE = {"width": 44, "ff": 88}


def lr_scale(batch): return math.sqrt(batch / 8)


def stretched_learning_rate(epoch, batch, batches, epochs, *, start=1e-3, end=1e-5):
    """gqt_crop_radius.learning_rate's shape over `epochs`: hold `start` for the first half, cosine to `end` by the
    last epoch (epochs = 20 reproduces it exactly)."""
    hold = epochs // 2
    if epoch <= hold: return float(start)
    within = (batch - 1) / max(batches - 1, 1)
    fraction = float(np.clip(((epoch - hold - 1) + within) / (epochs - hold - 1), 0.0, 1.0))
    return float(end + 0.5 * (start - end) * (1 + np.cos(np.pi * fraction)))


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

    def __init__(self, params, rate, predict_shift=None):
        import jax
        import jax.numpy as jnp
        import optax
        from pkanet.model import PKPDB_PK_MOD
        if predict_shift is None: from pkanet.ogqt import predict_shift
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
    width = params["head"]["w"].shape[0]; kb, ki = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 27183))
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
    (production_aux rsa-bound-v1, < 0 masked).
    ph_weight (--ph-weight, 2026-10-12): + ph_weight x structure-level experimental-pH cumulative BCE from the pooled
    head aux_ph (production_ph); PINDER AB branch and pKPDB, each structure weighted by eligibility x parent weight.
    The objectives then take ph=(target, weight) per structure (production_ph.PhTable.arrays). ph_probe (--ph-probe):
    the pH head reads stop_gradient(tokens) and trains under production_ph.separate_optimizer (trunk untouched).
    bind_weight (--bind-weight, 2026-10-12): + bind_weight x the paired binding head aux_bind (production_bind: weights
    for the Hadamard product / difference of the normalised AB and free site tokens from a small block on the two
    tokens) against the interface soft target, PINDER pKa-labelled sites; operand bind_operand."""

    def __init__(self, params, weight, rate=0.0, mode="ce", interface_branch="bound", heads="both", site_embeddings=None, geometry_weight=0.0, chi_weight=0.0,
                 ph_weight=0.0, ph_probe=False, bind_weight=0.0, bind_operand="product", shift_weights=None, gate_weight=0.0, gate_cys_weight=10.0,
                 residue_burial=False, residue_edge_weight=0.0, residue_burial_siamese=False, decoy_retype_weight=0.0, decoy_batch_weight=0.0,
                 within_weight=0.0, swap_weight=0.0, swap_mode="class", hotspot_mask=None, alt_rank_weight=0.0, alt_rank_margin=0.0,
                 alt_cons_edge_weight=0.0, alt_cons_pka_weight=0.0, alt_cons_stopgrad=False, alt_edge_only=False, quartet_weight=0.0,
                 chain_quartet_weight=0.0, chain_quartet_tau=1.0, chain_quartet_k=32):
        import jax
        import jax.numpy as jnp
        import optax
        from pkanet.model import PKPDB_PK_MOD, linear
        site_embeddings_indexed = site_embeddings
        if site_embeddings_indexed is None: from pkanet.site_model import site_embeddings_indexed
        from .gqt_regularization import decay_mask
        from .production_aux import BURIAL_RSA_CUTS, INTERFACE_CAP, INTERFACE_THRESHOLDS
        if mode not in ("ce", "ordinal"): raise ValueError(mode)
        if interface_branch not in ("free", "bound"): raise ValueError(interface_branch)
        if heads not in ("both", "burial", "burial-siamese"): raise ValueError(heads)
        if heads == "burial-siamese" and mode != "ce": raise ValueError("Siamese burial is implemented for mode ce")
        self.heads = heads; use_interface = heads == "both"; siamese = heads == "burial-siamese"
        self.mode = mode; self.interface_branch = branch = 1 if interface_branch == "free" else 0
        from . import production_ph
        make = lambda tree: optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(tree)))
        self.optimizer = production_ph.separate_optimizer(params, make) if ph_probe else make(params)
        reference = jnp.asarray(PKPDB_PK_MOD, jnp.float32); rate = float(rate); weight = float(weight)
        cuts = jnp.asarray(BURIAL_RSA_CUTS, jnp.float32); thresholds = jnp.asarray(INTERFACE_THRESHOLDS, jnp.float32)

        def burial_labels(rsa):
            return (rsa[..., None] < cuts).astype(jnp.float32) if mode == "ordinal" else (1.0 - jnp.clip(rsa, 0.0, 1.0))[..., None]

        def interface_labels(score):
            return (score[..., None] > thresholds).astype(jnp.float32) if mode == "ordinal" else \
                (jnp.log1p(jnp.minimum(score, INTERFACE_CAP)) / jnp.log1p(INTERFACE_CAP))[..., None]

        geometry_weight = float(geometry_weight); chi_weight = float(chi_weight); structural = bool(geometry_weight or chi_weight)
        self.ph_weight = ph_weight = float(ph_weight)
        self.bind_weight = bind_weight = float(bind_weight); self.bind_operand = bind_operand
        from . import production_bind
        from . import production_gate
        paired_delta = "delta_head" in params
        self.gate_weight = gate_weight = float(gate_weight); gate_cys_weight = float(gate_cys_weight)
        from . import production_residue
        residue_burial_siamese = bool(residue_burial_siamese); residue_burial = bool(residue_burial) or residue_burial_siamese
        self.residue_edge_weight = residue_edge_weight = float(residue_edge_weight)
        if (residue_burial or residue_edge_weight) and ("aa_types" not in params or heads != "burial" or mode != "ce"):
            raise ValueError("--residue-burial / --residue-edge-weight need --all-residue-tokens and --aux-heads burial (ce)")
        if gate_weight and "aa_types" not in params: raise ValueError("--gate-weight needs --all-residue-tokens")
        # decoy-rank-v1 (2026-10-29): margin ranking of true vs decoy partners on the residue edge head (production_residue)
        self.decoy_retype_weight = decoy_retype_weight = float(decoy_retype_weight); self.decoy_batch_weight = decoy_batch_weight = float(decoy_batch_weight)
        # negatives-v1 (2026-10-29): within-complex listwise and same-class swap negatives (production_residue)
        self.within_weight = within_weight = float(within_weight); self.swap_weight = swap_weight = float(swap_weight)
        if swap_mode not in ("class", "curv"): raise ValueError(swap_mode)
        negatives = bool(within_weight or swap_weight)
        decoys = bool(decoy_retype_weight or decoy_batch_weight) or negatives
        if decoys and not residue_edge_weight: raise ValueError("--decoy-retype-weight / --decoy-batch-weight / --within-weight / --swap-weight need --residue-edge-weight")
        # hotspot-mask-v1 (2026-10-30): masked third pass (production_residue.mask_nodes) replaces the re-typed pass; the
        # free edge BCE, its distillation toward the bound branch and the within listwise term read the masked embeddings
        self.hotspot_mask = hotspot_mask; hotspot = hotspot_mask is not None
        if hotspot:
            if hotspot_mask not in production_residue.MASK_ORACLES: raise ValueError(hotspot_mask)
            if not residue_edge_weight: raise ValueError("--mask-oracle needs --residue-edge-weight")
            if decoy_retype_weight: raise AssertionError("--mask-oracle replaces the re-typed pass: --decoy-retype-weight must be 0")
            if decoy_batch_weight or swap_weight: raise ValueError("--mask-oracle is defined with the within listwise term only (no --decoy-batch-weight / --swap-weight)")
            decoys = True
        # min-aug-v1 (2026-10-31): alt-vs-crystal ranking on the edge head (production_alt): an extra free-branch pass on the
        # crystal geometry (loader xtal_ fields); + w x mean softplus(m - (l_alt - l_xtal)) over alt visits' contact pairs,
        # gradient into both passes
        self.alt_rank_weight = alt_rank_weight = float(alt_rank_weight); alt_rank_margin = float(alt_rank_margin)
        if alt_rank_weight and not residue_edge_weight: raise ValueError("--alt-rank-weight needs --residue-edge-weight")
        from . import production_alt
        # anm-aug-v1 (2026-10-31): symmetric alt-vs-crystal consistency on the edge logits and the free-state pKa (production_alt D),
        # computed from the same crystal free pass as the rank term
        self.alt_cons_edge_weight = alt_cons_edge_weight = float(alt_cons_edge_weight); self.alt_cons_pka_weight = alt_cons_pka_weight = float(alt_cons_pka_weight)
        alt_cons_stopgrad = bool(alt_cons_stopgrad); consistency = bool(alt_cons_edge_weight or alt_cons_pka_weight)
        if consistency and not residue_edge_weight: raise ValueError("--alt-cons-*-weight needs --residue-edge-weight")
        if alt_cons_stopgrad and not consistency: raise ValueError("--alt-cons-stopgrad needs --alt-cons-edge-weight or --alt-cons-pka-weight")
        if consistency and paired_delta: raise ValueError("--alt-cons-*-weight is defined without the paired delta head")
        # anm-edge-v1 (2026-11-01): --alt-edge-only (production_alt E): pKa / burial / gate read the crystal free pass, the alt
        # free pass (and the hotspot-masked pass on the alt geometry) feeds only the edge head
        self.alt_edge_only = alt_edge_only = bool(alt_edge_only)
        if alt_edge_only:
            if alt_cons_pka_weight: raise ValueError("--alt-cons-pka-weight is refused with --alt-edge-only (the pKa never sees the alt geometry)")
            if not residue_edge_weight: raise ValueError("--alt-edge-only needs --residue-edge-weight")
            if paired_delta: raise ValueError("--alt-edge-only is defined without the paired delta head")
        # quartet-rank-v1 (2026-11-15): + quartet_weight x mean softplus(1 - (s_ij + s_kl - s_il - s_kj)) on the unmasked free-branch
        # aux_redge logits (production_residue.quartet_terms); aux slots 10-12 = (softplus loss, accuracy, covered complex fraction)
        self.quartet_weight = quartet_weight = float(quartet_weight)
        if quartet_weight < 0: raise ValueError("--quartet-weight must be >= 0")
        if quartet_weight:
            if not residue_edge_weight or not within_weight: raise ValueError("--quartet-weight needs --residue-edge-weight and --within-weight")
            if alt_rank_weight or consistency or alt_edge_only: raise ValueError("--quartet-weight is defined without the alt-geometry terms")
        # quartet-rank-v2 (2026-11-19): + chain_quartet_weight x mean softplus(1 - (S(A1,B1) + S(A2,B2) - S(A1,B2) - S(A2,B1))), S = soft top-k
        # pooled free-branch aux_redge logits over surface pairs (production_residue.chain_quartet_terms); batch fields chain_role /
        # chain_quartet (host side, chain_quartet_batch) are removed from graphs before anything else; aux slots 13-22 (CHAIN_SLOT)
        self.chain_quartet_weight = chain_weight = float(chain_quartet_weight); chain_tau = float(chain_quartet_tau); chain_k = int(chain_quartet_k)
        if chain_weight < 0: raise ValueError("--chain-quartet-weight must be >= 0")
        if chain_weight:
            if not residue_edge_weight or not within_weight: raise ValueError("--chain-quartet-weight needs --residue-edge-weight and --within-weight")
            if alt_rank_weight or consistency or alt_edge_only: raise ValueError("--chain-quartet-weight is defined without the alt-geometry terms")
            if chain_tau <= 0 or chain_k < 1: raise ValueError(("--chain-quartet-tau must be > 0 and --chain-quartet-k >= 1", chain_tau, chain_k))

        def predict_all(p, graph, key=None):
            tokens, query_site = site_embeddings_indexed(p, graph, key=key, dropout_rate=rate if key is not None else 0.0)
            # side-chain auxiliaries (production_inputs): predictions and targets from the untransformed graph
            geometry = {}
            if geometry_weight:
                geometry.update(zip(("position", "partner", "target", "mask", "score", "partner_mask"),
                                    (linear(p["aux_geom_pos"], tokens)[query_site], linear(p["aux_partner"], tokens)[query_site, 0], *geometry_targets(graph))))
            if chi_weight:
                geometry.update(zip(("chi", "chi_class", "chi_mask"), (linear(p["aux_chi"], tokens)[query_site], *chi1_targets(graph))))
            ph = production_ph.pooled_logits(p["aux_ph"], jax.lax.stop_gradient(tokens) if ph_probe else tokens, graph["site_mask"]) if ph_weight else jnp.zeros((len(production_ph.THRESHOLDS),), jnp.float32)
            # titratable gate (production_gate): one logit per all-residue token row, with its labels and weights
            gate = (production_gate.logits(p, tokens), *production_gate.targets(graph, gate_cys_weight)) if gate_weight else (jnp.zeros(1, jnp.float32),) * 3
            # per-residue burial and residue edges (production_residue), PINDER graphs that carry the residue table
            residue = {}; n_nodes = graph["nodes"].shape[0]
            if residue_burial and "rsa_residue" in graph:
                residue["burial"] = linear(p["aux_burial"], tokens[:n_nodes]); residue["rsa"] = graph["rsa_residue"]
            if residue_edge_weight and "edge_a" in graph:
                z = production_residue.residue_embedding(p, tokens, n_nodes)
                residue["edge"] = production_residue.pair_logits(p, z, graph["edge_a"], graph["edge_b"])
                residue["edge_target"] = graph["edge_target"]; residue["edge_weight"] = graph["edge_weight"]
                if decoy_batch_weight or negatives: residue["z"] = z
            return (8 * jnp.tanh(linear(p["head"], tokens)[:, 0]))[query_site], linear(p["aux_burial"], tokens)[query_site], \
                (linear(p["aux_interface"], tokens)[query_site] if use_interface else jnp.zeros((query_site.shape[0], 1), jnp.float32)), geometry, ph, \
                tokens[query_site] if bind_weight or paired_delta else jnp.zeros((query_site.shape[0], 1), jnp.float32), gate, residue

        def geometry_loss(g, sites, valid):
            """Weighted side-chain auxiliaries, each averaged per structure over sites with a target: geometry =
            functional-centroid squared error (A / 10, mean over xyz) + partner-score cross-entropy; chi = chi1-class
            softmax cross-entropy."""
            total = 0.0
            if geometry_weight:
                total = total + geometry_weight * (structure_mean(jnp.mean(jnp.square(g["position"] - g["target"]), axis=-1), sites * g["mask"], valid)
                                                   + structure_mean(optax.sigmoid_binary_cross_entropy(g["partner"], g["score"]), sites * g["partner_mask"], valid))
            if chi_weight:
                total = total + chi_weight * structure_mean(optax.softmax_cross_entropy_with_integer_labels(g["chi"], g["chi_class"]), sites * g["chi_mask"], valid)
            return total

        def flatten(graphs):
            batch, branches = graphs["nodes"].shape[:2]
            return jax.tree.map(lambda value: value.reshape((batch * branches,) + value.shape[2:]), graphs), batch, branches

        def paired_all(p, graphs, key=None):
            flat, batch, branches = flatten(graphs)
            if key is None: out = jax.vmap(predict_all, in_axes=(None, 0))(p, flat)
            else: out = jax.vmap(lambda graph, k: predict_all(p, graph, k))(flat, jnp.repeat(jax.random.split(key, batch), branches, axis=0))
            out = jax.tree.map(lambda value: value.reshape((batch, branches) + value.shape[1:]), out)
            if paired_delta:
                from .production_position import delta_correction
                correction = delta_correction(p["delta_head"], out[5][:,0], out[5][:,1])
                shift = out[0].at[:,0].add(correction / 2).at[:,1].add(-correction / 2)
                out = (shift, *out[1:])
            return out

        def structure_mean(per_site, sites, valid):
            n = jnp.sum(sites, axis=1); per = jnp.sum(per_site * sites, axis=1) / jnp.maximum(n, 1)
            counted = valid & (n > 0); return jnp.sum(jnp.where(counted, per, 0.0)) / jnp.maximum(jnp.sum(counted), 1)

        def cross_entropy(logits, labels): return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels), axis=-1)

        def burial_loss(logits, rsa, sites, valid):
            return structure_mean(cross_entropy(logits, burial_labels(rsa)), sites * (rsa >= 0), valid)

        def gate_loss(gate, valid):
            """Weighted BCE per structure over its token rows, then over structures (production_gate)."""
            logit, label, w = gate; total = jnp.sum(w, axis=-1)
            per = jnp.sum(optax.sigmoid_binary_cross_entropy(logit, label) * w, axis=-1) / jnp.maximum(total, 1e-6)
            counted = valid & (total > 0); return jnp.sum(jnp.where(counted, per, 0.0)) / jnp.maximum(jnp.sum(counted), 1)

        def decoy_edges(p, graphs, key):
            """Edge logits of the re-typed decoy pass: free branch, nodes re-typed (production_residue.retype_nodes), the
            same per-complex dropout key as the AB / free passes, the sampled pairs edge_a / edge_b."""
            free = jax.tree.map(lambda x: x[:, 1], graphs)
            free = {**free, "nodes": jax.vmap(production_residue.retype_nodes)(free["nodes"], free["decoy_idx"], free["decoy_type"], free["decoy_mask"])}

            def one(graph, k):
                tokens = site_embeddings_indexed(p, graph, key=k, dropout_rate=rate if k is not None else 0.0)[0]
                z = production_residue.residue_embedding(p, tokens, graph["nodes"].shape[0])
                return production_residue.pair_logits(p, z, graph["edge_a"], graph["edge_b"])
            if key is None: return jax.vmap(lambda graph: one(graph, None))(free)
            return jax.vmap(one)(free, jax.random.split(key, free["nodes"].shape[0]))

        def masked_embedding(p, graphs, key):
            """Residue embeddings z (B, n, W) of the hotspot-masked pass: free branch, type one-hot zeroed outside
            mask_keep_idx (production_residue.mask_nodes), the same per-complex dropout key as the AB / free passes."""
            free = jax.tree.map(lambda x: x[:, 1], graphs)
            free = {**free, "nodes": jax.vmap(production_residue.mask_nodes)(free["nodes"], free["mask_keep_idx"], free["mask_keep_mask"])}

            def one(graph, k):
                tokens = site_embeddings_indexed(p, graph, key=k, dropout_rate=rate if k is not None else 0.0)[0]
                return production_residue.residue_embedding(p, tokens, graph["nodes"].shape[0])
            if key is None: return jax.vmap(lambda graph: one(graph, None))(free)
            return jax.vmap(one)(free, jax.random.split(key, free["nodes"].shape[0]))

        def crystal_edges(p, crystal, key):
            """Edge logits (B, P) of the crystal free-branch pass (production_alt.split_crystal) on the sampled pairs, the same
            per-complex dropout key as the AB / free / masked passes."""
            def one(graph, k):
                tokens = site_embeddings_indexed(p, graph, key=k, dropout_rate=rate if k is not None else 0.0)[0]
                z = production_residue.residue_embedding(p, tokens, graph["nodes"].shape[0])
                return production_residue.pair_logits(p, z, graph["edge_a"], graph["edge_b"])
            if key is None: return jax.vmap(lambda graph: one(graph, None))(crystal)
            return jax.vmap(one)(crystal, jax.random.split(key, crystal["nodes"].shape[0]))

        def crystal_pass(p, crystal, key):
            """(edge logits (B, P), pKa (B, Q)) of the crystal free-branch pass, the same per-complex dropout key as the other
            passes; pKa = the head's 8 tanh output at the query sites (predict_all without the paired delta head)."""
            def one(graph, k):
                tokens, query_site = site_embeddings_indexed(p, graph, key=k, dropout_rate=rate if k is not None else 0.0)
                z = production_residue.residue_embedding(p, tokens, graph["nodes"].shape[0])
                return production_residue.pair_logits(p, z, graph["edge_a"], graph["edge_b"]), (8 * jnp.tanh(linear(p["head"], tokens)[:, 0]))[query_site]
            if key is None: return jax.vmap(lambda graph: one(graph, None))(crystal)
            return jax.vmap(one)(crystal, jax.random.split(key, crystal["nodes"].shape[0]))

        def alt_free_edges(p, graphs, key):
            """anm-edge-v1 E: (edge logits (B, P), residue embeddings z (B, n, W)) of the unmasked free pass on the drawn (alt)
            geometry graphs[:, 1], the same per-complex dropout key as the other passes."""
            free = jax.tree.map(lambda x: x[:, 1], graphs)

            def one(graph, k):
                tokens = site_embeddings_indexed(p, graph, key=k, dropout_rate=rate if k is not None else 0.0)[0]
                z = production_residue.residue_embedding(p, tokens, graph["nodes"].shape[0])
                return production_residue.pair_logits(p, z, graph["edge_a"], graph["edge_b"]), z
            if key is None: return jax.vmap(lambda graph: one(graph, None))(free)
            return jax.vmap(one)(free, jax.random.split(key, free["nodes"].shape[0]))

        def decoy_losses(p, graphs, residue, valid, key, z_free=None):
            """(re-typed, in-batch) margin losses on the free-branch edge logits (0.0 for a source that is off)."""
            true = residue["edge"][:, 1]; contact = production_residue.contact_weight(residue["edge_target"][:, 1], residue["edge_weight"][:, 1], valid)
            retyped = batched = 0.0
            if decoy_retype_weight:
                retyped = production_residue.margin_loss(true, decoy_edges(p, graphs, key), contact * graphs["decoy_valid"][:, 1, None])
            if decoy_batch_weight:
                z = residue["z"][:, 1]; a, b = graphs["edge_a"][:, 1], graphs["edge_b"][:, 1]
                j, bprime, pmask = production_residue.batch_decoy_index(b, contact, valid)
                za = z[jnp.arange(z.shape[0])[:, None], a][:, None]  # (B, 1, P, W)
                zb = z[j[..., None], bprime]  # (B, K, P, W): complex j's embedding of its own interface residue b'
                batched = production_residue.margin_loss(true[:, None], production_residue.pair_logits_from(p, za, zb), pmask)
            if not negatives: return retyped, batched
            within = swapped = 0.0; z = residue["z"][:, 1] if z_free is None else z_free; rows = jnp.arange(z.shape[0])
            if within_weight:  # negatives-v1 A: within-complex listwise, free branch
                za = z[rows[:, None], graphs["within_a"][:, 1]][:, :, None]  # (B, A, 1, W)
                zc = z[rows[:, None, None], graphs["within_c"][:, 1]]  # (B, A, C, W)
                within = production_residue.within_loss(production_residue.pair_logits_from(p, za, zc), graphs["within_t"][:, 1],
                                                        graphs["within_mask"][:, 1], graphs["within_valid"][:, 1], valid)
            if swap_weight:  # negatives-v1 B: same-class in-batch swaps, free branch
                a, b = graphs["edge_a"][:, 1], graphs["edge_b"][:, 1]
                curv = production_residue.curvature(graphs["edge"][:, 1], graphs["edge_mask"][:, 1], graphs["neighbors"][:, 1]) if swap_mode == "curv" else None
                j, bprime, pmask = production_residue.swap_decoy_index(b, contact, valid, graphs["stratum"][:, 1], curv)
                za = z[rows[:, None], a][:, None]; zb = z[j[..., None], bprime]
                swapped = production_residue.margin_loss(true[:, None], production_residue.pair_logits_from(p, za, zb), pmask)
            return retyped, batched, within, swapped

        def paired_objective(p, graphs, targets, mask, rsa, contacts, valid, key, ph=None):
            chain_batch = None
            if chain_weight:  # quartet-rank-v2: batch-level fields leave graphs first (flatten / vmap read per-complex leaves only)
                chain_batch = tuple(graphs[name] for name in production_residue.CHAIN_FIELDS)
                graphs = {name: value for name, value in graphs.items() if name not in production_residue.CHAIN_FIELDS}
            crystal = None
            if alt_rank_weight or consistency or alt_edge_only: graphs, crystal = production_alt.split_crystal(graphs)
            alt_graphs = graphs; x_free_edge = None
            if alt_edge_only:  # anm-edge-v1 E: bound + crystal free pass for pKa / burial / gate; alt free pass for the edge head only
                if crystal is None: raise ValueError("--alt-edge-only needs the loader's crystal free branch (xtal_ fields; AltConfig keep_crystal)")
                graphs = production_alt.crystal_graphs(graphs, crystal)
            predicted, burial_logits, interface_logits, geometry, ph_logits, site_tokens, gate, residue = paired_all(p, graphs, key)
            if alt_edge_only and "edge" in residue:
                x_free_edge = residue["edge"][:, 1]; alt_edge, alt_z = alt_free_edges(p, alt_graphs, key)
                residue = {**residue, "edge": jnp.stack((residue["edge"][:, 0], alt_edge), axis=1),
                           **({"z": jnp.stack((residue["z"][:, 0], alt_z), axis=1)} if "z" in residue else {})}
            edge_graphs = alt_graphs if alt_edge_only else graphs  # the geometry the edge head's free side reads
            expected = targets - reference[graphs["query_group"][:, 0]][:, None, :]
            state_error = jnp.square(predicted - expected) * mask[:, None, :]
            pair_error = jnp.square((predicted[:, 0] - predicted[:, 1]) - (expected[:, 0] - expected[:, 1])) * mask
            if shift_weights is not None:
                from .production_shift_weight import site_weights
                pair_error = pair_error * site_weights(targets, shift_weights)
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
            elif "burial" in residue and residue_burial_siamese:
                # per-residue Siamese burial (2026-10-20): each branch vs its own RSA, plus the predicted change
                # sigmoid(bound) - sigmoid(free) against clip(RSA_free) - clip(RSA_bound) (squared error), every residue
                rb, rr = residue["burial"], residue["rsa"]; ones = jnp.ones_like(rr[:, 1])
                states = (burial_loss(rb[:, 0], rr[:, 0], ones, valid) + burial_loss(rb[:, 1], rr[:, 1], ones, valid)) / 2
                both = ((rr[:, 0] >= 0) & (rr[:, 1] >= 0)).astype(jnp.float32)
                change = jax.nn.sigmoid(rb[:, 0, :, 0]) - jax.nn.sigmoid(rb[:, 1, :, 0])
                burial = states + structure_mean(jnp.square(change - (jnp.clip(rr[:, 1], 0.0, 1.0) - jnp.clip(rr[:, 0], 0.0, 1.0))), both, valid)
            elif "burial" in residue:  # every residue of the free branch (production_residue)
                burial = burial_loss(residue["burial"][:, 1], residue["rsa"][:, 1], jnp.ones_like(residue["rsa"][:, 1]), valid)
            else: burial = burial_loss(burial_logits[:, 1], rsa, sites, valid)
            interface = structure_mean(cross_entropy(interface_logits[:, branch], interface_labels(contacts)), sites, valid) if use_interface else 0.0
            # geometry auxiliary on both branches (the free branch's partner score sees only its own edges)
            geometric = (geometry_loss(jax.tree.map(lambda x: x[:, 0], geometry), sites, valid)
                         + geometry_loss(jax.tree.map(lambda x: x[:, 1], geometry), sites, valid)) / 2 if structural else 0.0
            # experimental pH on the AB branch (the deposited structure), one term per structure
            ph_loss = production_ph.loss(ph_logits[:, 0], ph[0], ph[1] * valid) if ph_weight else 0.0
            # paired binding head on the AB and free tokens of each pKa-labelled site
            bind_loss = structure_mean(optax.sigmoid_binary_cross_entropy(
                production_bind.logits(p["aux_bind"], site_tokens[:, 0], site_tokens[:, 1], bind_operand),
                (jnp.log1p(jnp.minimum(jnp.maximum(contacts, 0.0), INTERFACE_CAP)) / jnp.log1p(INTERFACE_CAP))),
                sites * (contacts >= 0), valid) if bind_weight else 0.0
            gated = (gate_loss(jax.tree.map(lambda x: x[:, 0], gate), valid) + gate_loss(jax.tree.map(lambda x: x[:, 1], gate), valid)) / 2 if gate_weight else 0.0
            edged = 0.0
            if "edge" in residue:  # (free + bound) / 2 + Siamese distillation of free toward bound (production_residue)
                ew = residue["edge_weight"][:, 1] * valid[:, None]; en = jnp.sum(ew, axis=1); ecount = en > 0
                emean = lambda per: jnp.sum(jnp.where(ecount, jnp.sum(per * ew, axis=1) / jnp.maximum(en, 1), 0.0)) / jnp.maximum(jnp.sum(ecount), 1)
                free_e, bound_e, target_e = residue["edge"][:, 1], residue["edge"][:, 0], residue["edge_target"][:, 1]
                if hotspot:  # hotspot-mask-v1: the free logits come from the masked pass
                    z_masked = masked_embedding(p, edge_graphs, key)
                    free_e = jax.vmap(production_residue.pair_logits, in_axes=(None, 0, 0, 0))(p, z_masked, edge_graphs["edge_a"][:, 1], edge_graphs["edge_b"][:, 1])
                    masked_bce = emean(optax.sigmoid_binary_cross_entropy(free_e, target_e))
                edged = (emean(optax.sigmoid_binary_cross_entropy(free_e, target_e)) + emean(optax.sigmoid_binary_cross_entropy(bound_e, target_e))) / 2 \
                    + emean(optax.sigmoid_binary_cross_entropy(free_e, jax.nn.sigmoid(jax.lax.stop_gradient(bound_e))))
            total = state_loss + pair_loss + weight * (burial + interface) + geometric + ph_weight * ph_loss + bind_weight * bind_loss + gate_weight * gated + residue_edge_weight * edged
            ranked = None; consistent = None; x_edge = x_pka = None
            if alt_edge_only:  # anm-edge-v1 E: the crystal side is the crystal free pass already run (no pKa term)
                x_edge = x_free_edge
            elif consistency and "edge" in residue and crystal is not None:  # anm-aug-v1: one crystal pass serves both terms
                x_edge, x_pka = crystal_pass(p, crystal, key)
            if alt_rank_weight and "edge" in residue and crystal is not None:  # min-aug-v1: l_alt = unmasked free logits on the drawn geometry
                contact = production_residue.contact_weight(residue["edge_target"][:, 1], residue["edge_weight"][:, 1], valid)
                ranked = production_alt.rank_loss(residue["edge"][:, 1], crystal_edges(p, crystal, key) if x_edge is None else x_edge, contact,
                                                  graphs["use_alt"][:, 1], alt_rank_margin)
                total = total + alt_rank_weight * ranked
            if alt_edge_only and consistency and x_edge is not None:  # anm-edge-v1 E: edge consistency only (pKa weight refused)
                if alt_cons_stopgrad: x_edge = jax.lax.stop_gradient(x_edge)
                consistent = (production_alt.cons_edge_loss(residue["edge"][:, 1], x_edge, residue["edge_weight"][:, 1], valid, graphs["use_alt"][:, 1]), 0.0)
                total = total + alt_cons_edge_weight * consistent[0]
            elif x_edge is not None and not alt_edge_only:  # anm-aug-v1 D: symmetric (alt - crystal)^2 on edges and free-state pKa, alt visits only
                if alt_cons_stopgrad: x_edge, x_pka = jax.lax.stop_gradient(x_edge), jax.lax.stop_gradient(x_pka)
                use = graphs["use_alt"][:, 1]
                consistent = (production_alt.cons_edge_loss(residue["edge"][:, 1], x_edge, residue["edge_weight"][:, 1], valid, use),
                              production_alt.cons_pka_loss(predicted[:, 1], x_pka, mask, valid, use))
                total = total + alt_cons_edge_weight * consistent[0] + alt_cons_pka_weight * consistent[1]
            quartet = None
            if quartet_weight and "edge" in residue:  # quartet-rank-v1: unmasked free-branch embeddings, gradient into the encoder
                quartet = production_residue.quartet_terms(p, residue["z"][:, 1], graphs["quartet_idx"][:, 1], graphs["quartet_mask"][:, 1], valid)
                total = total + quartet_weight * quartet[0]
            chained = None
            if chain_weight and "edge" in residue:  # quartet-rank-v2: unmasked free-branch embeddings, gradient into the encoder
                side_a, side_b = production_residue.chain_sides(graphs["nodes"][:, 1], graphs["node_mask"][:, 1], graphs["rsa_residue"][:, 1], chain_batch[0])
                chained = production_residue.chain_quartet_terms(p, residue["z"][:, 1], side_a, side_b, chain_batch[1], valid, chain_tau, chain_k)
                total = total + chain_weight * chained[0]
            if decoys and "edge" in residue:  # decoy-rank-v1; the aux tuple gains (re-typed, in-batch) margin losses
                terms = decoy_losses(p, edge_graphs, residue, valid, key) if not hotspot else decoy_losses(p, edge_graphs, residue, valid, key, z_masked)
                retyped, batched = terms[:2]
                if decoy_retype_weight: total = total + decoy_retype_weight * retyped
                if decoy_batch_weight: total = total + decoy_batch_weight * batched
                if negatives:  # negatives-v1: (re-typed, in-batch, within listwise, same-class swap)
                    if within_weight: total = total + within_weight * terms[2]
                    if swap_weight: total = total + swap_weight * terms[3]
                if hotspot:  # (re-typed 0, in-batch 0, within, swap 0, masked free edge BCE)
                    terms = (*terms, *([0.0] * (4 - len(terms))), masked_bce)
                if ranked is not None or consistent is not None:  # min-aug-v1: slot 5 = alt rank margin loss
                    terms = (*terms, *([0.0] * (5 - len(terms))), 0.0 if ranked is None else ranked)
                if consistent is not None: terms = (*terms, *consistent)  # anm-aug-v1: slots 6, 7 = cons edge, cons pKa
                if alt_edge_only: terms = (*terms, *([0.0] * (8 - len(terms))), burial, gated)  # anm-edge-v1: slots 8, 9 = burial, gate (crystal pass)
                if quartet is not None: terms = (*terms, *([0.0] * (QUARTET_SLOT - len(terms))), *quartet)  # quartet-rank-v1: slots 10-12
                if chained is not None:  # quartet-rank-v2: slots 13-22 = loss, accuracy, quartets, softplus sum, correct x 3, count x 3 (hetero, homo, abag)
                    terms = (*terms, *([0.0] * (CHAIN_SLOT - len(terms))), *chained[:4], *(chained[4][c] for c in range(3)), *(chained[5][c] for c in range(3)))
                return total, (state_loss, pair_loss, ph_loss, bind_loss, terms)
            if alt_edge_only:
                return total, (state_loss, pair_loss, ph_loss, bind_loss, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0 if ranked is None else ranked,
                                                                            *(consistent if consistent is not None else (0.0, 0.0)), burial, gated))
            if consistent is not None: return total, (state_loss, pair_loss, ph_loss, bind_loss, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0 if ranked is None else ranked, *consistent))
            if ranked is not None: return total, (state_loss, pair_loss, ph_loss, bind_loss, (0.0, 0.0, 0.0, 0.0, 0.0, ranked))
            return total, (state_loss, pair_loss, ph_loss, bind_loss)

        def pkpdb_objective(p, graphs, targets, eligible, rsa, valid, key, ph=None):
            keys = jax.random.split(key, graphs["nodes"].shape[0])
            predicted, burial_logits, _, geometry, ph_logits, _, gate, _ = jax.vmap(lambda graph, k: predict_all(p, graph, k))(graphs, keys)
            expected = targets - reference[graphs["query_group"]]
            per_structure = jnp.sum(jnp.square(predicted - expected) * eligible, axis=1) / jnp.maximum(jnp.sum(eligible, axis=1), 1)
            primary = jnp.sum(jnp.where(valid, per_structure, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
            return primary + weight * burial_loss(burial_logits, rsa, eligible.astype(jnp.float32), valid) + \
                (geometry_loss(geometry, eligible.astype(jnp.float32), valid) if structural else 0.0) + \
                (ph_weight * production_ph.loss(ph_logits, ph[0], ph[1] * valid) if ph_weight else 0.0) + \
                (gate_weight * gate_loss(gate, valid) if gate_weight else 0.0)

        def apply(p, state, pair_gradient, pk_gradient, rate_):
            gradient = jax.tree.map(lambda a, b: a + b, pair_gradient, pk_gradient)
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient)]))
            updates, state = self.optimizer.update(gradient, state, p)
            return optax.apply_updates(p, jax.tree.map(lambda value: value * rate_, updates)), state, finite

        self.predictions = jax.jit(lambda p, graphs: paired_all(p, graphs)[0])
        self.paired_auxiliary = jax.jit(lambda p, graphs: paired_all(p, graphs)[1:3])
        self.site_auxiliary = jax.jit(jax.vmap(lambda p, graph: predict_all(p, graph)[1], in_axes=(None, 0)))
        self.paired_binding = jax.jit(lambda p, graphs: production_bind.logits(p["aux_bind"], *(lambda t: (t[:, 0], t[:, 1]))(paired_all(p, graphs)[5]), bind_operand))
        self.ph_paired = jax.jit(lambda p, graphs: paired_all(p, graphs)[4][:, 0])
        self.ph_single = jax.jit(jax.vmap(lambda p, graph: predict_all(p, graph)[4], in_axes=(None, 0)))
        self.gate_paired = jax.jit(lambda p, graphs: paired_all(p, graphs)[6])
        self.paired_value_grad = jax.jit(jax.value_and_grad(paired_objective, has_aux=True))
        self._paired_objective = paired_objective  # unjitted (tests)
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
                 "pinder_burial_bound": ([], []), "interface_binding": ([], [])}; change = ([], [])
    for ids, batch in Prefetcher(pinder_source, pinder_source.policy.plans(pinder_records, np.random.default_rng(0)), config):
        graphs, _, mask, rsa, contacts, _, valid = batch
        burial, interface = map(np.asarray, engine.paired_auxiliary(params, graphs)); keep = mask & valid[:, None]
        for branch, name in ((0, "interface_bound"), (1, "interface_free")) if engine.heads == "both" else ():
            collected[name][0].append(interface[:, branch][keep]); collected[name][1].append(contacts[keep])
        if getattr(engine, "bind_weight", 0.0):
            bind = np.asarray(engine.paired_binding(params, graphs))[..., None]; k = keep & (contacts >= 0)
            collected["interface_binding"][0].append(bind[k]); collected["interface_binding"][1].append(contacts[k])
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


def architecture(width=None, sidechains=False, depth=2, loops=1):
    """ARCHITECTURE, or width W with ff = 2W (the size-sweep tiers; W divisible by the 4 attention heads); with
    sidechains, node_dim 152 (24 backbone + 128 side-chain features, production_sidechains); depth D residue blocks
    (default 2) or the 2 blocks tied and looped r times (production_depth). Defaults add no keys."""
    if width is not None and width % 4: raise ValueError("width must be divisible by the 4 attention heads")
    if depth != 2 and loops != 1: raise ValueError("--depth and --loops are separate variants")
    base = ARCHITECTURE if width is None else {"width": int(width), "ff": 2 * int(width)}
    return {**base, **({"node_dim": 24 + SIDECHAIN_WIDTH} if sidechains else {}), **({"depth": int(depth)} if depth != 2 else {}),
            **({"loops": int(loops), "injection": f"encoded input re-injected before loops >= 1, W_inject at {INJECT_SCALE} x default scale"}
               if loops != 1 else {})}


def parameter_count(params):
    import jax
    return int(sum(np.size(x) for x in jax.tree.leaves(params)))


def make_model(seed, query_norm="shared", paired_weight="none", reduction="structure", width=None, sidechains=False, depth=2, loops=1,
               orientation=False, sidechain_view="all", sidechain_drop=0.0, table=None, dihedrals=False, hbonds=False, variant="none",
               sidechain_source="original"):
    """(params, engine, predict): scratch oGQT and its engine; separate query norm starts as a copy of query norm1.
    depth / loops (production_depth), orientation / side-chain views (production_inputs): the engine returned is
    JointEngine, valid for depth alone; the others take DropoutEngine / AuxEngine with model_inputs(...) (train())."""
    import jax
    import jax.numpy as jnp
    from pkanet.ogqt import initialize as initialize_ogqt, predict_shift
    from .gqt_multitask_replay import JointEngine
    shape = architecture(width, sidechains)
    params = initialize_ogqt(jax.random.PRNGKey(seed), **shape)
    if depth != 2 or loops != 1 or orientation or sidechain_view != "all" or table or dihedrals or hbonds or sidechain_source != "original":
        architecture(width, sidechains, depth, loops)  # validates the combination
        if query_norm != "shared" or paired_weight != "none" or reduction != "structure":
            raise ValueError("depth / loops / input variants combine with the default objective only")
        if depth != 2: params = deepen(params, seed, depth, shape["width"], shape["ff"], shape.get("node_dim", 24))
        if loops != 1: params = add_injection(params, seed)
        if orientation: params = add_orientation(params, seed)
        if dihedrals: params = add_dihedral_inputs(params, seed)
        if hbonds: params = add_hbond_inputs(params, seed)
        from .production_position import initialize as initialize_variant
        params = initialize_variant(params, seed, variant)
        predict_shift = model_inputs(loops, orientation, sidechain_view, sidechain_drop, table, dihedrals, hbonds, variant, sidechain_source)[1]
        return params, JointEngine(params), jax.jit(jax.vmap(predict_shift, in_axes=(None, 0)))
    if query_norm == "separate":
        params = {**params, "query_context_norm": jnp.array(params["query"]["norm1"])}
        return params, VariantEngine(params, predict_shift_separate, paired_weight, reduction), jax.jit(jax.vmap(predict_shift_separate, in_axes=(None, 0)))
    if reduction != "structure":
        return params, VariantEngine(params, predict_shift, paired_weight, reduction), jax.jit(jax.vmap(predict_shift, in_axes=(None, 0)))
    engine = JointEngine(params)
    if paired_weight == "interface": engine = interface_weighted(engine)
    return params, engine, jax.jit(jax.vmap(predict_shift, in_axes=(None, 0)))


def model_inputs(loops=1, orientation=False, sidechain_view="all", sidechain_drop=0.0, table=None, dihedrals=False, hbonds=False, variant="none",
                 sidechain_source="original"):
    """(site_embeddings, predict_shift) for the looped encoder and the input variants (pkanet.site_model's by default);
    table: the loader appends backbone-v1 (default: with orientation); sidechain_source: production_packed."""
    if variant in ("none", "paired-delta"):
        return input_functions(model_functions(loops), sidechain_view, sidechain_drop, orientation, table, dihedrals, hbonds, sidechain_source)
    from .production_position import annotate, functions
    embeddings, predict = input_functions(functions(variant), sidechain_view, sidechain_drop, orientation, table, dihedrals, hbonds, sidechain_source)
    def wrap(fn):
        def call(params, graph, **kwargs): return fn(params, annotate(graph, variant), **kwargs)
        return call
    return wrap(embeddings), wrap(predict)


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


def code_hashes(alt=False, residue=False):
    src = Path(__file__).parents[1]
    paths = [Path(__file__), src / "pkatrain/production_loading.py", src / "pkatrain/production_graphs.py",
             src / "pkatrain/loading.py", src / "pkatrain/production_depth.py", src / "pkatrain/production_inputs.py", src / "pkatrain/gqt_multitask_replay.py", src / "pkatrain/gqt_paired_pinder.py",
             src / "pkatrain/site_graph_data.py", src / "pkanet/ogqt.py", src / "pkanet/model.py", src / "pkanet/triton_attention.py",
             src / "pkatrain/production_ph.py", src / "pkabench/experimental_ph.py",
             src / "pkatrain/production_shift_weight.py", src / "pkatrain/production_position.py"]
    if alt: paths += [src / "pkatrain/production_alt.py", src / "pkatrain/production_residue.py"]
    elif residue: paths += [src / "pkatrain/production_residue.py"]  # drsa-within-v1
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


def _pinder_rows(engine, params, source, records, config, alt=False):
    """Per-site validation rows with gqt_paired_pinder.evaluate's definitions. alt (min-aug-v1): also the set of complexes
    whose free branch was the side-store geometry (graph use_alt)."""
    from pkanet.model import PKPDB_PK_MOD
    reference_table = np.asarray(PKPDB_PK_MOD); rows = []; used = set()
    plans = source.policy.plans(records, np.random.default_rng(0))
    for ids, batch in Prefetcher(source, plans, config):
        graphs, targets, mask, _, _, metadata, valid = batch
        if alt: used.update(cid for slot, cid in enumerate(ids) if graphs["use_alt"][slot, 1] > 0.5)
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
    return (rows, used) if alt else rows


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
    from .production_shift_weight import validation_bins
    pinder["binding_shift_bins"] = validation_bins(pair_rows)
    pkpdb = None
    if "pkpdb-val" in sources:
        pk_rows = _pkpdb_rows(predict, params, sources["pkpdb-val"], select(manifests["pkpdb"], "val"), config); pkpdb = pkpdb_metrics(pk_rows)
        if out is not None: _write(out / f"predictions-pkpdb-epoch-{epoch:03d}.csv", pk_rows)
    if out is not None:
        _write(out / f"predictions-pinder-epoch-{epoch:03d}.csv", pair_rows); _write(out / f"predictions-benchmark-epoch-{epoch:03d}.csv", bench_rows)
    pinder_alt = None
    if "pinder-val-alt" in sources:  # min-aug-v1 --val-alt: free branch from the side-store record where one is usable
        alt_rows, used = _pinder_rows(engine, params, sources["pinder-val-alt"], select(manifests["pinder"], "val"), config, alt=True)
        pinder_alt = {"complexes": len(used), "alt": _metrics([r for r in alt_rows if r["complex_id"] in used]) if used else None,
                      "crystal_same_complexes": _metrics([r for r in pair_rows if r["complex_id"] in used]) if used else None}
        for name in ("alt", "crystal_same_complexes"):
            if pinder_alt[name] is not None: pinder_alt[name]["selection"] = pinder_alt[name]["state_mae"] + pinder_alt[name]["interface_paired_mae"]
        if out is not None and used: _write(out / f"predictions-pinder-alt-epoch-{epoch:03d}.csv", [{**r, "use_alt": r["complex_id"] in used} for r in alt_rows])
    ph = None
    if getattr(engine, "ph_weight", 0.0):
        from .production_ph import evaluate as ph_evaluate
        ph = ph_evaluate(engine, params, sources, manifests, engine.ph_table, config)
    return {"pinder": pinder, **({"pinder_alt": pinder_alt} if pinder_alt is not None else {}), **({"pkpdb": pkpdb} if pkpdb else {}), "benchmark": {"overall": overall, "bins": bins, "equal_bin_mae": equal_bin, **bench_squared},
            **({"ph": ph} if ph else {}), "selection": pinder["state_mae"] + pinder["interface_paired_mae"]}


QUARTET_SLOT = 10  # quartet-rank-v1: aux slots 10, 11, 12 = quartet softplus loss, accuracy, covered fraction (unweighted)
CHAIN_SLOT = 13  # quartet-rank-v2: aux slots 13-22 = chain loss, accuracy, quartets, softplus sum, correct (hetero, homo, abag), counts (same)
CHAIN_NAMES = ("pinder_chain_softplus_batch_mean", "pinder_chain_accuracy_batch_mean", "pinder_chain_quartets", "pinder_chain_softplus_sum",
               "pinder_chain_correct_hetero", "pinder_chain_correct_homo", "pinder_chain_correct_abag", "pinder_chain_n_hetero", "pinder_chain_n_homo", "pinder_chain_n_abag")
WITHIN_SLOT = 2


def chain_history(means):
    """quartet-rank-v2 history fields from the epoch means of the aux slots (CHAIN_SLOT on): pooled loss / accuracy over the
    epoch's quartets (sums / counts), quartets per batch, per class (hetero, homo, abag) accuracy and quartets per batch."""
    v = dict(zip(CHAIN_NAMES, map(float, means[CHAIN_SLOT:CHAIN_SLOT + len(CHAIN_NAMES)])))
    n = v["pinder_chain_quartets"]; cls = ("hetero", "homo", "abag"); ratio = lambda a, b: (a / b) if b > 0 else None
    correct = sum(v[f"pinder_chain_correct_{c}"] for c in cls)
    return {"pinder_chain_softplus": ratio(v["pinder_chain_softplus_sum"], n), "pinder_chain_accuracy": ratio(correct, n), "pinder_chain_quartets_per_batch": n,
            **{f"pinder_chain_accuracy_{c}": ratio(v[f"pinder_chain_correct_{c}"], v[f"pinder_chain_n_{c}"]) for c in cls},
            **{f"pinder_chain_quartets_{c}_per_batch": v[f"pinder_chain_n_{c}"] for c in cls},
            "pinder_chain_softplus_batch_mean": v["pinder_chain_softplus_batch_mean"]}


def quartet_grad_norms(engine, params, graphs, targets, mask, wb, wi, valid, key, epoch=None, number=None, updates=None):
    """quartet-rank-v1/v2 diagnostic: L2 norms of the gradients of the UNWEIGHTED within listwise term (aux slot 2), the
    UNWEIGHTED residue quartet term (slot 10, when --quartet-weight > 0) and the UNWEIGHTED chain quartet term (slot 13, when
    --chain-quartet-weight > 0) per top-level parameter key, on one batch with the training objective and dropout key;
    encoder = every key not starting with "aux_", trunk = encoder minus head / groups. Ratios term / within, unweighted and
    weighted (L x term) / (W_within x within)."""
    import jax
    import jax.numpy as jnp
    lq, lw, lc = float(engine.quartet_weight), float(engine.within_weight), float(getattr(engine, "chain_quartet_weight", 0.0))
    slots = [("within", WITHIN_SLOT)] + ([("quartet", QUARTET_SLOT)] if lq else []) + ([("chain", CHAIN_SLOT)] if lc else [])
    t0 = time.time(); fn = getattr(engine, "_quartet_norm_fn", None)
    if fn is None:
        def norms(p, *args):
            out = {}
            for name, slot in slots:
                g = jax.grad(lambda q, s=slot: engine._paired_objective(q, *args)[1][4][s])(p)
                out[name] = {k: jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(v))) for k, v in g.items()}
            return out
        fn = engine._quartet_norm_fn = jax.jit(norms)
    raw = jax.device_get(fn(params, graphs, targets, mask, wb, wi, valid, key))
    keys = sorted(raw["within"]); encoder = [k for k in keys if not k.startswith("aux_")]; trunk = [k for k in encoder if k not in ("head", "groups")]
    total = lambda d, ks: float(np.sqrt(sum(float(d[k]) ** 2 for k in ks)))
    ratio = lambda a, b: (a / b) if b > 0 else None
    weights = {"within": lw, "quartet": lq, "chain": lc}
    record = {"epoch": epoch, "batch": number, "updates_before": updates, "encoder_subtree": encoder, "norms": {}, "per_key": {},
              "weights": {name: weights[name] for name, _ in slots}}
    for name, _ in slots:
        record["norms"].update({f"{name}_encoder": total(raw[name], encoder), f"{name}_trunk": total(raw[name], trunk), f"{name}_aux_redge": total(raw[name], ["aux_redge"])})
        record["per_key"][name] = {k: float(raw[name][k]) for k in keys}
    n = record["norms"]
    for name, _ in slots[1:]:
        record.update({f"ratio_{name}_over_within_encoder": ratio(n[f"{name}_encoder"], n["within_encoder"]),
                       f"ratio_weighted_{name}_over_within_encoder": ratio(weights[name] * n[f"{name}_encoder"], lw * n["within_encoder"]),
                       f"ratio_{name}_over_within_trunk": ratio(n[f"{name}_trunk"], n["within_trunk"]),
                       f"ratio_{name}_over_within_aux_redge": ratio(n[f"{name}_aux_redge"], n["within_aux_redge"])})
    record["seconds"] = round(time.time() - t0, 1)
    if not all(np.isfinite(v) for v in n.values()): raise FloatingPointError(f"non-finite term gradient norms: {n}")
    print("[quartet-grad-norms] " + json.dumps({k: v for k, v in record.items() if k not in ("encoder_subtree", "per_key")}), flush=True)
    return record


def alt_edge_only_check(engine, params, graphs, targets, mask, wb, wi, valid, key, epoch, number, tolerance=1e-5):
    """anm-edge-v1 smoke assertion: on a training batch with alt visits, the edge-only objective's pKa state / paired, burial
    and gate terms equal those of the same batch made crystal-only (production_alt.crystalise: branch 1 = the crystal free
    branch, use_alt 0), same dropout key and jitted objective; the edge-side terms (within, masked free edge BCE, total)
    must differ. Raises AssertionError past `tolerance`; returns the record written to history."""
    import jax
    import numpy as np
    from . import production_alt
    fn = getattr(engine, "_check_objective", None) or jax.jit(engine._paired_objective); engine._check_objective = fn
    (ta, aa), (tc, ac) = (fn(params, g, targets, mask, wb, wi, valid, key) for g in (graphs, production_alt.crystalise(graphs)))
    core = {"pinder_state": (aa[0], ac[0]), "pinder_paired": (aa[1], ac[1]), "burial": (aa[4][8], ac[4][8]), "gate": (aa[4][9], ac[4][9])}
    edge = {"total": (ta, tc), "within": (aa[4][2], ac[4][2]), "masked_free_edge_bce": (aa[4][4], ac[4][4])}
    diff = lambda pair: float(abs(np.float64(pair[0]) - np.float64(pair[1])))
    out = {"epoch": epoch, "batch": number, "alt_complexes": int(np.sum(np.asarray(graphs["use_alt"][:, 0]) * np.asarray(valid))),
           "core_values": {k: float(v[0]) for k, v in core.items()}, "core_abs_diff": {k: diff(v) for k, v in core.items()},
           "edge_abs_diff": {k: diff(v) for k, v in edge.items()}, "tolerance": tolerance}
    out["core_max_abs_diff"] = max(out["core_abs_diff"].values()); out["edge_side_differs"] = out["edge_abs_diff"]["total"] > 0
    out["passed"] = bool(out["core_max_abs_diff"] <= tolerance and out["edge_side_differs"])
    print(json.dumps({"alt_edge_only_check": out}), flush=True)
    if not out["passed"]: raise AssertionError(("--alt-edge-only: pKa / burial / gate terms depend on the alt geometry", out))
    return out


def train(root, run, fraction=0.1, smoke=False, batch=BATCH, paired_weight="none", seed=SEED, query_norm="shared", reduction="structure",
          mask_pmin=None, dropout=0.0, aux_weight=None, aux_loss="ce", aux_interface_branch="bound", aux_heads="both", width=None,
          sidechains=False, depth=2, loops=1, orientation=False, sidechain_view="all", sidechain_drop=0.0, geometry_weight=0.0,
          dihedrals=False, chi_weight=0.0, hbonds=False, epochs=None, ph_weight=0.0, ph_sources="all", ph_probe=False, bind_weight=0.0, bind_operand="product", shift_weight="none", variant="none",
          sidechain_source="original", gate_weight=0.0, gate_cys_weight=10.0, drop_bonded_cys=False, residue_burial=False, residue_edge_weight=0.0,
          residue_burial_siamese=False, decoy_retype_weight=0.0, decoy_batch_weight=0.0, within_weight=0.0, swap_weight=0.0, swap_mode="class",
          mask_oracle=None, mask_store=None, mask_keep=0.25, mask_soft=0.0, mask_sides="both", mask_prob=1.0,
          alt_source=None, alt_prob=0.0, val_alt=False, alt_rank_weight=0.0, alt_rank_margin=0.0,
          alt_weights=None, alt_cons_edge_weight=0.0, alt_cons_pka_weight=0.0, alt_cons_stopgrad=False, alt_edge_only=False, within_target="distance",
          quartet_weight=0.0, quartet_count=None, smoke_batches=5, quartet_grad_norms_every_epoch=False,
          pinder_manifest=None, pinder_extra=None, mask_store_extra=None,
          chain_quartet_weight=0.0, chain_quartet_count=None, chain_quartet_tau=None, chain_quartet_k=None, chain_quartet_abag_min=None):
    import jax
    import jax.numpy as jnp
    from pkanet.ogqt import initialize as initialize_ogqt, predict_shift
    from .gqt_crop_radius import EPOCHS, MIN_DELTA, PATIENCE, learning_rate
    # --epochs N (2026-10-12): the schedule stretched over N epochs (hold to N / 2) and patience scaled to 8 N / 20;
    # the default keeps gqt_crop_radius's 20-epoch schedule and patience 8
    if epochs is not None and epochs != EPOCHS:
        total = int(epochs); PATIENCE = PATIENCE * total // EPOCHS
        learning_rate = lambda epoch, batch, batches: stretched_learning_rate(epoch, batch, batches, total)
    else: total = EPOCHS
    from .gqt_multitask_replay import JointEngine
    from .trainer import load_checkpoint, save_checkpoint
    # quartet-rank-v1 (2026-11-15): validated before the run directory is touched
    quartet_weight = float(quartet_weight); quartet_count = production_residue.QUARTET_COUNT if quartet_count is None else int(quartet_count)
    if quartet_weight < 0 or quartet_count < 1: raise ValueError((quartet_weight, quartet_count))
    if quartet_weight and (not within_weight or not residue_edge_weight or aux_weight is None or alt_source):
        raise ValueError("--quartet-weight needs --within-weight, --residue-edge-weight and --aux-weight, and no --alt-source")
    # quartet-rank-v2 (2026-11-19): cross-complex chain quartets, validated before the run directory is touched
    chain_weight = float(chain_quartet_weight)
    chain_count = production_residue.CHAIN_COUNT if chain_quartet_count is None else int(chain_quartet_count)
    chain_tau = production_residue.CHAIN_TAU if chain_quartet_tau is None else float(chain_quartet_tau)
    chain_k = production_residue.CHAIN_K if chain_quartet_k is None else int(chain_quartet_k)
    chain_abag_min = production_residue.CHAIN_ABAG_MIN if chain_quartet_abag_min is None else int(chain_quartet_abag_min)
    if chain_weight < 0 or chain_count < 1 or chain_tau <= 0 or chain_k < 1 or chain_abag_min < 2:
        raise ValueError(("chain quartet: weight >= 0, count >= 1, tau > 0, k >= 1, abag-min >= 2", chain_weight, chain_count, chain_tau, chain_k, chain_abag_min))
    if chain_weight and (not within_weight or not residue_edge_weight or aux_weight is None or alt_source):
        raise ValueError("--chain-quartet-weight needs --within-weight, --residue-edge-weight and --aux-weight, and no --alt-source")
    if not chain_weight and any(v is not None for v in (chain_quartet_count, chain_quartet_tau, chain_quartet_k, chain_quartet_abag_min)):
        raise ValueError("--chain-quartet-count / -tau / -k / -abag-min need --chain-quartet-weight > 0")
    if quartet_grad_norms_every_epoch and not (quartet_weight or chain_weight): raise ValueError("--quartet-grad-norms needs --quartet-weight or --chain-quartet-weight > 0")
    if int(smoke_batches) < 1 or (int(smoke_batches) != 5 and not smoke): raise ValueError("--smoke-batches needs --smoke (and >= 1)")
    smoke_batches = int(smoke_batches)
    # quartet-rank-v1b (2026-11-16): extra PINDER training records (production_extra; pool-v5), validated before the run directory is touched
    if (pinder_manifest is None) != (pinder_extra is None): raise ValueError("--pinder-manifest and --pinder-extra go together")
    if mask_store_extra is not None and pinder_extra is None: raise ValueError("--mask-store-extra needs --pinder-extra")
    if pinder_extra is not None:
        from . import production_extra
        production_extra.refuse(sidechains=sidechains, sidechain_source=sidechain_source != "original", alt_source=alt_source, val_alt=val_alt, ph_weight=ph_weight,
                                swap_weight=swap_weight, burial_siamese_aux=aux_heads == "burial-siamese", site_masking=mask_pmin is not None,
                                geometry_weight=geometry_weight, chi_weight=chi_weight, hbonds=hbonds)
        if mask_store is not None and mask_store_extra is None: raise ValueError("--mask-store with --pinder-extra needs --mask-store-extra")
    root = Path(root); out = run_dir(root, run); out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(out.parent / "compilation-cache"))
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb", "benchmark-val")}
    if pinder_manifest is not None:  # quartet-rank-v1b: the pool's PINDER manifest (store-v2 + extra records)
        pool_manifest = read(pinder_manifest); production_extra.check_manifest(pool_manifest, manifests["pinder"], pinder_extra); manifests["pinder"] = pool_manifest
    norms = normalization(manifests["pinder"], fraction); scale = lr_scale(batch)
    chain_meta = chain_bucket = chain_pool_tsv = None
    if chain_weight:  # quartet-rank-v2: classes / roles / groups / UniProt of every training complex, and its size bucket
        chain_pool_tsv = production_residue.chain_pool_path(root, manifests["pinder"]); chain_meta = production_residue.chain_metadata(chain_pool_tsv)
        chain_train = select(manifests["pinder"], "train", fraction); missing = [r["id"] for r in chain_train if r["id"] not in chain_meta]
        if missing: raise ValueError((f"{len(missing)} training complexes missing from {chain_pool_tsv}", missing[:5]))
        chain_policy = _policy(manifests["pinder"], batch); chain_bucket = {r["id"]: chain_policy.bucket(r["n"]) for r in chain_train}
    if variant != "none" and (not orientation or not dihedrals or loops != 1 or aux_weight is None):
        raise ValueError("positional ablations require base A orientation/dihedrals/burial settings")
    # hotspot-mask-v1 (2026-10-30): masked third pass on the free branch (production_residue.hotspot_keep), default off
    hotspot_store = None
    # hotspot-mask-v2 (2026-10-30): --mask-sides one / --mask-prob p (asymmetric augmentation); defaults = v1
    if mask_sides not in production_residue.MASK_SIDES or not 0.0 <= float(mask_prob) <= 1.0: raise ValueError((mask_sides, mask_prob))
    mask_v2 = (mask_sides, float(mask_prob)) != ("both", 1.0)
    if mask_oracle is None:
        if mask_store is not None or mask_soft: raise ValueError("--mask-store / --mask-soft need --mask-oracle")
        if mask_v2: raise ValueError("--mask-sides / --mask-prob need --mask-oracle")
    else:
        if mask_oracle not in production_residue.MASK_ORACLES: raise ValueError(mask_oracle)
        if decoy_retype_weight: raise AssertionError("--mask-oracle replaces the re-typed pass: --decoy-retype-weight must be 0")
        if not residue_edge_weight or decoy_batch_weight or swap_weight: raise ValueError("--mask-oracle needs --residue-edge-weight and no --decoy-batch-weight / --swap-weight")
        if sidechains: raise ValueError("--mask-oracle with --sidechains would leak residue types through the side-chain features")
        if not 0.0 < mask_keep <= 1.0 or not 0.0 <= mask_soft <= 1.0: raise ValueError((mask_keep, mask_soft))
        if mask_oracle != "random" and mask_store is None: raise ValueError("--mask-oracle pesto / own need --mask-store")
        if mask_store is not None: hotspot_store = production_residue.HotspotStore(mask_store)
    # min-aug-v1 (2026-10-31): alt free branch (production_alt; apo-negatome-v1's --apo-* renamed) and the alt-vs-crystal
    # rank term, default off
    from . import production_alt
    alt_store = None
    if (alt_prob or val_alt) and not alt_source: raise ValueError("--alt-prob / --val-alt need --alt-source")
    if alt_source and not (alt_prob or val_alt): raise ValueError("--alt-source without --alt-prob / --val-alt")
    if not 0.0 <= float(alt_prob) <= 1.0: raise ValueError("--alt-prob must be in [0, 1]")
    if alt_rank_margin and not alt_rank_weight: raise ValueError("--alt-rank-margin needs --alt-rank-weight")
    if alt_rank_weight and (not alt_prob or aux_weight is None or not residue_edge_weight or chunked(batch)):
        raise ValueError("--alt-rank-weight needs --alt-prob > 0, --aux-weight and --residue-edge-weight (batch <= 64)")
    # anm-aug-v1 (2026-10-31): several alt entries (store[:keysuffix],...), optional choice weights, consistency terms
    alt_entries = None; consistency = bool(alt_cons_edge_weight or alt_cons_pka_weight)
    if consistency and (not alt_prob or aux_weight is None or not residue_edge_weight or chunked(batch)):
        raise ValueError("--alt-cons-*-weight needs --alt-prob > 0, --aux-weight and --residue-edge-weight (batch <= 64)")
    if alt_cons_stopgrad and not consistency: raise ValueError("--alt-cons-stopgrad needs --alt-cons-edge-weight or --alt-cons-pka-weight")
    if alt_weights is not None and not alt_source: raise ValueError("--alt-weights needs --alt-source")
    # anm-edge-v1 (2026-11-01): --alt-edge-only (production_alt E)
    alt_edge_only = bool(alt_edge_only)
    if alt_edge_only:
        if alt_cons_pka_weight: raise ValueError("--alt-cons-pka-weight is refused with --alt-edge-only (the pKa never sees the alt geometry)")
        if not alt_prob or aux_weight is None or not residue_edge_weight or chunked(batch):
            raise ValueError("--alt-edge-only needs --alt-prob > 0, --aux-weight and --residue-edge-weight (batch <= 64)")
    if alt_source:
        if sidechains: raise ValueError("--alt-source supports the backbone-v1 side table only (no --sidechains)")
        alt_entries = production_alt.open_sources(alt_source); alt_store = alt_entries[0][0]
        if alt_weights is not None and len(alt_weights) != len(alt_entries): raise ValueError("--alt-weights needs one weight per --alt-source entry")
    shift_table = None
    if shift_weight != "none":
        if aux_weight is None or aux_heads != "burial": raise ValueError("shift weighting requires burial AuxEngine")
        from .production_shift_weight import fit
        shift_table = fit(manifests["pinder"], fraction, out.parent / f"invsqrt-weights-{fraction:.3f}-v1.json")
    protocol = {"version": "gqt-production-joint-v1", "run": run, "fraction": fraction, "seed": seed, "batch": batch,
        "architecture": {**architecture(width, sidechains, depth, loops), **({"orientation_edges": ORIENTATION_EDGE} if orientation else {}),
                         **({"dihedral_features": DIHEDRAL_FEATURES} if dihedrals else {}),
                         **({"hbond_features": HBOND_FEATURES} if hbonds else {})}, "initialization": f"scratch (pkanet.ogqt.initialize, PRNGKey({seed}))"
        + (f"; residue blocks 3-{depth} from production_depth.deepen" if depth != 2 else "") + ("; W_inject from production_depth.add_injection" if loops != 1 else ""),
        "objective": "gqt_multitask_replay.JointEngine: pKPDB state-shift MSE (train_mask) + PINDER AB/free state-shift MSE + binding-shift MSE; equal coefficients; no site weights",
        "optimizer": "AdamW weight decay 1e-4, global clip 1 after gradient summation",
        "schedule": f"gqt_crop_radius.learning_rate x {scale:.4f} (1e-3 hold to epoch 10, cosine to 1e-5 by epoch {EPOCHS}); patience {PATIENCE}, min delta {MIN_DELTA}"
            if total == EPOCHS else f"stretched_learning_rate x {scale:.4f} (1e-3 hold to epoch {total // 2}, cosine to 1e-5 by epoch {total}); patience {PATIENCE}, min delta {MIN_DELTA}",
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
        **({"sidechain_view": {"titratable": "side-chain features on D/E/H/C/Y/K/R residues only",
                               "non-titratable": "side-chain features on all residues except D/E/H/C/Y/K/R",
                               "dropout": f"side-chain features hidden per structure with probability {sidechain_drop} in training (AB/free share the draw); "
                                          "validation without side chains"}[sidechain_view]} if sidechain_view != "all" else {}),
        **({"geometry_auxiliary": f"+ {geometry_weight} x (functional-centroid MSE in the own frame, A / 10, + salt-bridge partner score cross-entropy) "
                                  "for D/E/H/C/Y/K/R query sites from the hidden side chains and backbone-v1 (production_inputs.geometry_targets); "
                                  "heads aux_geom_pos / aux_partner; PINDER both branches, pKPDB train_mask sites",
            "backbone_verification_sha256": {d: BackboneTable(root, d).verification_sha256 for d in ("pinder", "pkpdb", "benchmark-val")}}
           if geometry_weight else {}),
        **({"chi_auxiliary": f"+ {chi_weight} x chi1-class softmax cross-entropy (g+ / t / g-; production_inputs.chi1_targets) for "
                             "D/E/H/C/Y/K/R query sites from the hidden side chains and backbone-v1; head aux_chi; PINDER both branches, pKPDB train_mask sites"}
           if chi_weight else {}),
        **({"ph_auxiliary": f"+ {ph_weight} x structure-level experimental pH (production_ph): gated attention pooling of the site tokens "
                            f"(LayerNorm, sigmoid gate x value, per-channel normalised) -> cumulative BCE over [pH > t], t in {list(PH_THRESHOLDS)}; "
                            "PINDER AB branch and pKPDB; structure weight = eligibility x parent weight; no per-residue repetition"}
           if ph_weight else {}),
        **({"ph_sources": "crystallisation pH only; EM-buffer and NMR targets masked in training, fit and validation"} if ph_sources != "all" else {}),
        **({"ph_probe": "pH head on stop_gradient(site tokens) with its own optimizer and clip (production_ph.separate_optimizer): "
                        "the trunk trains on the pKa and burial objectives alone"} if ph_probe else {}),
        **({"binding_auxiliary": f"+ {bind_weight} x paired binding head (production_bind, operand {bind_operand}): LayerNorm'd AB and free "
                                 "site tokens a, b; per-channel weights w = W2 gelu(W1 [a; b]) + w0 (W2 zero-initialised); logit = sum w * "
                                 + ("(a * b)" if bind_operand == "product" else "(a - b)") + " + c; sigmoid cross-entropy vs the interface soft target "
                                 f"log(1 + min(c, {INTERFACE_CAP:g})) / log({INTERFACE_CAP + 1:g}) on PINDER pKa-labelled sites"} if bind_weight else {}),
        **({"hbonds": "production_inputs.backbone_hbonds: DSSP backbone H-bond energies (best donor / acceptor / 9.9, bonded flags, has-H / has-O) "
                       "from backbone-v1 over the residue-graph edges appended to the node features (6); embedding +6 rows"} if hbonds else {}),
        **({"dihedrals": "production_inputs.backbone_dihedrals: sin/cos phi, psi, omega from backbone-v1 appended to the node features (6); embedding +6 rows"}
           if dihedrals else {}),
        **({"orientation": {"definition": "production_inputs: residue edges + direction to i in j's frame, F_i^T F_j, signed sequence separation / 32 "
                                          "(13 features) from backbone-v1; residue and query block edge projections 20 -> 33",
                            "verification_sha256": {d: BackboneTable(root, d).verification_sha256 for d in ("pinder", "pkpdb", "benchmark-val")}}}
           if orientation else {}),
        **({"sidechains": {"definition": "pkanet.graph.sidechain_features appended to the node features (production_sidechains, sidechains-v1)",
                           "verification_sha256": {d: SidechainTable(root, d).verification_sha256 for d in ("pinder", "pkpdb", "benchmark-val")}}}
           if sidechains else {}),
        **({"sidechain_source": {"source": sidechain_source, "definition": (
                "sidechains-packed-v1 (production_packed: FASPR from backbone-v1 + residue types; PINDER free branch packed per chain)" if sidechain_source == "packed" else
                "sidechains-v1 and sidechains-packed-v1 loaded (256 columns); one of the two per structure with probability 1/2 (AB/free share the draw), "
                "packed without a key (production_inputs.input_functions source mix)"),
            "verification_sha256": {d: PackedSidechainTable(root, d).verification_sha256 for d in ("pinder", "pkpdb", "benchmark-val")}}}
           if sidechain_source != "original" else {}),
        **production_gate.provenance(root, gate_weight, gate_cys_weight, drop_bonded_cys),
        **production_residue.provenance(root, residue_burial or residue_burial_siamese, residue_edge_weight),
        **({"residue_burial_siamese": "burial on both PINDER branches, every residue, bound vs 1 - clip(RSA_bound), free vs 1 - clip(RSA_free) (cross-entropy, averaged), "
            "+ squared error of sigmoid(bound) - sigmoid(free) vs clip(RSA_free) - clip(RSA_bound) (residue-rsa-v1); pKPDB unchanged"} if residue_burial_siamese else {}),
        **production_residue.decoy_provenance(decoy_retype_weight, decoy_batch_weight),
        **production_residue.negatives_provenance(root, within_weight, swap_weight, swap_mode, within_target),
        **production_residue.hotspot_provenance(mask_oracle, mask_keep, mask_soft, hotspot_store, mask_sides, mask_prob),
        **production_alt.alt_provenance(alt_prob, alt_store, val_alt, alt_rank_weight, alt_rank_margin, alt_entries, alt_weights,
                                        alt_cons_edge_weight, alt_cons_pka_weight, alt_cons_stopgrad, *((True,) if alt_edge_only else ())),
        **production_residue.quartet_provenance(quartet_weight, quartet_count),
        **production_residue.chain_provenance(chain_weight, chain_count, chain_tau, chain_k, chain_abag_min, chain_pool_tsv),
        **({"smoke_batches": smoke_batches} if smoke and smoke_batches != 5 else {}),
        **({"pinder_pool": {"manifest": str(pinder_manifest), "manifest_sha256": digest(pinder_manifest), "pool": manifests["pinder"].get("pool"),
                            "pool_sha256": manifests["pinder"].get("pool_sha256"), "extra": str(pinder_extra),
                            "extra_verification_sha256": {t: digest(output(Path(pinder_extra), "pinder") / t / "verification.json") for t in production_extra.TABLES},
                            **({"mask_store_extra": str(mask_store_extra), "mask_store_extra_verification_sha256": digest(Path(mask_store_extra) / "verification.json")}
                               if mask_store_extra is not None else {}),
                            "rule": "training PINDER source reads store-v2 + side tables, or the extra copies for ids not in them (production_extra); validation unchanged",
                            "production_extra_sha256": digest(Path(production_extra.__file__))}}
           if pinder_extra is not None else {}),
        "pinder_weight_normalization": norms, "smoke": smoke, "test_data_included": False,
        "manifests": {d: digest(output(root, d) / MANIFEST) for d in manifests}, "code": code_hashes(alt_store is not None, within_target != "distance")}
    if shift_table is not None:
        protocol["binding_shift_weights"] = shift_table
        protocol["objective"] += "; binding-shift MSE weighted by capped inverse-sqrt training-bin count"
    if variant != "none": protocol["production_variant"] = variant
    if (out / "protocol.json").exists():
        stored = read(out / "protocol.json")
        if {k: v for k, v in stored.items() if k != "code"} != json.loads(json.dumps({k: v for k, v in protocol.items() if k != "code"})):
            raise AssertionError(f"{out} was registered with a different protocol")
    else: atomic_json(out / "protocol.json", protocol)
    config = LoaderConfig()
    ph_table = None
    if ph_weight:
        from .production_ph import PhTable
        ph_table = PhTable(root, manifests, fraction, ph_sources)
        stored = read(out / "ph-annotations.json") if (out / "ph-annotations.json").exists() else None
        if stored is not None and stored != json.loads(json.dumps(ph_table.provenance)): raise AssertionError(f"{out}: pH annotations changed")
        atomic_json(out / "ph-annotations.json", ph_table.provenance)
    side = {}
    for d in ("pinder", "pkpdb", "benchmark-val"):
        tables = (sidechain_tables(root, d, sidechain_source) if sidechains else []) + ([BackboneTable(root, d)] if orientation or geometry_weight or chi_weight or dihedrals or hbonds else [])
        side[d] = None if not tables else tables[0] if len(tables) == 1 else ConcatTable(*tables)
    contacts = ContactTable(root, RSA_BOUND_TABLE if aux_heads == "burial-siamese" else CONTACTS_TABLE) if aux_weight is not None else None
    train_pinder = AuxPinderSource(_with_policy(manifests["pinder"], batch), contacts, config=config, norms=norms, sidechains=side["pinder"]) if contacts else \
        PinderSource(_with_policy(manifests["pinder"], batch), config=config, norms=norms, sidechains=side["pinder"])
    sources = {"train": JointSource(train_pinder, PkpdbSource(_with_policy(manifests["pkpdb"], batch), config=config,
                                    weights=reduction != "structure" or mask_pmin is not None, aux=contacts is not None, sidechains=side["pkpdb"])),
               "pinder-val": PinderSource(_with_policy(manifests["pinder"]), config=config, norms=norms, sidechains=side["pinder"]),
               "benchmark": PkpdbSource(_with_policy(manifests["benchmark-val"]), config=config, mask="eval_mask", sidechains=side["benchmark-val"]),
               **({"pkpdb-val": PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask", sidechains=side["pkpdb"])}
                  if select(manifests["pkpdb"], "val") else {})}
    if gate_weight or drop_bonded_cys:  # production_gate: disulfide flags everywhere, bonded-Cys masking in training only
        for d, source in (("pinder", train_pinder), ("pkpdb", sources["train"].pkpdb), ("pinder", sources["pinder-val"]), ("benchmark-val", sources["benchmark"]),
                          *((("pkpdb", sources["pkpdb-val"]),) if "pkpdb-val" in sources else ())):
            source.disulfides = production_gate.DisulfideTable(root, d)
        train_pinder.drop_bonded_cys = sources["train"].pkpdb.drop_bonded_cys = bool(drop_bonded_cys)
    if gate_weight and (aux_weight is None or "allres" not in variant): raise ValueError("--gate-weight needs --aux-weight and --all-residue-tokens")
    residue_burial = residue_burial or residue_burial_siamese
    if residue_burial or residue_edge_weight:
        if aux_weight is None or "allres" not in variant or aux_heads != "burial": raise ValueError("--residue-burial / --residue-edge-weight need --aux-weight, --aux-heads burial and --all-residue-tokens")
        train_pinder.residues = production_residue.ResidueTable(root); train_pinder.sample_edges = bool(residue_edge_weight)
    if decoy_retype_weight or decoy_batch_weight:
        if not residue_edge_weight: raise ValueError("--decoy-retype-weight / --decoy-batch-weight need --residue-edge-weight")
        train_pinder.decoy_retype = bool(decoy_retype_weight)
    if within_target not in production_residue.WITHIN_TARGETS: raise ValueError(within_target)
    if within_target != "distance" and not within_weight: raise ValueError("--within-target drsa needs --within-weight")
    if within_weight or swap_weight:  # negatives-v1
        if not residue_edge_weight: raise ValueError("--within-weight / --swap-weight need --residue-edge-weight")
        train_pinder.within_sample = bool(within_weight) if within_target == "distance" else within_target  # drsa-within-v1: "drsa"
        if swap_weight: train_pinder.strata = production_residue.load_strata(root)
    if mask_oracle is not None:
        train_pinder.hotspot = (hotspot_store, mask_oracle, float(mask_keep), float(mask_soft)) + ((mask_sides, float(mask_prob)) if mask_v2 else ())
    if quartet_weight: train_pinder.quartet_count = quartet_count  # quartet-rank-v1
    if alt_prob: train_pinder.alt = production_alt.AltConfig(alt_entries, alt_prob, keep_crystal=bool(alt_rank_weight or consistency or alt_edge_only), weights=alt_weights,
                                                             **({"alt_rsa": False} if alt_edge_only else {}))
    if pinder_extra is not None: production_extra.attach(train_pinder, root, pinder_extra, mask_store_extra)  # quartet-rank-v1b
    if val_alt:
        sources["pinder-val-alt"] = PinderSource(_with_policy(manifests["pinder"]), config=config, norms=norms, sidechains=side["pinder"])
        sources["pinder-val-alt"].alt = production_alt.AltConfig(alt_entries, 1.0, weights=alt_weights)
        if gate_weight or drop_bonded_cys: sources["pinder-val-alt"].disulfides = production_gate.DisulfideTable(root, "pinder")
    if (reduction != "structure" or mask_pmin is not None) and chunked(batch): raise ValueError("this loss is not chunk-additive; use batch <= 64")
    if (width is not None or sidechains) and (query_norm != "shared" or paired_weight != "none" or reduction != "structure"):
        raise ValueError("--width / --sidechains combine with the default objective, dropout and auxiliaries only")
    if sidechain_view != "all" and not sidechains: raise ValueError("--sidechain-view needs --sidechains")
    if sidechain_source not in SIDECHAIN_SOURCES: raise ValueError(sidechain_source)
    if sidechain_source != "original" and (not sidechains or geometry_weight or chi_weight):
        raise ValueError("--sidechain-source needs --sidechains (and not --geometry-weight / --chi-weight, whose targets read the side-chain columns)")
    if (sidechain_view == "dropout") != (sidechain_drop > 0): raise ValueError("--sidechain-drop goes with --sidechain-view dropout")
    if (ph_sources != "all" or ph_probe) and not ph_weight: raise ValueError("--ph-sources / --ph-probe need --ph-weight")
    if bind_weight and (aux_weight is None or aux_heads == "burial-siamese" or aux_loss != "ce"):
        raise ValueError("--bind-weight runs through AuxEngine with contact scores (use --aux-weight with --aux-heads burial or both)")
    if ph_weight and aux_weight is None: raise ValueError("--ph-weight runs through AuxEngine (use --aux-weight, e.g. with --aux-heads burial)")
    if (geometry_weight or chi_weight) and (sidechain_view != "dropout" or aux_weight is None):
        raise ValueError("--geometry-weight / --chi-weight need --sidechains --sidechain-view dropout (side chains hidden from validation) and --aux-weight")
    table = bool(orientation or geometry_weight or chi_weight or dihedrals or hbonds)
    params, engine, predict = make_model(seed, query_norm, paired_weight, reduction, width, sidechains, depth, loops,
                                         orientation, sidechain_view, sidechain_drop, table, dihedrals, hbonds, variant, sidechain_source); state = engine.optimizer.init(params)
    custom_inputs = loops != 1 or table or sidechain_view != "all" or sidechain_source != "original"
    if custom_inputs and aux_weight is None and not dropout:
        raise ValueError("--loops / --orientation / --sidechain-view run through DropoutEngine / AuxEngine (use --dropout or --aux-weight)")
    if (depth != 2 or custom_inputs) and mask_pmin is not None: raise ValueError("depth / loops / input variants do not combine with site masking")
    embeddings_fn, shift_fn = model_inputs(loops, orientation, sidechain_view, sidechain_drop, table, dihedrals, hbonds, variant, sidechain_source)
    step_key = None
    if aux_weight is not None:
        if query_norm != "shared" or paired_weight != "none" or reduction != "structure" or mask_pmin is not None:
            raise ValueError("auxiliary heads combine with the defaults and dropout only")
        if chunked(batch): raise ValueError("the auxiliary objective runs the unchunked path; use batch <= 64")
        if aux_heads != "both" and aux_loss != "ce": raise ValueError("burial-only auxiliaries are implemented for --aux-loss ce")
        params = initialize_aux(params, seed, aux_loss, aux_heads)
        if geometry_weight: params = add_geometry_heads(params, seed)
        if chi_weight: params = add_chi_head(params, seed)
        if ph_weight:
            from .production_ph import initialize as initialize_ph
            params = initialize_ph(params, seed)
        if bind_weight:
            from .production_bind import initialize as initialize_bind
            params = initialize_bind(params, seed)
        if gate_weight: params = production_gate.initialize(params, seed)
        if residue_edge_weight: params = production_residue.initialize_edges(params, seed)
        engine = AuxEngine(params, aux_weight, dropout, aux_loss, aux_interface_branch, aux_heads, embeddings_fn, geometry_weight, chi_weight, ph_weight, ph_probe,
                           bind_weight, bind_operand, shift_table, gate_weight, gate_cys_weight, residue_burial, residue_edge_weight, residue_burial_siamese,
                           decoy_retype_weight, decoy_batch_weight, within_weight, swap_weight, swap_mode,
                           **({"hotspot_mask": mask_oracle} if mask_oracle is not None else {}),
                           **({"alt_rank_weight": alt_rank_weight, "alt_rank_margin": alt_rank_margin} if alt_rank_weight else {}),
                           **({"alt_cons_edge_weight": alt_cons_edge_weight, "alt_cons_pka_weight": alt_cons_pka_weight, "alt_cons_stopgrad": alt_cons_stopgrad}
                              if consistency else {}),
                           **({"alt_edge_only": True} if alt_edge_only else {}),
                           **({"quartet_weight": quartet_weight} if quartet_weight else {}),
                           **({"chain_quartet_weight": chain_weight, "chain_quartet_tau": chain_tau, "chain_quartet_k": chain_k} if chain_weight else {}))
        engine.ph_table = ph_table
        state = engine.optimizer.init(params)
        step_key = jax.random.PRNGKey(seed + 200003)
    elif dropout:
        if query_norm != "shared" or paired_weight != "none" or reduction != "structure" or mask_pmin is not None:
            raise ValueError("dropout combines with the defaults only")
        if chunked(batch): raise ValueError("dropout runs the unchunked path; use batch <= 64")
        engine = DropoutEngine(params, dropout, shift_fn); state = engine.optimizer.init(params); step_key = jax.random.PRNGKey(seed + 200003)
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
    epochs = 2 if smoke else total
    edge_check = None  # anm-edge-v1 smoke: alt_edge_only_check on the first batch with an alt visit
    for epoch in range(len(history) + 1, epochs + 1):
        if stale >= PATIENCE: break
        pair, pk = epoch_plans(manifests, fraction, epoch, batch, seed)
        chain_regroup = None; chain_counts = production_residue.chain_stats() if chain_weight else None
        if chain_weight: pair, chain_regroup = production_residue.chain_regroup(pair, chain_meta, chain_bucket, chain_abag_min)  # quartet-rank-v2
        if smoke: pair, pk = pair[:smoke_batches], pk[:smoke_batches]
        specs = list(zip(pair, pk)); deferred = DeferredScalars(every=50); began = time.time()
        prefetcher = Prefetcher(sources["train"], specs, config, to_device if not chunked(batch) else host_chunks)
        source_before = train_pinder.alt.source_snapshot() if alt_prob else None
        ph_losses = []; bind_losses = []; decoy_values = []; alt_values = []; mask_totals = np.zeros(4); side_totals = np.zeros(4)  # v2: unmasked, side 0, side 1, both
        quartet_norms = []  # quartet-rank-v1: debug_quartet_grad_norms (smoke / --quartet-grad-norms: last batch of the epoch, before its update)
        for number, ((pair_ids, pk_ids), (pair_batch, pk_batch)) in enumerate(prefetcher, 1):
            rate = jnp.asarray(scale * learning_rate(epoch, number, len(specs)), jnp.float32)
            if step_key is not None:
                graphs, targets, mask, wb, wi, _, valid = pair_batch
                k_pair, k_pk = jax.random.split(jax.random.fold_in(jax.random.fold_in(step_key, epoch), number))
                if chain_weight:  # quartet-rank-v2: batch-level chain quartets, stream [CHAIN_SALT, seed, epoch, batch number]
                    role, cq = production_residue.chain_quartet_batch(pair_ids, len(valid), chain_meta, chain_count,
                                                                      np.random.default_rng([production_residue.CHAIN_SALT, seed, epoch, number]), chain_counts)
                    graphs = {**graphs, "chain_role": jax.device_put(role), "chain_quartet": jax.device_put(cq)}
                if ph_table is not None:
                    ph_pair = jax.device_put(ph_table.arrays("pinder", pair_ids, len(valid)))
                    ph_pk = jax.device_put(ph_table.arrays("pkpdb", pk_ids, len(pk_batch[-1])))
                    (pair_total, pair_aux), pair_gradient = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, valid, k_pair, ph_pair)
                    state_loss, pair_loss, ph_loss, bind_loss = pair_aux[:4]
                    pk_loss, pk_gradient = engine.pkpdb_value_grad(params, *pk_batch, k_pk, ph_pk); ph_losses.append(ph_loss)
                else:
                    (pair_total, pair_aux), pair_gradient = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, valid, k_pair)
                    state_loss, pair_loss, _, bind_loss = pair_aux[:4]
                    pk_loss, pk_gradient = engine.pkpdb_value_grad(params, *pk_batch, k_pk)
                if len(pair_aux) > 4: decoy_values.append(jnp.stack([jnp.asarray(v, jnp.float32) for v in pair_aux[4]]))
                if (quartet_weight or chain_weight) and (smoke or quartet_grad_norms_every_epoch) and number == len(specs):
                    try: quartet_norms.append(quartet_grad_norms(engine, params, graphs, targets, mask, wb, wi, valid, k_pair, epoch, number, (epoch - 1) * len(specs) + number - 1))
                    except Exception as error:  # noqa: BLE001  (diagnostic only; the smoke must fail loudly)
                        if smoke: raise
                        quartet_norms.append({"epoch": epoch, "batch": number, "error": repr(error)}); print(f"debug_quartet_grad_norms_error {error!r}", flush=True)
                if alt_edge_only and smoke and edge_check is None and float(jnp.sum(graphs["use_alt"][:, 0] * valid)) > 0:
                    edge_check = alt_edge_only_check(engine, params, graphs, targets, mask, wb, wi, valid, k_pair, epoch, number)
                if "use_alt" in graphs: alt_values.append(jnp.sum(graphs["use_alt"][:, 0] * valid) / jnp.maximum(jnp.sum(valid), 1))
                if mask_oracle is not None:  # (kept, surface, fallback) of the valid complexes, free branch
                    ms = np.asarray(graphs["mask_stats"][:, 1]) * np.asarray(valid)[:, None]; mask_totals += np.r_[ms.sum(axis=0), np.asarray(valid).sum()]
                    if mask_v2:  # mask_side of the valid complexes: -1 / 0 / 1 / 2 -> slots 0..3
                        sd_ = np.asarray(graphs["mask_side"][:, 1])[np.asarray(valid, bool)]; side_totals += np.bincount(sd_ + 1, minlength=4)[:4]
            elif not chunked(batch):
                graphs, targets, mask, wb, wi, _, valid = pair_batch
                (pair_total, (state_loss, pair_loss)), pair_gradient = engine.paired_value_grad(params, graphs, targets, mask, wb, wi, valid)
                pk_loss, pk_gradient = engine.pkpdb_value_grad(params, *pk_batch)
            else:
                (pair_total, (state_loss, pair_loss)), pair_gradient = _accumulate(lambda *a: engine.paired_value_grad(params, *a), pair_batch)
                pk_loss, pk_gradient = _accumulate(lambda *a: engine.pkpdb_value_grad(params, *a), pk_batch)
            if step_key is not None and getattr(engine, "bind_weight", 0.0): bind_losses.append(bind_loss)
            params, state, finite = engine.apply(params, state, pair_gradient, pk_gradient, rate)
            losses = jnp.stack((pair_total + pk_loss, pk_loss, state_loss, pair_loss))
            if not bool(finite & jnp.all(jnp.isfinite(losses))): raise FloatingPointError(f"nonfinite update at epoch {epoch} batch {number}")
            deferred.add(losses)
        deferred.flush(); train_seconds = time.time() - began; values = np.asarray(deferred.values).reshape(-1, 4)
        telemetry = prefetcher.telemetry.summary()
        began = time.time(); validation = validate(engine, predict, params, sources, manifests, config); val_seconds = time.time() - began
        save_checkpoint(out / "checkpoints" / f"epoch-{epoch:03d}", params, state, {"epoch": epoch, "run": run})
        row = {"epoch": epoch, "updates": len(specs), "learning_rate_end": scale * learning_rate(epoch, len(specs), len(specs)),
               "train_loss": {**{name: float(values[:, i].mean()) for i, name in enumerate(("total", "pkpdb", "pinder_state", "pinder_paired"))},
                              **({"pinder_ph_bce": float(np.mean(np.asarray(jnp.stack(ph_losses))))} if ph_losses else {}),
                              **({"pinder_bind_ce": float(np.mean(np.asarray(jnp.stack(bind_losses))))} if bind_losses else {}),
                              **(dict(zip(("pinder_decoy_retype_margin", "pinder_decoy_batch_margin", "pinder_within_listwise", "pinder_swap_margin",
                                           "pinder_mask_free_edge_bce", "pinder_alt_rank_margin_loss", "pinder_alt_cons_edge", "pinder_alt_cons_pka",
                                           "pinder_alt_edge_only_burial", "pinder_alt_edge_only_gate",
                                           "pinder_quartet_softplus", "pinder_quartet_accuracy", "pinder_quartet_covered"),
                                          map(float, np.mean(np.asarray(jnp.stack(decoy_values)), axis=0))))
                                 if decoy_values else {}),
                              **(chain_history(np.mean(np.asarray(jnp.stack(decoy_values)), axis=0)) if chain_weight and decoy_values else {}),
                              **({"pinder_mask_kept_fraction": float(mask_totals[0] / max(mask_totals[1], 1.0)), "pinder_mask_kept": int(mask_totals[0]),
                                  "pinder_mask_surface": int(mask_totals[1]), "pinder_mask_fallbacks": int(mask_totals[2]), "pinder_mask_complexes": int(mask_totals[3])}
                                 if mask_oracle is not None else {}),
                              **({"pinder_mask_visit_fraction": float(side_totals[1:].sum() / max(side_totals.sum(), 1.0)),
                                  "pinder_mask_side_fraction": float(side_totals[2] / max(side_totals[1] + side_totals[2], 1.0)),
                                  "pinder_mask_side_counts": {"none": int(side_totals[0]), "chain0": int(side_totals[1]), "chain1": int(side_totals[2]), "both": int(side_totals[3])}}
                                 if mask_oracle is not None and mask_v2 else {}),
                              **({"pinder_alt_fraction": float(np.mean(np.asarray(jnp.stack(alt_values))))} if alt_values else {}),
                              **({"pinder_alt_source_counts": {label: {k: v - source_before[label].get(k, 0) for k, v in counts.items() if v - source_before[label].get(k, 0)}
                                                               for label, counts in train_pinder.alt.source_snapshot().items()}} if source_before is not None else {})},
               **({"debug_alt_edge_only_check": edge_check} if edge_check is not None and (not history or "debug_alt_edge_only_check" not in history[-1]) else {}),
               **({"loader_counts": {**({"alt": dict(train_pinder.alt.counts)} if train_pinder.alt is not None else {}),
                                     **({"val_alt": dict(sources["pinder-val-alt"].alt.counts)} if "pinder-val-alt" in sources else {})}}
                  if alt_store is not None else {}),
               **({"within_counts": {**production_residue.WITHIN_STATS, "dropped_fraction": production_residue.WITHIN_STATS["dropped_drsa"] / max(production_residue.WITHIN_STATS["anchors"], 1),
                                     "note": "cumulative over training visits (production_residue.within_sample)"}} if within_weight and within_target != "distance" else {}),
               **({"quartet_counts": {**production_residue.QUARTET_STATS, "pool_cache": len(production_residue.QUARTET_POOLS),
                                      "note": "cumulative over training visits (production_residue.quartet_fields)"}} if quartet_weight else {}),
               **({"debug_quartet_grad_norms": quartet_norms} if quartet_norms else {}),
               **({"chain_quartet_counts": chain_counts, "chain_regroup": chain_regroup} if chain_weight else {}),
               "validation": validation, "train_seconds": round(train_seconds, 1), "validation_seconds": round(val_seconds, 1),
               "loader_wait_fraction": telemetry.get("wait_fraction")}
        if chain_weight and not quartet_weight:  # quartet-rank-v2 chain-only: slots 10-12 are padding, not a residue quartet term
            for name in ("pinder_quartet_softplus", "pinder_quartet_accuracy", "pinder_quartet_covered"): row["train_loss"].pop(name, None)
        history.append(row)
        with open(out / "history.jsonl", "a") as handle: handle.write(json.dumps(row) + "\n")
        improved = best is None or validation["selection"] < best["validation"]["selection"] - MIN_DELTA
        if best is None or validation["selection"] < best["validation"]["selection"]: best = row
        stale = 0 if improved else stale + 1
        print(json.dumps({"epoch": epoch, "loss": row["train_loss"]["total"], "pinder_state_mae": validation["pinder"]["state_mae"],
                          "interface_paired_mae": validation["pinder"]["interface_paired_mae"],
                          "benchmark_mae": validation["benchmark"]["overall"]["mae"], "pkpdb_val_mae": validation.get("pkpdb", {}).get("site_mae"),
                          "selection": validation["selection"],
                          **({"terms": {k: row["train_loss"].get(k) for k in ("pinder_state", "pinder_paired", "pkpdb", "pinder_within_listwise", "pinder_mask_free_edge_bce",
                                                                             "pinder_alt_rank_margin_loss", "pinder_alt_cons_edge", "pinder_alt_cons_pka")},
                              "weighted": {"cons_edge": alt_cons_edge_weight * row["train_loss"].get("pinder_alt_cons_edge", 0.0),
                                           "cons_pka": alt_cons_pka_weight * row["train_loss"].get("pinder_alt_cons_pka", 0.0)}} if consistency else {}),
                          **({"alt_selection": (validation["pinder_alt"]["alt"] or {}).get("selection"), "alt_complexes": validation["pinder_alt"]["complexes"]}
                             if "pinder_alt" in validation else {}),
                          **({"ph_mae": {k: validation["ph"][k].get("mae") for k in ("pinder", "pkpdb", "benchmark") if k in validation["ph"]},
                              "ph_baseline_mae": {k: validation["ph"][k].get("baseline_mae") for k in ("pinder", "pkpdb", "benchmark") if k in validation["ph"]}}
                             if "ph" in validation else {}),
                          "train_s": row["train_seconds"], "val_s": row["validation_seconds"], "wait": row["loader_wait_fraction"]}), flush=True)
    selected = best["epoch"]
    params, state, _ = load_checkpoint(out / "checkpoints" / f"epoch-{selected:03d}", (params, state))
    final = validate(engine, predict, params, sources, manifests, config, out=out, epoch=selected)
    if contacts is not None:
        aux_sources = (AuxPinderSource(_with_policy(manifests["pinder"]), contacts, config=config, norms=norms, sidechains=side["pinder"]),
                       PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask", aux=True, sidechains=side["pkpdb"])
                       if "pkpdb-val" in sources else None)
        if gate_weight:
            for d, source in zip(("pinder", "pkpdb"), aux_sources):
                if source is not None: source.disulfides = production_gate.DisulfideTable(root, d)
        final["auxiliary"] = auxiliary_metrics(engine, params, aux_sources[0], select(manifests["pinder"], "val"), aux_sources[1],
                                               select(manifests["pkpdb"], "val"), config)
        for source in aux_sources:
            if source is not None: source.close()
    for source in sources.values(): source.close()
    atomic_json(out / "selection.json", {"selected_epoch": selected, "epochs_run": len(history), "parameter_count": parameter_count(params), "validation": final,
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
    if "sidechains" in protocol: raise NotImplementedError("aux-rescore does not load side-chain tables")
    arch = protocol["architecture"]; loops = arch.get("loops", 1)
    params, _, _ = make_model(protocol["seed"], width=arch["width"], depth=arch.get("depth", 2), loops=loops); params = initialize_aux(params, protocol["seed"], mode, heads)
    engine = AuxEngine(params, weight, dropout, mode, branch, heads, model_functions(loops)[0]); state = engine.optimizer.init(params)
    params, state, _ = load_checkpoint(out / "checkpoints" / f"epoch-{selection['selected_epoch']:03d}", (params, state))
    sources = (AuxPinderSource(_with_policy(manifests["pinder"]), contacts, config=config, norms=norms),
               PkpdbSource(_with_policy(manifests["pkpdb"]), config=config, mask="train_mask", aux=True))
    metrics = auxiliary_metrics(engine, params, sources[0], select(manifests["pinder"], "val"), sources[1], select(manifests["pkpdb"], "val"), config)
    for source in sources: source.close()
    result = {"run": run, "selected_epoch": selection["selected_epoch"], "mode": mode, "weight": weight, "interface_trained_branch": branch,
              "contacts_verification_sha256": contacts.verification_sha256, "auxiliary": metrics}
    atomic_json(out / "aux-rescore.json", result); return result


def protocol_inputs(protocol):
    """(sidechain_view, sidechain_drop, orientation, table, dihedrals, hbonds) as train() recorded them in protocol.json."""
    arch = protocol["architecture"]; view, drop = "all", 0.0
    if "sidechain_view" in protocol:
        text = protocol["sidechain_view"]
        view = "dropout" if text.startswith("side-chain features hidden") else "titratable" if text.startswith("side-chain features on D/E") else "non-titratable"
        if view == "dropout": drop = float(text.split("probability ")[1].split(" ")[0])
    orientation = "orientation_edges" in arch; dihedrals = "dihedral_features" in arch; hbonds = "hbond_features" in arch
    table = bool(orientation or dihedrals or hbonds or "geometry_auxiliary" in protocol or "chi_auxiliary" in protocol)
    return view, drop, orientation, table, dihedrals, hbonds


def selected_model(root, run=None):
    """(params at the selected epoch, model_inputs functions (site_embeddings, predict_shift)) rebuilt from a run's
    protocol: architecture, input variants and auxiliary heads, as train() built them."""
    import optax
    from .gqt_regularization import decay_mask
    from .trainer import load_checkpoint
    if run is None:
        from .production_defaults import RELATIONAL_RUN
        run = RELATIONAL_RUN
    out = run_dir(root, run); protocol = read(out / "protocol.json"); selection = read(out / "selection.json"); arch = protocol["architecture"]
    view, drop, orientation, table, dihedrals, hbonds = protocol_inputs(protocol)
    params, _, _ = make_model(protocol["seed"], width=arch["width"], sidechains="sidechains" in protocol, depth=arch.get("depth", 2),
                              loops=arch.get("loops", 1), orientation=orientation, sidechain_view=view, sidechain_drop=drop, table=table, dihedrals=dihedrals,
                              hbonds=hbonds, variant=protocol.get("production_variant", "none"), sidechain_source=protocol.get("sidechain_source", {}).get("source", "original"))
    if "auxiliary" in protocol:
        text = protocol["auxiliary"]; mode = "ce" if text.startswith("AuxEngine ce") else "ordinal"
        heads = "burial" if text.startswith("AuxEngine ce burial-only") else "burial-siamese" if text.startswith("AuxEngine ce burial-siamese") else "both"
        params = initialize_aux(params, protocol["seed"], mode, heads)
    if "geometry_auxiliary" in protocol: params = add_geometry_heads(params, protocol["seed"])
    if "chi_auxiliary" in protocol: params = add_chi_head(params, protocol["seed"])
    if "titratable_gate" in protocol: params = production_gate.initialize(params, protocol["seed"])
    if "residue_edges" in protocol: params = production_residue.initialize_edges(params, protocol["seed"])
    # the engines' optimizer (JointEngine / DropoutEngine / AuxEngine), only for the checkpoint's tree structure
    state = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1.0, weight_decay=1e-4, mask=decay_mask(params))).init(params)
    params, _, _ = load_checkpoint(out / "checkpoints" / f"epoch-{selection['selected_epoch']:03d}", (params, state))
    return params, model_inputs(arch.get("loops", 1), orientation, view, drop, table, dihedrals, hbonds, protocol.get("production_variant", "none"),
                                protocol.get("sidechain_source", {}).get("source", "original"))


def export_model(root, run, destination):
    """Write a run's selected parameters (no optimizer state) to `destination`: params.npz (one array per parameter,
    keyed by its tree path, e.g. "blocks/0/q/w") and model.json (architecture, parameter count, the run's protocol and
    selection metrics, checkpoint and params sha256). Load with load_exported."""
    import jax
    root = Path(root); out = run_dir(root, run); protocol = read(out / "protocol.json"); selection = read(out / "selection.json")
    if protocol.get("production_variant") == "paired-delta": raise NotImplementedError("paired-delta needs paired inference packaging")
    params, _ = selected_model(root, run)
    flat = {"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): np.asarray(v) for path, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "params.npz", **flat)
    atomic_json(destination / "model.json", {"run": run, "model": "oGQT (pkanet.ogqt), shift head predicts pKa - PKPDB_PK_MOD[group]",
        "architecture": protocol["architecture"], "parameter_count": parameter_count(params),
        "selected_epoch": selection["selected_epoch"], "epochs_run": selection["epochs_run"], "validation": selection["validation"],
        "protocol": {k: v for k, v in protocol.items() if k != "code"}, "code": protocol["code"],
        "checkpoint_sha256": selection["checkpoint_sha256"], "params_sha256": digest(destination / "params.npz")})
    return destination


def load_exported(path):
    """Parameter tree from export_model's params.npz (nested dicts; integer path parts become list indices)."""
    import jax.numpy as jnp
    tree = {}
    with np.load(Path(path) / "params.npz", allow_pickle=False) as f:
        for name in f.files:
            node = tree; parts = name.split("/")
            for part in parts[:-1]: node = node.setdefault(part, {})
            node[parts[-1]] = jnp.asarray(f[name])
    def lists(node):
        if not isinstance(node, dict): return node
        if node and all(k.isdigit() for k in node): return [lists(node[str(i)]) for i in range(len(node))]
        return {k: lists(v) for k, v in node.items()}
    return lists(tree)


def exported_functions(path):
    """model_inputs functions (site_embeddings, predict_shift) for an exported model, rebuilt from its model.json; use
    with load_exported's params on graphs carrying the node tables the model was trained with (model.json "protocol")."""
    model = read(Path(path) / "model.json"); view, drop, orientation, table, dihedrals, hbonds = protocol_inputs(model["protocol"])
    return model_inputs(model["architecture"].get("loops", 1), orientation, view, drop, table, dihedrals, hbonds, model["protocol"].get("production_variant", "none"),
                        model["protocol"].get("sidechain_source", {}).get("source", "original"))


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
    p.add_argument("--width", type=int, default=None, help="model width (ff = 2 x width); default ARCHITECTURE")
    p.add_argument("--sidechains", action="store_true", help="append side-chain node features (production_sidechains)")
    p.add_argument("--depth", type=int, default=2, help="untied residue-encoder blocks (production_depth); default 2")
    p.add_argument("--loops", type=int, default=1, help="tie the 2 residue blocks and loop them r times with input re-injection (production_depth)")
    p.add_argument("--orientation", action="store_true", help="13 orientation features on the residue edges (production_inputs, backbone-v1)")
    p.add_argument("--sidechain-view", choices=("all", "titratable", "non-titratable", "dropout"), default="all")
    p.add_argument("--sidechain-drop", type=float, default=0.0, help="with --sidechain-view dropout: per-structure probability of hiding side chains")
    p.add_argument("--gate-weight", type=float, default=0.0, help="titratable gate BCE weight on all-residue tokens (production_gate)")
    p.add_argument("--gate-cys-weight", type=float, default=10.0, help="BCE weight of cysteine side-chain rows")
    p.add_argument("--drop-bonded-cys", action="store_true", help="drop bonded-Cys queries from the training pKa loss (production_gate)")
    p.add_argument("--residue-burial", action="store_true", help="PINDER burial on every residue token (production_residue)")
    p.add_argument("--residue-burial-siamese", action="store_true", help="per-residue burial on both branches + paired change term (implies --residue-burial)")
    p.add_argument("--residue-edge-weight", type=float, default=0.0, help="residue contact-edge head weight (production_residue)")
    p.add_argument("--decoy-retype-weight", type=float, default=0.0, help="margin loss vs re-typed decoy partners on the edge head (production_residue; needs --residue-edge-weight)")
    p.add_argument("--decoy-batch-weight", type=float, default=0.0, help="margin loss vs in-batch cross-complex decoy partners on the edge head (production_residue)")
    p.add_argument("--within-weight", type=float, default=0.0, help="within-complex interface-vs-interface listwise loss on the edge head (production_residue, negatives-v1)")
    p.add_argument("--within-target", choices=("distance", "drsa"), default="distance",
                   help="within listwise target: soft(d) (negatives-v1) or soft(d) * min(dRSA_a, dRSA_c) with zero-target anchors dropped (drsa-within-v1)")
    p.add_argument("--swap-weight", type=float, default=0.0, help="margin loss vs same-stratum in-batch swapped partners, homodimers excluded (production_residue, negatives-v1)")
    p.add_argument("--swap-mode", choices=("class", "curv"), default="class", help="swap decoy residue: cyclic contact slot (class) or nearest curvature proxy (curv)")
    p.add_argument("--mask-oracle", choices=("pesto", "own", "random"), default=None,
                   help="hotspot mask (hotspot-mask-v1): free edge BCE + distillation + within term on a pass with residue types hidden outside the oracle's top surface residues")
    p.add_argument("--mask-store", default=None, help="oracle side store (pesto / own_edgemax per residue; production_residue.HotspotStore)")
    p.add_argument("--mask-keep", type=float, default=0.25, help="unmasked fraction of each chain's surface residues (top by the oracle)")
    p.add_argument("--mask-soft", type=float, default=0.0, help="additionally unmask this fraction of the remaining surface residues at random")
    p.add_argument("--mask-sides", choices=("both", "one"), default="both", help="hotspot-mask-v2: mask both chains (v1) or one chain drawn per visit, the other fully unmasked")
    p.add_argument("--alt-source", default=None, help="alt free-branch side store(s) (production_alt; apo-negatome-v1 schema); anm-aug-v1: comma-separated store[:keysuffix] entries, records <cid>::<suffix>")
    p.add_argument("--alt-prob", type=float, default=0.0, help="per complex visit probability of the alt free branch (production_alt)")
    p.add_argument("--val-alt", action="store_true", help="extra PINDER validation pass with the alt free branch (reported, not selected on)")
    p.add_argument("--alt-rank-weight", type=float, default=0.0, help="alt-vs-crystal margin term on the edge head (alt visits; extra crystal free pass)")
    p.add_argument("--alt-rank-margin", type=float, default=0.0, help="margin m in softplus(m - (l_alt - l_xtal)) (default 0)")
    p.add_argument("--alt-weights", type=lambda s: [float(x) for x in s.split(",")], default=None,
                   help="anm-aug-v1: comma-separated choice weights, one per --alt-source entry (default uniform)")
    p.add_argument("--alt-cons-edge-weight", type=float, default=0.0, help="anm-aug-v1: symmetric (l_alt - l_xtal)^2 on the sampled pairs (alt visits)")
    p.add_argument("--alt-cons-pka-weight", type=float, default=0.0, help="anm-aug-v1: symmetric (pKa_alt - pKa_xtal)^2 on free-state sites (alt visits)")
    p.add_argument("--alt-cons-stopgrad", action="store_true", help="anm-aug-v1: detach the crystal side of the consistency terms")
    p.add_argument("--alt-edge-only", action="store_true", help="anm-edge-v1: alt geometry feeds only the edge head (edge BCE, within, masked pass); "
                                                                  "pKa / burial / gate read the crystal free pass")
    p.add_argument("--mask-prob", type=float, default=1.0, help="hotspot-mask-v2: probability a visit is masked (else identity mask, losses as unmasked)")
    p.add_argument("--sidechain-source", choices=SIDECHAIN_SOURCES, default="original", help="with --sidechains: deposited, FASPR-packed or a per-structure mixture (production_packed)")
    p.add_argument("--geometry-weight", type=float, default=0.0, help="titratable side-chain geometry auxiliary weight (production_inputs)")
    p.add_argument("--dihedrals", action="store_true", help="sin/cos phi, psi, omega node features from backbone-v1 (production_inputs)")
    p.add_argument("--chi-weight", type=float, default=0.0, help="chi1 rotamer-class auxiliary weight (production_inputs)")
    p.add_argument("--hbonds", action="store_true", help="DSSP backbone H-bond node features from backbone-v1 (production_inputs)")
    p.add_argument("--epochs", type=int, default=None, help="schedule length (default 20; others stretch the schedule and patience)")
    p.add_argument("--ph-weight", type=float, default=0.0, help="structure-level experimental pH auxiliary weight (production_ph; needs --aux-weight)")
    p.add_argument("--ph-sources", choices=("all", "crystal"), default="all", help="pH targets: all deposited sources, or crystallisation pH only")
    p.add_argument("--bind-weight", type=float, default=0.0, help="paired binding head weight (production_bind; needs --aux-weight)")
    p.add_argument("--bind-operand", choices=("product", "difference"), default="product", help="binding head operand: a * b or a - b")
    p.add_argument("--shift-weight", choices=("none", "invsqrt"), default="none")
    p.add_argument("--variant", choices=("none", "fourier", "rope", "relations", "shared-projection", "paired-delta"), default="none")
    p.add_argument("--relation-maps", default=None, help="A,B relation maps in residue layers 1,2 (each 6, 4 or 1; needs --variant relations); "
                   "recorded as production variant relations-A-B (production_position.REMAP)")
    p.add_argument("--layer2-attention", choices=("graph", "full"), default="graph",
                   help="full: residue layer 2 attends to every residue (needs --relation-maps A,1; variant relations-A-1-full)")
    p.add_argument("--site-blocks", type=int, choices=(1, 2), default=1,
                   help="2: a second site-to-site attention block (needs --relation-maps; variant suffix -site2)")
    p.add_argument("--all-residue-tokens", action="store_true",
                   help="every residue is a site token with side-chain / N- / C-terminus outputs (needs --relation-maps; suffix -allres)")
    p.add_argument("--token-cutoff", default=None,
                   help="S or S,C: all-residue token layers use CA-CA <= S A same-chain, <= C A other-chain (needs --all-residue-tokens; suffix -cutS / -cutSxC)")
    p.add_argument("--ph-probe", action="store_true", help="pH head on stop-gradient site tokens with its own optimizer (trunk unaffected)")
    p.add_argument("--quartet-weight", type=float, default=0.0,
                   help="quartet-rank-v1: native-pairing quartet loss weight on the unmasked free-branch edge logits (production_residue.quartet_terms; 0 = off)")
    p.add_argument("--quartet-count", type=int, default=None, help="quartets per complex visit (default production_residue.QUARTET_COUNT = 128; half chemically matched)")
    p.add_argument("--quartet-grad-norms", action="store_true", help="quartet-rank-v1: debug_quartet_grad_norms on the last batch of every epoch (always on under --smoke)")
    p.add_argument("--smoke-batches", type=int, default=5, help="with --smoke: PINDER / pKPDB batches per smoke epoch (default 5)")
    p.add_argument("--pinder-manifest", default=None, help="quartet-rank-v1b: PINDER pool manifest replacing manifest-v1.json (e.g. training/pool-v5; production_extra)")
    p.add_argument("--pinder-extra", default=None, help="quartet-rank-v1b: runtime-shaped root of the extra records' store-v2 + side tables (production_extra)")
    p.add_argument("--mask-store-extra", default=None, help="quartet-rank-v1b: hotspot oracle store for the extra records (with --mask-store)")
    p.add_argument("--chain-quartet-weight", type=float, default=0.0,
                   help="quartet-rank-v2: cross-complex chain quartet loss weight, soft top-k pooled free-branch edge logits (production_residue.chain_quartet_terms; 0 = off)")
    p.add_argument("--chain-quartet-count", type=int, default=None, help="chain quartets per batch (default production_residue.CHAIN_COUNT = 16)")
    p.add_argument("--chain-quartet-tau", type=float, default=None, help="pooling temperature tau (default 1)")
    p.add_argument("--chain-quartet-k", type=int, default=None, help="pooling k: S = tau logsumexp(s / tau) - tau log k (default 32)")
    p.add_argument("--chain-quartet-abag-min", type=int, default=None, help="Ab/Ag complexes regrouped per Ab/Ag batch inside each size bucket (default 2)")
    p = sub.add_parser("count"); p.add_argument("widths", type=int, nargs="+"); p.add_argument("--sidechains", action="store_true")
    p.add_argument("--depth", type=int, default=2); p.add_argument("--loops", type=int, default=1)
    p = sub.add_parser("export"); p.add_argument("run"); p.add_argument("destination")
    p = sub.add_parser("rescore"); p.add_argument("runs", nargs="+")
    p = sub.add_parser("aux-rescore"); p.add_argument("runs", nargs="+")
    args = parser.parse_args(argv); root = Path(os.environ["PKABENCH_RUNTIME"])
    if args.action == "rescore":
        for run in args.runs: rescore(root, run)
        return
    if args.action == "export":
        print(export_model(root, args.run, args.destination)); return
    if args.action == "count":
        for w in args.widths:
            params = make_model(SEED, width=w, sidechains=args.sidechains, depth=args.depth, loops=args.loops)[0]
            print(json.dumps({"width": w, "ff": 2 * w, "sidechains": args.sidechains, "depth": args.depth, "loops": args.loops, "parameters": parameter_count(params),
                              "with_burial_head": parameter_count(initialize_aux(params, SEED, "ce", "burial"))}))
        return
    if args.action == "aux-rescore":
        for run in args.runs:
            r = aux_rescore(root, run)
            print(json.dumps({"run": run, **{n: [round(b["auroc"], 3) for b in v["boundaries"]] for n, v in r["auxiliary"].items()}}), flush=True)
        return
    if args.action == "train":
        if args.relation_maps is not None:
            from .production_position import relation_maps
            if args.variant != "relations": raise ValueError("--relation-maps needs --variant relations")
            args.variant = "relations-" + "-".join(c.strip() for c in args.relation_maps.split(","))
            relation_maps(args.variant)  # validates the counts
        if args.layer2_attention == "full":
            if args.relation_maps is None: raise ValueError("--layer2-attention full needs --relation-maps A,1")
            args.variant += "-full"
        if args.site_blocks == 2:
            if args.relation_maps is None: raise ValueError("--site-blocks 2 needs --relation-maps")
            args.variant += "-site2"
        if args.all_residue_tokens:
            if args.relation_maps is None: raise ValueError("--all-residue-tokens needs --relation-maps")
            args.variant += "-allres"
        if args.token_cutoff is not None:
            from .production_position import token_cutoff
            if not args.all_residue_tokens: raise ValueError("--token-cutoff needs --all-residue-tokens")
            radii = [c.strip() for c in args.token_cutoff.split(",")]
            args.variant += "-cut" + "x".join(radii); token_cutoff(args.variant)  # validates the radii
        result = train(root, args.run, args.fraction, args.smoke, args.batch, args.paired_weight, args.seed, args.query_norm, args.loss_reduction, args.site_mask_pmin,
                       args.dropout, args.aux_weight, args.aux_loss, args.aux_interface_branch,
                       args.aux_heads, args.width, args.sidechains, args.depth, args.loops, args.orientation, args.sidechain_view, args.sidechain_drop,
                       args.geometry_weight, args.dihedrals, args.chi_weight, args.hbonds, args.epochs, args.ph_weight,
                       args.ph_sources, args.ph_probe, args.bind_weight, args.bind_operand, args.shift_weight, args.variant, args.sidechain_source,
                       args.gate_weight, args.gate_cys_weight, args.drop_bonded_cys, args.residue_burial, args.residue_edge_weight,
                       args.residue_burial_siamese, args.decoy_retype_weight, args.decoy_batch_weight, args.within_weight, args.swap_weight, args.swap_mode,
                       args.mask_oracle, args.mask_store, args.mask_keep, args.mask_soft, args.mask_sides, args.mask_prob,
                       args.alt_source, args.alt_prob, args.val_alt, args.alt_rank_weight, args.alt_rank_margin,
                       args.alt_weights, args.alt_cons_edge_weight, args.alt_cons_pka_weight, args.alt_cons_stopgrad, args.alt_edge_only, args.within_target,
                       args.quartet_weight, args.quartet_count, args.smoke_batches, args.quartet_grad_norms,
                       args.pinder_manifest, args.pinder_extra, args.mask_store_extra, args.chain_quartet_weight, args.chain_quartet_count,
                       args.chain_quartet_tau, args.chain_quartet_k, args.chain_quartet_abag_min)
        print(json.dumps({"selected_epoch": result["selected_epoch"], "selection": result["validation"]["selection"]}))


if __name__ == "__main__":
    main()
