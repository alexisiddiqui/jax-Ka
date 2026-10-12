"""Structure-level experimental pH auxiliary for production oGQT training (2026-10-12, --ph-weight).

Target: the deposited structure-condition pH (crystal growth, EM buffer or NMR sample pH; pkabench.experimental_ph),
not binding-assay pH. One value per structure, shared by all its sites, so the head pools the site tokens and predicts
once per structure: never one loss term per residue.

- Annotations: pool-v4 training records from annotations/experimental-ph-v2/{pinder,pkpdb} (experimental_ph.build_v4),
  validation records from annotations/validation-ph-v1/{pinder-pool-v4, pkpdb-pool-v4, benchmark_active_val142}.
  Joined by structure id; every manifest record (training fraction and validation) must have an annotation with the
  manifest's split, else PhTable raises. ATLAS and PDBbind have no production store here; ATLAS pH stays masked.
- Weight per structure: ph_quality_weight (binary eligibility; 0 for masked targets) x parent_weight (1 / entries
  sharing the parent PDB in the split), so a PDB split into several PINDER dimers counts once.
- Head (aux_ph): gated attention pooling over the site tokens, LayerNorm first (the tokens are the un-normalised
  residual stream): g_i = sigmoid(W_g h_i), v_i = W_v h_i, z = sum_i m_i g_i * v_i / sum_i m_i g_i (per channel),
  logits = W_o z, one logit per threshold. PINDER: AB branch only (the deposited structure).
- Loss: cumulative binary cross-entropy over [pH > t] for t in THRESHOLDS (between the 0.5-unit heaps of deposited pH;
  P(pH > t) 0.91-0.96 / 0.78-0.88 / 0.54-0.73 / 0.20-0.22 / 0.07-0.13 on pool-v4 training), mean over thresholds,
  weighted mean over the batch's structures; a batch without eligible structures contributes 0.
- Evaluation (fit on training labels only, `fit`): expected pH = sum over the threshold bins of P(bin) x the training
  mean pH in that bin (P from the cumulative sigmoids, made monotone); compared with the training-mean baseline (MAE /
  RMSE), plus Spearman, per-threshold AUROC and mean BCE, overall and per source (crystal / EM buffer / NMR).
- Variants (2026-10-12, after the first pilot's head learned EM vs crystal rather than pH):
  sources "crystal" (--ph-sources crystal): only crystallisation-pH targets; EM-buffer and NMR structures are masked
  everywhere (training loss, the fit, validation), so the baseline is the crystal training mean.
  probe (--ph-probe): the head reads stop_gradient(tokens) and has its own optimizer (separate_optimizer: own global
  clip), so the pH loss cannot change the trunk; it measures what the pKa-trained trunk already encodes.
"""
from __future__ import annotations

import numpy as np

from pkabench.experimental_ph import FORMAT_V4, VALIDATION_V1, load_v4, load_validation

THRESHOLDS = (4.75, 5.75, 6.75, 7.75, 8.25)
SALT = 7_0_1_4  # fold_in salt for the head's key (initialisation independent of the other heads)
SOURCES = ("all", "crystal")
VALIDATION_COHORTS = {"pinder": "pinder-pool-v4", "pkpdb": "pkpdb-pool-v4", "benchmark-val": "benchmark_active_val142"}


def source_category(row):
    sources = set(row["ph_sources"])
    if any(s.startswith("crystallization") for s in sources): return "crystal"
    if "em_buffer" in sources: return "em_buffer"
    if "nmr" in sources: return "nmr"
    return "none"


class PhTable:
    """Per-structure pH annotations for the production manifests, verified against their membership."""

    def __init__(self, root, manifests, fraction, sources="all"):
        if sources not in SOURCES: raise ValueError(sources)
        self.rows = {"pinder": dict(load_v4(root, "pinder")), "pkpdb": dict(load_v4(root, "pkpdb")), "benchmark-val": {}}
        for dataset, cohort in VALIDATION_COHORTS.items():
            for cid, row in load_validation(root, cohort).items():
                if cid in self.rows[dataset]: raise ValueError(f"{dataset} {cid}: in both training and validation annotations")
                self.rows[dataset][cid] = row
        if sources == "crystal":
            masked = lambda r: {**r, "ph_target": None, "ph_mask": False, "ph_quality_weight": 0.0}
            self.rows = {d: {cid: r if not r["ph_mask"] or source_category(r) == "crystal" else masked(r) for cid, r in rows.items()}
                         for d, rows in self.rows.items()}
        self.coverage = {}
        for dataset, manifest in manifests.items():
            for split in ("train", "val"):
                records = [r for r in manifest["records"] if r["split"] == split and (split != "train" or r["min_fraction"] <= fraction + 1e-12)]
                if not records: continue
                missing = [r["id"] for r in records if r["id"] not in self.rows[dataset]]
                if missing: raise KeyError(f"{dataset} {split}: {len(missing)} records without pH annotation, e.g. {missing[:5]}")
                wrong = [r["id"] for r in records if self.rows[dataset][r["id"]]["split"] != split]
                if wrong: raise ValueError(f"{dataset} {split}: annotation split differs for {len(wrong)}, e.g. {wrong[:5]}")
                eligible = [self.rows[dataset][r["id"]] for r in records if self.rows[dataset][r["id"]]["ph_mask"]]
                self.coverage[f"{dataset}_{split}"] = {"structures": len(records), "eligible": len(eligible),
                    "by_source": {c: sum(source_category(r) == c for r in eligible) for c in ("crystal", "em_buffer", "nmr")}}
        self.train = [self.rows[d][r["id"]] for d in ("pinder", "pkpdb") for r in manifests[d]["records"]
                      if r["split"] == "train" and r["min_fraction"] <= fraction + 1e-12]
        self.stats = fit(self.train)
        self.provenance = {**({"sources": sources} if sources != "all" else {}), "training": f"annotations/{FORMAT_V4}", "validation": {d: f"{VALIDATION_V1}/{c}" for d, c in VALIDATION_COHORTS.items()},
                           "coverage": self.coverage, "fit": self.stats}

    def arrays(self, dataset, ids, size):
        """(target, weight) float32 arrays of length `size` for a batch's ids (padding: weight 0)."""
        target = np.zeros(size, np.float32); weight = np.zeros(size, np.float32)
        for slot, cid in enumerate(ids):
            row = self.rows[dataset][cid]
            if row["ph_mask"]: target[slot] = row["ph_target"]; weight[slot] = row["ph_quality_weight"] * row["parent_weight"]
        return target, weight


def fit(rows):
    """Training-label statistics (parent-weighted): mean pH (the baseline) and the mean pH in each threshold bin."""
    rows = [r for r in rows if r["ph_mask"]]
    values = np.asarray([r["ph_target"] for r in rows], float); weights = np.asarray([r["parent_weight"] for r in rows], float)
    edges = np.concatenate(([-np.inf], THRESHOLDS, [np.inf])); bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (values > lo) & (values <= hi)
        bins.append(float(np.average(values[inside], weights=weights[inside])) if inside.any() else float((max(lo, 0) + min(hi, 14)) / 2))
    return {"structures": len(rows), "weighted_structures": float(weights.sum()), "mean": float(np.average(values, weights=weights)),
            "bin_means": bins, "thresholds": list(THRESHOLDS),
            "positive_fraction": [float(np.average(values > t, weights=weights)) for t in THRESHOLDS]}


def initialize(params, seed):
    import jax
    import jax.numpy as jnp
    width = params["head"]["w"].shape[0]; kg, kv, ko = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), SALT), 3)
    head = lambda key, out: {"w": jax.random.normal(key, (width, out), jnp.float32) / jnp.sqrt(float(width)), "b": jnp.zeros(out, jnp.float32)}
    return {**params, "aux_ph": {"norm": jnp.stack((jnp.ones(width), jnp.zeros(width))), "gate": head(kg, width), "value": head(kv, width),
                                 "out": head(ko, len(THRESHOLDS))}}


def separate_optimizer(params, make):
    """Optimizer `make(subtree)` run separately on the trunk and on aux_ph (own global-norm clip and Adam state), so a
    stop-gradient probe's gradient cannot rescale the trunk update."""
    import optax
    trunk = lambda tree: {k: v for k, v in tree.items() if k != "aux_ph"}
    rest, head = make(trunk(params)), make(params["aux_ph"])

    def update(gradient, state, p=None):
        u, s = rest.update(trunk(gradient), state[0], None if p is None else trunk(p))
        v, t = head.update(gradient["aux_ph"], state[1], None if p is None else p["aux_ph"])
        return {**u, "aux_ph": v}, (s, t)
    return optax.GradientTransformation(lambda p: (rest.init(trunk(p)), head.init(p["aux_ph"])), update)


def pooled_logits(p, tokens, site_mask):
    """Gated attention pooling of one structure's site tokens -> (len(THRESHOLDS),) logits."""
    import jax
    import jax.numpy as jnp
    from pkanet.model import linear, norm
    h = norm(p["norm"], tokens); m = site_mask.astype(jnp.float32)[:, None]
    gate = jax.nn.sigmoid(linear(p["gate"], h)) * m
    z = jnp.sum(gate * linear(p["value"], h), axis=0) / jnp.maximum(jnp.sum(gate, axis=0), 1e-6)
    return linear(p["out"], z)


def loss(logits, target, weight):
    """Weighted mean over structures of the mean cumulative BCE; 0 without eligible structures."""
    import jax.numpy as jnp
    import optax
    labels = (target[:, None] > jnp.asarray(THRESHOLDS, jnp.float32)).astype(jnp.float32)
    per = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels), axis=-1)
    return jnp.sum(per * weight) / jnp.maximum(jnp.sum(weight), 1e-6)


def expected_ph(logits, stats):
    """Expected pH from cumulative logits: monotone survival S_k = P(pH > t_k), bin probabilities, training bin means."""
    survival = np.minimum.accumulate(1 / (1 + np.exp(-np.asarray(logits, float))), axis=-1)
    edges = np.concatenate((np.ones(survival.shape[:-1] + (1,)), survival, np.zeros(survival.shape[:-1] + (1,))), axis=-1)
    return np.sum((edges[..., :-1] - edges[..., 1:]) * np.asarray(stats["bin_means"]), axis=-1)


def metrics(collected, stats):
    """collected: [(row, logits)] for one cohort, eligible rows only."""
    from scipy.stats import spearmanr
    from .production_train import _auroc

    def block(items):
        if not items: return {"structures": 0}
        target = np.asarray([r["ph_target"] for r, _ in items], float); logits = np.stack([z for _, z in items]).astype(float)
        weight = np.asarray([r["parent_weight"] for r, _ in items], float)
        predicted = expected_ph(logits, stats); error = predicted - target; base = stats["mean"] - target
        labels = target[:, None] > np.asarray(THRESHOLDS)
        p = np.clip(1 / (1 + np.exp(-logits)), 1e-7, 1 - 1e-7)
        out = {"structures": len(items), "parent_weighted_structures": float(weight.sum()),
               "mae": float(np.average(np.abs(error), weights=weight)), "rmse": float(np.sqrt(np.average(error ** 2, weights=weight))),
               "baseline_mae": float(np.average(np.abs(base), weights=weight)), "baseline_rmse": float(np.sqrt(np.average(base ** 2, weights=weight))),
               "spearman": float(spearmanr(predicted, target)[0]) if len(items) > 2 and np.ptp(target) > 0 else None,
               "mean_bce": float(np.average(np.mean(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)), axis=1), weights=weight)),
               "thresholds": [{"t": t, "positive_fraction": float(labels[:, j].mean()), "auroc": _auroc(logits[:, j], labels[:, j])}
                              for j, t in enumerate(THRESHOLDS)]}
        out["mae_skill_vs_baseline"] = 1 - out["mae"] / out["baseline_mae"] if out["baseline_mae"] else None
        return out
    result = block(collected)
    result["by_source"] = {c: block([(r, z) for r, z in collected if source_category(r) == c]) for c in ("crystal", "em_buffer", "nmr")}
    return result


def evaluate(engine, params, sources, manifests, table, config):
    """pH section of validation: PINDER validation (AB branch), pKPDB validation and the benchmark set."""
    from .loading import Prefetcher
    from .production_loading import select
    out = {"fit": {"training_mean": table.stats["mean"], "bin_means": table.stats["bin_means"], "thresholds": list(THRESHOLDS)}}
    for name, dataset, paired in (("pinder", "pinder", True), ("pkpdb", "pkpdb", False), ("benchmark", "benchmark-val", False)):
        key = {"pinder": "pinder-val", "pkpdb": "pkpdb-val", "benchmark": "benchmark"}[name]
        if key not in sources: continue
        records = select(manifests[dataset], "val"); source = sources[key]; collected = []; total = 0
        for ids, batch in Prefetcher(source, source.policy.plans(records, np.random.default_rng(0)), config):
            logits = np.asarray(engine.ph_paired(params, batch[0]) if paired else engine.ph_single(params, batch[0]))
            for slot, cid in enumerate(ids):
                row = table.rows[dataset][cid]; total += 1
                if row["ph_mask"]: collected.append((row, logits[slot]))
        out[name] = {"records": total, **metrics(collected, table.stats)}
    return out
