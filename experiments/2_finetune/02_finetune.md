# 02 — Fine-tune an existing model on paired ΔpKa

**Initial diagnostic complete:** [results and interpretation](05_diagnostic_results.md).
All 13 runs passed. Fine-tuning did not establish a reliable validation improvement
over frozen pKAI under the registered interface-only pilot protocol.

**Timebox:** 1 day (a few hours on the 3090, rest is analysis)
**Hardware:** 1× 3090
**Runs concurrently with:** 03's dataset generation on the CPU node. No contention.

**Purpose is diagnostic, not a product.** This answers one question: *is the gap found
in 01 a training-signal problem or an architecture problem?* The answer determines how
much of 03 is actually necessary.

---

## Which model

**pKAI.** Reasons:

- Smallest and fastest; fine-tunes in minutes, so you can run the full ablation grid
- Trained on pKPDB, so its prior matches the tier-A distribution you'll use in 03
- Predicts a per-site value from a local environment — the simplest possible
  parameterisation, which makes it the cleanest test of whether paired supervision alone
  is enough
- Permissive tooling, no web-service rate limits

**Standalone CatBoost as the second arm.** Train on paired geometric features
(see Delta-learning below). This is a new baseline, not the released KaML-CBtree
wrapper, whose single-chain limitation excluded it from this benchmark.

Do **not** pick DeepKa (web-server-mediated) or KaML-GAT (heavier, worse than the trees
on the task it was designed for).

---

## Setup

### Siamese wrapper

```
ΔpKa_pred = f_θ(complex, site) − f_θ(separated, site)
```

Same weights, two passes, subtract. Not a two-tower model with a difference head.
The gradient of the difference is the difference of gradients, so any error the encoder
makes identically in both states contributes nothing to the loss. Error cancellation
becomes a training-time property, not just a hope.

Partner presence enters as input content only — the free state is the identical
structure with the partner's atoms removed. No architectural flag, no extra pathway.
The only input difference between the two passes is the thing being measured.

### Data

Use 01's fixed 500-complex training pilot and 151 frozen validation complexes.
The initial handoff contains 24,646 training and 8,365 validation teacher site pairs;
3,241 and 1,459 respectively are interface sites. Model-specific representation
checks may reduce these counts. There are usable teacher interface sites in 458
training and 144 validation complexes. Failed examples are not replaced.

Inherit the existing frozen sequence/antigen-family split and uncertainty masks;
do not resplit. Test inputs and labels are excluded from the handoff. Fit scaling,
residue normalization and weighting on training data only. Check paired pKAI
features against frozen predictions before fitting. The versioned handoff is
`_runtime/jax-Ka/pkabench/finetune/handoff-v2` outside the repository.
Timeout recovery produces a separate label version; it does not silently change
this handoff. The handoff passed 64,114 frozen-state prediction checks. The frozen
paired model gate also passed; training was launched on 2026-10-05 under the
[fixed diagnostic protocol](04_diagnostic_protocol.md). Initial fitting uses
3,105 training-interface sites; broader shell rows are retained for evaluation.

### Loss

Huber on ΔpKa, normalised per residue type. Upweight |ΔpKa| > 0.5 — the bulk of
near-zero surface sites will otherwise dominate and you'll learn the null model with
extra steps.

---

## Arms to run

| Arm | What it tests |
|---|---|
| Frozen pKAI, siamese, no training | Reproduces 01's result; sanity check |
| Fine-tune last layer only | Is a recalibration enough? |
| Fine-tune all layers | Full capacity of the existing parameterisation |
| Train pKAI architecture from scratch on paired data | Does pretraining help at all here? |
| CatBoost delta-learning (below) | The baseline to beat |

---

## Delta-learning baseline — build this, it may win

Do not fine-tune anything. Predict the *residual* of the physics.

```
target = ΔpKa_PypKa − ΔpKa_PROPKA3
```

PypKa is the teacher here, so using it as both teacher and residual baseline would
produce a trivial zero target. Features computed from the two-state geometry:

- ΔSASA of the site on binding
- partner heavy-atom coordination counts within 6 / 10 Å
- opposite-formal-charge contact counts and nearest distance
- potential donor/acceptor atom counts within 4 Å from the partner
- change in local formal-charge proxy within 6 / 10 Å
- change in local dielectric-ish proxy (heavy-atom density)
- residue type, one-hot
- distance to interface centroid

These are geometric proxies, not measured burial depth, directional hydrogen bonds
or inferred protonation states. Charge features use fixed residue identities and
never teacher labels. Preserve full chain/residue/insertion/group keys when joining.

CatBoost, separate models for acids and bases (KaML found this materially helps).

This inherits the physics' sign and magnitude behaviour, trains in minutes on CPU, needs
no GPU, and sets the bar. **If the transformer in 03 cannot beat this, the transformer
has not earned its place.** Report it as a first-class method, not an afterthought.

---

## Decision rules out of this section

| Observation | Action in 03 |
|---|---|
| Fine-tuned pKAI goes from zero skill to good skill | Gap is training signal. Proceed with 03 as planned; expect it to work. |
| Fine-tuned pKAI plateaus well below delta-learning | Single-site parameterisation is the limit → the intrinsic/pair factorisation is load-bearing. Strengthens the paper. |
| Delta-learning beats everything by a wide margin | Seriously consider shipping it as the headline method and the neural model as the differentiable alternative. Be honest about this. |
| From-scratch ≈ fine-tuned | pKPDB pretraining transfers little to ΔpKa → in 03, weight tier B over tier A. |

---

## Outputs

```
results/finetune/
  arms.csv                  # skill score, Spearman, sign acc per arm
  deltalearn_model.cbm
  feature_importance.png
  decision.md               # which branch of the table above fired
```

## Non-goals

No titration curves, no pH-dependence, no linkage integral, no coupled sites. These
models cannot produce them — that is precisely why 03 exists. Don't try to retrofit it.
