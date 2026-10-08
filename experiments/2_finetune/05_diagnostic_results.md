# Experiment 02: paired-learning diagnostic results

Completed 2026-10-05. All 13 runs and the independent aggregation audit passed.
Fine-tuning gives a small validation MAE reduction, but the paired group-bootstrap
intervals include zero improvement for every fine-tuned seed. This pilot does not
establish a reliable gain over frozen pKAI.

## Comparison on identical sites

The comparison uses 1,368 validation interface sites across 77 sequence groups.
All values below measure agreement with current PypKa labels. Neural checkpoints
were selected on this validation set; these are development results. The test set
was not read or used for this experiment.

| Arm | Validation group-macro MAE, range across seeds |
|---|---:|
| Frozen pKAI | 0.3501 |
| Last-layer tuning | 0.3456–0.3459 |
| All-layer tuning | 0.3433–0.3474 |
| Same architecture from scratch | 0.4937–0.5078 |
| CatBoost residual learning | 0.6153–0.6214 |
| PROPKA, same support | 0.8230 |
| Zero shift, same support | 0.8629 |

Last-layer seed 17 changes MAE by −0.00425, with a paired 95% group-bootstrap
interval of −0.01072 to +0.00186. Even the best all-layer seed changes MAE by only
−0.00683 (−0.02352 to +0.00936). These intervals also do not account for the
optimism from selecting neural checkpoints on validation.

All-layer tuning reduces training MAE to 0.170–0.175 but validation remains
0.343–0.347, consistent with limited transfer of the fitted improvement. The
scratch model is substantially worse under this fixed training budget, supporting
the usefulness of pretraining in this pilot. This is not an exhaustive comparison
of optimization schedules or proof of an architecture ceiling.

CatBoost improves on PROPKA by about 0.20 pKa MAE on the same sites; its paired
intervals exclude zero. It nevertheless remains substantially worse than frozen
pKAI. It is a standalone geometric residual baseline, not released KaML-CBtree.

## Antibody/general breakdown

The common validation support contains 184 antibody–antigen sites in 13 groups,
and 1,184 general-protein sites in 64 groups. Frozen pKAI MAE is 0.423 and 0.335,
respectively. Last-layer tuning gives about 0.414–0.415 and 0.332. All-layer tuning
is less consistent on antibody–antigen examples (0.415–0.435). The small antibody
sample and broad uncertainty limit subgroup conclusions.

## Data and checks

Training used 3,105 pKAI-representable interface sites from the fixed 500-complex
pilot, with the frozen split and uncertainty masks. This is an interface-only
pilot, not fitting on all 23,958 available training feature rows. CatBoost retained
its own valid training support; all reported comparisons use the intersection.
The 151 validation complexes were retained with their original groups. Missing
labels were not replaced. The ongoing timeout recovery is a separate label version.

Checks cover 64,114 handoff/frozen state predictions, exact identical-input
cancellation, final-layer gradient isolation, actual parameter updates, finite
predictions, split/group separation, matching teacher labels across arms, hashes,
and an independent pandas recomputation of all 468 reported group MAE/RMSE/skill values.
Figures were visually reviewed. Skill retains its original denominator and can be
unstable when reference shifts are nearly zero; MAE and skill need not rank methods
identically. Spearman and sign accuracy are explicitly pooled secondary metrics.

All scientific work ran on compute nodes, excluding comp1400, with 2 GB/core and
submission checks against the 400-core user-wide cap. Training jobs completed in
21 seconds to 2 min 45 s after setup; eight allocated cores supplied memory headroom,
while numerical work used two threads. The original benchmark environments and
results were not changed.

## Decision

Keep frozen pKAI as the main baseline. Retain last-layer tuning as a cheap candidate,
but do not claim a meaningful gain yet or use this pilot alone to justify a new
architecture. A subsequent data-expansion or training-support experiment should
be registered before fitting; the final test evaluation should wait until those
choices are fixed. Experimental Set 2 remains necessary for independent validation.

## Artifacts

Runtime directory: `/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1`.

- [Scores, strata and secondary shell results](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1/arms.csv)
- [Paired changes versus frozen pKAI](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1/comparisons.json)
- [Independent audit and physics/null baselines](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1/independent_audit.json)
- [Frozen run manifest](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1/manifest.json)
- [Collection verification](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1/verification.json)

![Validation MAE across seeds](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1/arms.png)

![CatBoost feature importance](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/diagnostic-v1/feature_importance.png)

Run jobs: 733185 and 733187–733198; collection 733199; independent audit 733200.
Models and histories are in each arm/seed subdirectory. The CatBoost baseline uses
separate `acid.cbm` and `base.cbm` files per seed; there is no single combined model.
