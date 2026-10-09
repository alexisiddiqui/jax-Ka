# oGQT multitask diagnostic campaign

This campaign separates five explanations for the current results: catastrophic forgetting, insufficient regularisation, insufficient paired-data diversity, insufficient capacity, and disagreement between the pKPDB/PypKa and PINDER/pKAI teachers.

All model experiments use seed 17, the frozen 5,000/400 PINDER cluster split, the unchanged 5,000/142 pKPDB split, backbone-only 20 A oGQT inputs, full float32, and no test data. The active 1:1 pKPDB/PINDER replay run is the baseline.

## Controlled arms

| Question | Comparison |
|---|---|
| Regularisation | baseline, 10% hidden dropout, 5% context masking |
| Unique paired data | nested 1,250, 2,500, and 5,000 PINDER clusters |
| Capacity | 67,725-parameter oGQT and width-92/FF-184 oGQT after matched pKPDB pretraining |
| Undertraining | conditional epoch-20-to-40 continuation when validation is still improving |
| Teacher mismatch | current PypKa versus pKAI on exact AB/A/B structures and matched sites for all 400 validation complexes |

Smaller PINDER subsets cycle to the full cohort's optimizer-update count. Consequently every data arm receives the same number of PINDER batches, pKPDB replay batches, and learning-rate steps. Dropout and masking apply to both data streams. Corresponding AB/free branches receive the same stochastic draw, and context masking never removes supervised centres.

## Comparable diagnostics

Every checkpoint is evaluated without augmentation on fixed 400-complex training subsets and both complete validation sets. Reports include state and paired MSE/MAE, interface error, residue and target-shift bins, distance bins, gradient norms and cosine, fixed-PK_MOD performance, and the zero binding-shift baseline. Selected-checkpoint comparisons use 2,000 paired cluster bootstrap replicates.

The continuation gate requires both training loss and validation selection to improve by at least 0.005 between epoch 16 and the epoch-17-to-20 window. If it opens, the exact epoch-20 optimizer state and sampling streams continue through epoch 40 while the learning rate decays smoothly from 1e-6 to 1e-7; it is not restarted.

## Gates

- Regularisation requires at least 0.01 pKa clean-validation improvement with a paired-bootstrap interval excluding zero and pKPDB MAE within epoch zero plus 0.05.
- More data require monotonic improvement from 25% through 100% and a non-zero paired-bootstrap difference between the endpoints.
- More capacity requires improvement in both fixed-training and validation metrics.
- Undertraining requires continued improvement in both training and validation after epoch 20.
- Teacher comparisons retain explicit failures and are qualified when fewer than 80% of eligible sites match.

GPU jobs use at most one A40, eight CPUs, and 16 GB each. Current-PypKa generation uses two cores and 4 GB per complex on responsive CPU nodes, with at most 400 CPU cores allocated across concurrent waves.

The component gate passed deterministic dropout, branch-matched context masking, nested subsets, one finite Triton-backed update for every small arm, and one finite update for the larger pKPDB pretraining model. The smoke receipts are stored under `training/ogqt-diagnostic-v1/*/smoke` and `pretraining/ogqt-large-site-v1/smoke` in the runtime tree.
