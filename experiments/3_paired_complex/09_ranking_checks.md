# Ranking checks — 2026-10-05

Audit job 735866 completed. All comparisons below use 1,284 validation interface
sites, 139 complexes and 75 frozen sequence groups. No test predictions were read.
All 01 macro metrics are calculated per complex, averaged over complexes within
sequence groups, then over groups. Sign accuracy uses |reference ΔpKa| >= 0.5;
603 sites qualify. Ranges cover seeds 17/29/43, except the two constant controls.

| Method | MAE ↓ | Skill (01 macro) ↑ | Spearman (01 macro) ↑ | Sign accuracy (01 macro) ↑ |
|---|---:|---:|---:|---:|
| PypKa intrinsic + interaction oracle, native MC | 0.000 | 1.000 | 1.000 | 100% |
| pKAI last-layer tuning | 0.335 | 0.676–0.680 | 0.725–0.730 | 96.6% |
| Frozen pKAI | 0.338 | 0.681 | 0.750 | 96.7% |
| pKAI all-layer tuning | 0.337–0.342 | 0.618–0.672 | 0.709–0.721 | 96.9–97.1% |
| CatBoost intrinsics + PypKa interactions | 0.563–0.583 | 0.032–0.053 | 0.547–0.589 | 85.3–85.7% |
| PROPKA + CatBoost residual | 0.600–0.604 | −1.740 to −1.382 | 0.598–0.619 | 84.3–84.8% |
| PROPKA | 0.787 | −0.482 | 0.363 | 69.9% |
| Zero-shift null | 0.832 | 0.000 | undefined | 0% |
| JAX-Ka, untrained | 0.857 | −0.639 | 0.304 | 70.2% |
| Constant-intrinsic controls + PypKa interactions | 0.900–0.903 | −0.455 to −0.450 | 0.219–0.236 | 64.9–66.4% |

## Skill aggregation matters

01's skill is `1 − MSE / MSE_null` per complex, followed by macro aggregation.
Small null denominators can dominate this average of ratios. The audit also reports
`1 − group_weighted_MSE / group_weighted_MSE_null`, using the same complex/group
weights for numerator and denominator, plus pooled metrics. These are distinct
estimands; the secondary ratio does not replace 01's convention.

| Method | Ratio of group-weighted MSEs: skill |
|---|---:|
| Frozen pKAI | 0.800 |
| Last-layer tuning | 0.809–0.810 |
| All-layer tuning | 0.810–0.815 |
| CatBoost intrinsic hybrid | 0.370–0.415 |
| PROPKA + CatBoost | 0.448–0.456 |
| PROPKA | 0.005 |
| JAX-Ka | −0.251 |
| Null | 0.000 |

Thus the MAE ordering is not a universal accuracy ordering. In particular, the
PROPKA residual correction improves overall weighted squared error while worsening
the average per-complex skill ratio. Fine-tuned pKAI does not dominate frozen pKAI
across all metrics. Its checkpoints were selected on validation; these remain
development results.

## pKAI structural zeros and G3

Conditional fraction: `|prediction| < 0.01` among `|reference| >= 0.1`.

- All-method common interface support: **12/1,114 = 1.08%** using production
  predictions; **10/1,114 = 0.90%** using unrounded frozen-network predictions.
- Broader pKAI-valid interface support: **13/1,192 = 1.09%** (unrounded: 0.92%).
- For |reference| >= 0.5 on common support: 2/603 = 0.33% with production predictions.

The frozen-smoke G3 check passed: maximum bound/free response 1.11 pKa, no difference
under chain relabeling. Installed pKAI's `Residue.calc_cutoff_atoms` iterates over
`self.protein.iter_atoms()` across chains, and the wrapper supplies the complete AB
structure. This is not a chain-isolated representation. In the actual training
handoff, AB/free feature vectors differ at all 3,105 training and all 1,379 validation
interface sites. Rounded and unrounded zero fractions are reported separately.

## Training-set check

On the same 3,105 training interface sites (255 groups):

| Model | Training MAE |
|---|---:|
| Frozen pKAI | 0.3434 |
| All-layer seed 17 | 0.1727 |
| All-layer seed 29 | 0.1711 |
| All-layer seed 43 | 0.1753 |

The model fits the training data; the limited validation improvement is not a
failure to update parameters or identical bound/free input representations.

## Explicit oracle

Rechecked the actual saved teacher replay outputs against original raw labels for
all **426 states** in the 142-complex hybrid cohort. Maximum occupancy-curve and
midpoint differences are both **exactly zero**. The oracle row is formed from those
replayed AB/free predictions on the same 1,284 common sites.

This hybrid uses **PypKa native tautomer Monte Carlo**, preserving the original
microstate interaction matrix. It does not convert to scalar pair terms or use
JAX-Ka mean field. Its replay/conversion floor is therefore zero here. The hybrid
error reflects substituting the learned intrinsic energies in this retained teacher
system, including their effects on coupled sites. This does not establish an oracle
floor for any future PypKa-to-JAX-Ka conversion, which needs its own test.

## Artifacts

[Full audit report](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/hybrid-mc-v1/ranking-checks-v2/report.md),
[all metric definitions and values](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/hybrid-mc-v1/ranking-checks-v2/scores.csv),
[structural zeros](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/hybrid-mc-v1/ranking-checks-v2/structural_zeros.csv),
[verification and source hashes](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/hybrid-mc-v1/ranking-checks-v2/verification.json).

The initial reporting attempt 735865 stopped on a duplicate metadata keyword after
computing score tables. Version 2 fixes that reporting error; no predictions or
scientific settings were changed.
