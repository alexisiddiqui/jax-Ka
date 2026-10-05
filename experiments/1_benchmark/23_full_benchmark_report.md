# Full structural benchmark report — initial non-JAX round

Date: 2026-10-05. All 1,452 frozen complexes have result receipts for PypKa, PROPKA, pKAI, pKAI+ and the zero-shift baseline. The report is complete with explicit method failures; JAX-Ka was deferred at the user’s request. No model training has been performed for these scores.

## Main outcome

**1,358/1,452 complexes have usable teacher interface labels (93.5%).** There are **78,964 usable teacher paired sites**, of 106,599 mask-eligible sites, and **75,823 sites shared by all retained comparison methods and the teacher**. Counts span the frozen training, validation and test splits; only training-eligible sites can enter optimization.

The dataset and split remain 778 training / 151 validation / 523 test complexes. The methods are compared to newly generated PypKa 2.10.0 labels, not experimental ground truth. Historical pKPDB configuration equivalence remains unresolved. pKAI and pKAI+ share the PypKa training lineage, so their agreement must not be interpreted as independent physical validation.

## Computation and failures

| Method | Complete complex calculations | Calculations with failures |
|---|---:|---:|
| PypKa reference | 1,387 | 65 |
| PROPKA | 1,452 | 0 |
| pKAI | 1,445 | 7 |
| pKAI+ | 1,445 | 7 |
| Zero shift | 1,452 | 0 |

State-level exception counts: {'TimeoutExpired': 140, 'RuntimeError': 24}. A complex can fail in multiple states; these counts are not counts of distinct complexes. Failed, unreported and out-of-range midpoints are excluded explicitly. Partial state successes remain usable only when the matching bound/free pair is valid.

The 20 missing receipts left after worker retirement were recovered before collection. Final collection checked all 7,260 complex-method receipts and independently recomputed 2,952 group macro metrics. It completed in 10 min 44 s with a 32 GB allocation. Original failures and retry provenance are retained.

## Interface coverage

| Split | Mask-eligible sites | Teacher-valid sites | All-method common sites | Common-support groups |
|---|---:|---:|---:|---:|
| train | 7,171 | 5,346 | 5,045 | 256 |
| val | 1,823 | 1,459 | 1,364 | 77 |
| test | 6,158 | 4,591 | 4,178 | 195 |

Interface membership uses residue ΔSASA >10 Å². Coverage is after the frozen, split-aware uncertainty masks. Method-specific and teacher-matched coverage remain available in coverage.csv; common support excludes every site missing any retained method.

## Bound-minus-free pKa agreement

The target is pKa(AB) − pKa(free partner). Metrics are computed per complex, averaged within sequence groups and then equally across groups. Confidence intervals bootstrap sequence groups, not individual sites. The tables below use all-method common interface support. No held-out result was used to tune these pretrained methods.

| Method | Train MAE (95% CI) | Validation MAE (95% CI) | Test MAE (95% CI) | Test macro RMSE | Test macro skill |
|---|---:|---:|---:|---:|---:|
| PROPKA | 0.815 (0.751–0.880) | 0.820 (0.725–0.912) | 0.811 (0.739–0.887) | 1.023 | -16.534 |
| pKAI | 0.338 (0.309–0.370) | 0.350 (0.305–0.395) | 0.337 (0.308–0.369) | 0.446 | 0.143 |
| pKAI+ | 0.535 (0.490–0.586) | 0.514 (0.459–0.572) | 0.480 (0.442–0.520) | 0.645 | 0.229 |
| Zero shift | 0.888 (0.816–0.969) | 0.859 (0.772–0.946) | 0.825 (0.761–0.893) | 1.068 | 0.000 |

Skill is 1 − MSE(predicted shift)/mean(reference shift²), evaluated per complex before group averaging. Small reference-shift denominators can produce large negative skill; a lower macro MAE need not imply positive macro skill. RMSE is also a macro average, not pooled site RMSE. The zero-shift baseline has skill zero by construction. These definitions were not changed after seeing results.

## Antibody versus general protein interfaces

| Test role | Method | Sites | Groups | Macro MAE (95% CI) |
|---|---|---:|---:|---:|
| antibody_antigen | PROPKA | 1027 | 49 | 1.064 (0.943–1.177) |
| antibody_antigen | pKAI | 1027 | 49 | 0.461 (0.401–0.528) |
| antibody_antigen | pKAI+ | 1027 | 49 | 0.593 (0.517–0.657) |
| antibody_antigen | Zero shift | 1027 | 49 | 1.039 (0.922–1.139) |
| general | PROPKA | 3151 | 150 | 0.731 (0.657–0.820) |
| general | pKAI | 3151 | 150 | 0.299 (0.268–0.334) |
| general | pKAI+ | 3151 | 150 | 0.444 (0.400–0.488) |
| general | Zero shift | 3151 | 150 | 0.755 (0.683–0.832) |

Antibody/antigen grouping follows the frozen antigen-family split with separate antibody-novelty annotations. Roles and groups are not rebalanced to improve this table. Missing role rows mean no scored common support under that exact stored role label.

## Linkage limits and figures

Complete accepted charge coverage permits linkage for 398 of 7,260 complex-method combinations. 6180 are blocked by masks and 682 by incomplete charge coverage. Partial masked charge sums are not reported as full binding free-energy curves.

![Interface MAE](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/report_figures/interface_mae.png)

![Interface coverage](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/report_figures/interface_coverage.png)

## Experiment 02 handoff and next work

The fixed 500-complex training pilot currently contains 458 complexes with usable teacher interface labels, 24,646 teacher paired sites and 23,818 sites shared by the retained methods. Those are coverage results, not permission to replace failed pilot complexes with validation/test examples. The handoff adds the 151 frozen validation complexes and excludes test inputs and labels.

Paired pKAI features must reproduce the frozen model predictions before fitting. CatBoost residual targets are PypKa ΔpKa minus PROPKA ΔpKa; subtracting PypKa from itself would leak the target. Density, formal-charge and donor/acceptor-proximity features are explicitly geometric proxies. Scaling and weighting are fitted on training rows only.

A separate recovery run retries 61 timed-out states across 26 pilot complexes with a three-hour per-state budget. Completed states and the original label version remain unchanged. The two preparation-failed states of one complex are handled separately. A recovery overlay must be verified and versioned before it can replace handoff labels.

Remaining benchmark work includes experimental Set 2 curation and independent overlap checks. JAX-Ka recovery is deferred. The full benchmark has not established experimental accuracy, and experiment 02 training outcomes are not part of this report.

## Artifacts and provenance

- [Verification](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/verification.json)
- [Complete scores and intervals](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/scores_set1.csv)
- [Coverage](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/coverage.csv)
- [Representability by shell](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/representability.csv)
- [Failure receipts](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/completion.json)
- [Manifest](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-nojax-v1/manifest.json)

Manifest SHA-256: `d1f1f06294cce0adbb4ad511a124060c045339807f590ca2d7c940a84a10892d`.
The inherited scoring_report.json interpretation sentence still says “engineering smoke”; that is a reused metadata template. This report’s scope is the complete frozen 1,452-complex structural benchmark. Numeric artifacts were not altered to correct that wording.
