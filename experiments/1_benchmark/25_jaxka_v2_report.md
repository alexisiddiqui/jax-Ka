# JAX-Ka production-1024-v2 and the six-method full report

Date: 2026-10-05. Addendum to [the non-JAX full report](23_full_benchmark_report.md). JAX-Ka was deferred from the first production round after its workers were OOM-killed. This round reruns it with a memory-reduced execution path and scores all six methods together. No model training has been performed for these scores.

## Main outcome

**JAX-Ka now completes all 1,452 frozen complexes** (v1: 205 completed, 288 OOM-killed of 472 attempted). Its predictions are equivalent to the validated 1024-step model. Out of the box it agrees with PypKa slightly less well than PROPKA: test interface ΔpKa macro MAE **0.849 (0.774–0.929)** versus PROPKA 0.788 (0.719–0.859) on identical six-method common sites. The confidence intervals overlap. JAX-Ka is an untrained, differentiable PROPKA-parameterized surrogate; its intended use is gradients for fine-tuning and design, not off-the-shelf accuracy.

## What changed in v2

The equations, PROPKA 3.0 parameters, 1024-step damped solver, pH grid (−2 to 16, 0.25 steps), shared-grid midpoint readout and validity rules are unchanged. Only execution changed:

- Structural arrays are jit arguments rather than closed-over constants. Previously every cache array was embedded in the compiled program; compile-time memory scaled with the cache and was the v1 OOM mechanism.
- The structural cache is restricted to the native sequence and stored compactly ([N, Ke, 9, 1] instead of 20 identity columns). Disallowed identities are rejected by the model.
- The environment-term padding mask is applied to the sequence factor, avoiding unfused full-size temporaries.
- Host copies of the cache are released after device transfer.

| Largest campaign complex (1,460 residues) | v1 execution | v2 execution |
|---|---:|---:|
| Peak resident memory per state | 7.5 GB | 1.96 GB |
| Cache size | 2.2 GB | 0.45 GB |

Code: commits `32df76e` and `4f03165` on branch `solver-newton`.

## Release gate and equivalence

Before release, the v2 worker re-ran all 50 accepted smoke complexes (150 states) and was compared with the accepted 1024-step predictions.

| Check | Result | Tolerance |
|---|---:|---:|
| Site status mismatches | 0 | exact |
| Maximum curve difference | 6.6 × 10⁻⁷ | 10⁻⁶ |
| Maximum midpoint difference | 1.9 × 10⁻⁶ pKa | 10⁻⁴ |

After production, the 205 complexes that v1 completed were compared again: 0 status mismatches and midpoint differences ≤ 3.8 × 10⁻⁶. Ten complexes have curve differences of 1.1–2.1 × 10⁻⁶, slightly above the smoke tolerance; these are float32 roundoff with no status or midpoint consequence. The tolerance was not changed after seeing this.

## Computation and failures

All 1,452 complex calculations completed with no process failures. A 140-worker pool (2 CPUs / 4 GB each) finished in 51 min wall time, using 92.8 worker-hours. Median wall time per complex (three states) was 189 s, 90th percentile 469 s, maximum 1,507 s. Structural cache construction accounted for 74% of compute.

Twelve complexes have one state each whose pH grid did not meet the fixed-point residual tolerance (2 × 10⁻⁵) at 1024 iterations; maximum residuals were 2.4 × 10⁻⁵ to 2.7 × 10⁻³. Every site in those states is recorded as failed. Separately, 10,356 bracketed site-states have a sampled upward segment and remain invalid under the existing monotonicity rule. Failed (12,449), out-of-range (3,221) and not-reported (2,220) site-states are excluded explicitly, never imputed.

## Interface coverage

| Split | PROPKA | JAX-Ka | pKAI | pKAI+ | Zero shift | Six-method common sites |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.971 | 0.895 | 0.779 | 0.791 | 1.000 | 4,632 |
| val | 0.987 | 0.934 | 0.770 | 0.780 | 1.000 | 1,294 |
| test | 0.969 | 0.884 | 0.746 | 0.767 | 1.000 | 3,828 |

Values are method-valid / mask-eligible interface sites. Adding JAX-Ka reduces all-method common support (test: 4,178 five-method sites to 3,828 six-method sites; all splits: 75,823 to 73,122 paired sites). Report 23’s five-method common-support table therefore remains the reference for the other methods; numbers below are on the six-method support and are not interchangeable with it.

## Bound-minus-free pKa agreement

Definitions are unchanged from report 23: per-complex metrics, equal complex means within sequence groups, equal group means, sequence-group bootstrap intervals. Six-method common interface support (test: 3,828 sites, 193 groups).

| Method | Train MAE (95% CI) | Validation MAE (95% CI) | Test MAE (95% CI) | Test macro RMSE | Test macro skill |
|---|---:|---:|---:|---:|---:|
| PROPKA | 0.777 (0.712–0.840) | 0.787 (0.698–0.881) | 0.788 (0.719–0.859) | 0.993 | −17.972 |
| JAX-Ka | 0.842 (0.770–0.913) | 0.858 (0.752–0.969) | 0.849 (0.774–0.929) | 1.082 | −18.287 |
| pKAI | 0.317 (0.289–0.348) | 0.335 (0.294–0.379) | 0.315 (0.290–0.342) | 0.415 | 0.093 |
| pKAI+ | 0.494 (0.451–0.540) | 0.495 (0.440–0.553) | 0.448 (0.413–0.485) | 0.594 | 0.229 |
| Zero shift | 0.820 (0.749–0.895) | 0.822 (0.736–0.906) | 0.766 (0.708–0.827) | 0.987 | 0.000 |

Pairwise support (each method’s own valid sites matched to the teacher), test split: JAX-Ka 0.838 (0.766–0.915) on 4,143 sites; PROPKA 0.798 (0.730–0.869) on 4,500 sites. Both PROPKA-family methods have strongly negative macro skill, driven by small reference-shift denominators, as discussed in report 23.

## Antibody versus general protein interfaces

| Test role | Method | Sites | Macro MAE (95% CI) |
|---|---|---:|---:|
| antibody_antigen | PROPKA | 886 | 0.999 (0.891–1.110) |
| antibody_antigen | JAX-Ka | 886 | 1.096 (0.960–1.229) |
| general | PROPKA | 2,942 | 0.720 (0.631–0.811) |
| general | JAX-Ka | 2,942 | 0.769 (0.674–0.861) |

Six-method common support. Other methods’ role rows are in scores_set1.csv.

## Linkage

Complete accepted charge coverage permits linkage for 493 of 8,712 complex-method combinations; 7,416 are blocked by masks and 803 by incomplete charge coverage.

## Solution uniqueness diagnostic

The mean-field equations can have more than one stable solution under strong coupling. A follow-up diagnostic solved 42 complexes (30 size quantiles plus, separately, the 12 with a nonconverged v2 state) at every pH with optimistix Newton from the cold start, ascending and descending pH continuation and Levenberg–Marquardt, and evaluated the mean-field free energy of each converged solution. The published v2 curves were included as a candidate. No production prediction was changed.

| Size-quantile set (90 states) | Result |
|---|---:|
| States with ≥ 2 converged solutions at some pH | 15 / 90 |
| Titrating sites with ≥ 2 solutions | 148 / 8,246 (1.8%) |
| v2-converged pH points where v2 is the lowest free-energy solution | 6,553 / 6,570 (99.7%) |
| Largest free-energy excess of v2 where it is not lowest | 0.16 kT ln 10 |
| Non-monotonic bracketed sites: v2 / lowest-free-energy curves | 281 / 283 |

Where v2 converged it almost always found the lowest free-energy solution, so the v2 predictions are retained. Non-monotonic curves are mostly genuine features of single solutions rather than jumps between solutions; selecting the lowest-energy solution does not remove them.

A lowest-free-energy rule over optimistix candidates only (no damped solver) reproduced v2 at 6,559 of 6,570 pH points on the quantile set, finding lower-energy solutions at the remainder and never missing one, with 7,889 versus 7,891 valid sites. On the 12 harder complexes it rescued pH points where v2 did not converge (valid site-midpoints 2,145 to 4,208) but missed v2's lower-energy solution at 85 pH points in 9 strongly coupled bound states (excess ≤ 0.34). It was 5.2× faster in total but only 1.4× on the largest state. Because a full v2 rerun now costs about 30 worker-hours after cache vectorization, the solver is not a bottleneck; the damped 1024-step solver remains the production solver, and the optimistix path remains an opt-in tool for implicit-gradient design work.

## Interpretation and next work

JAX-Ka reproduces the PROPKA-family behaviour it approximates and is now operationally viable on every benchmark complex within a 4 GB worker. It is not more accurate than PROPKA out of the box. Its value is differentiability; any accuracy claim requires fine-tuning on training-split labels only, with validation for selection and test held out. As for all methods here, agreement is with PypKa 2.10.0 labels, not experimental ground truth.

Structural cache construction was the dominant JAX-Ka cost in this run (68.2 of 92.8 worker-hours). It has since been vectorized (commit `89b414c`). The loop implementation is retained as `build_cache_reference`, and every cache array is bitwise identical: all 4,356 production state fingerprints recorded in the v2 receipts are reproduced, and the production worker gives identical predictions on the six largest complexes. Cache time across the campaign falls from 68.2 to 6.25 worker-hours (10.9×; 605 s to 43 s for the largest state), with peak memory ≤ 2.03 GB per state. The predictions reported here are unchanged; a rerun would cost about 30 worker-hours, now dominated by the 1024-step solver. The opt-in optimistix Levenberg–Marquardt solver with implicit gradients converges on all profiled complexes but is not used for these predictions.

## Artifacts and provenance

- [Six-method verification](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-full-v2/verification.json): 8,712 receipts checked, 3,708 group metrics recomputed.
- [Scores and intervals](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-full-v2/scores_set1.csv) and [coverage](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-full-v2/coverage.csv).
- [JAX-Ka v2 release gate](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-1024-v2/release_gate.json) and [v1 comparison](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-1024-v2/v1_comparison.json).
- [JAX-Ka v2 receipts](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/production-1024-v2/jobs/jaxka) with per-state cache fingerprints, timings and peak memory.
- [Solution-uniqueness diagnostic](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/diagnostics/solution-branches/br2/summary.json) and [solver comparison](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/diagnostics/solver-compare/cmp1/summary.json).
- [Vectorized-cache fingerprint sweep](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/diagnostics/cache-sweep/sweep1/summary.json) and [worker comparison](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/diagnostics/worker-vectorized/run1/comparison.json).

Six-method manifest SHA-256: `b92e282631320a15ee79767ba3bada83a38e038b9f7e5deba17bae29706727cd`.
JAX-Ka v2 manifest SHA-256: `611c6ce7e7e85592bedde81572361c3332567558290e373d1e1432b01530bde8`.
The inherited scoring_report.json interpretation sentence still says “engineering smoke”; this report’s scope is the complete frozen 1,452-complex benchmark. Numeric artifacts were not altered to correct that wording.
