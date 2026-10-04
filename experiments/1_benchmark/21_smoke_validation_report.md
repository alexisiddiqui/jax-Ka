# Benchmark smoke validation report

Date: 2026-10-04. Scope: 50 selected protein–protein complexes, three states
(AB, A, B), current PypKa reference labels, and five comparison methods.

## Outcome

The smoke pipeline now completes all 50 PypKa calculations. JAX-Ka with
1024 solver iterations converges for all 150 states, compared with nine
nonconverged states at 64 iterations. Every isolated 64-iteration baseline
reproduces the original prediction statuses and passes curve/midpoint equivalence
checks. No previously valid midpoint becomes invalid at 1024 iterations.

JAX-Ka failed site-state predictions decrease from 540 to 39. The remaining
invalid predictions are retained as failures and excluded from midpoint scoring;
no numerical acceptance threshold was relaxed. Process isolation and reusing
the computed titration grid resolved the previously memory-failed diagnostic.

The candidate score report passed receipt checks and independent recomputation
of group macro metrics. These results support a versioned production wrapper
using 1024 iterations and isolated state processes. They do not establish that
larger production complexes will all converge or fit the same runtime budget.
Production and training have not started; existing model defaults are unchanged.

## Dataset and reference configuration

The engineering sample contains 24 training, 10 validation and 16 test
complexes, with 46–349 residues. Selection favored small structures and included
12 antibody complexes, eight glycan-bearing complexes, 13 buffer-bearing
complexes and four other ligand-bearing complexes; these categories overlap.
It is not a representative sample for final benchmark performance claims.

The underlying frozen structural dataset remains 778 training / 151 validation /
523 test usable interface pairs. Frozen assignments, structures and split-aware
site masks were unchanged throughout this work. Interface sites have
residue ΔSASA >10 Å². The reported shell covers sites within 20 Å of the partner.

Reference labels use PypKa 2.10.0, internal/solvent dielectric 15/80, ionic
strength 0.1 M, grid size 81, nonperiodic boundaries, G54A7, hydrogen
optimisation enabled, ions removed, temperature 298.15 K and SER/THR titration
disabled. Curves span pH −2 to 16 at 0.25 spacing. All 150 completed teacher
states were audited against the effective saved configuration, including
200000 Monte Carlo steps, 1000 equilibration steps and seed 1234567.
Temperature is pinned in requests; PypKa omits it from its resolved listing.

Historical pKPDB configuration equivalence remains unresolved. These are newly
generated labels under the stated configuration, not an exact reconstruction of
historical pKPDB outputs. See [the evidence amendment](18_historical_pkpdb_settings.md).

## Numerical validation

The 1024-iteration candidate was selected from training-only convergence
experiments before validation on held-out structures. Damping, residual tolerance
(2e-5), slope threshold and sampled-monotonicity checks were unchanged.

Each AB/A/B state and iteration configuration ran in its own subprocess.
Midpoints were extracted by the existing `_grid_pka_result` from the already
computed curves, avoiding a duplicate full-structure solve and compilation.
Baseline statuses must match exactly; curves use absolute tolerance 1e-6 and
midpoints 1e-4 for equivalence checks. These are comparison tolerances, not
relaxed acceptance criteria.

| Check | Original 64 iterations | Candidate 1024 iterations |
|---|---:|---:|
| Nonconverged states, out of 150 | 9 | 0 |
| Valid JAX-Ka site-state midpoints | 4100 | 4597 |
| Failed site-state predictions | 540 | 39 |
| Out-of-range site-state predictions | 8 | 12 |
| Not titrating | 196 | 196 |
| Not reported | 16 | 16 |
| Previously valid midpoints lost | — | 0 |

Site-state counts include masked sites and count bound and free predictions
separately. They must not be read as numbers of training sites or complexes.
The earlier midpoint review found non-monotonic curves with a single 50%
crossing; additional iterations did not consistently remove their reversals.
A converged grid therefore does not guarantee an accepted midpoint.

## Usable supervision and coverage

| Measure | Original smoke | Completed teacher, original JAX | Completed teacher, JAX 1024 |
|---|---:|---:|---:|
| Eligible paired sites before method coverage | 1219 | 1219 | 1219 |
| Valid teacher paired sites | 964 | 999 | 999 |
| Paired sites shared by all comparison methods and teacher | 662 | 696 | 902 |
| Complexes with at least one usable teacher interface pair | 47/50 | 47/50 | 47/50 |

All teacher jobs are complete, but three complexes still lack usable teacher
interface pairs under the current masks and output coverage. The operational
80% teacher-interface requirement passes at 94%; this is not a claim of full
site coverage or independent physical accuracy. Missing and invalid labels are
never imputed.

| Split | Eligible interface sites | Teacher sites | JAX 1024 sites | Teacher–JAX matched sites | All-method common sites |
|---|---:|---:|---:|---:|---:|
| Training | 205 | 178 | 197 | 175 | 162 |
| Validation | 123 | 105 | 123 | 105 | 97 |
| Test | 141 | 115 | 130 | 109 | 95 |

JAX test-interface coverage increases from 90/141 in the original smoke to
130/141. Test-shell coverage increases from 186/389 to 365/389. Teacher and
method columns describe their individual coverage; matched support requires
both methods at the same site in both states.

## Bound-minus-free pKa agreement

The target is ΔpKa = pKa(AB) − pKa(free partner). Errors below compare predicted
shifts with current PypKa shifts. Metrics are computed per complex, averaged
within sequence group, and then averaged equally across groups. Bootstrap
confidence intervals resample whole sequence groups, not individual sites.

The following table uses the same all-method common interface sites: 162 sites
in 21 training groups, 97 in 10 validation groups, and 95 in 16 test groups.
MAE and RMSE are in pKa units; RMSE is also a group macro statistic, not pooled
site RMSE. The zero-shift baseline predicts no binding-induced pKa change.

| Method | Train MAE | Validation MAE | Test MAE (95% group-bootstrap CI) | Test RMSE | Test skill |
|---|---:|---:|---:|---:|---:|
| PROPKA | 0.539 | 0.400 | 0.461 (0.336–0.592) | 0.587 | −0.078 |
| JAX-Ka, 1024 | 0.578 | 0.399 | 0.495 (0.350–0.657) | 0.636 | −0.266 |
| pKAI | 0.284 | 0.209 | 0.255 (0.170–0.348) | 0.316 | −0.359 |
| pKAI+ | 0.416 | 0.297 | 0.308 (0.222–0.403) | 0.387 | 0.626 |
| Zero shift | 0.663 | 0.484 | 0.537 (0.420–0.658) | 0.669 | 0.000 |

Skill is computed per complex as 1 − MSE(predicted shift)/mean(reference
shift²), then aggregated by group. Tiny reference-shift denominators can
produce very negative values; a lower macro MAE does not necessarily imply
positive macro skill. The registered metric was retained rather than altered
to improve apparent performance. Full tables include pairwise support, residue
and shift strata, sign accuracy and correlations.

pKAI has the lowest test MAE on this common-support sample, but this does not
establish a statistically significant or independent experimental ranking.
pKAI/pKAI+ share PypKa training lineage, the sample is small and selected, and
historical pretraining overlap/equivalence is not resolved by this smoke.
The iteration change establishes improved numerical coverage; before/after
aggregate score differences also reflect changing support and must not be
presented as a pure accuracy improvement.

## Linkage, figures and runtime

Only 37 of 300 complex-method combinations have complete accepted charge
coverage for linkage calculations. Another 228 are blocked by uncertainty masks
and 35 by incomplete prediction coverage. Partial masked charges are not
reported as complete binding free-energy curves.

- [Interface skill figure (PNG)](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/plots/interface_skill.png)
  and [PDF](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/plots/interface_skill.pdf).
- [Bound/free error cancellation (PNG)](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/plots/error_cancellation.png)
  and [PDF](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/plots/error_cancellation.pdf).

All scientific jobs ran on compute nodes, excluding comp1400, with 2 CPUs /
4 GB per job and the 400-core submission cap. Fifty solver workers requested
100 cores. Scientific workers remain pinned to one CPU; the second allocation
provides memory headroom. The longest validation job took 27 min 56 s and
included six state/configuration subprocesses, not just candidate inference.
The previously memory-failed complex completed in 17 min 37 s. Final collection,
scoring and verification took 22 seconds. The four teacher retries took about
30–37 minutes each under a 90-minute total budget.

## Production recommendation and remaining work

1. Promote the validated 1024-iteration configuration and isolated/shared-grid
   execution into an explicitly versioned production wrapper, retaining all
   validity flags and bounded timeouts. Do not silently change old runs or
   treat this report as a change to the existing adapter defaults.
2. Run production predictions on the frozen split, retaining per-method
   failures and reporting coverage. Runtime and memory on larger structures
   remain unvalidated by this small-structure smoke.
3. Train only on training-eligible sites with valid reference labels; retain
   separate provenance for historical pKPDB labels. Test structures are not
   available for fitting or selecting model parameters.
4. Finish experimental Set 2 curation and overlap checks before independent
   accuracy claims. Numerical agreement with PypKa does not replace this step.

No dataset rebalancing, mask relaxation, structural reprediction, or historical
configuration assumption was introduced to obtain these results.

## Reproducibility record

- [Solver contract](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-isolated-v1/contract.json): fixed before the 50-pair validation.
- [Solver validation](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/solver_validation.json): all 150 state comparisons and validity counts.
- [Verified score report](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/verification.json): 300 complex-method shards, frozen masks/split and recomputed group metrics.
- [Coverage](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/coverage.csv) and [full scores](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-jax1024-v1/scores_set1.csv).
- [Teacher-only update](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-teacher-complete-v1/verification.json): separates teacher recovery from the solver change.
- [Implementation and job record](20_isolated_solver_validation.md): initialization 730852, teacher report 730853, validations 730854–730903, final report 730904.

Candidate manifest SHA-256:
`78f014a5d2c7e0232380369a4db8fbfaabf5cf7e4463720ab0c038449d637d57`.
Scorer SHA-256:
`ef6dff8314a04b7978e1532203a20452cfbc4650f501eb5ea4031be0de23d9f9`.
The original smoke, partial attempts, timeout receipts and both memory-failed
attempts remain preserved.
