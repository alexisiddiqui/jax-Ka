# Frozen-input smoke benchmark and current-preparation gate

The user authorized updating the preparation gate and production scorer, then running approximately 50 pairs end to end. The subsequent historical-configuration question was clarified: exact historical pKPDB configuration is needed to claim reproduction of deposited labels, not to generate and benchmark new labels under our own explicit configuration.

## Gate revision

The new `frozen-smoke` path uses `G2_current`, requiring verified frozen inputs/masks, the exact current teacher configuration, environment/code/weight hashes, and native teacher curves from a freshly run AB/A/B reference. `G2_historical` remains unresolved and is not silently marked passed. Earlier historical diagnostics and their gates remain intact. Generated labels are current PypKa labels, not reconstructed historical pKPDB outputs.

Teacher settings remain εin=15, εsolvent=80, ionic strength=0.1, grid size=81, nonperiodic, G54A7, hydrogen optimization enabled, ions stripped, 298.15 K and SER/THR titration disabled. Each job records actual input/config/implementation/environment hashes. A fresh preflight checks joint-system behavior for PROPKA, JAX-Ka, pKAI and pKAI+, pKAI trainability, and PDB2PQR rebuilding. KaML stays excluded because its released predictor is single-chain.

A teacher gate failure has no substitute teacher. Method gate failures block batch dispatch. Structural rejection rates are not reapplied to the already accepted frozen structures. This explicit gate applies to the bounded engineering smoke; production remains disabled until results/coverage are reviewed.

## Sample and execution

Runtime: `_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-v1`.

Select 50 small-first complexes (46–349 residues) with distinct sequence groups preferred, covering 24 train / 10 validation / 16 test pairs, 12 antibody pairs, 8 glycan cases, 13 buffer cases and 4 other-ligand cases (component categories overlap). Selection uses preparation/size/role metadata, not prediction results. This is an engineering sample, not a representative benchmark or a replacement split. Test labels may be generated for pipeline validation but must not drive model fitting/hyperparameter choices.

Methods: PypKa teacher, PROPKA, JAX-Ka, pKAI, pKAI+ and zero-shift null. One Slurm job per complex runs methods sequentially, fast methods first and teacher last, with 2 CPUs / 4 GB. Fast-method timeout is 600 seconds each; teacher timeout is 1,800 seconds per complex across AB/A/B. The runtime remains single-threaded with reserved headroom. At most 100 requested worker CPUs for this batch, subject to the global 400-core limit; comp1400 excluded. Scheduler failures and timeouts remain explicit missing coverage.

## Scoring contract (written before predictions)

The scorer joins frozen `training_eligible`/`evaluation_eligible` masks by full site key and reports each split separately. Legacy `sites.supervision_mask` is not consulted. Native eligible paired AB/free midpoints must both have `status=ok`; failures, unsupported groups and out-of-range midpoints are excluded explicitly and counted, never imputed.

For each method, report pairwise teacher/method support and common support across all scored methods. Primary subsets are interface (ΔSASA >10 Å²) and the 0–20 Å partner-distance shell. Additional strata cover residue type, shift magnitude, residue ΔSASA, role and radial shells. Coverage is relative to all eligible sites, including teacher failures.

Compute site metrics within each complex, average complex metrics equally within sequence groups, and average groups equally. Skill is `1 − mean((predicted ΔpKa − reference ΔpKa)^2) / mean(reference ΔpKa^2)` per complex; zero denominators remain undefined. Other metrics: MAE, RMSE, Spearman, sign accuracy for |reference ΔpKa|≥0.5, and absolute-error cancellation correlation. Report the number of defined groups for each metric.

Percentile confidence intervals resample entire sequence groups: 2,000 replicates for primary subsets and 400 for descriptive strata, seed 20261004, at least five defined groups required. These intervals quantify variability in the selected smoke sample, not representative population uncertainty. Representability reports group-macro structural-zero fractions (<0.01 predicted shift for ≥0.1 reference shift) by radial shell.

Linkage requires complete paired charge curves and all sites passing the current masks. No partial masked charge sum is presented as full-system ΔG. Native versus HH curves are labelled; unsupported charge groups invalidate full linkage.

Artifacts: `scores_set1.json/.csv`, `scores_per_complex.csv`, `scores_per_group.csv`, `coverage.csv`, `representability.csv`, `linkage.json`, `scoring_report.json`, strict merge receipts and per-method job evidence. These are PB-agreement diagnostics; pKAI/pKAI+ share the teacher lineage and are not independently validated by this comparison.

## Validation and status

Scorer tests passed: 5/5 (job 730738), covering unequal group/site counts, null and undefined metrics, frozen masks overriding legacy masks, split exclusions, missing predictions, complete-charge linkage and deterministic group resampling. Sample initialization completed as 730739; fresh gates are running as 730740. Batch and result status will be appended after completion.

Fresh gates passed in 730740: G1/G1b, G2_current, retained-method G3 checks, G4 and G5. G2_historical remains unresolved/nonblocking for current labels. Native PROPKA's fresh control did not show order sensitivity at this target; the earlier order-sensitive control remains a caveat rather than being overwritten.

Dispatcher 730742 submitted all 50 complex workers (730743–730792), with no unsent tasks. A dependency-controlled collector runs even after scheduler failures so they can be recorded rather than silently dropped. Secondary pooled scores are in `scores_pooled_secondary.csv`, teacher interface coverage in `coverage_gate.json`, and diagnostic skill/cancellation figures in `plots/`.

Six JAX-Ka cases exceeded the initial 600-second total AB/A/B budget. The failure type was `TimeoutExpired`; some had completed AB and exhausted the remaining budget on A/B. Their initial JSON/Parquet receipts are preserved in `timeout-history/`, and raw attempt directories remain intact. A single 1,800-second retry per affected complex runs as 730797–730802 with unchanged scientific inputs/configuration. First-attempt runtime failures must be reported separately from final prediction coverage. Original pending collector/verification 730793/730795 were cancelled and replaced with dependencies covering these retries.


## Completed smoke results

Final collection/plots completed as 730803; independent verification passed as 730804. Diagnostic audits completed as 730814/730815. This completes the gate/scorer/50-pair smoke task, not the full production benchmark.

| Method | Complete AB/A/B worker jobs | Runtime/failure notes |
|---|---:|---|
| PROPKA | 50/50 | Median 2.59 s per complex |
| JAX-Ka | 50/50 | Six initial 600 s timeouts recovered with a single 1,800 s retry; final-attempt median 134 s, maximum 1,249 s. Worker completion does not imply every site converged. |
| pKAI | 50/50 | Median 6.57 s |
| pKAI+ | 50/50 | Median 5.34 s |
| Null | 50/50 | Zero-shift baseline |
| PypKa | 46/50 | Four B-state timeouts after successful AB/A; median 261 s and maximum 1,722 s among complete jobs. Valid partial paired labels are retained explicitly. |

Teacher labels cover at least one eligible interface site in **47/50 pairs (94%)**, passing the specified 80% pair-coverage check. This is distinct from full-job completion and full-site coverage. Across all eligible sites there are **964 teacher-paired sites out of 1,219**, and **662 sites** shared across every scored method. For interface sites specifically, the corresponding counts are **376/469** teacher-covered and **279** common to all methods. Coverage is also reported by split and method; no prediction failure is imputed.

The scorer emitted 608 method/split/support/subset summaries. Independent verification recomputed **3,648 group-macro metrics**, checked all 300 complex–method shards, and confirmed unchanged frozen masks/splits. Five targeted scorer tests passed before execution. Current teacher resolved-parameter auditing checked 138 states from fully completed jobs, including seed 1234567, 200,000 Monte Carlo steps and 1,000 equilibration steps. Temperature is pinned to 298.15 K in requests/config hashes; upstream `get_clean_params()` deliberately omits it from the saved resolved listing.

### Numerical validity and production follow-up

JAX-Ka returned **540 failed site-state rows** despite process completion. The saved residuals and adapter logic distinguish **523 rows due to grid nonconvergence across seven complexes** from **17 invalid bracketed midpoint readouts**. Failed outputs touch 13 complexes overall, including rows already excluded by the frozen masks. They remain excluded; no tolerance or validity flag was relaxed. Test-shell JAX coverage is 186/389 eligible sites, making the coverage issue material even though all workers completed.

Full-charge linkage was valid for 36 complex–method combinations. Another 228 were suppressed by uncertainty masks and 36 lacked complete charge coverage. These are explicit unavailable full-system results, not zero linkage effects.

Some per-complex skill scores have very large negative magnitude because the reference ΔpKa denominator is tiny. The registered definition and group weighting were retained; inspect MAE/RMSE, coverage, common-support results and magnitude strata alongside skill. This selected engineering sample does not support production performance or independent experimental-accuracy claims.

Before production: investigate JAX grid/midpoint validity and review its coverage; use `runtime_by_residues.csv` to set appropriate per-state/complex budgets, including the four teacher timeouts. Do not release the full benchmark simply because worker exit codes passed. Historical pKPDB reconstruction remains unrelated to these current numerical/runtime issues.

### Artifacts

- [Skill diagnostic](../../../_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-v1/plots/interface_skill.png) and [error cancellation](../../../_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-v1/plots/error_cancellation.png), also PDF.
- `scores_set1.json/.csv`, per-complex/per-group CSVs, `scores_pooled_secondary.csv`, `coverage.csv`, `representability.csv`, `linkage.json`.
- `completion.json`, `verification.json`, `coverage_gate.json`, `resolved_teacher_audit.json`, `diagnostics.json`, `runtime_by_residues.csv`.
- Archived `scoring_sources/` with hashes and `timeout-history/` retaining initial JAX attempts. No frozen structural artifact or original proposal was changed.
