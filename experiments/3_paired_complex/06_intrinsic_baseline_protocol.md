# Native-state intrinsic baseline protocol — 2026-10-05

## Execution status

Initialization job 733248 completed. First feature shard 733292 completed with
10,156 tautomer-state rows and no missing functional-atom geometry. Submission of
the remaining jobs is blocked by the automatic approval service reporting model
capacity exhaustion; the requests were not executed. This is not a safety finding.
No baseline fitting or validation scoring has run yet, and the new aggregation
regression test is staged but not yet executed.

Resume through `_HPC/submission/jax-Ka/pkabench/launch-intrinsic-v1.sh` in the parent
workspace. It records each successful submission and schedules assembly, three
seed fits and collection with dependencies, using the existing 400-core admission
checks. The prior authorization remains in place; do not interpret this service
failure as a request to bypass approval review or run scientific work on login nodes.

## Registered configuration

This configuration is fixed before fitting. Use the verified `tierB/native-v2`
training/validation export, with frozen masks and groups. No test data enters.

Predict each nonreference tautomer's intrinsic pKa minus its G54A7 model-compound
pKa from observed local geometry. Use a single shared CatBoost model for AB and
free-partner states. Subtract matched state predictions to evaluate binding-induced
intrinsic shifts; retain the original tautomer identities and reference convention.
This does not project or learn the interaction matrix.

## Inputs and training

Use all eligible finite native site-state energies, including sites whose coupled
midpoint lies outside the sampled pH range. This differs from midpoint-only training.
The masks apply at site level. Match bound/free labels by full original site key and
tautomer, never by row order. Unmatched states contribute only to absolute evaluation.

Features are residue/tautomer category, heavy/N/O/S atom counts within 3/6/10/15 Å,
and fixed formal-charge neighborhood proxies, evaluated at the functional-atom
centroid and the first two named functional atoms. Exclude atoms from the site's
own residue. Absent functional-atom slots remain missing with explicit indicators.
The field proxy uses fixed ASP/GLU −1 and LYS/ARG +1 charges; it is not a teacher
protonation state or an exact electrostatic potential. No teacher energies, pKa
labels, occupancies, structure IDs, residue numbers, interface flags, or bound/free
state flags are model inputs. The latter ensures identical geometry yields the
same predicted intrinsic value in either state.

Read model-compound constants from the installed force-field `.st` files and pin
their hashes. This preserves tautomer-specific references, rather than substituting
one residue-average pKa. Add constants back at inference.

CatBoost uses 500 iterations, depth 6, rate 0.03, Huber delta 1, and seeds 17/29/43.
There is no validation checkpoint selection or hyperparameter search. Weights give
equal groups, equal complexes per group, equal site-state records per complex and
equal tautomers per site-state; normalize to mean one. Use two numerical threads
within an eight-CPU/16 GB fitting allocation. Feature shards use two CPUs/4 GB.
All requests remain under the 400-core user-wide cap and exclude comp1400.

## Baselines and reporting

Compare force-field model-compound values and a training-only, weighted
per-tautomer constant. Both predict zero AB-minus-free intrinsic shift.

Report absolute and paired-shift MAE/RMSE, averaging tautomers within sites, then
sites within complexes, then complexes within sequence groups. Use 2,000 group
bootstrap replicates and paired changes from each baseline. Report all eligible
sites and interface sites, plus antibody/general strata. Include seed dispersion,
coverage and feature importance. These are intrinsic-state targets, not coupled
midpoint pKas; do not compare their MAE numerically with experiment 02's midpoint
MAE or claim experimental accuracy.

Expected artifacts under `tierB/intrinsic-baseline-v1`: feature shards and hashes,
three saved models, predictions, scores, paired bootstrap comparisons, figures,
a report, and verification. Scalar pair reduction remains unvalidated.
