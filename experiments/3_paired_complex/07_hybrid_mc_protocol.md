# Hybrid native-state MC validation — 2026-10-05

## Execution status

### Parallel scheduling revision

At the user's request, replace the sequential per-complex schedule with independent
state/method jobs. The old dependency-blocked jobs 733459–733461 were cancelled;
the remaining workers in 733438 were stopped after replacement-worker preflights
733463/733464 exactly reproduced the completed teacher and CatBoost-17 curves,
midpoints, and site metadata. Completed scientific outputs remain unchanged.

Dispatcher 733465 preserved **178 completed calculations** and initially launched
62 ready calculations concurrently. It requests two CPUs/4 GB per calculation and
admits new batches under the shared submission lock, counting every queued/running
user job against 400 cores. Only ready tasks enter Slurm: a state's teacher replay
must pass before its five variants become ready. Larger ready tasks start first
within replay/variant priority classes. Unfinished work directories are archived;
they are never mistaken for completed results.

The dispatcher completes and scores the pilot before admitting the remaining
122 validation complexes, then generates the full report automatically. It verifies
that all 178 preserved result hashes remain unchanged at completion. There are no
dependency-blocked production arrays consuming the CPU budget. Runtime scheduling
receipts, current status, interrupted directories, and pinned execution code hashes
are under `hybrid-mc-v1/parallel-v1/`.

The scientific helper module and scoring module remain unchanged. New results keep
`code_sha256` for the shared frozen scientific code and add `execution_code_sha256`
for the independent-task worker. Single-calculation seeds and settings are unchanged.
Both the energy/masking regression tests and dispatcher-readiness tests passed.

### Original schedule (superseded)

Initialization 733437 completed. There are 142 eligible validation complexes:
four of 151 lack a native state, and five have no eligible paired interface sites.
The fixed pilot has ten antibody–antigen and ten general complexes, spanning
62–978 residues. Its array is 733438 (two CPUs/4 GB per task).

Scoring preflight 733458 passed both the energy/masking contract test and the
group-weighting regression test, then scored the first completed complex. All
three original-state replays for that complex reproduced curves and midpoints
exactly. This one-complex preflight is not the pilot result.

Pilot collection/gate 733459 waits for all pilot tasks. Remaining-validation array
733460 (122 complexes) waits for that successful gate, and each worker additionally
checks the gate artifact. Full collection/report 733461 waits for the remaining
array and reuses completed pilot results. No model-performance threshold gates
the extension. A failed prerequisite prevents dependent work from running.
The user-wide queued/running total was 334 CPUs after submission, including
unrelated jobs. Pilot and final scientific results are pending.

## Registered diagnostic

Use the completed intrinsic CatBoost models (seeds 17/29/43), with no retraining,
new PB solves, structure prediction, or test-set access. This is a hybrid diagnostic:
PypKa supplies the complete interaction matrix and masked sites' intrinsic energies.
It is not yet an independent replacement for PypKa.

Select up to 20 validation complexes before measuring errors, allocating equal
quotas to antibody/antigen and general complexes and evenly spaced ranks by residue
count within each role, including size extremes. Require all three saved native
states and at least one eligible paired interface site. Document other exclusions.

For each AB/A/B state, first replay original saved energies with the corrected
native site order. Require maximum curve discrepancy <=1e-12 and midpoint discrepancy
<=1e-10, including identical missing-midpoint status. Then replace every eligible
site with complete predictions for all its nonreference tautomers, using common
replacement support for all methods. Convert pKint to dimensionless energy with
`g = ln(10) * pKint * (1 - 2*occupancy)`. Keep reference energies, padding, occupancies,
site identities, interaction lookup, and the entire pair matrix unchanged. Retain
masked or unpredicted sites' teacher energies as physical context.

Run model-compound, training-only per-tautomer constant, and all three CatBoost
variants. Reuse the original 200,000 MC steps, 1,000 equilibration steps, seed 1234567,
73 pH points from -2 to 16, and teacher temperature/configuration. Do not interpret
unmodified auxiliary `states_ddG` as a recomputed decomposition of substituted
intrinsics; the MC consumes `possible_states_g`.

Report curves, midpoint pKas, and bound-minus-free shifts against the saved teacher.
Report missing midpoint coverage rather than silently dropping sites. Compare
frozen pKAI only for midpoint and paired-shift metrics on common support. Primary
analysis is paired eligible interface sites; include all eligible sites and role
strata. Aggregate sites within complexes and complexes within frozen sequence groups;
bootstrap groups (2,000 replicates), report seed range and retained teacher-site
fraction. Changes from controls use paired group bootstrap support.

Extend to every eligible validation complex only after pilot implementation gates
pass. The gate tests replay, finite curves, site alignment, provenance, and retained
energies; it does not require CatBoost to beat a control. Reuse completed pilot
outputs. No selection or tuning on the test set.

Runtime artifacts: `_runtime/jax-Ka/pkabench/tierB/hybrid-mc-v1`.
Jobs request two CPUs and 4 GB each, exclude comp1400, and count all queued and
running user CPUs against the 400-core cap.
