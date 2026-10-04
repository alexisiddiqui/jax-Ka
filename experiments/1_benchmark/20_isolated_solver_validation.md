# Isolated JAX solver validation and complete-teacher report

**Completed:** all 50 validations and final scoring/verification passed. See
[the consolidated results report](21_smoke_validation_report.md). Final collection
was job 730904; all 150 candidate states converged and no previously valid
midpoint became invalid. The launch-time plan below is retained for provenance.

## Scope and contract

The user authorized isolating the remaining memory-heavy calculation, validating
an increased solver iteration budget, and updating the scored report. The
original smoke artifacts, model defaults and numerical acceptance criteria remain
unchanged. No production or training job has been launched.

The candidate is 1024 iterations, selected from the preceding training-only
convergence diagnostics. All 50 smoke complexes are validated at both the
original 64 and candidate 1024 iterations. Held-out examples assess this fixed
candidate; they do not select its iteration count or fit model parameters.

Each state/configuration executes in a fresh subprocess. Midpoint extraction
uses the existing `_grid_pka_result` on the computed curves, avoiding a second
full-structure compilation. Every baseline must match original prediction
statuses, curves (absolute tolerance 1e-6) and reported midpoints (absolute
tolerance 1e-4). These equivalence-test tolerances do not change solver validity
thresholds. The source/configuration contract was saved before validation.

All scientific work runs on Slurm, excluding comp1400, with 2 CPUs / 4 GB per
job. The shared submission wrapper enforces at most 400 user-requested cores.
Fifty validation workers request 100 cores; report jobs request 2 cores each.

## Complete-teacher report

Job 730853 assembled and verified a separate report using the four successful
PypKa retries. Its runtime directory is
`_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-teacher-complete-v1`.
All 50 teacher calculations completed. Current masked teacher coverage increased
from 964 to 999 paired sites; all-method common support rose from 662 to 696.
Usable interface pairs remain 47/50. Completed jobs do not guarantee reported
pKa values for every residue type or eligible site.

This report retains the original 64-iteration JAX predictions. It passed
receipt/input checks and independent recomputation of group macro metrics;
150 teacher state configurations can now be audited. Dataset masks/splits and
scoring definitions are unchanged. Initial teacher timeout receipts remain in
the numerical follow-up directory.

## Solver jobs and outputs

- Initialization and contract: 730852.
- Isolated validations: 730854–730903; the first is the previously memory-failed
  complex ebcb3f31129dcaa6.
- Input/output root: `campaigns/frozen-smoke-isolated-v1` beneath the runtime.
- Dependent collection generates `campaigns/frozen-smoke-jax1024-v1` only when
  all 50 baseline-equivalence validations are present and receipt hashes pass.
  Failure leaves an explicit `collection_status.json`; no partial candidate
  score is presented as a completed report.

The candidate report will retain nonconverged states and non-monotonic sites as
invalid and report any losses of previously valid midpoints. It is a diagnostic
report, not automatic approval for production. Engineering-smoke agreement with
PypKa is not independent experimental accuracy.
