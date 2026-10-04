# Frozen smoke numerical follow-up (2026-10-04)

The user authorized proceeding after the historical-settings review. The original
`frozen-smoke-v1` predictions, scores and verification remain unchanged.
Follow-up artifacts are written separately under
`_runtime/jax-Ka/pkabench/campaigns/frozen-smoke-numerical-v1`.

Four PypKa jobs retry the full AB/A/B calculation with a 5400-second total
budget, increased from 1800 seconds. Scientific configuration is unchanged.
The original failed receipt and retry policy are retained. Jobs: 730819–730822.
The new predictions must be reviewed before merging into a successor score report.

Thirteen JAX-Ka jobs inspect every complex with failed numerical site rows.
They save residuals for every pH, full curves, midpoint validity, monotonicity,
slopes, exact configurations and structure hashes. All splits reproduce the
64-iteration baseline. Only training structures explore 256 and 1024 iterations;
no teacher labels are read, no validity thresholds are relaxed and no new solver
configuration is automatically adopted. Jobs: 730818 and 730823–730834.

The source explains two distinct validity checks: grid-wide fixed-point
convergence (one failing pH invalidates the state) and bracketed midpoint
requirements including sampled monotonicity and negative slope. Additional
wall time does not itself change the fixed 64 solver iterations. All jobs use
2 CPUs / 4 GB, exclude comp1400, and pass through the shared 400-core cap.
The 17 jobs request 34 cores total. Scientific execution stays on compute nodes.

This is an in-progress diagnostic, not a completed numerical correction or a
production release. Training and production prediction batches remain unstarted.

## Initial review and memory retry

Twelve of thirteen JAX diagnostics completed; 730831 exceeded 4 GB after
saving its AB/64 baseline. Retry 730841 writes to a separate `jaxka-retry`
directory and clears JAX compilation caches and collected model objects between
configurations. Allocation stays at 2 CPUs / 4 GB. The suspected accumulation
of compiled configurations is being tested; successful recovery is not yet
established. Original partial outputs are retained.

Review 730842 found that all 17 original bracketed midpoint failures on
converged grids have a sampled upward segment, despite exactly one strict
50% crossing. Several reversals persist unchanged with additional iterations.
Newly converged training cases also reveal non-monotonic sites, with individual
upward steps as large as approximately 0.095 in protonated fraction. Thus
monotonicity failures cannot all be described as roundoff. No validity rule has
been changed and these remain excluded under the current contract.

All three training complexes with original grid nonconvergence converge at
1024 iterations; two still fail at 256. Held-out structures were only evaluated
with the existing 64-iteration configuration. This supports testing an increased
iteration budget, but does not yet validate a new production configuration.

Two teacher retries completed with no worker exceptions in approximately
30.1 and 30.3 minutes; the other two were still running at this review.
A dependent review is scheduled after all remaining jobs terminate. Its
`review.json` reports the latest receipts and uses a completed memory retry
in preference to the retained partial diagnostic.
