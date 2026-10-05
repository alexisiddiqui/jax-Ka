# Native teacher export and experiment 03 entry gate

Export and native-order MC replay are complete; see [results](05_native_export_results.md).
The combined `readiness.json` is authoritative for modelling readiness; the earlier
export-only `verification.json` does not certify an unmodified PypKa reload path.

The next dataset version uses all 778 frozen training and 151 validation complexes.
Test inputs and labels are excluded. No structures or PB calculations are regenerated.
The fixed 500-complex recovery overlay replaces only 54 successfully retried states;
seven retries still failed, and the separate preparation failures remain unresolved.
Original successful states and all validation labels must compare unchanged.

## Native state representation

PypKa 2.10.0 exports multiple microstates per titratable site, including tautomers
and a reference state. Preserve this resolution. A site-level scalar coupling matrix
is not equivalent without a separately validated reduction; do not feed native
microstate interactions directly into a binary-site solver.

Installed source evidence: `pypka/mc/run_mc.py:53–114`, `pypka/tautomer.py:777–835`,
`pypka/titsite.py:172–177,272–280`, and `pypka/mc/mc.pyx:113–120`.

For each site, the native ordered nonreference states precede its reference state.
Nonreference energies obey `g = ln(10) * pKint * (1 - 2*occupancy)` in kBT;
reference energy is zero. The pH-dependent term is `ln(10)*pH*occupancy`.
The native interaction matrix is indexed by microstate, in kBT. Same-site blocks
are undefined placeholders, not observations; the export zeroes those blocks and
retains an owner array so they can always be masked. Off-site blocks must be finite,
free of sentinel values and symmetric. Terminal numbering offsets are removed
before mapping back to original chain, residue number and insertion code.

Site and pair supervision must use the frozen uncertainty eligibility flags.
For direct pair losses, both endpoint sites must be eligible. Masked sites still
remain in the complete physical system when replaying native curves; dropping
those sites would change the teacher calculation. Out-of-range midpoints do not
by themselves invalidate finite native state energies/curves.

## Export and checks

Candidate output: `_runtime/jax-Ka/pkabench/tierB/native-v2`.
Each complex has paired midpoint/curve targets, per-state native arrays and mapped
site metadata. The state index retains source hashes, the state-specific recovery
choice and split/component assignment. The collector checks all receipts and
hashes; export failures remain explicit. Rejected intermediates cannot be treated
as training-ready targets.

The first shard in `native-v1` was rejected because its curve check compared raw
float64 values directly with schema-stored float32 curves. Version 2 compares
exactly after the required float32 conversion; it does not relax an error tolerance.
A compute-node regression test checks the units and this round trip. Version 1
is an incomplete failed attempt and must not be used.

Before neural training, replay at least one small training-state calculation from
saved MC energies, with the original settings, and compare native curves. Decide
whether the model should retain native states or validate a reduction before
implementing scalar pair supervision. Frozen pKAI remains the main midpoint
baseline; experiment 02 did not establish a reliable fine-tuning improvement.
