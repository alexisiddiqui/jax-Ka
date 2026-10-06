# Small graph-query pretraining pilot

2026-10-06: user chose graph-query pretraining before direct JAX-Ka parameter
training. The JAX-Ka launch controller is stopped and a durable TRAINING_HOLD.json
prevents a restarted controller from launching seeds. Existing checks are retained.

## Architecture and objective

Approximately 30k parameters (the run records the exact count): two sparse
residue-attention layers, width 32, four heads, feed-forward width 72; one
titratable-site query cross-attention block and a scalar head. Query tokens use
residue embeddings plus a group embedding. Inputs are soft residue probabilities,
terminal/disulfide flags and raw backbone geometry. Geometry uses a 20 Å radius
graph, 16 distance RBFs, local-frame directions and same-chain indicators. All
neighbors are retained; attention tapers over 18–20 Å. Degenerate local frames
have zero directional features and an explicit validity flag.

Predict final pKa as MODEL_PKA[group] + 8*tanh(head). These are output midpoints,
not native tautomer energies or binary intrinsic parameters. Single-state masked
pKa MSE is the first objective: no invented curves and no solver in this phase.
The encoder is in pkanet; optimization, losses, sampling and checkpoints are in
pkatrain. A later LocalTerms head can reuse the encoder, but it is not implemented
or scientifically validated by this scalar pilot. Coordinate derivatives are
also not claimed: geometry is precomputed for frozen structures in this pilot.

## Data and registration

Runtime: `_runtime/jax-Ka/pkabench/pretraining/graph-pilot-v1`.
Inherit the 477 train / 142 validation complexes and sequence-component split
from shared-v4-float32. Use AB only, with all finite eligible scalar pKas,
including non-interface sites. Retain the existing split-specific uncertainty
masks. Record exclusions for missing midpoints and quality masks. No test data.
Labels are current native-v2 PypKa, not historical pKPDB.

Seed 17; 20 epochs; full float32; Adam 1e-3; gradient clipping 1; accumulate eight
complexes per update. Sample components uniformly and complexes uniformly within
component. Final epoch is primary, with validation reporting each epoch and no
checkpoint selection. Compare against fixed model values and per-type constants
fitted on train only with matching sampling weights. Report group-macro MAE and
2,000 component-bootstrap replicates. Tests cover rigid-motion and permutation
invariance, padding, finite gradients to the encoder, learning and checkpoint
round-trip. Run on one A40 after preparation/tests pass; measure runtime rather
than reuse the JAX-Ka solver-training estimate.

## Capacity comparison added at the user's request

Run 10,229 parameters (width 20, FF 32) and 49,709 parameters (width 44,
FF 88) alongside the original 28,565 (width 32, FF 72). All retain two
encoder layers, four heads and one site-query block. The variants share the
exact immutable prepared graphs, labels, split, seed 17, sampling schedule,
optimizer, 20 epochs and final-epoch reporting. Only widths change. Each has
its own manifest, compilation cache, checkpoints and test gate. Original source
files are preserved under the first run's `source-snapshot/`; a regression test
checks exact original initialization and output equivalence after generalization.
No model is chosen on test data; this is a single-seed capacity comparison.

## Background pKPDB acquisition

Runtime: `_runtime/jax-Ka/pkabench/pretraining/pkpdb-v1`.
Download the official `https://bucket.pypka.org/pkas.csv` export with a content
hash, derive unique PDB IDs, and fetch their deposited asymmetric-unit mmCIF files
from RCSB with two concurrent requests. Each structure has a provenance/hash
receipt; download/parse errors are recorded and resumable. Job 737122 requests
2 CPUs / 4 GB, excludes comp1400, and shares the user-wide 400-core admission cap.
The acquired export contains 121,294 PDB IDs and no non-PDB identifiers.

These are deposited coordinates, not guaranteed historical pKPDB-prepared
structures. Downloads do not imply training admission. Before full pKPDB
pretraining, reconcile label numbering/assembly scope, apply quality masks, and
exclude sequence relatives of frozen validation/test and experimental reserves.
A raw-versus-cleaned comparison needs matched architecture, split and evaluation;
the current pilot alone cannot establish a cleaning benefit.
