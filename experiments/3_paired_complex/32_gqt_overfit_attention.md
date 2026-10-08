# Best-versus-late GQT attention audit

Compare each 50k, 200k and 800k GQT at its lowest-validation-MAE checkpoint
with its last completed checkpoint from the stopped long continuation. Use the
same 128 component-unique training complexes and every frozen validation
complex for every checkpoint. No test data are read.

For each query site, layer and head, measure attention entropy, effective
neighbor count, maximum mass, mean attended distance, distance-shell mass,
self mass, cross-chain mass and titratable-neighbor mass. Relate the change in
these quantities to the site's change in absolute prediction error. Stratify
prediction error by teacher-shift magnitude and residue group.

Attention alone is descriptive. Repeat predictions after removing all context,
titratable neighbors, individual distance shells, residue identity, and edge
geometry. Compare the causal prediction effects and ablated MAE between best
and late checkpoints. Report both train and validation behavior so increased
training-specific context use can be separated from generalizable context use.

Repeat the audit with epoch 55 as the late checkpoint for every size. This
matched-exposure comparison is primary for differences between capacities; the
original last-completed comparison remains useful for describing each stopped
run but confounds checkpoint age across sizes.
