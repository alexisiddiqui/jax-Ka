# Strict backbone-only longer training

User-approved 2026-10-06: retain the backbone-only target and test 100 epochs with
learning-rate decay across three seeds before changing model geometry.

Use the original 49,709-parameter architecture (width 44, FF 88, two encoder
layers, four heads and one site-query block). Train from scratch with seeds
17, 29 and 43. Each seed has a separate immutable manifest, optimizer state,
checkpoints, compilation cache and final report. The same 477 training / 142
validation complexes and all site labels/masks are reused. No test access.

Strict backbone inputs: residue identity, N–CA–C local frames, Cα distances,
same-chain indicator, true-terminal flags and frame validity. Disable the
SG-derived disulfide flag (column 22). No side-chain coordinates. Original
prepared structures and labels remain the full-structure reference; they are
not recalculated after stripping atoms. The earlier 20-epoch pilot retained
the disulfide flag, so it is not an exact training-duration control.

Keep 20 Å **Cα–Cα** radius graphs for this experiment, with taper over 18–20 Å.
This is not a heavy-atom contact cutoff or the atom-distance uncertainty mask.
Attention layers expand the indirect receptive field; they do not create direct
edges to every distant residue. The larger-radius hypothesis remains untested.

Full float32; Adam; clipping 1; group-uniform sampling; eight complexes per
optimizer update. Learning rate 1e-3 through 20 epochs (1,200 updates), then
cosine decay to 1e-5 over the remaining 80 epochs (4,800 updates). Schedule
counters are part of the Optax checkpoint. Preserve epoch-20 validation
predictions as the within-run control. Final epoch 100 is primary; no early
stopping or validation-selected checkpoint. Resumption truncates any history
written after the last committed checkpoint.

Jobs 737387/737388/737389 request one A40, 8 CPUs and 16 GB per seed.
The shared model tests now check schedule endpoints, scheduled-optimizer
checkpoint equivalence and removal of the disulfide flag. Final reporting
plots training loss, validation MAE and learning rate, reports per-type errors,
and bootstraps the epoch-100 minus epoch-20 change over sequence components.
Three seeds characterize this run's variability; they do not prove all sources
of uncertainty are covered. JAX-Ka parameter training remains on hold.
