# Joint oGQT critical-batch audit

The 282,285-parameter capacity arm was cancelled on 2026-10-09 at the user's
request.  It is replaced by a five-epoch batch-size study of the current
67,725-parameter backbone-only oGQT on the full frozen 5,000-cluster PINDER
training cohort plus pKPDB replay.

The arms scale every PINDER size-bucket batch and the pKPDB batch by 0.5, 1, 2,
or 4.  All start from the identical selected pKPDB checkpoint, see every
training structure once per epoch, use no stochastic augmentation, and are
evaluated on the unchanged validation cohorts.  AdamW's learning rate follows
`1e-4 * sqrt(batch scale)`; beta and epsilon remain fixed.  Fixed epoch 5 is the
primary comparison, avoiding checkpoint-selection bias in this short audit.

The gradient-noise measurement samples 64 independent PINDER complexes and 64
pKPDB structures at the common parent checkpoint.  It estimates the mean and
trace covariance of each task gradient separately, then combines independent
task covariances for the joint update:

`B_noise = (tr(Sigma_PINDER) + tr(Sigma_pKPDB)) / |G_PINDER + G_pKPDB|^2`.

One unit of this batch measure is one PINDER complex plus one independently
sampled pKPDB structure.  The estimate and the empirical sweep answer different
questions: the former locates the diminishing-returns scale, while the latter
measures validation accuracy, throughput, updates, and peak VRAM under the
actual padded graph implementation.  No final test records are read.
