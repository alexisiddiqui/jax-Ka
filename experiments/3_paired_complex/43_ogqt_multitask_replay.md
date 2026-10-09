# Joint pKPDB and paired-complex oGQT training

## Question

Can one shared backbone-only oGQT learn the paired PINDER objective without catastrophically forgetting its pKPDB representation?

## Registered experiment

- Initialize the 67,725-parameter oGQT from the selected epoch-16 pKPDB checkpoint.
- Keep the existing 20 A backbone residue graph and explicit orientation-aware site-token graph.
- Use no cropping, context masking, dropout, or test data.
- At every optimizer update, load one pKPDB batch and one PINDER paired batch.
- Compute three equally weighted losses: pKPDB state-shift MSE, PINDER AB/free state-shift MSE, and PINDER binding-shift MSE.
- Sum the pKPDB and PINDER gradients, clip their combined global norm to 1, and apply one AdamW update with weight decay `1e-4`.
- Use learning rate `1e-4` through epoch 10, followed by cosine decay to `1e-6` at epoch 20.

This is replay-based multitask training: the pKPDB examples remain present throughout paired training rather than being used only for initialization.

## Sampling

Each epoch visits the full 5,000-complex PINDER training cohort once, subject to the existing size buckets. An independently sampled pKPDB batch is paired with every PINDER batch. If one sampled pKPDB pass is shorter than the PINDER plan, another independently sampled pass supplies the remaining batches.

## Validation and selection

Evaluate both unchanged validation sets after every epoch. Epoch zero is the original pKPDB checkpoint and remains eligible.

Select the checkpoint with the lowest

`PINDER state MAE + PINDER interface paired MAE`

among checkpoints whose pKPDB group-macro MAE is no more than `0.05` above epoch zero. Report the PINDER state, paired, interface-paired, and distance-bin metrics together with pKPDB overall and target-shift-bin metrics.

## Execution gate

The Slurm job stages the three immutable mmap stores on comp1400 local storage, then performs one joint optimizer update as a smoke test. Full training starts only if that update is finite and its provenance receipt is written. The job requests one A40, 16 CPUs, and 32 GB, satisfying the 2 GB per requested CPU rule. The mmap stores remain on local disk and are read in batches, so their combined on-disk size does not determine the memory request. It does not count against the 400-core CPU-node queue cap agreed for the separate CPU workload.

