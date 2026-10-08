# Paired-training diagnostic protocol — 2026-10-05

Execution: setup 733184; frozen gate 733185 (passed); last-layer seeds 17/29/43
in jobs 733187–733189; all-layer 733190–733192; scratch 733193–733195;
CatBoost 733196–733198. The frozen validation interface group-macro MAE is
0.35317 on pKAI's own support; final arm comparisons use common support.
All runs completed. Collection 733199 and independent audit 733200 passed;
[results](05_diagnostic_results.md) retain all seeds and paired uncertainty.

This protocol is fixed before experiment 02 fitting. Inputs are the verified
`finetune/handoff-v2`; outputs go to `finetune/diagnostic-v1` under the runtime root.
There are no test inputs or labels. The pilot recovery does not change this run.

## Support and loss

Fit training-interface sites only (ΔSASA >10 Å²): 3,105 representable pKAI sites.
The broader 23,958 training and 8,157 validation feature rows remain available for
secondary shell evaluation. Each method retains its own representation coverage;
arm comparisons use identical site keys on the intersection of supports.

Neural loss is Huber on paired ΔpKa divided by the training-only residue-type
standard deviation, floored at 0.25 pKa. Weights start with equal sequence groups,
equal complexes within each group and equal sites within each complex, then double
for |teacher ΔpKa| >0.5 and normalize to mean one. Dropout is disabled in both
branches so identical environments cancel exactly. Weights remain shared.

## Fixed arms

| Arm | Settings |
|---|---|
| Frozen pKAI | Native weights; seed 17; sanity and comparison baseline |
| Last layer | AdamW, learning rate 0.001, only final linear layer trainable |
| All layers | AdamW, learning rate 0.0001 |
| From scratch | Same architecture, fresh Linear-style Kaiming-uniform weights/biases; rate 0.001 |
| CatBoost residual | Separate acid/base models; 500 iterations, depth 5, rate 0.03, Huber delta 1 |

Trained arms use seeds 17, 29 and 43. Neural runs use batch size 128, weight decay
0.0001, gradient clipping 5, at most 40 epochs and patience 8. Select checkpoints
by validation interface sequence-group macro MAE, including the epoch-zero model.
CatBoost uses the preregistered fixed iteration count without validation fitting;
its residual target is PypKa ΔpKa minus PROPKA ΔpKa. It uses group/complex/site and
large-shift weights, with Huber applied to the unscaled residual. Feature names are
explicitly allowlisted, excluding teacher values and absolute pKa predictions.

## Validation and interpretation

Require the frozen neural output to agree with the handoff baseline (allowing its
original two-decimal rounding), finite losses/predictions, exact cancellation for
identical inputs and actual parameter updates. Check hashes and split separation
before every arm. The separate environment uses NumPy 1.26 and the existing frozen
Torch 1.13.1 package; the benchmark environments are not modified.

Report group-macro MAE, RMSE and skill; 2,000 sequence-group bootstrap replicates;
paired MAE changes from frozen pKAI; seed dispersion; and antibody/general strata.
Pooled Spearman and sign accuracy for |reference ΔpKa| >0.5 are secondary and named
as pooled metrics. Validation selected the neural checkpoint, so these development
results are not unbiased held-out test estimates. Do not infer that architecture
is necessarily limiting just because this optimization configuration fails.

CPU execution uses two numerical threads, with eight allocated cores providing
16 GB memory at the required 2 GB/core. Jobs exclude comp1400 and the submission
wrapper counts all user jobs against the 400-core cap. The available scheduler
lists A40 GPUs, not the originally planned 3090; this small diagnostic starts on CPU.
