# 50k GQT regularization screen

Authorized 2026-10-08 after stopping the 100-epoch capacity continuation. The
experiment holds the cleaned 5,000-structure historical-pKPDB cohort, frozen
142-complex current-PypKa validation set, explicit signed-shift target, uniform
within-structure loss, backbone graph, 49,709-parameter architecture, batch size
eight, and full float32 arithmetic fixed.

| Arm | Context masking | Optimizer |
|---|---:|---|
| baseline | 0% | Adam |
| mask | 5% | Adam |
| adamw | 0% | AdamW, weight decay 1e-4 |
| mask-adamw | 5% | AdamW, weight decay 1e-4 |

AdamW decays learned matrices and group embeddings; biases and normalization
vectors are excluded. Dropout is zero. Every arm uses a per-update cosine
schedule from 1e-3 to 1e-5 beginning with the first update. Runs are capped at
20 epochs and select minimum validation group-macro MAE with `min_delta=0.001`
and patience eight. Seeds 17, 29 and 43 change initialization, sampling order,
and deterministic per-structure context masks. Full validation is unaugmented.

The twelve jobs run as an array capped at four simultaneous A40 allocations on
comp1400. Each task requests eight CPUs and 16 GB host memory. The full-protein
encoder uses the validated float32 indexed Triton attention backend; the much
smaller query-attention layer remains native. A four-arm finite-update smoke gate
precedes production. No test structures or labels are read.

An initial native-attention submission was cancelled before epoch one completed
and is retained as `pretraining/gqt-50k-regularization-v1`. Registered production
artifacts use `pretraining/gqt-50k-regularization-v2-triton`.
