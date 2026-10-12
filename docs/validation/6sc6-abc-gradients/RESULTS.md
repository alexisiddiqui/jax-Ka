# 6SC6 A–B interface gradient and GPU-memory check

Interface: any protein heavy-atom contact within 5 Å. All 604 A–B–C residues remain model context; gradients are masked to 39 A–B interface residues. Chain C contributes to prediction but its input gradients are masked to zero. Nonprotein material is omitted. The original structure contains zinc, sulfate and chloride; this tests numerical derivatives of the model, not experimental accuracy for this metal-containing complex.

GPU memory was measured with JAX preallocation disabled, float32 inputs/parameters and highest matmul precision (the raw report records output dtype). Each workload ran in a separate Python process. Whole-process peaks are sampled by nvidia-smi; short transient peaks may be missed. JAX allocator high-water marks are also reported. Torch-only counters do not include JAX allocations.

| Model | Forward process peak (MiB) | Forward+backward process peak (MiB) | JAX forward allocator peak (MiB) | JAX forward+backward allocator peak (MiB) |
|---|---:|---:|---:|---:|
| released 24-feature checkpoint | 1010.0 | 1014.0 | 224.4 | 193.7 |
| initialized 152-feature width44 ff88 seed4 | 1010.0 | 1014.0 | 160.3 | 180.8 |

Fresh processes loading both cached executables peaked at **754 MiB** total
GPU memory for each model, versus **1014 MiB** for the first forward+backward
process (including backward compilation). JAX allocator peaks on cached runs
were 128.56 MiB (backbone) and 128.60 MiB (side-chain). The first forward process
compiled a cold executable; its allocator high-water mark can exceed that of a
separate cached-forward/backward process. These are single-complex measurements,
not a guarantee for larger batches. Dense geometry scales quadratically.

The run used NVIDIA GH200 120GB, JAX 0.11.2 and Torch 2.14.1+cu130. Inputs and
parameters were float32; model output was float64 under the process JAX settings.
Slurm job 7349803 completed in 1m58s (exit 0). Remote CPU regressions: 20 passed;
GPU structural bridge regression: 1 passed. Local inference regressions: 8 passed,
1 GPU-dependent skip. Local CPU directional checks also passed for both models.

## Gradients and cache reuse

- backbone: directional checks passed; finite gradients; maximum gradient outside interface 0.0; restart forward/backward disk-cache hits 1/1; largest gradient change across restart 3.05e-05.

| Input group | Chain | Autograd directional derivative | ε | Finite difference | Absolute error | Relative error |
|---|---|---:|---:|---:|---:|---:|
| sequence | A | 0.609773 | 0.01 | 0.609475 | 0.000298 | 0.000489 |
| sequence | A | 0.609773 | 0.003 | 0.609719 | 5.4e-05 | 8.85e-05 |
| sequence | B | 0.3876 | 0.01 | 0.387593 | 6.29e-06 | 1.62e-05 |
| sequence | B | 0.3876 | 0.003 | 0.387593 | 6.7e-06 | 1.73e-05 |
| backbone | A | 0.321839 | 0.01 | 0.321807 | 3.15e-05 | 9.8e-05 |
| backbone | A | 0.321839 | 0.003 | 0.321726 | 0.000113 | 0.000353 |
| backbone | B | 0.176427 | 0.01 | 0.176406 | 2.14e-05 | 0.000121 |
| backbone | B | 0.176427 | 0.003 | 0.176538 | 0.000111 | 0.000629 |
| sidechains | A | 0 (expected) | — | — | — | — |
| sidechains | B | 0 (expected) | — | — | — | — |

- sidechain: directional checks passed; finite gradients; maximum gradient outside interface 0.0; restart forward/backward disk-cache hits 1/1; largest gradient change across restart 1.5e-08.

| Input group | Chain | Autograd directional derivative | ε | Finite difference | Absolute error | Relative error |
|---|---|---:|---:|---:|---:|---:|
| sequence | A | 0.0146787 | 0.01 | 0.0146799 | 1.21e-06 | 8.25e-05 |
| sequence | A | 0.0146787 | 0.003 | 0.0146799 | 1.28e-06 | 8.74e-05 |
| sequence | B | 0.0168982 | 0.01 | 0.016898 | 1.28e-07 | 7.59e-06 |
| sequence | B | 0.0168982 | 0.003 | 0.0168988 | 6.52e-07 | 3.86e-05 |
| backbone | A | 0.00655308 | 0.01 | 0.00655188 | 1.19e-06 | 0.000182 |
| backbone | A | 0.00655308 | 0.003 | 0.00654959 | 3.48e-06 | 0.000531 |
| backbone | B | 0.00710635 | 0.01 | 0.00710594 | 4.16e-07 | 5.85e-05 |
| backbone | B | 0.00710635 | 0.003 | 0.00711016 | 3.81e-06 | 0.000536 |
| sidechains | A | 0.00155468 | 0.01 | 0.00155542 | 7.33e-07 | 0.000471 |
| sidechains | A | 0.00155468 | 0.003 | 0.00155469 | 5.67e-09 | 3.64e-06 |
| sidechains | B | 0.00176885 | 0.01 | 0.00176871 | 1.31e-07 | 7.43e-05 |
| sidechains | B | 0.00176885 | 0.003 | 0.00177413 | 5.28e-06 | 0.00298 |

The scalar objective is the mean predicted shift at supported interface query sites. Checks use normalized gradient-aligned directions restricted separately to each chain and input group, with zero-sum sequence-feature perturbations. Central differences use ε=0.01 and 0.003; acceptance is absolute error ≤5e-5 + 2% of the analytic derivative. These are directional checks, not exhaustive Jacobian validation. Site identities, atom presence and topology are fixed. The side-chain model uses initialized parameters (width44/ff88, seed4), not a trained side-chain checkpoint.

Raw timing, allocator statistics, checks and gradient arrays accompany this report.
