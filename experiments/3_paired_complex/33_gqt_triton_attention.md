# Indexed Triton attention

Status: **accepted as an optional full-float32 CUDA backend**. Native JAX remains the default. The active 50k/200k/800k epoch-21-to-100 continuation jobs started before this change and were left unchanged.

## Problem

The native encoder constructs K and V once per residue and then gathers both across every padded neighbor slot. For the measured batch-8 200k case with 512 residues, 256 neighbor slots, four heads and head width 23, the node-level K/V arrays occupy 2.9 MiB while the logical gathered K/V arrays occupy 736 MiB per encoder block.

The optional backend calls a graph-specific kernel through [`jax_triton.triton_call`](https://jax-ml.github.io/jax-triton/triton_call/). Each program loads indexed neighbors directly, computes logits, learned edge bias, masks, distance-switch renormalization and the existing `1e-8` floor, and writes only the attention message. The backward kernel recomputes the weights and atomically accumulates node-level K/V gradients. The query-attention layer remains native because its tensors are much smaller.

Environment: JAX/JAXLIB 0.6.2, `jax-triton==0.3.0`, `triton==3.3.0`, Python 3.11, full float32, one comp1400 A40. The two optional packages live in `_runtime/jax-Ka/pkabench/overlays/jax-triton-0.3.0`; the shared training environment was not mutated. The repository exposes the same pins through the `triton` package extra.

## Registered gates

- Full loss absolute difference at most `1e-5`.
- Full parameter-gradient relative L2 at most `2e-4`.
- Triton full gradient step faster than native JAX.
- Empty-neighbor rows remain exactly zero.
- The forced normalization-floor case remains within the float32 numerical gate.
- Existing batched `jax.vmap` training and its reverse-mode gradient work without per-complex kernel launches.

## Full-step results

These are complete loss-and-gradient steps on real batch-8 graph batches. Only the two encoder attention messages differ.

| Model | Parameters | N/K/Q capacity | Native step | Triton step | Speedup | Native temp | Triton temp | Temp reduction | Gradient rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | 49,709 | 384/192/160 | 48.8 ms | 21.5 ms | 55.9% | 306.0 MiB | 66.0 MiB | 78.4% | 2.05e-7 |
| 50k | 49,709 | 512/256/288 | 80.8 ms | 39.3 ms | 51.3% | 645.2 MiB | 219.0 MiB | 66.0% | 2.46e-7 |
| 200k | 209,645 | 384/192/160 | 65.0 ms | 29.8 ms | 54.2% | 543.1 MiB | 85.5 MiB | 84.3% | 4.49e-7 |
| 200k | 209,645 | 512/256/288 | 106.3 ms | 52.9 ms | 50.2% | 533.8 MiB | 353.7 MiB | 33.7% | 1.89e-7 |
| 800k | 790,093 | 384/192/160 | 115.4 ms | 58.1 ms | 49.6% | 153.6 MiB | 122.6 MiB | 20.2% | 3.64e-7 |
| 800k | 790,093 | 512/256/288 | 190.6 ms | 119.2 ms | 37.5% | 258.4 MiB | 206.9 MiB | 19.9% | 3.74e-7 |

All six gates passed. Five losses were bit-identical; the remaining 800k loss differed by `1.19e-7`. The widest model saves a smaller fraction of compiled temporary memory because its dense projections and feed-forward layers contribute more of the full step.

## Kernel findings

Eight warps is the selected A40 configuration. Two warps severely underoccupied the GPU. At eight warps, operator forward time improved by 6–29%; the direct atomic backward was 10–16% slower in the repeat used for the final report. Removing the two large gathered intermediates allowed the surrounding compiled model to fuse and schedule better, producing the 37–56% full-step gains above.

The two measured operator shapes had zero XLA-reported Triton temporary allocation versus 225–392 MiB for native backward. Forward relative error was at most `2.5e-7`; Q/K/V/edge-bias gradient relative error was at most `3.0e-7`. Empty rows were exactly zero. The forced floor case matched at `9.6e-8` relative L2. Atomic ordering at the model `vmap` boundary added at most `1.4e-7` relative variation.

## Implementation and evidence

- Production kernel and custom VJP/vmap: `src/pkanet/triton_attention.py`.
- Explicit predictors: `predict_shift_indexed` and `predict_pkpdb_indexed` in `src/pkanet/model.py`.
- CUDA regression: `tests/test_pkanet_triton.py`; 11 model tests passed in Slurm job 743929, and the enhanced focused direct-inference/batched-gradient test passed in job 743934.
- Full-step benchmark: `benchmarks/profile_gqt_triton_fullstep.py`.
- Microbenchmark: `benchmarks/probe_gqt_triton_attention.py`.
- Consolidated runtime report: `_runtime/jax-Ka/pkabench/audits/gqt-triton-attention-v1/report.md`.
- Machine-readable gate record: `_runtime/jax-Ka/pkabench/audits/gqt-triton-attention-v1/verification.json`.

The backend is CUDA-only and rejects non-float32 Q/K/V at trace time. Triton is imported lazily, so native CPU use and normal installs do not require it. Subsequent training manifests must record the indexed backend explicitly; existing experiment manifests and checkpoints are not reinterpreted.
