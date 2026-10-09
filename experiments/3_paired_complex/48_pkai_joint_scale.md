# Joint pKAI data-scale experiment

The experiment trains the native 3,608,001-parameter pKAI architecture on both the revised pKPDB state dataset and the leakage-filtered PINDER paired dataset. The registered arms are backbone-only at 10%, 50% and 75%, plus native all-atom pKAI at 10% and 50%. Every arm uses seed 17 and its matching nested fraction from both datasets. Validation and test memberships remain fixed across fractions.

Each update combines three task-normalized losses: pKPDB signed shift from the native fixed `PK_MOD`, PINDER AB/free signed state shifts, and the PINDER Siamese binding shift. The two branches use one shared pKAI model. Losses are normalized by task rather than concatenating rows, preventing the larger state dataset and abundant near-zero paired sites from setting the objective implicitly.

Production registration waits for the final pKPDB pool-v2 and environment manifests. The completed PINDER pool-v2 counts are:

| Fraction | Complexes | Clusters | Labelled sites | Interface sites |
|---:|---:|---:|---:|---:|
| 10% | 3,592 | 1,586 | 184,096 | 48,178 |
| 50% | 17,347 | 7,937 | 908,299 | 230,004 |
| 75% | 26,087 | 11,905 | 1,361,481 | 343,327 |

## Smoke gate

Job `747150` exercised both representations on real frozen pilot records. Each mode used 64 pKPDB sites and 51 matched PINDER AB/free sites, evaluated all three losses through one shared model, and performed one Adam update.

| Mode | pKPDB loss | PINDER state loss | Siamese loss | Peak allocated VRAM | Result |
|---|---:|---:|---:|---:|---|
| Backbone-only | 0.97077 | 0.85195 | 0.000118 | 124 MiB | pass |
| All atom | 0.97075 | 0.85202 | 0.000118 | 124 MiB | pass |

Every component gradient was finite and nonzero. The shared parameters changed, input arrays remained bit-identical, and swapping both Siamese branches left the paired loss unchanged. No production pool or test data was read. The machine-readable receipt is `training/pkai-joint-scale-v1/smoke.json` in the benchmark runtime.

## First production screen

The first production screen is restricted to the nested 10% pools and uses a 2 x 2 design: backbone versus native all-atom input, and pKPDB-only versus joint pKPDB/PINDER training. All four models use seed 17 and scratch initialization. Batch size is 256 and the square-root learning-rate rule gives `1e-6 * sqrt(256/64) = 2e-6`. State losses use the registered burial weights, the Siamese loss uses the registered interface-distance weights, and model selection uses unweighted validation metrics.

| Stage | Slurm job |
|---|---:|
| Real-record feature smoke | 747967 |
| Retry pKPDB pool-v3 with 64 GiB (the 16-GiB job 747925 was OOM-killed) | 747973 |
| Freeze 10% records after pool-v3 | 747974 |
| Backbone + all-atom preparation, 200 x 2 CPU tasks | 747975 |
| Immutable mmap packing | 747976 |
| Four A40 training arms | 747977 |

CPU preparation excludes comp1400 and caps the array at 400 queued cores. Each GPU task requests exactly two CPUs and 8 GiB host RAM. Every downstream stage uses `afterok`, so failed feature validation cannot fall through into training.

## Loader update (2026-10-09)

`train_scale` uses the shared loader protocol (`pkatrain.loading`, `loading_torch.PackedSiteSource`):
- the packed arrays stay on the GPU when they fit, otherwise they're streamed through pinned asynchronous copies;
- one fused finite check per step replaces the per-parameter checks;
- losses are read back in groups.

Batch membership, order and the loss arithmetic are unchanged. On synthetic data the losses are bit-identical, and
throughput rises from 81 to 117 steps/s (resident) or 88 (streaming); see
[06_loader_protocol_and_transfer.md](../4_backbone/06_loader_protocol_and_transfer.md). Each history row now includes
`loader_wait_fraction`. pKPDB validation reads the compact package `pretraining/pkpdb-val-pkai-v1` (the same pilot
validation rows, checked against `rows.json`) when it exists.
