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

## Move to Isambard-AI and feature-preparation fixes (2026-10-10)

- Runs on Isambard-AI (GH200, one GPU and 72 cores per job), in a separate uv venv (`pkai` optional group: torch
  2.11.0+cu128, CUDA 12.8 aarch64 wheels). The native pKAI package is copied unchanged (model sha256 0f808c24...);
  `architecture_gate` passes. Structures are read from the squashfs dataset images through `scripts/sqfs_run.sh`.
- The original smoke (5k-pilot arrays) is not repeated; `feature-smoke` and `register` passed. Registered 10% arms:
  5,878 pKPDB structures (157,674 declared sites), 3,422 PINDER complexes (176,871), 400 PINDER validation complexes.
- First preparation (72 processes per dataset, about 20-30 s each) found two bugs in the pKPDB path:
  - **Chain selection:** `defects.json` lists chains by `label_asym_id`, but `_pkpdb_atoms` filtered the
    author-field structure on them. 166 structures whose author and label chain IDs differ kept no atoms ("0 models",
    e.g. 5ma7 label A = author E), and where letters overlap the wrong chain could be kept. Chains are now selected on
    the label IDs in the CIF rows before the structure is built. All pKPDB features were recomputed.
  - **No candidate sites:** `np.asarray([])` is float, so `kf & kb & ...` raised a TypeError for structures with no
    side-chain training sites (19). The mask is now built as bool, so these report "no mapped pKPDB sites".
- After the fixes, 171 of the 192 failed structures prepare. Records that cannot give features are excluded and listed
  under `excluded` in `packed/verification.json`; any other error still blocks packing:
  - pKPDB: 19 with no usable side-chain sites, 6qsz (too large for the PDB format the pKAI parser reads), and 2xgc
    (coincident atoms);
  - PINDER: 8 training complexes with no paired sites. The 400 validation complexes are unaffected.

## Batch-size sweep, fraction scaling and the released pKAI reference (2026-10-10, Isambard)

All MSEs are at the best (selection) epoch on the same fixed validation sets: the frozen 5k-pilot pKPDB validation
package and the 400-complex PINDER validation cohort. PINDER validation is now reported for pKPDB-only arms too, but
their selection stays on pKPDB MSE alone; the earlier pKPDB-only runs were scored afterwards from `best.pt`
(`posthoc-validation.json`). Early stopping is unchanged (delta 0.001, patience 8); `train` takes an optional batch size
and epoch cap (sqrt LR rule, separate run directories), and `PKAI_FRACTION` selects the pool-v3 fraction (output
`training/pkai-joint-scale-v1-f<pct>`; 10% keeps the registered path).

Batch sweep, 10% full/joint, cap 400 (the registered b256/cap-100 run hit its cap at 0.406 pKPDB MSE):

| Batch | LR | Epochs | Selection | pKPDB | PINDER state | PINDER paired | Wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2e-6 | 177 | 0.1594 | 0.369 | 0.079 | 0.030 | 10.8 min |
| 1,024 | 4e-6 | 183 | 0.1586 | 0.366 | 0.079 | 0.030 | 2.9 min |
| 4,096 | 8e-6 | 231 | 0.1544 | 0.357 | 0.077 | 0.030 | 1.2 min |

Batch 4,096 / LR 8e-6 / cap 400 was chosen for the fraction runs. Every run stopped on the delta threshold with its best
epoch at or near the last, so convergence is slow rather than finished; a higher base LR or a schedule is untested.

Fraction scaling (batch 4,096, LR 8e-6, cap 400; peak GPU memory under 9 GB of 96):

| Model | Fraction | pKPDB val MSE | PINDER state MSE | PINDER paired MSE |
|---|---|---:|---:|---:|
| Released pKAI (all-atom) | - | 0.324 | 0.0070* | 0.0012* |
| Backbone, pKPDB-only | 10% / 50% | 1.075 / 1.104 | 0.866 / 0.870 | 0.212 / 0.215 |
| Backbone, joint | 10% / 50% | 1.077 / 1.088 | 0.847 / 0.840 | 0.205 / 0.205 |
| All-atom, pKPDB-only | 10% / 50% | 0.405 / 0.383 | 0.147 / 0.082 | 0.040 / 0.029 |
| All-atom, joint | 10% / 50% | 0.357 / 0.314 | 0.077 / 0.055 | 0.030 / 0.022 |

\* The PINDER validation labels were produced by this pKAI model (Siamese-task relabelling), so its PINDER columns are
a floor rather than a target. Its pKPDB figure is likely optimistic: it was trained on pKPDB PypKa labels, probably
including these validation structures, and on more data than these subsets. Computed with exp 48's validation code in
float32 (`native-pkai-validation.json`).

- All-atom features improve from 10% to 50% on every metric. Joint training beats pKPDB-only at both fractions (50%:
  pKPDB 0.314 vs 0.383; PINDER paired 0.022 vs 0.029), and the 50% joint model edges past the released pKAI on pKPDB.
- Backbone features do not improve with more data (about 1.08-1.10 pKPDB MSE), and joint training changes little.
- Training sites: pKPDB 154,539 (10%) and 872,450 (50%). 75% and 100% are running (jobs 7212900-7212915).
