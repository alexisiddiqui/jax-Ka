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
- Training sites: pKPDB 154,539 (10%) and 872,450 (50%).

### 75% and 100% (atom16)

| Arm | 10% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|
| All-atom, joint: pKPDB val MSE | 0.357 | 0.314 | 0.309 | 0.305 |
| All-atom, joint: PINDER state / paired | 0.077 / 0.030 | 0.055 / 0.022 | 0.050 / 0.021 | 0.048 / 0.020 |
| All-atom, pKPDB-only: pKPDB val MSE | 0.405 | 0.383 | 0.381 | 0.381 |
| Backbone, joint: pKPDB val MSE | 1.077 | 1.088 | 1.077 | 1.066 |
| Backbone, pKPDB-only: pKPDB val MSE | 1.075 | 1.104 | 1.078 | 1.075 |

- The all-atom joint model improves at every fraction with shrinking gains (0.044, 0.005, 0.004); at 100% it beats the
  released pKAI on pKPDB validation (0.305 vs 0.324). The pKPDB-only model plateaus from 50% (0.38).
- Training sites at 75% / 100%: pKPDB 1,306,448 / 1,724,738; PINDER 1,337,559 / 1,793,175. Exclusions 230 + 61 /
  315 + 70 records (all of the recorded kinds).
- The 100% joint arms do not fit on the GPU (about 170 GB of packed features), so `PackedSiteSource` streams them from
  Lustre: loader wait about 25% (flagged, criterion 5%), about 17 min per arm. Staging to node-local `/tmp` (334 GB
  tmpfs) would remove the wait if 100% runs become routine.

## Neighbour-slot encodings: 20 amino acids and combined (2026-10-10)

Native pKAI fills each of its 250 neighbour slots (nearest N/O/S environment atoms within 15 A, by distance) with a
16-class functional-atom one-hot holding 1/d^2, plus an 8-class site one-hot (4,008 inputs). Two alternatives
(`pkai_scratch.feature_matrix(protein, encoding)`, `pkai_backbone_pinder_eval._features(..., encoding)`, exp 48
`PKAI_ENCODING`; the defaults are unchanged):
- `aa20`: a 20-amino-acid one-hot of the neighbour atom's residue instead of the atom class (5,008 inputs). Terminal
  atoms take their residue's amino acid; common modified residues map to their parent (MSE->MET etc.); anything else
  would be a recorded exclusion (none occurred).
- `atom16aa20`: both, the 16 atom classes then the 20 residue types in each slot (9,008 inputs), in the native slot
  order. Its atom-class half is bit-identical to `atom16` (checked on training records and the whole validation set).

The first layer widens to the input size (training from scratch). The pKPDB validation rows were re-encoded from the
5k-pilot `input.pdb` files (`build-validation <encoding>`, packages `pretraining/pkpdb-val-pkai-<encoding>-v1`); the
same builder in `atom16` reproduces `pkpdb-val-pkai-v1` exactly. The data are otherwise identical (same records, sites
and exclusions). Batch 4,096, LR 8e-6, cap 400.

| Arm | Fraction | atom16 | aa20 | atom16aa20 |
|---|---|---:|---:|---:|
| Backbone, pKPDB-only: pKPDB val MSE | 10% / 50% | 1.075 / 1.104 | 1.017 / 0.999 | 0.996 / 0.968 |
| Backbone, joint: pKPDB val MSE | 10% / 50% | 1.077 / 1.088 | 0.999 / 0.986 | 0.977 / 0.955 |
| Backbone, joint: PINDER state MSE | 10% / 50% | 0.847 / 0.840 | 0.736 / 0.678 | 0.734 / 0.663 |
| Backbone, joint: PINDER paired MSE | 10% / 50% | 0.205 / 0.205 | 0.192 / 0.183 | 0.193 / 0.183 |
| All-atom, pKPDB-only: pKPDB val MSE | 10% / 50% | 0.405 / 0.383 | 0.592 / 0.543 | 0.418 / 0.389 |
| All-atom, joint: pKPDB val MSE | 10% / 50% | 0.357 / 0.314 | 0.543 / 0.509 | 0.373 / 0.311 |
| All-atom, joint: PINDER state MSE | 10% / 50% | 0.077 / 0.055 | 0.250 / 0.217 | 0.086 / 0.059 |
| All-atom, joint: PINDER paired MSE | 10% / 50% | 0.030 / 0.022 | 0.063 / 0.053 | 0.033 / 0.023 |

- Backbone: residue identity helps (the backbone slots only ever held N or O). Combined is best, about 12% below atom16
  on pKPDB and 21% on PINDER state at 50%, and unlike atom16 it improves from 10% to 50%.
- All-atom: residue identity alone loses the atom type (e.g. backbone O vs carboxylate O) and is much worse. Combined
  is close to atom16: behind at 10%, level at 50% on pKPDB (0.311 vs 0.314) and slightly behind on PINDER. The residue
  types add little once atom classes are present.
- Joint beats pKPDB-only under every encoding.
- The combined 50% joint arms stream their features (about 190 GB; loader wait about 55%, 15-21 min); peak GPU
  allocation 32 GB.

Isambard file limit: per-record feature files are archived into `features/<dataset>.tar` (+ `.tar.json`) after packing
(`$S/_submission/jaxka_pkai_archive_features.sbatch`; about 338k files so far). Nested fractions can share one
`features/` (the combined 10% run links to the 50% one), since `pack` now selects only the run's registered records.

## Feature store (2026-10-10)

The per-record `.npz` files, their tar archives and every run's dense `packed/*.npy` arrays are replaced by one
compressed store per encoding, `training/pkai-features-v2/<encoding>/{pkpdb,pinder}/store/` (17 files per encoding),
covering the 100% pool plus the PINDER validation cohort and shared by every fraction (`pkatrain.pkai_joint_scale`:
`store-ids`, `import`, `prepare`, `pack-store`; Isambard job `jaxka_pkai_store.sbatch`).

- Each site is stored in compact slot form (`pkai_scratch.compact`): the 250 slot values (1/d²), each slot's atom
  class and/or residue type, and the site class; 1.25–1.5 kB per site against 16–36 kB dense. `compact()` checks that
  `expand()` rebuilds the dense row exactly; training expands batches on the GPU (`expand_torch`).
- One zstd frame per structure, written in shards by 72 prepare tasks (or imported from an earlier run's archive),
  concatenated in the old file-name row order and verified record by record.

| Encoding | pKPDB records / sites | PINDER records / sites | Store | Dense equivalent |
|---|---:|---:|---:|---:|
| atom16 | 62,559 / 1,724,738 | 35,386 / 1,804,630 | 2.8 GB | 171 GB |
| aa20 | same | same | 3.0 GB | 214 GB |
| atom16aa20 | same | same | 3.3 GB | 384 GB |

Exclusions are unchanged (315 pKPDB, 70 PINDER). For all eight earlier runs (atom16 10/50/75/100%, aa20 and
atom16aa20 10/50%) `compare-packed` found the store's selection, expanded, bit-identical to the run's dense arrays
(`packed-comparison.json`); the dense arrays (711 GB) and per-record archives were then removed (the JSON receipts stay).
Runs now load their compact rows resident on the GPU: a 2-epoch check of 50% atom16aa20 full/joint reproduced the
original run's epoch 1–2 metrics exactly at 2.8 s per epoch (was ~120 s, loader wait 57–95%; now 0.02%).

## 100%, all three encodings (2026-10-10)

Batch 4096, LR 8e-6, cap 400, early stopping (delta 0.001, patience 8); aa20 and atom16aa20 trained from the feature
store (jobs 7230716–7230724, 1.5–9 min each). Best epoch's pKPDB val MSE / PINDER state / paired (epochs run):

| Arm | atom16 | aa20 | atom16aa20 |
|---|---:|---:|---:|
| BB pkpdb | 1.075 (43) | 0.984 (43) | **0.948** (53) |
| BB joint | 1.066 / 0.825 / 0.203 (70) | 0.974 / 0.669 / 0.181 (57) | **0.941 / 0.650 / 0.179** (57) |
| AA pkpdb | 0.381 (43) | 0.539 (35) | **0.380** (43) |
| AA joint | 0.305 / **0.048 / 0.020** (81) | 0.500 / 0.205 / 0.050 (74) | **0.296** / 0.050 / 0.020 (81) |

At 100% atom16aa20 is best or level everywhere: AA joint pKPDB 0.296 (atom16 0.305, released pKAI 0.324); BB joint
0.941 vs 1.066. No arm reached the cap: the joint AA arms stopped at epoch 81 with the best epoch the last one (gains
below the 0.001 stopping delta), so they were still improving slowly.

## Learning-rate sweep, 100% atom16aa20 full/joint (2026-10-10)

The paper (Reis et al., pKAI) trained with Adam at batch 256, LR 1e-6, weight decay 1e-4, 16-bit, early stopping
(Δ 1e-3, 5 steps). Our sqrt rule anchors 1e-6 at batch 64, not 256, so 8e-6 at batch 4096 is already 2× the paper's
setting under the same rule. Explicit rates (train arg 6, runs `-lr<rate>`), batch 4096, same stopping:

| LR | Epochs (best) | pKPDB val | PINDER state | paired | train MSE |
|---|---:|---:|---:|---:|---:|
| 8e-6 | 81 (81) | 0.296 | 0.050 | 0.020 | 0.122 |
| 3.2e-5 | 42 (42) | 0.286 | 0.046 | 0.019 | 0.107 |
| 1.28e-4 | 26 (22) | **0.285** | **0.044** | **0.017** | 0.104 |

Higher rates are better on every metric and converge faster; 1.28e-4 is the first to stop on a plateau (best epoch 22).

## Cosine schedule and the 12 arms at 100% (2026-10-10)

Rate/schedule on 100% atom16aa20 full/joint (selection MSE / pKPDB val / state / paired, epochs (best)):
5.12e-4 constant 0.118 / 0.288 / 0.046 / 0.019, 21 (17); 1.28e-4 cosine over 40 epochs 0.114 / 0.282 / 0.044 / 0.018,
32 (24); **5.12e-4 cosine over 40 epochs 0.113 / 0.279 / 0.042 / 0.017, 40 (34)** (train args `4096 40 5.12e-4 cosine`).

All arms at 5.12e-4 cosine/40 (pKPDB val / PINDER state / paired; epochs (best)); 8e-6 constant results above for comparison:

| Arm | atom16 | aa20 | atom16aa20 |
|---|---:|---:|---:|
| BB pkpdb | 1.045 / 0.865 / 0.210, 12 (4) | 0.972 / 0.739 / 0.190, 10 (2) | **0.931** / 0.707 / 0.194, 13 (5) |
| BB joint | 1.043 / 0.815 / 0.204, 24 (16) | 0.966 / 0.667 / 0.180, 11 (3) | **0.921 / 0.653 / 0.182**, 12 (4) |
| AA pkpdb | 0.319 / 0.109 / 0.027, 37 (35) | 0.500 / 0.269 / 0.067, 40 (33) | **0.311** / 0.116 / 0.033, 40 (35) |
| AA joint | 0.287 / **0.034 / 0.016**, 40 (34) | 0.498 / 0.202 / 0.045, 13 (9) | **0.279** / 0.042 / 0.017, 40 (34) |

Every arm improves over 8e-6 constant; AA pKPDB-only gains most (0.381 → 0.311–0.319). The backbone arms peak within
2–5 epochs (before the rate has decayed) and then overfit, so they would likely gain from a lower rate. AA joint:
atom16aa20 is best on pKPDB (0.279), atom16 on PINDER (state 0.034).

## 20 Å cutoff with 540 slots (2026-10-10)

At 15 Å the 250 slots almost never fill (stored features: median ~100 all-atom / ~85 backbone; <0.4% of sites at 250).
Recounted on a pKPDB sample (315 structures, ~33k sites), candidates per site rise from a median of 118 (p99 256) at
15 Å to 240 (p99 538) at 20 Å for all-atom, and 95 (182) to 184 (376) for backbone, so 20 Å with 250 slots would
truncate 46% of all-atom sites. 540 slots hold the 20 Å p99 (inputs 8,648 / 10,808 / 19,448; 7.3M / 9.0M / 15.9M
parameters). `PKAI_CUTOFF=20 PKAI_SLOTS=540` (stores, runs and validation package tagged `-r20s540`); 100%, 5.12e-4
cosine/40, the same arms. pKPDB val / PINDER state / paired, with the 15 Å / 250 result in brackets:

| Arm | atom16 | aa20 | atom16aa20 |
|---|---:|---:|---:|
| BB pkpdb | 1.044 (1.045) | 0.964 (0.972) | 0.923 (0.931) |
| BB joint | 1.042 / 0.806 / 0.204 (1.043 / 0.815) | 0.956 / 0.676 / 0.183 (0.966 / 0.667) | **0.911 / 0.653 / 0.180** (0.921 / 0.653) |
| AA pkpdb | 0.321 (0.319) | 0.535 (0.500) | 0.318 (0.311) |
| AA joint | 0.291 / 0.036 / 0.017 (0.287 / 0.034) | 0.497 / 0.202 / 0.050 (0.498 / 0.202) | 0.281 / 0.043 / 0.019 (**0.279** / 0.042) |

The longer range gives backbone a small gain (~0.01 pKPDB) and all-atom nothing or a small loss: with all atoms, the
15 Å environment already carries the information, and the extra 2× input width adds parameters without new signal.
