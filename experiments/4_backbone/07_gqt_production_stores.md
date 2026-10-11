# Production GQT graph stores (2026-10-10)

The graphs for the pool-v3 datasets are built on Isambard-AI (user decision: faster than transferring the 206 GB of
coulson pKPDB `graph.npz` files). Module: `pkatrain.production_graphs`; runner:
`$S/_submission/jaxka_gqt_production.sbatch {ids,build,pack,compare} {pkpdb,pinder}` (one GH200, 72 single-threaded
processes per job, sources read from the squashfs images through `scripts/sqfs_run.sh`, JAX on CPU).

## Same rules as the existing builds

No graph rule changes. Each builder repeats an existing construction:

- **pKPDB**: the residue graph of `pkabench.pkpdb_mask_all.clean`, i.e. the full build's `graph.npz`:
  - the same conformer resolution, residue selection, node features and 20 A geometry;
  - selected chains are label IDs from `defects.json` sequences, and sites map by author chain, resnum and icode;
  - queries are every mapped site in `sites.json` order, with labels;
  - added per site: `train_mask`, `eval_mask`, `rsa`, `w_burial`, `chain_distance_A` (NaN for monomers), plus the
    site graph;
  - every structure is checked against its receipt (n, k, q) and its recorded conformer selection.
- **PINDER**: `gqt_paired_pinder._prepare_one`:
  - `load_topology` with the gap cap, 20 A geometry, paired rows, partner-removed branch masks and site graph;
  - every pool-v3 complex uses the training masks, and the 400 validation complexes use the evaluation masks;
  - complexes without a graph-mapped paired interface site are recorded and skipped, as in `prepare`.
- **Site tokens**: these exist only for candidate sites. A pKPDB terminal label on a residue that is not the chain's
  first or last sequence position has no token. The build's terminal rule never trains or evaluates such a site, so
  it keeps `query_site = -1`, and the builder asserts it is masked.

The full-pool builds of both datasets share the 20 A graph and the 20 A RBF basis. The 25 A basis exists only in the
5k-pilot replay store (`gqt-site-tokens-v1`, regraphed for the neighbour-cutoff screen), so only a warm start from the
old pKPDB checkpoint would carry it.

## Check against the coulson graphs (`compare`)

60 randomly chosen pKPDB pool structures and 60 PINDER factorial complexes (50 training complexes that are in the
pool, plus 10 validation) were copied from coulson and rebuilt on Isambard:

| Dataset | Result |
|---|---|
| pKPDB vs `pkpdb-full-v1/entries/*/graph.npz` | 60/60 identical: nodes, neighbours, masks, switch, queries and labels are exact. Edge features match within 1e-6 (max 2.4e-7, only in the 3 direction components; numpy 2.5.3 on aarch64 vs 2.2.6 on x86 round the `einsum` differently in the last float32 bit). |
| PINDER vs `ogqt-pinder-factorial-v1/graphs` | 60/60 have identical geometry (residue graph, site graph, branch masks; edges exact). 23 are fully identical. In the other 37, only the per-site fields differ, and in every case the complex's `sites.json`/`labels.json` changed after the factorial registration (recorded source hashes), so the rebuild uses the current labels. |

Peak memory is about 1 GB per process; one structure takes about 0.15 s.

## Store layout

`<runtime>/training/gqt-production-v1/<dataset>/`:
- `ids.json`: pool order, then validation;
- `shards/<t>-of-<T>/`: one raw `.bin` per field, written one structure at a time;
- `store-v1/`: one flat `.npy` per field, `index.npz` (dims n, k, q, s, sk and offsets), `records.json`,
  `metadata.json` and `verification.json`;
- `compare.json`.

`ProductionStore(path).raw(id)` returns read-only views with the recorded shapes. `pack` re-reads every record and
checks its arrays' sha256 against the build record before the store is installed.

## Build results (Isambard, 2026-10-10)

| Store | Structures | Sites | Skipped / failed | Build (72 processes, one job) | Pack and verify | Size |
|---|---:|---:|---:|---:|---:|---:|
| `pinder/store-v1` | 35,456 | 1,846,388 | 0 / 0 | 65 s | 13 min | 299 GB |
| `pkpdb/store-v1` | 62,874 | 7,243,547 | 0 / 0 | 129 s | 31 min | 526 GB |

Both `verification.json` files pass. The build shards were removed after packing. Each store is about 30 files.

## Compressed stores (store-v2, zstd records)

The first packed layout (`store-v1`: one flat `.npy` per field, uncompressed) was 825 GB. A squashfs image of it
compressed well (PINDER 299 -> 123 GB) but was unusable for random access: about 10 structures/s at any thread count,
because each structure is read from 28 field files at random offsets and `squashfuse_ll` 0.5.2 spends about 0.1 s per
seek in a 1.8M-block file, without thread scaling. Sequential reads were fine (1.2-1.5 GB/s). (An earlier "4,900
structures/s" image figure was wrong: that benchmark did not read the data.)

Three layouts were compared on 4,000 random PINDER structures (every field materialised; fresh structures per run):

| Layout | Size (raw 32.9 GB) | 8 threads | 32 threads | 72 threads |
|---|---:|---:|---:|---:|
| store-v1, uncompressed per-field mmap on Lustre | 32.9 GB | 138/s | 343/s | 555/s |
| one uncompressed `.npz` per structure in a zstd squashfs image | 13.5 GB (2.4x) | 172/s | 351/s | 322/s |
| **one zstd frame per structure + offset index (store-v2)** | **10.4 GB (3.2x)** | **416/s** | **395/s** | **365/s** |

Decision (user: keep the stores compressed): `store-v2`.
- `records.bin` holds one zstd frame (level 3) per structure, containing every field's raw bytes in FIELDS order.
- `index.npz` holds ids, dims and byte offsets, so each structure is individually addressable.
- `ProductionStore.raw(id)` does one `pread` plus a decompress, is safe from many threads, and needs no FUSE.
- Build tasks write compressed records directly (`shards/<t>-of-<T>/records.bin`) and `pack` concatenates them, so
  no uncompressed intermediate is ever written.
- `pack` and `compress` (the v1 -> v2 conversion) decompress every record and check it against the build's
  `arrays_sha256` before installing the store.
- New dependency: `zstandard>=0.23` (train extra; 0.25.0 in the Isambard venv).

Conversion of the full stores (job 7213750, 72 threads; store-v1 removed after verification):

| Store | Structures | Raw | Compressed | Ratio | Time |
|---|---:|---:|---:|---:|---:|
| `pinder/store-v2` | 35,456 | 299.2 GB | 94.8 GB | 3.16x | 226 s |
| `pkpdb/store-v2` | 62,874 | 525.8 GB | 179.4 GB | 2.93x | 460 s |

Random reads from the full PINDER store-v2 (all fields decoded): 797/s at 8 threads, 856/s at 32 and 1,215/s at 72
(6.5-9.8 GB/s of arrays), against 112/s and 285/s for store-v1 on the same node type. Squashfs remains in use for the
source datasets only.

## Production loader (`pkatrain.production_loading`)

- **Manifests** (`<dataset>/manifest-v1.json`, built under `sqfs_run.sh` because they read pool-v3.tsv):
  - bucket policy `PRODUCTION`: bounds 128-1536, batch sizes 24/12/8/6/4/4/3/2/2 from the 3,072-residue budget,
    still provisional until the GH200 memory check;
  - capacities (n, k, q, s, sk, rounded to 32) taken over all records, so every pool fraction compiles the same shapes;
  - per record: split, dims and the pool-v3 `min_fraction`. Training subsets are `select(manifest, "train",
    fraction)`. PINDER records also carry per-complex `w_burial`/`w_interface` means, and `normalization(manifest,
    fraction)` is the factorial's definition (mean complex-level weight over that fraction's training complexes).
- **`PinderSource`**: `gqt_paired_pinder._load_one` per record (AB and partner-free branches).
- **`PkpdbSource`**: the `SiteBatchLoader` format, with `eligible` = `train_mask` (or `eval_mask`). Queries without a site token
  (`query_site` -1, never eligible; counted per record as `untokenised_sites`) point at site 0.
- Both pad to the bucket's fixed batch size, with unsupervised copies and `valid` False.
- Unit tests: synthetic store, manifest, subset selection, normalisation, padding and masks (4 pass).

Check on the real stores (job 7214034; 20 batches compared with the store records, then 300 batches through
`Prefetcher`, prefetch 4):

| Dataset | Training structures (100%) | Batches / epoch | Content | 8 workers | 32 workers |
|---|---:|---:|---|---:|---:|
| PINDER | 35,056 (+400 validation) | 6,619 | pass | 104 batches/s (544 structures/s) | 108 (564) |
| pKPDB | 62,874 | 10,929 | pass | 94 batches/s (552 structures/s) | 164 (964) |

Training sizes per bucket (100%): PINDER 128: 2,499; 256: 6,993; 384: 6,846; 512: 5,350; 640: 4,314; 768: 3,383;
1024: 3,475; 1280: 1,489; 1536: 707. pKPDB 128: 3,546; 256: 11,232; 384: 15,625; 512: 10,668; 640: 7,905;
768: 5,981; 1024: 7,915; 1280: 2.

Open: the pKPDB store has no validation records (the pool is training-only). The pKPDB validation set (the 142
benchmark structures) is not on Isambard and needs a store built with the same rules.

## Benchmark validation store (`benchmark-val`)

The 142-complex PypKa validation set (`training/shared-v4-float32` `val`; the benchmark validation split both pools
were screened against) is a third store, built by `graph_data.prepare`'s rules:
- AB state, `load_topology` with the gap cap, 20 A geometry;
- `supervision_eligible` finite PypKa midpoints as queries;
- the site graph;
- node column 22 (disulfide) zeroed, as `strict_backbone` does at load.

Inputs (711 files, 66 MB; record JSON, AB structures with symlinks dereferenced, PypKa exports and raw results) were
copied to the same relative runtime paths and checked by sha256. The records' absolute coulson paths are mapped onto
the local runtime.

Result:
- 142/142 are identical to `pretraining/graph-pilot-v1` (the original 20 A graphs), edges included.
- The store is 337 MB (zstd records).
- `PkpdbSource(manifest, mask="eval_mask")` serves it; the loader content check passes, at about 1,100 structures/s.

## GH200 batch-size calibration (`pkatrain.production_calibration`)

One `JointEngine` gradient step (current 67,725-parameter backbone oGQT, width 44 / ff 88) was run per dataset, bucket
and batch size, each in a fresh process, on one GH200 (76.5 GB usable). Memory never binds: the largest peak is 17.8 GB
(PINDER, 1,536-residue bucket, batch 64). Throughput (structures/s; median of 3 steps after compilation):

| Bucket | PINDER 4 | 8 | 16 | 32 | 64 | pKPDB 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 164 | 327 | 653 | 1,143 | 1,744 | 165 | 327 | 653 | 1,296 | 2,246 |
| 256 | 145 | 224 | 321 | 405 | 451 | 165 | 304 | 476 | 691 | 863 |
| 384 | 127 | 180 | 237 | 278 | 306 | 159 | 261 | 386 | 498 | 585 |
| 512 | 105 | 144 | 174 | 192 | 203 | 141 | 229 | 313 | 404 | 450 |
| 640 | 111 | 144 | 178 | 192 | 199 | 136 | 211 | 275 | 337 | 378 |
| 768 | 91 | 120 | 136 | 142 | 150 | 123 | 180 | 227 | 262 | 285 |
| 1024 | 76 | 95 | 107 | 110 | 114 | 103 | 142 | 181 | 201 | 216 |
| 1280 | 71 | 88 | 94 | 105 | 108 | 125 | 186 | 244 | 282 | 315 |
| 1536 | 70 | 81 | 88 | 91 | 95 | | | | | |

The residue-budget policy gives 2-3 structures per batch for the large buckets, where the GPU is least efficient (about
55-70 structures/s). Throughput saturates from about 16 structures per batch in the large buckets, while small buckets
keep scaling. Experiment 45 estimated the joint gradient-noise scale at about 18 structures per task batch, and found
the 1x-2x batch scales (8-16) equivalent on validation. Estimated PINDER epoch time from these rates:
- residue budget (24/12/8/6/4/4/3/2/2): about 275 s;
- constant 16 per batch: about 190 s;
- constant 32 per batch: about 172 s.
Results: `<runtime>/training/gqt-production-v1/calibration-sweep.json`.

## Production joint training (`pkatrain.production_train`)

User decisions (2026-10-10): constant batch of 16 in every bucket; scratch initialisation; the experiment 43 joint
objective with the 67,725-parameter backbone oGQT; first run on the 10% pool.
- **Engine:** `gqt_multitask_replay.JointEngine`, unchanged. Each update sums one PINDER gradient (AB/free state-shift
  MSE + binding-shift MSE) and one pKPDB state-shift gradient (train_mask sites), then one AdamW step (weight decay
  1e-4, clip 1). Equal coefficients, no site weights.
- **Sampling:** one pass over the fraction's PINDER complexes per epoch, with one pKPDB batch per PINDER batch from a
  continuing shuffled pKPDB stream. Plans are deterministic in (seed, epoch), so runs resume from their last checkpoint.
- **Schedule:** `gqt_crop_radius.learning_rate` (hold 1e-3 through epoch 10, cosine to 1e-5 by epoch 20), scaled by
  sqrt(16/8) per experiment 45's rule; patience 8.
- **Validation every epoch:** PINDER 400 (state MAE, interface paired MAE) and benchmark-val 142 (group-macro MAE).
- **Selection:** lowest PINDER state MAE + interface paired MAE.
- **Outputs:** `training/gqt-production-v1/runs/<run>/` (protocol, history, checkpoints, selection, predictions of the
  selected epoch).
- **Runner:** `$S/_submission/jaxka_gqt_train.sbatch RUN FRACTION` (one GH200, 32 CPUs).

Smoke test (2 epochs x 5 batches, 10% manifests): finite losses; validation improved (selection 1.217 -> 1.172);
loader wait fraction about 0; validation takes 1.5 s once compiled. The 10% pilot is `runs/pilot-10pct` (job 7215696).

### 10% pilot (`runs/pilot-10pct`, job 7215696, one GH200)

3,422 PINDER and 5,878 pKPDB training structures; 214 joint updates per epoch.
- **Time:** about 27.5 s of training per epoch (102 s for epoch 1, including compilation) plus 1.6 s of validation;
  11.5 min in total for 20 epochs.
- **Loader:** wait fraction 1-4%.
- **Selection:** epoch 17 (patience never triggered).

| Epoch | Train loss | pKPDB train loss | PINDER state MAE | Interface paired MAE | Benchmark group-macro MAE | Selection |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.731 | 0.825 | 0.569 | 0.348 | 0.657 | 0.917 |
| 5 | 1.081 | 0.545 | 0.487 | 0.310 | 0.582 | 0.797 |
| 10 | 0.932 | 0.471 | 0.458 | 0.295 | 0.568 | 0.753 |
| 13 | 0.850 | 0.425 | 0.473 | 0.290 | 0.570 | 0.762 |
| **17** | 0.750 | 0.377 | **0.453** | **0.295** | **0.559** | **0.747** |
| 20 | 0.708 | 0.353 | 0.454 | 0.298 | 0.560 | 0.752 |

Selected epoch 17:
- **PINDER validation:** 11,758 sites; state MAE 0.453; paired MAE 0.147; interface paired MAE 0.295 (5,271 sites).
- **Benchmark validation:** 7,986 sites, 142 complexes, 76 component groups; group-macro MAE 0.559, RMSE 0.789,
  Spearman 0.753, sign accuracy 0.937.

The benchmark column comes from `rescore` (`runs/pilot-10pct/rescore.json`). The run's own history scored it with
`component_id` missing from the benchmark manifest, which collapsed the group-macro MAE into one group. Manifests now
carry `component_id`. Selection uses PINDER only and is unchanged.

For orientation only, since the cohorts and label revisions differ: experiment 45's pretrained joint arms (5k PINDER
cohort, pKPDB-pretrained parent) reached PINDER state MAE 0.464-0.473, interface paired MAE 0.295-0.300 and benchmark
pKPDB MAE 0.548-0.565. The scratch 10% production pilot is in the same range.

### Batch-size sweep on the 10% pool (2026-10-10)

`production_train train RUN --fraction 0.1 --batch B` sets a constant B structures per batch in every bucket and
scales the learning rate by sqrt(B / 8). Validation always runs at 16 per batch. All runs are `runs/pilot-10pct-b{B}`
(B = 16 is `runs/pilot-10pct`), use seed 17 and the same 20-epoch schedule, each on one GH200.

**Large batches use exact gradient accumulation.**
- A single PINDER step at 256 in the 1,536-residue bucket needs a 45 GiB allocation and fails.
- Chunks of 128 also failed at batch 256 (a 21 GiB allocation with prefetched chunks resident on the device), and
  so did chunks of 64 while prefetched chunks were still put on the device.
- So batches above 64 are split on the host into chunks within a residue budget of 64 x 1,536 = 98,304 slots (the
  largest divisor of the batch that fits), and chunks move to the device only when their step is dispatched.
  At batch 256 this means chunks of 256 up to the 384-residue bucket, 128 up to 768, and 64 above.
- Both objectives are means over the valid structures, so weighting each chunk's value and gradient by its share of
  valid structures gives the full-batch step. Checked on a GH200 for both datasets in two buckets: losses agree to 6
  digits, and the gradient differs from the full-batch gradient by at most 1.8e-5 of its norm (float32 summation
  order; a repeated full-batch step differs by about 1e-7).
- Batch 128 ran unchunked (it fits in every bucket); batch 256 ran with chunks of 64 in every bucket.

| Batch | Updates/epoch | Selected epoch | PINDER state MAE | Interface paired MAE | Selection | Benchmark MAE | Spearman | Final train loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 432 | 17 | 0.457 | 0.301 | 0.759 | 0.564 | 0.765 | 0.692 |
| **16** | 219 | 17 | **0.453** | 0.295 | **0.747** | **0.559** | 0.772 | 0.708 |
| 32 | 112 | 17 | 0.457 | **0.294** | 0.751 | 0.564 | 0.745 | 0.784 |
| 64 | 59 | 17 | 0.462 | 0.294 | 0.756 | 0.569 | 0.745 | 0.828 |
| 128 | 31 | 20 | 0.468 | 0.301 | 0.769 | 0.572 | 0.762 | 0.883 |
| 256 | 18 | 18 | 0.476 | 0.307 | 0.783 | 0.578 | 0.758 | 0.936 |

Benchmark MAE is the group-macro MAE over 76 component groups (benchmark-val, 142 complexes).

- Batches 8-32 lie within 0.012 of each other on selection; 16 is marginally best on every validation metric except
  interface paired MAE. Selection worsens steadily from 64 upward.
- Batch 128 selected its last epoch and has a higher training loss, so with 31 updates per epoch it is short of steps,
  not converged. Batch 256 (peak learning rate 5.7e-3) diverged in epoch 1 (train loss 22, PINDER state MAE 2.9) and
  recovered by epoch 5.
- These are single seeds on a 10% pool; differences below about 0.01 are within epoch-to-epoch noise.

Validation MSEs on the training losses' scales (`production_train.squared_errors`, added to `validate`; every
checkpoint rescored with `rescore`, `runs/*/rescore.json`; per-epoch curves for all six runs in
`<runtime>/training/gqt-production-v1/batch-sweep-curves.csv`). Selected epochs:

| Batch | pKPDB val MSE (site) | PINDER state MSE | PINDER paired MSE | Interface paired MSE | Train pKPDB MSE | Train state MSE | Train paired MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.838 | 0.515 | 0.148 | 0.325 | 0.383 | 0.317 | 0.048 |
| **16** | **0.812** | **0.497** | **0.144** | **0.315** | 0.377 | 0.324 | 0.049 |
| 32 | 0.834 | 0.514 | 0.146 | 0.322 | 0.415 | 0.348 | 0.051 |
| 64 | 0.850 | 0.530 | 0.149 | 0.330 | 0.437 | 0.373 | 0.055 |
| 128 | 0.836 | 0.528 | 0.146 | 0.322 | 0.442 | 0.383 | 0.059 |
| 256 | 0.862 | 0.572 | 0.165 | 0.364 | 0.488 | 0.422 | 0.063 |

- pKPDB val is the benchmark validation set (PypKa labels, scored as pKPDB shifts); the production pKPDB pool has no
  held-out split. Its MSE is about twice the pKPDB training loss (labels from a different method), and PINDER state
  validation MSE is about 1.5x its training loss.
- Validation MSEs are site-level means; the training losses average per structure first.
- Batch 16 is lowest on every validation MSE.

Time per run (one GH200, 32 CPUs, 8 loader workers per source, zero-fill padding):

| Batch | Job wall time | Epoch 1 | Later epochs (median) | Training total | Loader wait |
|---:|---:|---:|---:|---:|---:|
| 8 | 11:44 | 101 s | 27.7 s | 627 s | 2.6% |
| 16 | 11:29 | 102 s | 27.5 s | 624 s | 3.2% |
| 32 | 11:06 | 97 s | 26.8 s | 606 s | 2.9% |
| 64 | 11:40 | 102 s | 28.0 s | 636 s | 2.3% |
| 128 | 12:25 | 116 s | 29.9 s | 683 s | 1.5% |
| 256 | 13:29 | 75 s | 35.2 s | 749 s | 1.4% |

Epoch time does not fall with batch size: the GPU is about 100% busy during training (nvidia-smi on the b256 job),
and the per-structure GPU cost was the same from batch 8 to 64. Validation takes about 46 s per run (about 15 s in
epoch 1, then 1.6 s per epoch).

## Training-step profile and padding (2026-10-10)

`scripts/profile_train_step.py` replays 60 real production steps (epoch-1 plan, 10% pool, batch 16, steps 40-99)
after compiling every bucket shape, and captures only that window under nsys (`--cuda-graph-trace=node`).
`scripts/profile_report.py` maps each kernel to its JAX primitive through the optimized HLO. Runner:
`$S/_submission/jaxka_gqt_profile.sbatch NAME [BATCH]`; outputs in `<runtime>/training/gqt-production-v1/profile/`.

**Before (zero-fill padding, `profile/b16`):** 0.135 s per step unprofiled; 6.49 s of kernel time in the window.

| Kernel | Share of GPU time | Calls | Mean |
|---|---:|---:|---:|
| Triton indexed attention, backward | 76.5% | 360 | 13.8 ms |
| XLA scatter-add (backward of the K/V gathers in `model.attend`, `model.py:53-54`, used by the site-token query attention) | 15.9% | 360 | 0.5-4.0 ms |
| Triton indexed attention, forward | 2.7% | 360 | 0.49 ms |
| All matmuls, reductions and elementwise ops | about 4% | | |

The backward kernel took 28x its forward. Both backward scatters (Triton `tl.atomic_add` of dK/dV; XLA scatter-add)
accumulate into the gathered rows, and the padding sent most gather indices to one row.

**Padding census (`scripts/padding_census.py`, 64 training structures per bucket, 10% pool, both PINDER branches;
`profile/padding-census.json`).** Share of each padded tensor that is real, per graph:

| PINDER bucket | Capacity n / k / s / sk | Nodes | Edge slots | Mean real neighbours | Sites | Site-edge slots | Query gathers | Residue gather indices on row 0 | Site gather indices on row 0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 / 128 / 96 / 64 | 77% | 29% | 48 | 36% | 9% | 13% | 67% | 89% |
| 256 | 256 / 224 / 160 / 96 | 77% | 22% | 63 | 43% | 9% | 11% | 77% | 90% |
| 384 | 384 / 256 / 192 / 96 | 82% | 23% | 73 | 54% | 12% | 14% | 75% | 87% |
| 512 | 512 / 288 / 256 / 96 | 88% | 25% | 82 | 52% | 12% | 14% | 74% | 87% |
| 640 | 640 / 256 / 320 / 96 | 91% | 31% | 88 | 57% | 15% | 18% | 67% | 84% |
| 768 | 768 / 288 / 320 / 96 | 92% | 31% | 98 | 66% | 19% | 21% | 67% | 81% |
| 1024 | 1024 / 288 / 416 / 96 | 87% | 30% | 98 | 64% | 18% | 20% | 69% | 81% |
| 1280 | 1280 / 256 / 576 / 96 | 88% | 36% | 106 | 60% | 19% | 23% | 62% | 80% |
| 1536 | 1536 / 256 / 608 / 96 | 91% | 39% | 110 | 69% | 22% | 28% | 60% | 77% |

- pKPDB is similar: nodes 78-91% real, edge slots 30-42%, sites 38-60%, site-edge slots 9-20%, and 58-71% of residue
  gather indices on row 0.
- Real edges that end on row 0 are only 0.02-0.2% of slots, so nearly every row-0 index is padding.
- Batch fill is 97.7% (PINDER) and 99.6% (pKPDB) of batch slots.
- Most edge padding is masked slots on real nodes (52-62% of slots): k is set by the most-connected residue of the
  bucket, while the mean residue has about 40% of that. Padded nodes add 8-23%.
- The Triton kernel also rounds the neighbour block to a power of two, so k = 288 runs as 512 slots.

**Fix: spread the padding (`production_loading.spread_padding`, on by default in both sources).**
- Masked neighbour slots, in the residue and site graphs, point at other rows instead of row 0. In mode "rotate" (the
  default), masked slot j of row i points at row (i + j + 1) mod rows; mode "own" points it at row i.
- Padded sites point at residue (index mod real residues) instead of residue 0.
- Masked slots carry exactly zero attention weight and padded tokens zero upstream gradient, so the loss is unchanged
  and gradients change only in float summation order. No model or kernel code changes (the registered experiments'
  code hashes are untouched).

`scripts/padding_spread_bench.py` (same batch-16 batches of real training ids, gradient step median of 10;
`profile/padding-spread-bench-v2.json`):

| Bucket | PINDER zero fill | PINDER spread | Speed-up | pKPDB zero fill | pKPDB spread | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6.1 ms | 3.4 ms | 1.8x | 3.7 ms | 2.3 ms | 1.6x |
| 256 | 31.6 ms | 13.0 ms | 2.4x | 15.7 ms | 7.2 ms | 2.2x |
| 384 | 51.8 ms | 20.2 ms | 2.6x | 24.8 ms | 11.0 ms | 2.3x |
| 512 | 82.5 ms | 29.8 ms | 2.8x | 33.0 ms | 14.1 ms | 2.3x |
| 640 | 87.5 ms | 33.2 ms | 2.6x | 40.3 ms | 17.4 ms | 2.3x |
| 768 | 105.9 ms | 43.5 ms | 2.4x | 55.3 ms | 22.6 ms | 2.4x |
| 1024 | 141.4 ms | 57.3 ms | 2.5x | 78.3 ms | 30.7 ms | 2.6x |
| 1280 | 142.8 ms | 65.1 ms | 2.2x | | | |
| 1536 | 175.1 ms | 76.3 ms | 2.3x | | | |

- Losses are bit-identical in every bucket. Gradients differ from the zero fill by 1e-7 to 2e-6 of their norm, the
  same as a repeated zero-fill step (atomics are nondeterministic).
- "own" and "rotate" give the same times within 1%.

**After (spread padding):**

| Window (60 steps) | Seconds per step (unprofiled) | Loader wait | Kernel time | Attention backward mean |
|---|---:|---:|---:|---:|
| Zero fill, 8 workers, prefetch 2 (`profile/b16`) | 0.135 | 6% | 6.49 s | 13.8 ms |
| Spread "own", 8 workers, prefetch 2 (`profile/b16-spread`) | 0.097 | 28% | 2.61 s | 5.5 ms |
| Spread "rotate", 15 workers, prefetch 4 (`profile/b16-rotate-w15`) | 0.073 | 10% | 2.63 s | 5.6 ms |

- End to end the step is 1.85x faster (0.135 -> 0.073 s). With the GPU work cut by 2.5x, the loader became the limit
  at 8 workers; 15 workers per source and prefetch depth 4 are now the training job's defaults
  (`jaxka_gqt_train.sbatch`).
- The XLA scatters fell from up to 4.0 ms to 0.13-0.44 ms.
- The attention backward is still 77% of GPU time and 11x its forward. Every masked slot (60-70% of slots) still
  issues a zero `atomic_add`; only a kernel change (masking the atomics with `edge_mask`) can skip them. Shorter
  neighbour blocks (k = 288 runs as 512) would need a kernel that loops over neighbour blocks. Both mean changing
  `pkanet/triton_attention.py`, which registered experiments hash, so neither is done here.
- The batch-size sweep above ran with zero-fill padding; spread padding changes speed only.

## gqt-production-v2: pool-v4 stores and the 10% pilot (2026-10-10)

Stores for pool-v4 (option B; `05_pinder_dataset.md`) under `training/gqt-production-v2/` (the default
`PKATRAIN_GQT_VERSION`; v1 stores and runs are unchanged). Every structure built: PINDER 28,733 (27,933 training +
800 validation), pKPDB 56,991 (56,191 + 800 validation, split 'val', scored on `train_mask`), benchmark-val 142.
Store-v2 sizes 79.9 GB and 163.2 GB; build + pack about 3 min (PINDER) and 6 min (pKPDB) on one GH200 job each.
Entries bundle-v1 lacked are read from `<runtime>/overlay/`.

Pilot `gqt-production-v2/runs/pilot-10pct` (job 7232756): batch 16, 10% pool (2,783 PINDER, 5,224 pKPDB), spread
padding, 15 loader workers, prefetch 4. 7.5 min in total; about 12.4 s of training and 4.1 s of validation per epoch
after epoch 1 (v1 pilot: 27.5 s on a larger 10% pool with zero-fill padding). Loader wait 4-9%.

| Epoch | Train loss | PINDER state MAE | Interface paired MAE | pKPDB val MAE | Benchmark MAE | Selection |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.886 | 0.596 | 0.315 | 0.651 | 0.647 | 0.911 |
| 5 | 1.155 | 0.540 | 0.295 | 0.621 | 0.623 | 0.835 |
| 10 | 1.025 | 0.534 | 0.265 | 0.600 | 0.621 | 0.799 |
| **17** | 0.848 | **0.473** | **0.264** | **0.557** | **0.573** | **0.737** |
| 20 | 0.798 | 0.474 | 0.270 | 0.560 | 0.577 | 0.744 |

Selected epoch 17:
- **PINDER validation (800 complexes):** 29,906 eval-mask sites, 7,703 at the interface; state MAE 0.473 (MSE 0.542),
  paired MAE 0.077 (MSE 0.067), interface paired MAE 0.264.
- **pKPDB validation (800 structures, pKPDB labels):** 21,666 sites; MAE 0.557, MSE 0.733; structure-macro MAE 0.448.
- **Benchmark (142, PypKa):** group-macro MAE 0.573, RMSE 0.818, Spearman 0.764.

Not comparable with the v1 pilot's numbers: the PINDER validation set, the training pools and the 10% subsets all
changed.

### Interface-weighted paired loss, 10% pool (2026-10-10)

`--paired-weight interface` multiplies the paired (AB - free) squared error by the normalised `w_interface`
(experiment 41's interface-arm formula; `production_train.interface_weighted`, `JointEngine` unchanged). Validation and
selection stay unweighted. Run `gqt-production-v2/runs/pilot-10pct-winterface` (job 7233332), otherwise identical to
`pilot-10pct`; both selected epoch 17.

| Selected epoch 17 | Unweighted | Interface-weighted | Paired bootstrap difference (95% CI, 800 complexes) |
|---|---:|---:|---|
| Selection | 0.7375 | 0.7317 | -0.0058 (-0.0115 to +0.0001) |
| Interface paired MAE | 0.2644 | 0.2605 | -0.0039 (-0.0082 to +0.0006) |
| Interface paired MSE | 0.2525 | 0.2407 | -0.0117 (-0.0236 to -0.0009) |
| Paired MAE, <= 4 A from partner | 0.5349 | 0.5224 | |
| State MAE | 0.4730 | 0.4712 | -0.0019 (-0.0053 to +0.0019) |
| State MSE | 0.5419 | 0.5503 | |
| pKPDB val MAE | 0.5572 | 0.5548 | |
| Benchmark MAE (group-macro) | 0.5728 | 0.5747 | |

- Interface weighting lowers interface paired error, mostly at sites within 4 A of the partner, without hurting state,
  pKPDB or benchmark error. The interface MSE gain is outside the validation-sampling CI; the MAE and selection gains
  are borderline.
- One seed each. The CIs cover only which validation complexes were sampled, not seed-to-seed variation.

### Query norm x paired weighting, two seeds, 10% pool (2026-10-10)

2 x 2 x 2: site-token query attention with the shared query/context LayerNorm (default) or separate affine parameters
(`--query-norm separate`: `query_context_norm`, initialised as query `norm1`; `production_train.VariantEngine`, checked
to reproduce `JointEngine` at initialisation: identical losses, gradients equal to 3e-7) x unweighted or
`w_interface`-weighted paired loss x seeds 17 and 29 (initialisation and batch order). Batch 16; about 6-7 min per run.
Runs `gqt-production-v2/runs/pilot-10pct[-winterface][-qsep...][-s29]`.

| Norm | Paired loss | Seed | Epoch | Selection | State MAE | Interface paired MAE | Interface paired MSE | pKPDB val MAE | Benchmark MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared | unweighted | 17 | 17 | 0.7375 | 0.4730 | 0.2644 | 0.2525 | 0.5572 | 0.5728 |
| shared | unweighted | 29 | 17 | 0.7355 | 0.4694 | 0.2660 | 0.2558 | 0.5598 | 0.5790 |
| shared | interface | 17 | 17 | 0.7317 | 0.4712 | 0.2605 | 0.2407 | 0.5548 | 0.5747 |
| shared | interface | 29 | 17 | 0.7414 | 0.4736 | 0.2678 | 0.2615 | 0.5623 | 0.5832 |
| separate | unweighted | 17 | 17 | 0.7413 | 0.4733 | 0.2680 | 0.2583 | 0.5559 | 0.5678 |
| separate | unweighted | 29 | 16 | 0.7397 | 0.4731 | 0.2667 | 0.2629 | 0.5622 | 0.5825 |
| separate | interface | 17 | 17 | 0.7343 | 0.4731 | 0.2612 | 0.2465 | 0.5589 | 0.5800 |
| separate | interface | 29 | 15 | 0.7435 | 0.4759 | 0.2677 | 0.2532 | 0.5640 | 0.5872 |

Main effects (mean over seeds and the other factor; 95% complex-bootstrap CI over the 800 PINDER validation complexes):

| Effect | Selection | State MAE | Interface paired MAE | Interface paired MSE |
|---|---|---|---|---|
| separate - shared | +0.0032 (+0.0008, +0.0055) | +0.0020 (+0.0008, +0.0032) | +0.0012 (-0.0007, +0.0030) | +0.0026 (-0.0027, +0.0077) |
| interface - unweighted | -0.0007 (-0.0044, +0.0027) | +0.0012 (-0.0006, +0.0031) | -0.0020 (-0.0047, +0.0009) | -0.0069 (-0.0133, -0.0006) |

- Seeds differ by about 0.004 in mean selection (17: 0.7362, 29: 0.7400), as large as either effect; the bootstrap
  intervals cover validation sampling only, not seed variation.
- The seed-17 interface-weighting gain did not repeat at seed 29 (shared: -0.0058 at seed 17, +0.0059 at seed 29).
  Averaged, weighting lowers interface paired MSE slightly and leaves selection unchanged.
- Separate query/context norms are slightly worse here (mostly state MAE), unlike the earlier three-seed test
  (`training/ogqt-query-norm-v1`, -0.0022 selection), which used the auxiliary objective, a warmup with the norm split
  afterwards, and the old 400-complex validation set.
- Weighting x norm interaction on selection: -0.0017.
- Neither change is adopted from this evidence: the default (shared norm, unweighted paired loss) is kept.

### Experiment 48's loss weighting on GQT, two seeds, 10% pool (2026-10-10)

`--loss-reduction`: `structure` (default; each loss a per-structure site mean, then a mean over structures,
unweighted), `site` (each loss a mean over the batch's supervised sites, unweighted) and `pkai` (experiment 48,
`pkai_joint_scale._weighted_mse`: sum(w err^2) / sum(w) over the batch's sites, `w_burial` on the pKPDB and PINDER
AB/free state losses, `w_interface` on the paired loss; pKPDB sites without a burial weight get weight 0). Batch 16,
seeds 17 and 29, shared norm; runs `gqt-production-v2/runs/pilot-10pct[-site|-pkaiw][-s29]`. Validation unweighted.

| Arm | Seed | Epoch | Selection | State MAE | Interface paired MAE | Interface paired MSE | pKPDB val MAE | pKPDB val MSE | Benchmark MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 17 | 17 | 0.7375 | 0.4730 | 0.2644 | 0.2525 | 0.5572 | 0.7328 | 0.5728 |
| default | 29 | 17 | 0.7355 | 0.4694 | 0.2660 | 0.2558 | 0.5598 | 0.7568 | 0.5790 |
| site | 17 | 19 | 0.7335 | 0.4658 | 0.2677 | 0.2562 | 0.5535 | 0.7130 | 0.5695 |
| site | 29 | 18 | 0.7406 | 0.4728 | 0.2678 | 0.2533 | 0.5583 | 0.7242 | 0.5784 |
| pkai | 17 | 17 | 0.7404 | 0.4746 | 0.2658 | 0.2582 | 0.5575 | 0.7327 | 0.5738 |
| pkai | 29 | 18 | 0.7459 | 0.4768 | 0.2692 | 0.2496 | 0.5669 | 0.7524 | 0.5893 |

Differences of seed means (95% bootstrap over the 800 PINDER and 800 pKPDB validation structures):

| Contrast | Selection | State MAE | Interface paired MAE | pKPDB val MAE | pKPDB val MSE |
|---|---|---|---|---|---|
| pkai - default | +0.0067 (+0.0025, +0.0110) | +0.0044 (+0.0024, +0.0064) | +0.0022 (-0.0011, +0.0055) | +0.0037 (+0.0011, +0.0063) | -0.0022 (-0.0107, +0.0062) |
| site - default | +0.0006 (-0.0034, +0.0042) | -0.0019 (-0.0041, +0.0001) | +0.0025 (-0.0004, +0.0054) | -0.0026 (-0.0053, +0.0003) | -0.0262 (-0.0346, -0.0176) |
| pkai - site | +0.0061 (+0.0021, +0.0104) | +0.0064 (+0.0045, +0.0084) | -0.0003 (-0.0033, +0.0031) | +0.0063 (+0.0036, +0.0088) | +0.0239 (+0.0157, +0.0324) |

- Experiment 48's weighting makes GQT slightly worse on every unweighted validation metric (both seeds), mostly state
  and pKPDB error; the interface-weighted Siamese term does not improve interface paired error.
- Site-level averaging alone matches the default on selection and lowers pKPDB validation MSE by 3.5%. The weights
  undo that.
- Seed spread in selection: 0.002 (default), 0.007 (site), 0.006 (pkai); intervals cover validation sampling only.
- Default kept; `site` is a candidate if pKPDB MSE matters.

### Stochastic site masking by weight percentile, two seeds, 10% pool (2026-10-10)

User proposal: instead of weighting the losses, mask sites at random with the inverse of the weights.
`--site-mask-pmin P` (`production_train.MaskedEngine`):
- **Percentile:** each weight is mapped to its percentile u among the training pool's sites (mid-rank, so ties share a
  value; tables in each run's `site-mask-tables.json`).
- **Keep probability:** P + (1 - P) * u, redrawn every step. `w_burial` masks the state losses (PINDER AB and free
  share a draw; pKPDB its own), `w_interface` masks the paired loss (an independent draw).
- **Mean keep fraction:** (1 + P) / 2, whatever the weight's scale. Raw weights as probabilities would have kept 77%
  for burial and about 21% for the Siamese loss: `w_burial` spans 0.40-1.0 (median 0.77), and 60% of `w_interface`
  values sit at its 0.05 floor (1.0 within 4 A of the partner).
- **Losses:** JointEngine's per-structure means over the kept sites; a structure with no kept site leaves that step's
  mean. Validation is unmasked.

With P = 0.2, sites at the `w_interface` floor (percentile 0.33) are kept 46% of the time and contact sites 97%.
Runs `pilot-10pct-mask2[-s29]` (P = 0.2, 60% kept on average) and `pilot-10pct-mask5[-s29]` (P = 0.5, 75% kept).

| Arm | Seed | Epoch | Selection | State MAE | Interface paired MAE | Interface paired MSE | pKPDB val MAE | pKPDB val MSE | Benchmark MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 17 | 17 | 0.7375 | 0.4730 | 0.2644 | 0.2525 | 0.5572 | 0.7328 | 0.5728 |
| default | 29 | 17 | 0.7355 | 0.4694 | 0.2660 | 0.2558 | 0.5598 | 0.7568 | 0.5790 |
| P = 0.2 | 17 | 17 | 0.7416 | 0.4767 | 0.2649 | 0.2546 | 0.5587 | 0.7339 | 0.5775 |
| P = 0.2 | 29 | 17 | 0.7574 | 0.4810 | 0.2765 | 0.2627 | 0.5663 | 0.7567 | 0.5963 |
| P = 0.5 | 17 | 18 | 0.7365 | 0.4724 | 0.2641 | 0.2507 | 0.5574 | 0.7188 | 0.5764 |
| P = 0.5 | 29 | 17 | 0.7442 | 0.4754 | 0.2688 | 0.2641 | 0.5642 | 0.7600 | 0.5769 |

| Contrast (seed means; 95% bootstrap) | Selection | State MAE | Interface paired MAE | pKPDB val MAE |
|---|---|---|---|---|
| P = 0.2 - default | +0.0131 (+0.0093, +0.0167) | +0.0076 | +0.0055 | +0.0040 |
| P = 0.5 - default | +0.0039 (+0.0003, +0.0077) | +0.0027 | +0.0012 | +0.0023 |

- Masking makes every validation metric worse, more so the more it masks. It gives no interface gain, and seed spread
  grows (0.002 default, 0.008 at P = 0.5, 0.016 at P = 0.2).
- On the 10% pool for 20 epochs, dropping sites mainly removes training signal. Whether masking helps as a
  regulariser on larger pools or longer schedules is untested. Default kept.

### Dropout, 10% and 50% pools, two seeds (2026-10-10)

`--dropout 0.1` (`production_train.DropoutEngine`): oGQT's built-in residual dropout (`pkanet.model.dropout`) on every
attention and feed-forward output (residue blocks, site query, site-site block) during training, the experiment 17 /
diagnostic campaign rate. Each complex's AB and free branches share one key (coordinated dropout); pKPDB structures
draw their own. Objectives, optimizer and schedule as JointEngine; validation without dropout. Runs
`pilot-{10,50}pct[-drop10][-s29]`; the 10% defaults are the existing `pilot-10pct[-s29]`.

| Pool | Arm | Seed | Epoch | Selection | State MAE | Interface paired MAE | pKPDB val MAE | pKPDB val MSE | Benchmark MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | default | 17 | 17 | 0.7375 | 0.4730 | 0.2644 | 0.5572 | 0.7328 | 0.5728 |
| 10% | default | 29 | 17 | 0.7355 | 0.4694 | 0.2660 | 0.5598 | 0.7568 | 0.5790 |
| 10% | dropout | 17 | 17 | 0.7292 | 0.4695 | 0.2597 | 0.5578 | 0.7306 | 0.5808 |
| 10% | dropout | 29 | 17 | 0.7300 | 0.4695 | 0.2605 | 0.5544 | 0.7211 | 0.5863 |
| 50% | default | 17 | 11 | 0.6794 | 0.4360 | 0.2434 | 0.5370 | 0.7104 | 0.5556 |
| 50% | default | 29 | 8 | 0.6926 | 0.4511 | 0.2415 | 0.5509 | 0.7392 | 0.5588 |
| 50% | dropout | 17 | 17 | 0.6730 | 0.4299 | 0.2431 | 0.5215 | 0.6445 | 0.5344 |
| 50% | dropout | 29 | 17 | 0.6802 | 0.4360 | 0.2441 | 0.5214 | 0.6289 | 0.5353 |

| Contrast (seed means; 95% bootstrap over validation complexes/structures) | Selection | State MAE | Interface paired MAE | pKPDB val MAE | pKPDB val MSE |
|---|---|---|---|---|---|
| dropout - default, 10% | -0.0069 (-0.0114, -0.0026) | -0.0018 | -0.0051 (-0.0084, -0.0020) | -0.0024 | -0.0190 (-0.0284, -0.0099) |
| dropout - default, 50% | -0.0094 (-0.0148, -0.0044) | -0.0106 (-0.0140, -0.0075) | +0.0012 | -0.0225 (-0.0267, -0.0183) | -0.0881 (-0.1058, -0.0708) |
| 50% - 10%, default | -0.0505 (-0.0551, -0.0455) | -0.0277 | -0.0228 | -0.0146 | -0.0200 |
| 50% - 10%, dropout | -0.0530 (-0.0587, -0.0472) | -0.0365 | -0.0165 | -0.0346 | -0.0891 |

- Dropout is the first variant to beat the default: selection improves at both pool sizes, in all four seed pairs.
- At 50% the default overfits: it selects epochs 11 and 8 (the seed 29 run early-stopped at epoch 16), seed spread
  0.013. With dropout both seeds select epoch 17, spread 0.007, and pKPDB val MSE falls 12% (0.725 to 0.637); the
  benchmark MAE also improves (0.557 to 0.535). The interface gain seen at 10% does not appear at 50%.
- Cost: 50% epochs take 64 s default and 67-70 s with dropout. Train loss is higher with dropout, as expected.
- Production default changed to dropout 0.1 is a decision for the 100% run; rate not tuned (0.1 only).

### Auxiliary target audit: ordinal burial and additive interface contacts (2026-10-10)

Data for replacing the exp 47 regression heads (MSE on normalised `w_burial` / `w_interface`) with ordinal BCE targets.
`scripts/auxiliary_target_audit.py` on 3,000 random pool-v4 training structures per dataset (154,477 labelled PINDER
sites, 83,052 pKPDB train_mask sites; Isambard job 7241362), summarised by `scripts/auxiliary_target_report.py`.
Output: `audits/auxiliary-targets-v1/{sample,report}.json`.

**Burial (free-state RSA).** Quartiles 0.22 / 0.39 / 0.54 (PINDER free), 0.20 / 0.37 / 0.53 (pKPDB); under 1% of
sites have RSA > 1. Mean |pKa - PKPDB_PK_MOD| falls steeply with RSA:

| RSA | PINDER share | PINDER mean abs shift | pKPDB share | pKPDB mean abs shift |
|---|---:|---:|---:|---:|
| < 0.05 | 7.7% | 2.77 | 8.6% | 3.15 |
| 0.05-0.1 | 4.6% | 1.75 | 5.2% | 2.01 |
| 0.1-0.2 | 10.2% | 1.22 | 11.1% | 1.44 |
| 0.2-0.3 | 12.9% | 0.84 | 13.4% | 0.99 |
| 0.3-0.4 | 15.9% | 0.60 | 16.1% | 0.72 |
| 0.4-0.5 | 17.3% | 0.43 | 16.5% | 0.56 |
| 0.5-0.6 | 14.7% | 0.34 | 13.4% | 0.42 |
| 0.6-0.8 | 14.0% | 0.23 | 13.6% | 0.27 |
| >= 0.8 | 2.7% | 0.20 | 2.1% | 0.23 |

Spearman(1 - clip(RSA), abs shift) = 0.61 (PINDER), 0.63 (pKPDB).

**Interface contacts.** Per labelled PINDER site, the residue-level minimum heavy-atom distance from the site residue
to every partner-chain residue within 10 A of AB.cif.gz. The minimum reproduces `partner_distance_A` to 0.005 A (30 of
41,174 sites stored at exactly 10.0 A fall outside by rounding). 26.6% of sites have a partner residue within 10 A.
Scores sum f(d) over those partner residues; plateau(L) = min(1, exp(-(d - 4) / L)), the `w_interface` shape per residue.

| Score | Share > 0 / 0.5 / 1 / 2 / 4 / 8 | Mean abs(AB - free) per band (<=0, 0-0.5, ..., >8) | Spearman, sites within 10 A | AUROC abs delta > 0.5 |
|---|---|---|---:|---:|
| plateau L = 1.5 | 26.6 / 14.4 / 11.2 / 7.5 / 3.3 / 0.34% | 0.005, 0.10, 0.21, 0.41, 0.67, 1.15, 1.85 | 0.636 | 0.965 |
| **plateau L = 3** | 26.6 / 20.9 / 17.0 / 12.1 / 6.9 / 1.7% | 0.005, 0.07, 0.11, 0.18, 0.39, 0.77, 1.47 | 0.630 | 0.964 |
| plateau L = 5 | 26.6 / 23.1 / 20.2 / 15.7 / 9.9 / 3.7% | 0.005, 0.07, 0.09, 0.13, 0.24, 0.54, 1.19 | 0.622 | 0.963 |
| exp(-d / 3) | 26.6 / 12.6 / 7.7 / 2.5 / 0.15 / 0.0% | 0.005, 0.11, 0.35, 0.69, 1.31, 2.03, - | 0.634 | 0.965 |
| exp(-d / 5) | 26.6 / 19.6 / 14.9 / 9.1 / 3.1 / 0.2% | 0.005, 0.08, 0.14, 0.27, 0.59, 1.21, 1.97 | 0.624 | 0.964 |
| count <= 4.5 A | 9.6 / 9.6 / 5.6 / 3.3 / 0.8 / 0.02% | | 0.578 | 0.890 |
| count <= 10 A | 26.6 / 26.6 / 22.6 / 19.9 / 15.5 / 9.0% | | 0.601 | 0.960 |

Nearest partner distance alone ranks abs(AB - free) about as well (Spearman 0.851 over all sites, AUROC 0.966), so the
decay length barely changes ranking; it decides how the 0-8 thresholds populate. With plateau L = 3 every threshold
holds at least 1.7% of sites and the mean shift roughly doubles per band, matching the doubling thresholds; L = 1.5 and
the pure exponentials leave the top thresholds nearly empty, and plain counts separate worse.

### Auxiliary burial/interface heads: cumulative BCE vs soft-target cross-entropy, 10% and 50% (2026-10-11)

`--aux-weight W` (`production_train.AuxEngine`, targets in `pkatrain.production_aux`), all on top of dropout 0.1, two
seeds; baseline `pilot-{10,50}pct-drop10[-s29]`. Contact-score table `pinder/contacts-v1` (28,733 structures, 1.45M
sites; nearest distance matches `partner_distance_A` to 0.005 A, score > 0 equals the stored interface flag on every
site). Two loss forms:
- `--aux-loss ordinal` (runs `-aux01`, `-aux1`): cumulative BCE, burial [RSA < 0.1 / 0.25 / 0.5], interface
  [c > 0 / 0.5 / 1 / 2 / 4 / 8];
- `--aux-loss ce` (runs `-ce01`, `-ce1`; user revision): one logit per head, cross-entropy against the soft targets
  1 - clip(RSA, 0, 1) and log(1 + min(c, 8)) / log 9, unweighted.

| Contrast vs dropout-only (seed means; 95% bootstrap) | Selection | State MAE | Interface paired MAE | pKPDB val MSE |
|---|---|---|---|---|
| ordinal W 0.1, 10% | -0.0041 (-0.0077, -0.0005) | -0.0037 | -0.0005 | -0.003 |
| ordinal W 1, 10% | +0.0049 (-0.0003, +0.0104) | -0.0076 | +0.0124 (+0.008, +0.017) | -0.003 |
| ce W 0.1, 10% | -0.0014 (-0.0045, +0.0019) | -0.0016 | +0.0001 | +0.003 |
| ce W 1, 10% | +0.0049 (+0.0003, +0.0096) | -0.0038 | +0.0087 (+0.005, +0.012) | +0.007 |
| ordinal W 0.1, 50% | +0.0016 (-0.0032, +0.0065) | -0.0011 | +0.0026 | +0.019 (+0.010, +0.028) |
| ordinal W 1, 50% | +0.0094 (+0.0041, +0.0146) | +0.0008 | +0.0086 | +0.023 (+0.013, +0.033) |
| ce W 0.1, 50% | +0.0075 (+0.0027, +0.0123) | +0.0036 | +0.0039 | +0.027 (+0.017, +0.037) |
| ce W 1, 50% | -0.0017 (-0.0066, +0.0030) | -0.0033 (-0.005, -0.001) | +0.0016 | +0.005 |

Head discrimination on validation (seed means; interface on the PINDER bound branch, burial on the PINDER free branch):

| Arm | Interface AUROC, c > 0 / > 2 / > 8 | Interface recall at 10% FPR, c > 0 | Burial AUROC, RSA < 0.1 / 0.25 / 0.5 |
|---|---|---|---|
| ordinal W 0.1, 10% | 0.61 / 0.68 / 0.81 | | 0.96 / 0.94 / 0.91 |
| ordinal W 1, 10% | 0.97 / 0.98 / 0.99 | | 0.97 / 0.96 / 0.94 |
| ce W 0.1, 10% | 0.62 / 0.69 / 0.78 | 0.19 | 0.95 / 0.93 / 0.89 |
| ce W 1, 10% | 0.95 / 0.97 / 0.98 | 0.88 | 0.96 / 0.95 / 0.92 |
| ordinal W 0.1, 50% | 0.93 / 0.97 / 0.99 | | 0.98 / 0.97 / 0.95 |
| ordinal W 1, 50% | 0.99 / 1.00 / 0.99 | | 0.98 / 0.98 / 0.96 |
| ce W 0.1, 50% | 0.91 / 0.97 / 0.98 | 0.76 | 0.97 / 0.96 / 0.93 |
| ce W 1, 50% | 0.99 / 0.99 / 0.99 | 0.98 | 0.98 / 0.97 / 0.96 |

- No auxiliary arm improves pKa selection at both pool sizes. The only interval excluding zero in favour is ordinal
  W 0.1 at 10% (-0.004), which reverses at 50%. W 1 consistently lowers state MAE but raises interface paired MAE at
  10%; at 50% most arms raise pKPDB val MSE, and seed spread grows (selected epochs 12-14 in four 50% runs).
- The heads learn: burial is easy (AUROC 0.89-0.98 at every weight); interface needs W 1 at 10% (AUROC 0.62 at
  W 0.1) and reaches 0.99 with 98% recall at 10% FPR for ce W 1 at 50%. ce and ordinal discriminate alike.
- Of the auxiliary arms, ce W 1 at 50% is the only one with no pKa cost (selection -0.002, n.s.; state MAE -0.003),
  so it is the candidate if the interface prediction is to be used (e.g. in the Siamese loss, the user's next step).
  Production default unchanged (no auxiliary heads).

**Interface head tested on the free branch** (`production_train aux-rescore`, job 7243593; selected checkpoints of the
16 bound-trained runs above; `aux-rescore.json` per run). Same head and bound-state contact-score labels, scored on the
free branch, where the partner is not in the graph. AUROC for c > 0 / > 2 / > 8:

| Arm | 10% bound | 10% free | 50% bound | 50% free |
|---|---|---|---|---|
| ordinal W 0.1 | 0.61 / 0.68 / 0.81 | 0.51 / 0.45 / 0.41 | 0.93 / 0.97 / 0.99 | 0.48 / 0.43 / 0.37 |
| ordinal W 1 | 0.97 / 0.98 / 0.99 | 0.57 / 0.49 / 0.43 | 0.99 / 1.00 / 0.99 | 0.52 / 0.45 / 0.41 |
| ce W 0.1 | 0.62 / 0.69 / 0.78 | 0.49 / 0.47 / 0.43 | 0.91 / 0.97 / 0.98 | 0.48 / 0.48 / 0.49 |
| ce W 1 | 0.95 / 0.97 / 0.98 | 0.57 / 0.57 / 0.57 | 0.99 / 0.99 / 0.99 | 0.51 / 0.49 / 0.50 |

- Without the partner the interface head is at chance (0.41-0.57; below 0.5 for the cumulative BCE heads at high
  thresholds). Its bound-branch accuracy is reading partner residues from the graph, not interface propensity; the
  bound-branch numbers in the table above are not evidence of interface prediction.
- Burial is unaffected: it was always trained and scored on the free branch (and pKPDB), AUROC 0.88-0.98.
- The free-branch training runs launched by mistake (`-cefree*`) were cancelled after 4 minutes; their partial run
  directories are not results.

### Burial-only auxiliary and Siamese burial, 10% pool (2026-10-11)

User decision: drop the interface head; keep burial. `--aux-heads burial` (burial cross-entropy vs 1 - clip(RSA_free)
on the PINDER free branch and pKPDB train_mask sites) and `--aux-heads burial-siamese`: the burial head on each branch
learns its own state (bound 1 - clip(RSA_bound), free 1 - clip(RSA_free); cross-entropy, the two averaged), plus the
squared error of the predicted bound - free burial (sigmoid outputs) against clip(RSA_free) - clip(RSA_bound), mirroring
the pKa state + Siamese losses. RSA_bound comes from `pinder/rsa-bound-v1` (`production_aux build-rsa-bound`; sites.json
rsa_bound in the store's site order; 28,733 structures, 1,451,907 sites, none missing, none with clipped bound RSA above
free; site counts, rsa_free and targets checked against the store). All W = 1, dropout 0.1, two seeds; runs
`pilot-10pct-drop10-bur1[-s29]` (off) and `pilot-10pct-drop10-bursiam[-s29]` (on).

| Arm | Seed | Epoch | Selection | State MAE | Interface paired MAE | pKPDB val MAE | pKPDB val MSE | Benchmark MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dropout only | 17 | 17 | 0.7292 | 0.4695 | 0.2597 | 0.5578 | 0.7306 | 0.5808 |
| dropout only | 29 | 17 | 0.7300 | 0.4695 | 0.2605 | 0.5544 | 0.7211 | 0.5863 |
| burial (off) | 17 | 17 | 0.7195 | 0.4633 | 0.2562 | 0.5537 | 0.7246 | 0.5709 |
| burial (off) | 29 | 18 | 0.7198 | 0.4621 | 0.2577 | 0.5513 | 0.7116 | 0.5743 |
| Siamese burial (on) | 17 | 18 | 0.7245 | 0.4624 | 0.2622 | 0.5537 | 0.7140 | 0.5675 |
| Siamese burial (on) | 29 | 18 | 0.7203 | 0.4621 | 0.2582 | 0.5482 | 0.7068 | 0.5673 |

| Contrast (seed means; 95% bootstrap) | Selection | State MAE | Interface paired MAE | pKPDB val MSE |
|---|---|---|---|---|
| burial - dropout only | -0.0100 (-0.0135, -0.0063) | -0.0068 (-0.009, -0.005) | -0.0032 (-0.006, -0.000) | -0.008 (-0.017, +0.001) |
| Siamese burial - dropout only | -0.0072 (-0.0108, -0.0035) | -0.0073 (-0.009, -0.005) | +0.0001 | -0.015 (-0.024, -0.007) |
| Siamese burial - burial | +0.0028 (-0.0001, +0.0057) | -0.0005 | +0.0032 (+0.001, +0.006) | -0.008 (-0.015, -0.001) |

- Burial alone is the largest gain over dropout-only of any auxiliary arm (selection -0.010, seed spread 0.0003),
  improving state, interface paired and pKPDB MAE together. With the interface head (ce W 1, earlier) the same
  weight gave +0.005 at 10%, so the interface head was what hurt.
- Siamese burial learns the burial change well (predicted bound - free burial vs truth: AUROC 0.98 for a change
  > 0.1, 6.4% of validation sites, recall 0.97 at 10% FPR; MAE 0.014; bound-branch burial AUROC 0.92-0.96) but does
  not help pKa beyond burial alone: selection +0.003 (n.s.), interface paired MAE +0.003 worse, pKPDB MSE slightly
  better. Burial-only W 1 is the auxiliary candidate; the 50% burial-only runs are pending.

**Burial-only at 50% and W 0.1** (`pilot-{10,50}pct-drop10-bur{01,1}[-s29]`, two seeds, vs dropout only):

| Contrast | Selection | State MAE | Interface paired MAE | pKPDB val MSE |
|---|---|---|---|---|
| burial W 0.1, 10% | +0.0018 (-0.0011, +0.0050) | +0.0008 | +0.0010 | -0.001 |
| burial W 1, 10% | -0.0100 (-0.0135, -0.0063) | -0.0068 | -0.0032 | -0.008 |
| burial W 0.1, 50% | +0.0005 (-0.0040, +0.0049) | -0.0013 | +0.0018 | +0.010 (+0.001, +0.019) |
| burial W 1, 50% | -0.0004 (-0.0056, +0.0049) | -0.0027 (-0.005, -0.000) | +0.0023 | -0.001 |

The 10% gain of burial W 1 does not hold at 50%: selection is unchanged (state MAE still slightly better, -0.003).
W 0.1 does nothing at either size. Like dropout's interface gain at 10%, it looks like a small-pool regulariser.
Production default unchanged (dropout 0.1 is the only change that held at both pool sizes).
