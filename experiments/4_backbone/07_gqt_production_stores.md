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
