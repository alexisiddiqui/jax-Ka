# Loader protocol, prefetching, staging and dataset transfer (2026-10-09)

Preparation for production training on the pool-v3 subsets on another cluster. User decisions: new modules, so the
code-hashed experiment loaders (`gqt_paired_pinder`, `graph_batches`, `site_graph_data`, `trainer`) stay untouched;
the protocol covers JAX and Torch, and experiment 48 (`pkai_joint_scale.py`) uses it; no sampling changes; the pKPDB
graph files are an optional export part.

## Shared loader protocol (`src/pkatrain/loading.py`)

- `BatchSource` protocol: `load(spec)` (thread-safe host assembly), `close()`, `provenance()`. Plans come unchanged
  from the existing plan functions.
- `LoaderConfig`: `workers` (assembly threads, default `min(8, cpus - 1)`, `PKATRAIN_LOADER_WORKERS`) and `prefetch`
  (batches in flight, default 2, `PKATRAIN_PREFETCH_DEPTH`).
- `Prefetcher`: bounded, in-order pipeline. The host-to-device transfer runs in the prefetch thread (`jax.device_put`,
  or pinned non-blocking Torch copies on a side stream). Worker errors re-raise at the consumer, and an early exit
  cancels pending work. Telemetry records per-batch wait and assembly time and the wait fraction, flagged above 5%
  (the experiment 29 criterion).
- `DeferredScalars`: losses stay on the device and are read back in groups. One fused finite check per step remains.
- `BucketPolicy`:
  - `LEGACY_PAIRED` reproduces `gqt_paired_pinder._plans` exactly (tested).
  - `PRODUCTION` uses bounds 128 / 256 / 384 / 512 / 640 / 768 / 1024 / 1280 / 1536, covering every pool-v3 structure
    (max: PINDER 1,500, pKPDB 1,053; previously 5,694 PINDER and 7,917 pKPDB pool structures exceeded 768).
  - Production batch sizes come from a residue budget (`budget // bound`). The budget is 3,072 = 8 × 384, giving
    24 / 12 / 8 / 6 / 4 / 4 / 3 / 2 / 2. It's provisional until a GPU memory check on each bucket's largest structure.
- Sources:
  - `loading_gqt.PairedSource`: PairedMMap + `_load_one`; bucket policy from the manifest.
  - `loading_gqt.LegacyLoaderSource`: wraps `BatchLoader`/`SiteBatchLoader` with a workers-sized pool.
  - `loading_gqt.JaxStepRunner`: calls the engines' jitted steps without `float()`.
  - `loading_torch.PackedSiteSource`: resident on the GPU when the arrays fit 60% of free memory, else streamed.
  - `loading_torch.CombinedSource`.

### Checks (`pkatrain.loading_checks`, results in `audits/loader-protocol-v1/`)

| Check | Result |
|---|---|
| `PairedSource` vs `Loader.batch`, 60 factorial batches | bit-identical |
| Paired GQT loop, 200 batches, A40 (old → new) | 2.88 → 3.20 steps/s (+11%); max loss difference 7e-7; wait fraction 4.9% (max 1.8 s; unstaged shared-FS mmap) |
| Exp 48 data path, synthetic 60k rows × 4,008, 235 steps (old → resident / streaming) | 81 → 117 / 88 steps/s; losses bit-identical; wait fraction 0.1% |
| Unit tests (`test_loading.py`, `test_dataset_transfer.py`) | 13 pass under pytest; the Torch source test passes in finetune-v1 (no pytest there) |

Experiment 48 `train_scale` now uses `PackedSiteSource` + `Prefetcher`:
- the same `rng.permutation` slices and wrap-around padding;
- one fused finite check per step;
- deferred losses;
- validation sums accumulated in the original order;
- `loader_wait_fraction` in each history row.

The pKPDB validation reads the compact package `pretraining/pkpdb-val-pkai-v1`, checked against the pilot row
selection, when it is present.

## Idempotent staging and transfer (`src/pkabench/dataset_transfer.py`)

**Staging** (`stage SRC --local DIR`):
- The local copy is keyed by sha256 of the store's `verification.json` and its file sizes.
- An unchanged source is reused; a changed one is copied to a new directory.
- Copies go to a pending directory under an flock and are verified (size, and sha256 where the store records it)
  before an atomic rename. Pending directories of dead processes are cleaned.
- On the 29 GB factorial store: first stage 86 s; second 0.17 s (no-op); three concurrent jobs → one copy.
- `_HPC/submission/jax-Ka/pkabench/train-template-staged.sbatch` replaces the bash `stage_store()` pattern.

**Export/import** (`export {pinder,pkpdb,validation} OUT [--graphs] [--scope pool|all]`, `import`, `verify`):
- Deterministic shards of up to 2 GB per dataset part, written in parallel: gzip level 1 for core, plain tar for graphs.
- `<dataset>-bundle.json` records per-file size and sha256, per-shard sha256, source hashes and the git commit.
- pKPDB `graph.npz` and `sites.json` are checked against `pilot.json` during export.
- Import verifies each shard, extracts it to a pending directory, verifies every file, installs it, and writes a
  per-shard marker under `<root>/.imports/`. Re-runs are no-ops, interrupted imports resume, and corrupt shards or
  unexpected members (path traversal) are rejected.
- Paths are relative to `PKABENCH_RUNTIME`.

Contents:
- **pinder**: pool-v3 ∪ the 400 validation complexes (all entry files except `labels_noterm.json`); `index/`,
  `pool-v3`, exclusions v3, summaries, the validation cohort.
- **pkpdb**: pool-v3 entries' JSON files and source structures; `pilot`/`protocol`/`verification`/`references`,
  `labels.sqlite`, `pool-v3`. The optional `graphs` part holds `graph.npz`.
- **validation**: the compact pKAI validation package (rows from the frozen 5k pilot plus provenance) instead of the
  2 × 7.7 GB pilot feature files.

Runner: `_HPC/submission/jax-Ka/pkabench/dataset-export.sbatch {pinder,pkpdb,validation} [--graphs]`; it exports, then
verifies.

### Export results (2026-10-09, `exports/bundle-v1/`)

| Bundle | Shards | Size | Files | Wall (incl. verify) |
|---|---:|---:|---:|---:|
| pinder (core; pool-v3 + 400 validation) | 6 | 8.3 GB | 390,223 | 40 min |
| pkpdb (core; pool-v3, no graphs) | 14 | 11.1 GB | 503,001 | 40 min |
| validation (compact pKAI package) | 1 | 11 MB | 6 | < 1 min |

All three passed `verify` against the source tree. The optional pKPDB graphs part (~215 GB, `--graphs`) was not
exported.

Import: the first full-scale import test (old code) installed 4 of 6 PINDER shards correctly, at about 11 min per
shard. Random-access reads of gzip tars restart decompression on every seek. Import now makes one streaming pass per
shard, hashing the shard and every file while extracting, and runs shards in parallel. That version passes the unit
tests, but the full-scale import test was stopped at the user's request before it finished, so it has no measured
timing yet.

## Transfer to Isambard-AI and squashfs images (2026-10-10)

- `rsync` of `bundle-v1` (19.4 GB, 22 files plus validation) to `u6xq.aip2.isambard:$SCRATCH/_transfer/bundle-v1`:
  6 min 55 s over 4 parallel streams (about 47 MB/s; one stream about 15 MB/s). Sizes match the source.
- The validation bundle was imported and verified there (6 files).
- Loose import does not fit. The scratch project allows 1,024,000 files (inodes) and was already at about 466k without
  these data; PINDER (390k files) plus pKPDB (503k) exceed the limit. The pKPDB import failed with
  `Disk quota exceeded`, and the PINDER import was cancelled and its partial files cleaned.
- Replacement: `dataset_transfer squash` builds one squashfs image per large bundle directory in one streaming pass,
  checking every file's size and sha256 against the manifest on the way:
  - `pinder-pkai-v1` (390,222 files, 11.5 GB);
  - `pkpdb-full-v1` (377,253 files, 20.4 GB);
  - `pkpdb-v1` (125,748 source structures, 9.3 GB).

  Images go to `<runtime>/images/<name>.sqfs` with a JSON marker (key, file count, sha256); the few remaining files are
  installed loose; `<runtime>/pretraining/<name>` becomes a symlink to `/tmp/$USER-sqfs/<name>`.
- `scripts/sqfs_run.sh <command>` mounts the images read-only with `squashfuse_ll` inside a private user and mount
  namespace, then runs the command as the calling user.
  - Isambard has no per-job `/tmp` or mount namespace, so a plain FUSE mount would be shared by every job on a node and
    would die with whichever job started it. The private namespace avoids that.
  - Tested on a GH200 node: the mount is private, the uid is unchanged, the GPU is visible, and writes to Lustre are
    owned by the user.
  - Outside the wrapper the dataset paths are dangling symlinks, so a missing mount fails at once. Image directories
    are read-only, so prep outputs (e.g. per-entry `graph.npz`) must go elsewhere, preferably straight into the packed
    stores, which also keeps the file count down.
- Tools: `squashfs-tools` 4.7.5 from conda-forge in its own prefix (`$S/_install/squashfs-tools`; the uv venv stays the
  main environment). `squashfuse_ll` and `/dev/fuse` are provided by the system.
- No Python squashfs reader (`PySquashfsImage`) was added: the mount serves the existing path-based readers unchanged.
  It is only worth adding as a fallback if FUSE becomes unavailable.
- Unit tests (Isambard compute node, `.venv`): build, idempotent re-run, file-by-file comparison through `unsquashfs`,
  refusal of a directory in the way, and rejection of a corrupted shard. All 8 transfer tests pass.

Results (Isambard job 7211842; 32 CPUs; both datasets built in parallel):

| Image | Files | Content | Image size | Build | Verify (all files, through the mount) |
|---|---:|---:|---:|---:|---:|
| `pinder-pkai-v1.sqfs` | 390,222 | 11.5 GB | 7.9 GB | 143 s | 9 s (390,223 incl. the loose `cohort.json`) |
| `pkpdb-full-v1.sqfs` | 377,253 | 20.4 GB | 1.1 GB | 145 s (both pKPDB images) | 12 s (503,001, both images) |
| `pkpdb-v1.sqfs` | 125,748 | 9.3 GB | 9.0 GB (already-compressed `.cif.gz`) | | |

All files passed. Project file count went from 627k (partial loose import) to 468k; the three images add 6 files.
Runners: `$S/_submission/jaxka_dataset_squash.sbatch` (build and verify) and `jaxka_dataset_import.sbatch` (loose
import, used for the 6-file validation bundle).
