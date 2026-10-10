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
