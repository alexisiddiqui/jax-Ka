Packed augmentation feature release
===================================

``pkatrain.augmentation_cache_generate`` builds all PINDER and pKPDB training
and validation parents from the existing immutable coordinate release. ATLAS is
outside this release. It performs no ANM generation, reconstruction, minimisation
or training. Native graphs are rebuilt transiently to verify partner mappings,
then discarded. The retained payload contains each repaired zero and selected
augmentation as free-unit features, plus a compact native partner-partition
store and native-input fingerprints. Live native training stores remain required.

The comp1400-only Slurm entry point is
``_HPC/submission/augmentation-cache-full.sbatch``: 48 CPUs, 96 GiB, 24
single-threaded workers and no GPU request. Each worker writes its own independent
level-1 zstd frames in at most 2-GiB shards. Arrays stay inside the producing
process. Workers checkpoint packed features before the corresponding packed
parent coverage records. A restart accepts an identical inventory and source
recipe, skips committed parents, reconstructs partially committed parents and
reuses immutable completed worker releases. No per-parent filesystem entries
are created. Source code is frozen outside the published payload.

Every generated graph is compared against an independent full free-complex
construction. Worker publication checks framing, hashes and finite values.
Consolidation checks source-derived parent/view coverage, partner agreement on
the native complex and canonical maps, adopts worker payloads without copying,
and streams every adopted frame once against published worker hashes while
checking the feature schema. Dimension indexes come from this verified metadata,
avoiding another payload scan. An outer ``manifest.json`` is written last and
defines the reader/transfer allowlist. Worker checkpoints, logs and source code
are not reader dependencies. Failed coordinate parents remain explicit native
fallbacks; unexpected feature failures prevent publication.

``augmentation-cache-canary.sbatch`` first runs the relevant regression tests,
then the same production path on pilot-overlapping parents spanning dataset,
split, partner, size extremes and source failure classes. All resulting arrays
must equal the independent published pilot bit-for-bit before canary publication.
Run this before the full build. Native compatibility on Isambard and training
activation are separate downstream steps; the build never changes training.

The pilot projected approximately 1.41 TB of compressed feature payload and
roughly 663 consolidated reader files before compact coverage/partition metadata.
Parallel worker shard tails and metadata increase the final count modestly;
the working release is limited to 5,000 entries including directories. The
initial full-build planning range is 13--15 hours on Coulson, excluding queue
and transfer; live production timings supersede this projection. The user
approved proceeding with the documented PINDER smallest-bucket throughput
exception: about 87 percent of native throughput versus a 90-percent target,
and about 98 percent when weighted over PINDER sizes. Training is still disabled.

The completed production build took 13 hours 20 minutes on comp1400. Its
561,990 feature records occupy 1,424,126,222,985 compressed bytes. The verified
reader closure contains 711 files; the working release used 1,035 entries
including directories. There were no unexpected feature failures: 112,433
parents have cached views, while 2,024 upstream coordinate failures are explicit
native fallbacks. Coverage includes all 114,457 training and validation parents.

Isambard transfer
-----------------

``_HPC/submission/augmentation-cache-transfer.sbatch`` runs the transfer on
comp1400 using four CPUs and no GPUs. The frozen
``benchmarks/augmentation_cache_transfer.py`` copies only the outer manifest's
allowlist with two resumable rsync streams. It checks the current project byte
and inode hard limits with a reserve for staging and control files. Shared
storage receives packed payloads, aggregate metadata and one frozen source
archive; no source or structure trees are extracted there.

The destination initially ends in ``.incoming``. A sibling ``.transfer``
directory contains the expected source manifest, verifier, source audit archive
and verification progress. After copying, two hash workers verify every copied
file against its source SHA-256. Unlisted files or mismatches prevent publication.
Only then is the original manifest installed and the staging directory renamed
atomically to the final destination. Tests cover rejection of corrupt, unlisted,
escaping and linked payloads, plus restart between marker installation and
directory rename. This establishes byte identity; native-store compatibility
and training activation remain separate steps.
