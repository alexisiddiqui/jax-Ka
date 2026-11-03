Packed free-partner feature-cache pilot
======================================

``pkatrain.augmentation_cache_pilot`` rehearses conversion and loading of the
published pKPDB/PINDER coordinate release. It neither activates augmentation in
training nor updates weights. The probability sample has 256 parents spread
across dataset, split, PINDER partner and five residue-count strata. Largest
structures and rare preparation/zero/partial failures are additional zero-weight
stress controls. The source release and sample are hashed.

Only the current base-A inputs are cached: residue/site graph fields plus the
14-column backbone table. Orientation and dihedrals remain device-computed;
unused sidechain tables and geometry-dependent labels are omitted. Each record
contains one free partner and explicit canonical residue/site/query mappings.
The repaired zero and each selected conformer use independently indexed zstd
level-1 records in shards no larger than 2 GiB. Native bound inputs are not copied
into every conformer record. A single native control per complex is retained in
the pilot solely for transport and independent checking. Full production reuses
the existing native stores instead of duplicating these controls.

The builder compares each partner reconstruction against an independent full
free-complex graph build, using the same arithmetic precision as the source.
Failures remain in the diagnostics and block the benchmark. Tests cover
sampling weights, native bound preservation, partner isolation and the packed
single-branch pKPDB/two-branch PINDER loader. A separate packing probe measures
serial encoding/writing and final verification rather than folding those costs
into a misleading ideal worker-scaling estimate.

The optimized loader reads complete pKPDB variants directly and writes PINDER
partner rows into the final padded buffer. PINDER also uses a compact five-int
dimension index and a manifest-pinned dtype schema, so the hot path can construct
read-only array views without parsing and validating a JSON header per read.
The standard decoder remains the independent validation path. The two index
files do not change any compressed frame and add about 11 MB at full scale.
The full builder should emit dimensions while writing and export a flat native
partner-partition table; the pilot's transport-only control graphs are not a
production storage dependency.

The CPU entry point is ``_HPC/submission/augmentation-cache-build.sbatch``.
``augmentation-cache-isambard.sbatch`` requests exactly one GH200, one task,
72 CPUs and 2 GiB/CPU. It extracts the frozen training source and isolated reader
overlay into node-local scratch. Shared storage holds only compact archives,
published data shards/indexes and aggregate logs. Transfer is by manifest
allowlist; worker checkpoints, source trees, packing probes and loose structures
are excluded.

Before timing, all transferred payload hashes and native graphs/backbone tables
are checked against the live production stores, and feature-source hashes must
match. The hot loader reads the live native ProductionStore/BackboneTable and
one selected partner record; metadata partitions are small read-only tables.
It preserves native bound inputs, changes only the selected free partner, and
uses deterministic shuffled parent cycles rather than a fixed first example.
Both partners can later be assembled independently without storing 16 combined
conformations. No per-conformer disk cache is created while loading.

GPU probes use the live base-A protocol (batch 16, width 64, feedforward width
128, orientation and dihedrals), single-branch pKPDB and paired PINDER. A matched
synthetic objective exercises finite forward/backward compute without labels or
optimiser updates. Each represented size bucket is compiled for native and
augmented capacities, then measured for 20 batches, three repeats, workers
8/16/32 and prefetch 2/4. A setting passes only if every bucket retains at least
90 percent of native throughput with at most 5 percent steady loader wait.
Verification and compilation warm the pilot; full-dataset cold random reads are
explicitly outside the result.

The initial diagnostic explored all three worker counts. Follow-up runs use
16 workers and preserve earlier results separately: ``bench-v3`` contains the
direct-to-batch loader, and ``bench-v4`` retests PINDER with the compact index.
pKPDB's measured decoder path is unchanged. Resident native/augmented compute
probes help separate input overhead from network computation. A PINDER benchmark
example changes one sampled partner; simultaneous A/B augmentation is outside
the measured hot path.

``benchmarks/augmentation_cache_report.py`` calibrates cache-byte and build-time
projections to exact known counts of parents with retained views within the
sampling strata. Source failures need no feature allocation. The budget includes
a 30 percent payload allowance, the full coordinate release, bounded transfer
partials and a conservative worker file/directory allowance. It requires both
effective project limits to remain below 90 percent and fewer than 5,000 added
inodes. Native compatibility, quotas and both dataset throughput gates are
required before the summary marks a full feature-cache build ready.
