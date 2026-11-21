Dense native MaSIF annotation timing
===================================

The measurements below describe the historical Python preprocessing pilot.
New invocations default to the verified original-paper backend; the submission
scripts select separate native pKPDB training chains. See
``masif_paper_batch.rst`` for the current CPU/GPU batch builder, packed-reader
contract and resume/merge commands.

``benchmarks/masif_annotation_pilot.py`` samples the existing pool-v4 cohort,
including validation: 56,991 pKPDB assemblies, 57,466 separated PINDER partners
and optionally 28,733 native bound PINDER complexes. There are no ANM views,
zero controls, minimisation or training operations. Two probability samples per
family and residue-count bin (<=150, <=300, <=600, >600) give 24 structures;
three largest-structure stress cases have zero population weight.

The output is a frozen MaSIF-search all-features 80-dimensional descriptor at
every molecular-surface vertex, using the checkpoint's 12-A geodesic cutoff,
200-vertex patch cap, straight orientation and disabled inference dropout.
This differs from the earlier 32-anchor invariance audit. Timings separate
native structure export, hydrogenation, MSMS/PyMesh surface construction,
PDB2PQR/APBS electrostatics, geodesic distances, patch/MDS construction, model
construction/restore, inference and output packing.

MSMS's default follows one molecular component and can silently miss most of a
disconnected native structure. The pilot explicitly partitions the expanded
atomic spheres into connected components, meshes each component at the same
probe/density, then regularizes the combined surface. Component atom counts,
represented components and surface residue coverage are recorded. The legacy
single-component diagnostic is retained only as a comparison, not as the final
whole-unit annotation ETA.

Native canonical residue and atom order comes from the dataset's frozen
preprocessing source. The PINDER CIF reader does not import training or solver
modules. Protein chains and backbone gaps become separate PDB segments;
source residue indices are retained for mapping the resulting surface owners
back to the native graph. Reduce uses ``-HIS -NOFLIP``. PDB2PQR uses PARSE,
``--noopt`` and ``--nodebump``. Original heavy atoms must survive without
coordinate movement; added atoms and uniquely matched coordinate-preserving
terminal-name aliases are recorded. No native geometry is relaxed.

``masif_annotation_bridge.py`` runs inside the existing frozen author container.
Surface distances are computed in bounded Dijkstra blocks and retained in CSR
format. Per-center MDS, angular coordinates, DDC, other features and neural
inference retain the upstream definitions. This avoids both the dense N-by-N
angle array and repeated inner-neighbour shortest-path searches of the old
selected-anchor diagnostic. Every completed structure checks three random real
surface centers against the independently upstream-validated original adapter.
Model and binary hashes are verified against the earlier environment record.

The MaSIF graph/checkpoint stays loaded within each persistent container worker.
Inference, BLAS and APBS are single-threaded. The pilot reserves four CPUs and
8 GiB per worker and monitors the complete host/container/subprocess RSS tree.
The main pilot uses eight workers on comp1400, with no GPU allocation. Cold
model restore is reported separately and excluded from per-structure production
projections. A memory-aware 48-CPU estimate includes a 25-percent worker margin
and ten-percent memory reserve; its higher concurrency is an extrapolation.

All PDB, PQR, APBS grids and per-patch tensors stay in node-local scratch. Only
surface vertices/faces/normals, residue ownership, normalized chemistry and
float32 descriptors are retained in independently compressed level-1 zstd frames
inside bounded shards. Permanent output has no per-structure filesystem entries.
Receipts include failures, source hashes, charge-completion details and timings.

``masif_annotation_report.py`` generates native-only CPU runtime and storage
estimates from exact population counts and sample weights. Probability cases
are bootstrapped within strata; the stress cases constrain memory and numerical
checks without biasing population averages. Failed inputs stay explicit; an
incomplete cell is not evidence that every requested structure can be annotated.
Queues, data transfer, full-dataset I/O and GPU inference are outside the CPU
estimate. Full annotation has not been launched by this pilot.

Measured native pilot
---------------------

The corrected all-component pilot (job 799444) completed in 13 minutes 34 seconds
on comp1400 with eight CPU workers. All 24 probability-sampled structures passed;
one of three largest-structure stress cases passed and two failed nonfinite
surface-curvature checks. Three random reference fingerprints per completed
structure matched the independently validated adapter exactly. Peak complete
worker process-tree RSS was 6.39 GiB.

pKPDB plus separated PINDER partners (114,457 surfaces) projects to 4,265
single-worker hours and 292 GB of packed float32 annotations. Including native
bound PINDER complexes (143,190 surfaces) gives 5,538 worker-hours and 374 GB.
The measured eight-worker layout projects to 22.2 or 28.8 days. Conservative
memory budgeting within a 48-CPU/96-GiB allocation gives ten workers and 17.8
or 23.1 days. This extrapolation has not been run at full-data concurrency.

Approximately 51 percent of measured steady annotation time is neural inference,
22 percent patch/MDS construction and 21 percent electrostatics. These are CPU
author-container estimates; GPU inference is not measured. A larger RAM
allocation, heterogeneous worker reservations or accelerated inference would
need a separate throughput check before reducing the ETA. Known curvature
failures also need an explicit production handling policy.

Subsequent GPU validation exposed NaN parameters in the bundled checkpoint's
shape-index branch. Legacy TensorFlow CPU maximum reduction suppresses them to
zero branch activations, so descriptor finiteness alone did not expose the
problem. The GPU port reproduces the CPU outputs, but a clean validated
checkpoint is needed before full annotation. See ``masif_gpu_pilot.rst`` and
the recorded independent activation probe.

New runs require an explicit validated checkpoint and record its hashes in
receipts and packed-release provenance. The submission wrapper selects the
validated active profile; see ``masif_checkpoint.rst``. Historical fingerprints
are not relabelled as outputs of the replacement model.
