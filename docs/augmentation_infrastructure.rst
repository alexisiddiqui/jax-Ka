ANM augmentation infrastructure pilot
====================================

This release is a separate input for later interface detection. It does not
activate augmentation in pKa training, change epoch sampling, replace validation,
or generate the full dataset. Native validation remains the selection set.

Storage and interfaces
----------------------

``pkatrain.augmentation_store.ReleaseWriter`` writes independently compressed
zstd level-3 parent records into shards of at most 2 GiB. Publication requires
complete inventory and record verification. Partial releases resume from a
checkpoint; changed provenance, duplicate IDs and corrupt committed records fail.
``finalize_workers`` adopts verified worker shards already underneath the release
directory without copying their payloads. Use distinct worker directories.

``AugmentationStore.raw(parent)`` returns read-only metadata and NumPy array
views after one pread/decompression. ``conformer(parent, candidate)`` selects one
coordinate array. Checksum verification defaults on. The benchmark hot path
disables per-read hashing after full release verification, matching production.

``augmentation_evidence.CoordinateInputs`` joins coordinates to original input
masks pinned to that exact coordinate record. ``source_present`` describes the
repair input. ``original_input_present`` and ``original_input_known`` describe
upstream observation, separately from repair completion. Unknown atoms must not
be promoted to observed atoms. PINDER teacher numbering is not native numbering.

There are no per-conformer files in the published release. Native source files
and historical pilot outputs are inputs, not copied into the release. Production
repair workers must use bounded, job-specific local staging and retain only
selected coordinates, topology/masks, diagnostics and provenance on shared storage.

Graph correctness
-----------------

The pilot retains canonical residue/site/query order. Paired bound graphs and
features stay native. Free partners are rebuilt independently and remapped;
residue and site edges cannot cross partners. Backbone tables and sidechains are
branch specific. Original geometry-dependent targets are not attached to variants.

The benchmark derives capacities from saved native and augmented records. Native
controls use their original capacities; augmented controls use capacities covering
all variants. Later full graph stores remain deferred until interface inputs are
settled. The paired pilot duplicates bound inputs to test the existing engine;
this is not a commitment to that storage layout for a full augmentation release.

Running the pilot
-----------------

Use ``_HPC/submission/augmentation-pilot-cpu.sbatch`` on Coulson. It excludes
comp1400, runs the infrastructure tests, packs/audits the existing native+min
pilot, builds canonical pilot graphs, measures materialised host reads and writes
``readiness.json``. All dataset processing requires a Slurm allocation.

The command interfaces are::

  python -m pkatrain.augmentation_pilot pack PILOT OUTPUT/coordinates
  python -m pkatrain.augmentation_evidence OUTPUT/coordinates OUTPUT/evidence PILOT
  python -m pkatrain.augmentation_pilot graphs OUTPUT/coordinates OUTPUT/graphs \
    --runtime PKABENCH_RUNTIME --atlas-tar PILOT/atlas-records.tar
  python -m pkatrain.augmentation_pilot verify-graphs OUTPUT/graphs
  python -m pkatrain.augmentation_report OUTPUT PILOT

Transfer only verified packed release files to Isambard; do not extract loose
structures or put hot graph records inside squashfs/gzip archives. Refresh the
SSH certificate with ``clifton auth`` if authentication fails. Inspect numeric
Lustre project IDs and effective project/default byte and file limits before any
large transfer. Account for transfer partials, repair staging and logs. Do not
clean unrelated datasets, caches or installations.

On Isambard use ``augmentation-pilot-isambard.sbatch`` with one GPU, 72 CPUs and
2 GiB per CPU. The live base-A ``protocol.json`` supplies the batch size and its
architecture is checked; ``AUGMENTATION_PRODUCTION_PROTOCOL`` can override its
default path. This verifies native controls against frozen production graph,
backbone and sidechain stores, then measures synchronised width-64 orientation/
dihedral forward/backward compute. The loss is a compute-only paired/burial proxy;
it creates no augmented labels, optimiser updates or trained weights.

Readiness criteria
------------------

CPU read timings materialise graph arrays and report uncontrolled cache state;
they cannot establish GPU loader readiness. GPU tests require every represented
bucket, native and augmented finite gradients, at least 20 timed steps after
compilation, three repeats, workers 8/16/32 and prefetch 2/4. A viable setting
must have at most 5% steady loader wait and at least 90% of native throughput.
Verification and compile probes warm the pilot working set. Passing this gate
does not measure random I/O across a future full augmented graph store; that
requires its own loading check once interface-detection inputs are settled.

``augmentation_preflight budget`` consumes a quota snapshot and explicit peak
additional bytes/files. Effective limits must be verified, remain below 90% of
their hard limits and respect the 5,000 additional-file cap. For the final report,
the snapshot must pin ``coordinate_manifest_sha256``; the requested budget must
cover the population projection and worker file allowance, plus staging/transfer
costs. Zero quota values are not evidence of effective project headroom.

Repair projections retain failures in their denominator. The 380-worker estimate
is ideal compute scaling, excluding queues, contention, transfers and validation.
Determine concurrency per node with ``augmentation_report.worker_budget`` using
measured RSS, a 25% worker margin and a 10% node memory reserve. Free CPU count
alone is insufficient. A failed or missing readiness check blocks full generation.

Short GH200 repair probe
-----------------------

``augmentation_repair_benchmark bundle PILOT COORDINATES inputs.tar`` selects
eight size-spread complete parents per dataset, retaining failed zero controls.
Run bundling in a CPU Slurm allocation. The single archive contains audited
completed templates, portable OpenMM systems and coordinate mappings.

``_HPC/submission/augmentation-repair-isambard.sbatch`` requests exactly one GPU,
one task and 72 CPU cores, with 2 GiB per core. It compares 24 CPU workers with
one and four CUDA processes sharing that one GPU, using five modes and all
24 candidates plus the zero control. Each phase publishes a verified packed
coordinate release and reports timings, quality and selected RMSDs. GPU identity,
package versions, commit and input hashes are recorded. The isolated environment
at ``_submission/augmentation-gh200-repair-v1/env`` needs ``openmm[cuda12]``
and ProDy. The probe uses OpenMM 8.6.1, ProDy 2.6.1 and NVRTC 12.6.85;
CUDA extensions are installed separately from the CPU-only base wheel.
Install all probe dependencies with ``env/bin/python -m pip install -r
_HPC/submission/augmentation-repair-requirements.txt`` in that isolated environment.

This probe starts after source completion: source reading, hydrogen/system
preparation and conditional CG2ALL completion are excluded. Its sample covers
sizes rather than population frequencies, so aggregate throughput is not a
population-weighted whole-dataset runtime estimate. Cold context startup and
packing are included; full graph generation remains separate.

Mixed CPU/CUDA repair probe
---------------------------

Bundle with ``--all-complete`` to include all 493 successfully prepared pilot
parents. The new input format stores independently compressed members in one
indexed tar file. Workers read the exact byte ranges and verify member hashes;
there is no outer gzip scan or extraction of loose structures. Old 24-parent
gzip bundles remain readable.

``augmentation-mixed-isambard.sbatch`` requests one GPU and 72 CPUs. It uses
four CUDA processes and compares CPU-only 64-worker repair with concurrent
32/48/64 CPU workers plus the CUDA workers. All processes claim units from one
queue, ordered largest first. CPU and CUDA workers have disjoint core affinity,
with cores reserved for the writer. The report includes per-platform counts,
sampled aggregate process RSS and GPU memory, and exact target hashes.

A full CUDA warmup precedes the comparisons; GPU-only controls before and after
the mixed phases check drift. All phases use the same input inventory, controls,
five modes, candidates, filters and packed output verification. This remains a
completed-template probe: source completion and hydrogen/system preparation are
separate costs, and measured repair-worker memory does not cover those stages.
