PINDER and pKPDB coordinate generation
=====================================

``pkatrain.augmentation_generate`` generates the five-mode ANM 1/1/2 release
for pool-v4 pKPDB structures and separated PINDER partners, including validation.
ATLAS is excluded. The inventory contains 56,991 pKPDB units and 57,466 PINDER
units (114,457 total). This produces coordinates and atom observation masks;
graph generation and training activation are separate stages.

The displacement targets are 0.5, 1 and 2 A. Initial candidate budgets are
2/2/4, with same-bin replacements up to eight directions per bin. The geometry
gate and selection policy are described in ``augmentation_recipe_readiness``.
Each parent uses a deterministic source-specific seed independent of worker
assignment. Native heavy atoms are transported by residue and receive the same
100-step restrained cleanup as the zero control. Conditional cg2all completion
runs once per source when required. Hydrogens are used internally in cleanup;
the packed output contains float32 heavy-atom coordinates only.

Registration checks every local source, defect record and original atom-mask
file, freezes pool/inventory/task hashes and preparation source hashes, and
assigns parents to workers by size. A short CPU-only preflight on comp1400 runs
the tests and six fresh parents spanning both pools and multiple lengths.
It checks repeated zero cleanup and compares persisted float32 geometry flags
with the independent original full-precision implementation. Production workers
use generalaccess CPUs, with one CPU/2 GiB for parents up to 500 residues and
two CPUs/4 GiB for larger parents. No GPU is requested.

Run with a frozen directory containing ``src``, ``tests`` and the audited legacy
preparation modules. The three submission files are
``augmentation-pools-preflight.sbatch``, ``augmentation-pools-generate.sbatch``
and ``augmentation-pools-finalize.sbatch``. After the preflight succeeds, submit
the small array with ``--array=0-159%112`` and offset 0, and the large array with
``--array=0-63%64 --cpus-per-task=2`` and offset 160. The combined ceiling is
240 allocated CPUs. Finalisation depends on successful completion of both
arrays. These arrays may queue when generalaccess CPUs are occupied.

The authorised OC alternative uses ``--partition=oc --array=0-159%20`` for
small parents and ``--partition=oc --array=0-63%14 --cpus-per-task=2`` for large
parents. Append ``--allow-comp1400`` after the offset argument to both generator
scripts. Submit finalisation with ``--partition=oc`` and append the same flag
after its frozen-code argument. This caps generation at 48 allocated CPUs
(20 small workers and 14 large workers) without requesting GPUs. Cancel the
superseded queued arrays and their finaliser before submitting replacements;
the same inventory, frozen generation code and worker checkpoint paths apply.
The OC allocation is smaller, so budget approximately one to two running days
rather than the generalaccess projection. ``launch.json`` records active jobs
and superseded submissions.

Each parent runs in a fresh subprocess with bounded node-local scratch and a
20-minute timeout. Only indexed zstd level-1 records and aggregate metadata
persist; no per-candidate PDB tree is retained. Parent records include native
coordinates, repaired zero, selected views, original atom masks, topology,
source hashes, quality results and preparation timings. Failures retain their
identity and reason. Unmet quotas remain explicit. Worker checkpoints permit
resuming the identical code and recipe; interruption can repeat at most the
uncommitted 16-parent tail. Published workers are immutable and verified on
resume. Packed shards are limited to 2 GiB. Finalisation checks all worker
payloads, exact parent coverage and the 5,000-file budget before publishing a
root manifest. Downstream consumers must use that published root manifest.

The working release root on Coulson is
``_runtime/jax-Ka/anm-pilot-v1/pools-112-v1``. Inventory and final summary files
record the scope and observed outcome. Queue time and source-specific failures
must be distinguished from the earlier stratified pilot runtime projection.
