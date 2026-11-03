Five-mode amplitude quota readiness
==================================

The recipe retains four augmented views per eligible parent: one at 0.5 A,
one at 1 A, and two at 2 A target CA displacement. Five ANM modes are fixed.
The initial candidate budgets are 2/2/4, using the same independent modal draws
at each amplitude. Capped native transport/minimisation and geometry checks are
unchanged. No local surface-preservation or APBS gate is applied.

``pkatrain.augmentation_selection`` plans candidate generation and ranks the
geometry-valid candidates independently within each amplitude bin by maximin
aligned post-repair CA RMSD, seeded by the repaired zero-displacement reference.
Failed candidates are replaced by the next direction in the same amplitude,
only until that bin has enough valid candidates; at most eight directions per
bin are attempted. Insufficient quotas remain explicit and never borrow views
from another amplitude. Source/preparation errors and invalid zero controls
remain in coverage denominators.

``benchmarks/augmentation_recipe_readiness.py`` replays this policy on the
saved 500-parent, 24-candidate native+min pilot. It does not generate new
structures or activate augmentation in training. The ``audit`` command reports
initial and replacement yields, selected amplitudes/RMSDs, pairwise diversity,
and full-population runtime estimates using the original dataset/split/length
stratum weights. Runtime accounting retains fixed ANM construction, source
completion/preparation and context costs, includes the successful zero repair,
and replaces the old candidate/selection components with the proposed policy.
Graph computation is excluded. Ideal worker scaling and a separate 30 percent
planning allowance are reported; these are not a new end-to-end throughput
measurement or a guarantee of current queue, memory or filesystem capacity.

The ``pack`` command creates a new coordinate release with zstd level 1,
2 GiB shards, float32 heavy coordinates, original source identities, repaired
zero, four selected views and original-input atom observation masks. Original
input masks and their source provenance are embedded in each parent record,
avoiding separate files per conformer or residue. Raw source coordinates remain
as evidence and as the independent chirality baseline; the reference view is
explicitly ``zero_coordinates``. Records with failed preparation or zero
controls are retained with unusable status. An independent streaming check
verifies all file/frame hashes, geometry classifications, source masks and
persisted quotas before downstream use.

The writer's compression level is explicit and pinned in checkpoint signatures.
Legacy level-3 checkpoints can resume at level 3; changing levels during resume
is rejected. Worker finalisation propagates the actual common compression and
shard size and rejects mixed layouts. Existing immutable releases remain
readable. Consolidated and conservative multi-worker file-count projections
are reported separately from an effective user/project quota.

The ``graphs`` command selects one quota-complete parent per dataset/length
bin (15 parents), preferring validation in bins 1 and 3 and training elsewhere.
It uses the previously verified canonical graph mapping, retains every native
bound input, and rebuilds free geometry and sidechain features for the repaired
zero reference and all four augmented views. Graph masks and indices are
independently verified. Different capacities may add zero padding to the bound
arrays, but changing any bound value is a hard error. Geometry-derived target
labels are omitted. This is an input compatibility check, not a training run.

The ``reads`` command exercises the existing materialising host prefetch loader
with one/two workers and prefetch depth two, touching model and sidechain data.
Matching checksums and byte counts are required across worker settings and
repeats. The ``native`` rows in the legacy loader report denote the repaired
zero free reference with the unchanged bound branch. These host-only timings
use a small, uncontrolled warm working set; they cannot establish full-dataset
random-I/O performance, on-the-fly graph construction cost, or GPU input wait.

Run the stages with ``_HPC/submission/augmentation-recipe-readiness.sbatch`` and
a frozen source directory containing ``src``, ``tests`` and ``benchmarks``.
Order the stages ``tests``, ``audit``, ``pack``, ``graphs``, ``reads``, ``report``
with successful-job dependencies. Coulson CPU allocations use 2 GiB per CPU and
the compute guard. The report summarises only these authorised pilot checks;
full-dataset generation and training activation are separate actions.
