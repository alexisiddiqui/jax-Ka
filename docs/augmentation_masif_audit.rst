Frozen MaSIF surface audit
=========================

This CPU-only Coulson diagnostic measures existing ANM conformers. It does not
train a network, regenerate coordinates, activate augmentation in pKa training,
or evaluate binding accuracy. MaSIF-search's all-feature pretrained checkpoint
is held fixed; ``flip=False`` and inference dropout is disabled.

Inputs and selection
--------------------

``augmentation_masif_audit.py select`` consumes a verified augmentation coordinate
store, the historical native+min pilot and the previous surface-audit CSV. It
chooses one typical and one upper-tail protein from each of five length bins in
ATLAS, PINDER and pKPDB: 30 parents. The sample is deliberately enriched for local
changes; reported percentages are not full-dataset estimates.

Each parent contributes prepared native, current repaired zero, four selected
conformers, historical zero, and the same preselected direction at 0.5, 1 and
2 Angstrom. Historical atom identity and native coordinates must match the
current release. Exact coordinate duplicates share a result while retaining
their comparison roles. Source residue keys, chain/segment boundaries and atom
completion masks are preserved.

Environment and commands
------------------------

The author container ``pablogainza/masif`` runs via user-installed udocker in P1
mode. Its image manifest, checkpoint, source modules, binaries and compatibility
libraries are hashed. Coulson's GCC-8 libstdc++/libgcc are mounted only inside the
diagnostic container because its bundled PyMesh needs CXXABI_1.3.11. The existing
training environments are not modified. The first setup resolves the image;
subsequent runs reuse that recorded image rather than pulling a changing tag.

Submit these stages in sequence, checking the smoke report before ``full``::

  sbatch _HPC/submission/augmentation-masif-setup.sbatch
  sbatch --cpus-per-task=4 _HPC/submission/augmentation-masif-audit.sbatch prepare
  sbatch --cpus-per-task=12 _HPC/submission/augmentation-masif-audit.sbatch smoke
  sbatch _HPC/submission/augmentation-masif-audit.sbatch full

All dataset workloads require Slurm and 2 GiB per allocated CPU. Full processing
uses eight workers with disjoint CPU affinities in a 32-CPU allocation. Scripts
copy their code to a job-specific snapshot before starting. No GPU is required.

Surface and chemistry controls
------------------------------

MSMS and PyMesh construct and regularise the whole unit's molecular surface.
Default pretrained geometry is retained: 1.5 Angstrom probe, 1 Angstrom mesh
resolution, 12 Angstrom geodesic patches and at most 200 vertices per patch.
The retained 200-vertex support can have a smaller effective radius; that radius
is reported. This differs from the earlier 10 Angstrom Euclidean SAS audit.

Reduce adds hydrogens without heavy-atom flips. Histidine hydrogen inventory is
fixed to the previously prepared source. Hydrogen records remain grouped with
their residues. PDB2PQR uses PARSE with both ``--noopt`` and ``--nodebump``:
``--noopt`` alone does not prevent heavy-atom debumping. Every stage verifies
heavy-atom inventory and coordinates within PDB rounding tolerance. PARSE atom
charge/radius identities must match across a parent's views; there is no pH
titration. These hydrogens belong only to diagnostic scratch, not the coordinate
release.

Up to 32 surface anchors follow corresponding residue frames. Projection must
remain on the same residue and within 2.5 Angstrom of the transported center;
unmapped anchors are reported explicitly. The adapter computes upstream fast-MDS
geodesic coordinates only at selected centers, avoiding the original quadratic
all-center theta allocation. Per-center seeds are deterministic. Preflight
compares features, radii and fingerprints with the unmodified upstream pipeline
on a closed test mesh. Smoke controls exercise a rigidly transformed fixed mesh,
an independently rebuilt rigid transform and a second MDS seed.

Measurements and artifacts
--------------------------

The report separates native-to-zero repair, selected-to-zero ANM, selected-to-
native total change, historical amplitude sweeps and zero-backend drift.
It reports fingerprint L2/cosine changes, global CA RMSD and molecular surface
distance, locally fitted CA/heavy RMSD and surface distance, and separate shape,
curvature, hydrogen-bond, electrostatic and hydropathy feature changes.

Retrieval uses the current repaired-zero gallery. Same-protein distractors must
have less than 10 percent surface-support overlap; other-protein distractors
exclude the same source sequence/complex group. A third pool uses the 32 other-
protein patches closest in mean chemical features. These are patch-identity
diagnostics, not the MaSIF paper's complementary-binding classification task.

Statistics weight parents equally, views equally within parents and patches
equally within views. Distance-margin uncertainty uses 2000 protein-level
bootstrap draws. The output includes per-patch CSV, mapping coverage, summary
JSON, PNG/SVG distributions, an RST report and indexed independently compressed
surface/fingerprint records. Stage times and peak memory are recorded per view.

Frame caches are reused only when input, configuration, code/environment hashes
and saved-result checksums match. Parent results checkpoint after each completion.
Published releases are immutable; failed attempts are retained separately before
an explicit rerun. Failures do not become empty or imputed fingerprints.

Patch radius sweep
------------------

Submit ``sbatch _HPC/submission/augmentation-patch-scale.sbatch`` to compare
6, 9, 12 and 15 A geodesic patches using the immutable full-pilot surfaces.
This requires eight Coulson CPUs and no GPU, repair, meshing or APBS rerun.
Override radii by appending ``--radii 6 9 12 15`` to the submission command.
The job snapshots its code and writes a new ``results/patch-scale-JOBID``
directory, preserving the original 12 A frozen-fingerprint results.

Surface graph edges are deduplicated before shortest-path computation. All
vertices within the requested radius are retained, without the MaSIF
200-vertex cap. Each corresponding anchor must be mapped in both views and
have at least four noncollinear reference patch residues at every radius;
otherwise it is excluded at all radii for that comparison. Exclusions,
actual radii, surface areas, vertex counts and residue counts are recorded.
The local fit uses the same reference residue identities in both views.

Outputs include patch-wise CSVs, parent-balanced quantiles, dataset strata,
a radius plot and RMSD/surface-distance distributions. Geometry uses both
global and local alignment. Chemistry uses locally aligned nearest surface
vertices and the saved normalized shape, hydrogen bond, electrostatic and
hydropathy features. Residue exposure TV measures redistribution of patch
surface area between residue identities; it is not solvent-accessible area.
Three-parent independently rebuilt rigid controls estimate preprocessing
variability separately. No frozen neural descriptor is recomputed at an
unvalidated patch radius.

Held-out surface-filter pilot
----------------------------

``sbatch _HPC/submission/augmentation-surface-filter.sbatch`` rebuilds one
random rigid orientation for each of the 30 audit parents. Before observing
those controls, 12 evaluation parents are fixed: four spread size bins per
dataset, split equally between typical and tail examples. The other 18 parents
calibrate the filter. All 24 historical repaired ANM candidates are evaluated
for each held-out parent, retaining existing geometry rejection decisions.

The filter uses uncapped 9 A geodesic patches. A patch is flagged when its
locally aligned surface distance, normalized shape/H-bond/electrostatic/
hydropathy RMSE, or residue area-fraction TV exceeds the corresponding
parent-balanced 95th percentile in calibration rigid controls. Missing mapped
anchors count as flagged patches. The allowed flagged-patch fraction is the
95th percentile across calibration parents. These are empirical noise
thresholds, not validated biological invariance limits. Held-out rigid-control
acceptance checks whether the calibration transfers to different proteins.

Three views are chosen from passing candidates by greedy farthest aligned
whole-chain CA RMSD, including the zero-displacement structure in the initial
reference set. Pairwise diversity and the full candidate decisions are saved.
No extra ANM candidates, repair steps, neural inference or model training are
performed. New surface/chemistry frames use exactly the existing MaSIF
preprocessing, with descriptor and polar-coordinate construction skipped;
three same-coordinate comparisons verify equivalence of the saved raw surface.

The pilot uses 24 workers with four disjoint cores each. Its ETA projects one
zero surface plus 24 candidate surfaces per full-data parent, weighted by the
ATLAS/PINDER/pKPDB population counts. The enriched length distribution remains
a limitation. ANM, preparation and repair costs are reported separately using
the earlier Coulson measurements; queues, data transfers and production quota
checks are outside the compute projection. Packed production storage remains
necessary: the pilot frame directories are unsuitable for a full release.

``augmentation-surface-filter-cpu.sbatch PILOT_JOB`` checks nine matched raw
surface rebuilds with one core per worker. Surface equality allows a bijective
vertex permutation while requiring the same coordinates, faces, normals,
chemistry, residue owners, areas and anchor mappings. Exact array order is not
required for a physically identical mesh. ``--verify-existing`` in the Python
entry point verifies already generated artifacts without rebuilding them.

After both jobs finish, submit ``augmentation-surface-filter-summary.sbatch
PILOT_JOB CPU_RESULT_JOB`` with suitable Slurm ``afterok`` dependencies. The
final report weights surface costs by dataset and recorded-length population
bins, and adds the earlier 500-parent weighted coordinate-generation costs.
It also simulates deferring electrostatics: a candidate failing the full allowed
bad-patch fraction on non-electrostatic features alone is rejected immediately;
remaining candidates are tested in greedy diversity order until three pass.
The pilot verifies that this produces exactly the exhaustive full-filter
selection. That cost estimate sums measured stages; it is not an end-to-end
benchmark of the staged implementation.

A one-core throughput advantage does not imply that every CPU can safely host
one worker. The heaviest-candidate probe (``--heaviest --workers 3`` in the CPU
entry point, with enough allocated memory) samples complete host/container/
subprocess RSS rather than only the bridge's Python process. Final worker budgets
apply a 25% worker margin and 10% memory reserve under the existing 2 GB/core
allocation rule. If four cores are needed per worker for memory, the report uses
the measured four-core speed rather than leaving the extra cores idle. Faster
memory-aware mixed scheduling has not been benchmarked. Uniform worst-case
worker reservations are conservative and do not establish a full-data bound.

Measured deferred electrostatics benchmark
-----------------------------------------

``sbatch _HPC/submission/augmentation-surface-staged.sbatch`` benchmarks the
actual staged implementation on the same 12 held-out parents. Every geometry
and electrostatic field is computed cold. The reference mesh and electrostatic
field are rebuilt once, then each geometry-valid candidate is meshed without
APBS. Shape, exposure, hydrogen bonds and hydropathy reject candidates that
already exceed the calibrated allowed fraction. Greedy global-diversity
selection then queries APBS as needed until three candidates pass or none remain.

Hydrogenated PDBs and meshes stay on node-local scratch. Deferred APBS uses the
retained hydrogenated PDB and the original double-precision meshing coordinates,
so it does not remesh, reprotonate, or interpolate onto rounded saved vertices.
Only selected heavy coordinates, receipts/timings and validation summaries are
persisted. Frozen input anchor targets are reused to match the earlier pilot;
exhaustive surface features are consulted only after the timed selection ends.
Canonical geometry, chemistry, charge inventories, candidate query order and
selected IDs must match the exhaustive pipeline. Missing electrostatic features
are explicit in the cheap-screen rows and cannot enter a full-feature decision.

Each parent gets four disjoint CPU cores. The report measures complete worker
process-tree RSS by phase, wall-clock time including bridge startup and local
I/O, and electrostatic query counts. It weights runtime by dataset/length
population counts, interpolating the unmeasured middle bin from its adjacent
bins. ANM/preparation/repair are added from the earlier 500-parent baseline,
which is separate from this existing-coordinate benchmark. Full-data storage
and high-concurrency I/O are not validated by this 12-parent run.

Ten-mode matched benchmark
-------------------------

``augmentation-modes10.sbatch FROZEN_CODE_DIRECTORY`` runs the same 12 held-out
proteins with ten modes, eight directions and 0.5/1/2 A target CA RMSD. The first
five modal coefficients and eigenvectors are retained from the original run;
five independent, inverse-square-root-eigenvalue-weighted components are added
before renormalization. Reconstructing all original five-mode targets after PDB
rounding verifies the pairing. Extra eigenvectors must be orthogonal to the old
basis and the lowest-five eigenspaces must agree.

Prepared atom inventories and systems, native translation, 100-step restrained
minimization, geometry gates, 9 A surface gate and 18-parent calibration remain
fixed. Local anchor offsets follow the corresponding five-mode candidate;
previously geometry-rejected candidates use the fixed-reference offset. Every
new candidate is evaluated; APBS is skipped only when non-electrostatic features
already make acceptance impossible. Unlike the selection benchmark, this does
not stop after three passing views. Geometry rejection counts as a failed
candidate in the denominator of 96 per amplitude.

The report compares passing counts, paired protein-bootstrap differences and
within-protein pairwise aligned CA RMSD. It does not infer population-wide rates
from this enriched pilot. General CPU nodes replace the unavailable AMD nodes;
wall-clock speed is not directly compared across the two hardware pools.

The ten-mode benchmark additionally repeats one original five-mode direction at
each amplitude on the replacement CPU pool (36 controls). Capped minimization
and remeshing can change borderline decisions, so this agreement is reported
alongside mode-count effects. For slow or failed parents, four candidate groups
can each replay the same repair sequence and evaluate disjoint surface subsets.
The merge requires exactly candidates 1--24 and matching mode directions.
Persistent non-finite curvature or other external surface errors are recorded
as unevaluable cases, with an upper bound showing their potential effect on the
passing rate; thresholds are never changed to make a candidate evaluable.

Experimental conformational reference
------------------------------------

``augmentation_conformation_reference.py`` compares saved five-mode, eight-
direction ANM candidates at 0.5, 1 and 2 A against independent experimental
PINDER apo/holo links for the same proteins. The ``prepare`` command resolves
links from the frozen PINDER index, downloads individual monomer PDBs with
SHA256 receipts, and verifies that the holo coordinates reproduce the saved
source (aligned CA RMSD <=0.05 A). No full dataset download is needed.

Sequence correspondence requires 100% identity over aligned resolved residues
and at least 90% coverage of each sequence. Whole-chain solvent-accessible
surfaces retain all atoms for occlusion; they are never cropped to common atoms
before surface construction. Patch neighbourhoods near unmatched atom
inventories, unresolved residues or backbone breaks are excluded on both
experimental states. A pair needs at least four comparable patches at each
radius. Length bins are interleaved with deterministic hash order and proteins
are deduplicated by UniProt, without selecting on observed displacement.

Reference surface anchors and alignment members are held fixed across views.
The patch radii are 6, 9 and 12 A Euclidean distances, independently of the ANM
amplitude; these are not the geodesic SES patches of the earlier MaSIF audit.
The inexpensive calculation uses 256 samples per inflated heavy-atom sphere.
An increased-density check uses 512 samples for the first three parents and
one fixed direction per amplitude. Rigid-motion and atom-permutation controls
verify correspondence and finite-sampling invariance.

The ``parent`` command reuses every geometry-valid saved candidate, reporting
global CA RMSD, local CA and surface displacement after global and local
alignment, atom exposure, exposed amino-acid composition, and residue-class
chemistry. Repair-only and candidate-versus-native comparisons accompany the
candidate-versus-repaired-zero analysis. No surface-preservation threshold is
applied. The report gives equal protein, then equal conformer, then equal patch
weight; it records exclusions and descriptive overlap with experimental
differences rather than calling overlap a biological validity test.
The summary distributions use the same measurable anchor intersection across
all geometry-valid views, with all raw rows and small-patch sample dropouts
saved separately. The report also shows total native-to-candidate changes;
candidate-versus-zero distances cannot be treated as directly equivalent to
raw experimental differences when source repair itself moves the surface.

Use ``augmentation-conformation-reference.sbatch prepare FROZEN_CODE_DIR``
followed by a ``parent`` job array with one task per accepted pair, then run
``augmentation_conformation_reference_report.py --output OUTPUT`` on a CPU
allocation. All workloads are CPU-only and use the existing compute guard.
Experimental structure differences include side-chain, crystal and resolution
effects as well as conformational motion. The comparison does not establish
thermodynamic accessibility or the quality of a trained representation.
