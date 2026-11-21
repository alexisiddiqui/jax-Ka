Original-paper MaSIF batch builder
=================================

V5 antibody partner assemblies and validated surfaces
-----------------------------------------------------

Use ``--v5-antibody-root /path/to/_runtime/abdata-v5`` instead of
``--coordinates``. This reads the pinned v5 training/validation tables and
selects rows whose actual ``stratum`` is ``Ab/Ag``. The current inventory is
553 training and 129 validation complexes, or 1,364 isolated partner units.
These counts cover PINDER only. The combined v5 training setup also inherits
the unchanged pool-v4 pKPDB tables. The pKPDB entry point is described below;
the PINDER pilot is not a combined-dataset antibody run.
``--split train`` selects only training. ANARCI IMGT H/K/L domains identify
the antibody partner; exactly one partner must be positive. Ambiguous roles
fail explicitly. UNDEFINED UniProt IDs and alphabetical chain labels are not
used as role definitions.

``masif_partner_units.py`` encodes ``v5-ab-ag-partner-assembly-v1``. Each
partner's entire A/B residue map is exported as one surface/electrostatics
calculation, including all supplied heavy/light chains and gap fragments.
``--separate-chains``, bound complexes and legacy chain cohorts are rejected
for this mode. IDs are ``pinder-v5/<complex>/Ab`` and ``.../Ag``; records retain
the original A/B state, split/group, chain labels, source hashes and canonical
assembly residue indices. The current PINDER sources are dimers: a missing
heavy or light chain is not reconstructed from another structure. Partner
grouping cannot turn a single supplied Ig chain into a complete Fab.

New paper builds use the separate preparation policy
``terminal-oxt-mesh-coverage-v2``. The original MaSIF runtime and model hashes
remain pinned, while policy/source hashes distinguish these outputs from
the historical cache. This policy:

* Completes only absent C-terminal OXT for each explicitly exported free
  fragment, using a planar carboxylate and 1.25 Angstrom bond. Existing source
  atoms are retained in export; added atoms have recorded canonical IDs.
  Missing side-chain atoms are not silently invented or zero-filled.
* Checks external command exit codes, stderr, APBS output size/finiteness and
  vertex correspondence. Electrostatics remain PDB2PQR PARSE plus APBS using
  the original generated solver input.
* Removes collapsed, duplicate and zero-area triangles after the final
  vertex merge, removes unusable three-vertex islands, and validates raw
  MSMS coverage and source heavy-atom extent. Every raw surface vertex must
  lie within 3 Angstrom of the remeshed surface and every source heavy atom
  inside its bounding box expanded by 2 Angstrom. These numerical integrity
  thresholds do not establish physical or thermodynamic plausibility.
* If the original cleanup loses coverage, retries remeshing from raw MSMS
  without the destructive outer-hull step. The receipt records which path
  was used and requires the same coverage checks. Remaining errors are
  quarantined; nonfinite features are never replaced with zeros.

``masif-v5-abag.sbatch`` is the dedicated v5 entry point. It uses the same
bounded zstd-1 packed store and GPU queue as the existing builder. Publication
and merge require matching preparation and partner-unit policies; old and
new surfaces cannot silently share a release. Existing releases are immutable.
The earlier per-chain pKPDB release remains held pending its diagnostic repair
and quality review; this v5 run does not republish it.

pKPDB antibody/antigen assemblies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``masif_pkpdb_abag.py prepare`` pins both inherited pKPDB membership tables
and exports their unique chain sequences from the trusted, locally generated
v5 pool-builder snapshot. ``number`` scans every sequence with ANARCI IMGT
H/K/L (one HMMER CPU per process). ``inventory`` maps domains back to native
chain labels and canonical training residues. Missing numbering evidence for
an antibody-labelled chain is quarantined instead of labelling it antigen.
Ig-only assemblies have no resolved non-Ig antigen under this rule and are
excluded. Constant-only antibodies without a numbered variable domain, and
antibodies binding another Ig, require additional role evidence; this is a
conservative variable-domain-defined Ab/Ag cohort.

The unit policy is ``pkpdb-ab-ag-training-assembly-v1``. Chains form a graph
using native heavy-atom contacts within 5 Angstrom (at least three atoms of
one chain near the other). Each connected component containing both Ig and
non-Ig protein chains contributes two units. All its antibody chains are
surfaced together and all its antigen chains together. Disconnected copies
remain separate; multiple copies that contact remain in the same component.
This is an operational partition of the supplied training assembly, not a
claim that crystal contacts establish a unique biological assembly. Protein
oligomers and all H/L chains in the component are preserved. Nonprotein
ligands are outside the canonical protein surface, as in the training graph.

For example, pKPDB ``1i8i`` becomes Ab chains A+B and Ag chain C. ``2ybr``
has two contact components, each with its own grouped Ab and Ag units.
Source hashes, all partner chain labels, contact edges, split/group and exact
canonical residue IDs are pinned in ``inventory.json``. No absent chain is
reconstructed and no protein chain is split into separate fingerprint units.

Run ``masif_annotation_pilot.py --pkpdb-abag-inventory .../inventory.json``
or ``masif-pkpdb-abag.sbatch``. The inventory is mutually exclusive with the
coordinate and PINDER inventory modes. Per-chain expansion is rejected;
the same validated surface policy, model checks, packed zstd-1 storage and
publication checks apply. The initial inventory contains 40 training and
8 validation parent assemblies, yielding 41+8 Ab/Ag pairs (98 partner units).

``masif_pkpdb_release.py`` validates and compacts the full worker output into
one independent training-transfer store, retaining zstd level 1 and 2-GiB
shard limits. Every transferred record is decoded again and checked against
its source mapping, model/runtime profile, surface coverage, finite 80D
descriptors and triangle geometry. ``partner-pairs.json`` records each grouped
Ab/Ag pair and ``quality.json`` is the final QA marker.

Publication rejects failures by default. An explicit ``--quarantine`` JSON
can list reviewed failure IDs, reasons and the pinned inventory hash. It must
match all actual failures exactly; an additional unreviewed failure holds
publication. Both members of an affected pair are withheld together and
listed in ``quarantine.json``. Dataset membership tables are never rewritten.
The current native-source issue is validation antigen ``6ln2``: 416 missing
heavy atoms out of 3,597 expected (11.57 percent), exceeding PDB2PQR's 10
percent repair limit. This pair requires explicit reconstruction before it
can receive a complete paired annotation; the repair limit is not raised.

``masif_annotation_pilot.py`` now defaults to ``--preprocessing paper``. This
routes native units through the verified original surface source, FLANN 1.6.14,
PDB2PQR/PARSE and APBS, original MATLAB curvature and classical MDS, radius-
shrinking 200-vertex patches, original feature assembly and the finite original
paper checkpoint. It never falls back to the newer Python geometry. The old
backend is available explicitly as ``--preprocessing legacy-python`` for
historical diagnostics only.

The immutable runtime is installed by ``masif_paper_runtime.py install`` from
the verified source artifacts. Its manifest pins every scientific source and
dependency file, exported weights and the paired checkpoint hashes. Each batch
job verifies this manifest before starting. MATLAB's original per-file process
exit is disabled in a renamed batch entry point; numerical code is unchanged.
The original PLY serialization is retained, including its coordinate rounding.
Vertex/face correspondence, canonical residue indices, component coverage,
feature finiteness and descriptor provenance are checked explicitly.

MSMS is run on every disconnected expanded-sphere component rather than silently
following only one. Single-component units use the original function unchanged.
The molecular-surface union, original cleanup and whole-unit electrostatics
retain every represented component. Reduce may adjust side chains as in the
paper; this backend does not promise fixed heavy coordinates. Source residue
identity and mapping to the training assembly remain fixed.

Scope and bounded storage
--------------------------

The submission wrappers select current pool-v4 pKPDB training only and
``--separate-chains``. Metadata lists 101,767 chain units from 56,191 training
assemblies, including monomers. The subsequent canonical-chain inventory found
101,704 chains with protein residues and 63 without; the full-build cohort must
exclude those nonprotein chains explicitly. IDs are
``pkpdb/<assembly>/chain-0000`` etc.
Each record carries the original chain label, actual residue count, and
``canonical_residue_ids`` mapping local indices back to the assembly's training
graph. Metadata average chain lengths are used only for scheduling/sampling;
the stored record uses the verified actual count.

Canonical filtering can remove a listed nonprotein chain. This does not
renumber the remaining chain indices or invalidate sibling protein chains.
Export validates the selected index and source label inventory; requesting a
chain with no canonical protein residues fails explicitly.

PINDER A/B and whole assemblies remain selectable through ``--dataset``,
``--split`` and ``--whole-assemblies``; bound PINDER requires ``--include-bound``.
The default direct entry point samples a size panel. ``--full`` processes the
entire selected cohort; ``--only-id`` repeated selects explicit integration
cases. ``--shards N --shard-index I`` deterministically partitions disjoint
units across node jobs. The sample/config manifest pins the input cohort.

Surface grids, PDB/PQR/PLY, MATLAB matrices and patch tensors stay in node-local
scratch and are deleted after each batch (default 16 units). Only dense float32
80D descriptors, vertices/faces/normals, canonical ownership and four raw
surface feature fields enter the existing indexed release. Independently
compressed level-1 zstd records are packed into 2-GiB shards. There are no
per-structure permanent files. The existing ``AugmentationStore`` reader and
Isambard training transfer format remain unchanged.

CPU and GPU execution
---------------------

CPU inference is supported without a GPU. For one-node CPU/GPU operation,
request one GPU and pass ``--gpu-python /path/to/cuda/python``; the launcher
starts and stops one persistent consumer. Surface and MATLAB batches stay on
CPU. Real original-paper patches are submitted in chunks of at most 1,024,
and every structure checks three GPU descriptors against TF1 CPU inference.

For distributed operation, start independently allocated one-GPU consumers
using ``masif-gpu-consumer.sbatch``, all pointing to the same dedicated shared
spool. CPU jobs use ``--gpu-spool /shared/spool``. Consumers atomically claim
requests; one outstanding chunk per producer bounds temporary bytes and inode
count. Producers pre-create a pending reply before publishing each request and
poll its contents. Polling for a missing reply filename caused approximately
30-second NFS negative-cache delays on both remote CPU architectures; the
existing-file protocol reduced measured reply waits to about 0.3 seconds.
Chunks are deleted after replies. Model/profile mismatches, malformed
inputs, nonfinite outputs and missing GPU consumers fail explicitly. There is
no implicit CPU fallback. After all producers finish, the owner stops consumers
by creating the dedicated spool's ``stop`` file.

``masif-paper-build.sbatch`` supplies a training-only full-build invocation;
``masif-annotation-pilot.sbatch`` retains pilot selection. These scripts alone
do not launch a full run. Multi-worker MATLAB license concurrency and full-data
I/O still require measurement before assigning an all-node ETA.

Resume and publication
-----------------------

``--resume`` requires an identical cohort, worker count, batch size, runtime,
model and inference backend. Committed frames are verified before append;
uncommitted scratch is regenerated. Worker state is saved before each bounded
data checkpoint. Scientific failures are retained in the report and never
replaced with zero fingerprints. A published partial result fails the command
and is immutable; repaired inputs need a new output release.

Disjoint complete node outputs can be combined with
``masif_paper_batch.py --merge <common-root>/node0 <common-root>/node1 ... --output <common-root>``.
Node outputs must be descendants of that common root. Merge rejects overlapping
IDs, incomplete nodes and mismatched checkpoint/preprocessing profiles. It
publishes one verified index over the existing shards without copying their
payloads. Only the verified final manifest makes a release readable.

Integration results (2026-11-16)
--------------------------------

Four native pKPDB training chains completed with two CPU workers and one A40,
producing 14,378 fingerprints. Every structure's independent TF1 checks agreed
with GPU inference within 1.7e-6. The two 1ake chains map to disjoint canonical
indices whose union recovers all 428 training residues. The existing reader
verified framing/hashes and original-paper feature ranges. The four-unit cache
requires five reader files (two payload shards plus three root metadata files);
there are no permanent per-structure files. Node merge was verified without
copying payload bytes, and immutable publication was checked.

The default CPU backend and completed-run resume also passed. A native PINDER
partner with two disconnected molecular components retained both and generated
10,778 fingerprints; GPU/TF1 reference error was below 1e-6. The component-union
extension therefore avoids the historical single-component MSMS loss.
These are integration checks, not a full-data concurrency or runtime benchmark.

Distributed throughput benchmark (2026-11-16)
--------------------------------------------

Four size-stratified panels attempted 242 annotations on AMD EPYC, older Intel,
and comp1400 CPUs while sharing one A40. Nine further native annotations checked
the export and reply-protocol corrections, giving 251 attempts over 130 unique
protein chains. All 130 unique chains succeeded after the corrections. Packed
release hashes, canonical mappings, finite features and independent TF1/GPU
agreement passed. The benchmark GPU allocation was stopped after all producers
finished.

For 101,704 native pKPDB training protein chains, the corrected pipeline projects
about 28 hours with 99 preprocessing workers, or 35 hours including a 25 percent
planning margin. Capping nodes at their measured worker counts gives 73 workers
and about 47 hours with the same margin. These estimates include source export,
surface generation, PARSE/APBS, MATLAB, patch assembly, GPU inference and packing.
Batch size 16 is projected by amortising measured batch-size-eight overhead.
Corrected inference timings come from four native chains per remote CPU type;
the entire large panel was not repeated after the transport correction.

The GPU consumer reached approximately eight GiB MaxRSS during the pilot. Its
submission template now reserves eight CPUs/sixteen GiB to add memory margin.
CPU workers retain four CPUs/eight GiB each; the largest observed worker tree
was 4.71 GiB. One A40 appears sufficient for the projected CPU supply. Four-GPU
capacity is an extrapolation and also reserves additional host CPUs.

Projected packed payload is about 161 GiB, with approximately 84--182 reader
files after accounting for independent two-GiB writer shards and root metadata.
Recovery state and logs add files proportional to the worker count. The source
metadata's 63 chains without protein residues must be excluded from a full-build
cohort. Aggregate MATLAB/Statistics licence capacity could not be queried, so
the ETA remains conditional on sufficient licences and continued CPU/RAM
availability. Submission templates retain twelve-hour limits: a full launch
must override these limits or coordinate resumable segments for both producers
and GPU consumers. No full build was launched by this benchmark.

Raw panels, immutable source snapshots, per-case timings, probability weights,
native follow-ups, cache verification and the final resource model are under
``_runtime/jax-Ka/anm-pilot-v1/masif-paper-benchmark-v1`` in the workspace.
