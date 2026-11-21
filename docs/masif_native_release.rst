Native MaSIF release for the current v5 pools
============================================

``benchmarks/masif_release_plan.py`` selects audited legacy pKPDB frames and
validated current Ab/Ag, intact-antibody and recovery frames. Existing pool
membership files and source caches are unchanged. The release has an explicit
availability mask: an assembly is usable only when every required surface is
available. A failed required partner withholds the whole assembly.

Current coverage
----------------

The November 2026 selection contains 101,235 dense surfaces from 56,277 complete
assemblies. It is a partial-pool release:

=================  ================  =================
Dataset / split    Covered           Membership
=================  ================  =================
pKPDB training     55,589             56,191
pKPDB validation   7                  800
PINDER training    553                28,352
PINDER validation  128                800
=================  ================  =================

PINDER coverage is the completed antibody/antigen subset. Ordinary PINDER
structures and most pKPDB validation structures still need annotation.
This availability mask does not redefine the benchmark validation population.
Training experiments must declare the covered subset rather than imply full
pool coverage.

Intact antibody surfaces replace the selected heavy/light chain surfaces.
Antigen partner assemblies remain separate, with supplied chain copies and
canonical residue indices preserved. The planner checks v5 PINDER membership
hashes and the unchanged inherited pKPDB membership hashes. PINDER raw
membership residue counts can include residues omitted by the canonical
protein topology; the release uses the production topology's actual residue
order/count. The raw membership count is retained separately.

Legacy preparation and review
-----------------------------

Every retained legacy frame passed the complete source/geometry/field audit.
Confirmed defective frames are excluded unless replaced by validated recovery
outputs. Nonmanifold-edge flags remain explicit and are not silently converted
into claims of manifold geometry. The stratified replay panel did not certify
every flagged record. Legacy frames retain ``original-v1`` preparation tags;
current frames retain ``terminal-oxt-mesh-coverage-v2`` tags. They share the
same pinned original-paper MaSIF model/runtime, with 80-dimensional descriptors.

The two largest preparation extrema were three- and seven-residue chains.
Their independently validated current outputs replace the old frames. Actual
added-OXT positions were used to review terminal-local descriptor changes.
The seven-residue case's current mesh has no nonmanifold edges; the
three-residue case still has two, explicitly recorded. Agreement of two
preprocessing runs cannot rule out a shared surface defect.

Packing and reading
-------------------

``masif_release_pack.py`` copies the existing zstd level 1 frames byte for byte
into bounded two-GiB shards. Each copied frame is hashed and decoded, checked
for finite arrays and the pinned scientific profile, and checked against its
original preparation receipt. Actual canonical residue ranges must form a
complete disjoint partition for every selected training assembly. The release
manifest is written only after all checks pass.

The 170.36 GB selection uses 80 payload shards and seven metadata files.
Compute scratch, worker checkpoints and source snapshots are not transfer
payloads. The transfer helper supports both native-record releases and the
existing augmentation feature-cache format, with quota checks, resumable
staging, file-hash verification and atomic publication.

``pkatrain.masif_store.MaSIFStore`` uses the existing indexed zstd reader:

.. code-block:: python

   from pkatrain.masif_store import MaSIFStore

   with MaSIFStore(release_path) as features:
       available = features.assemblies(dataset="pkpdb", split="train")
       item = features.load_assembly("pkpdb", available[0]["cid"], split="train")
       for surface in item["units"]:
           descriptors = surface["arrays"]["descriptors"]
           residue_ids = surface["surface_canonical_residue_ids"]

The reader rejects held assemblies and split mismatches. It verifies canonical
partitioning, including joined antibody chains and disjoint PINDER partners.
It exposes the dense fingerprints without inventing per-residue pooling or
changing the training model architecture.

Isambard checks
---------------

``masif_loader_pilot.py`` checks assembly lookup, canonical mapping and optional
GPU uploads of actual fingerprints. ``masif-loader-isambard.sbatch`` requests
exactly one GPU, 72 CPUs and a 15-minute maximum. A small size/profile pilot is
transferred and tested first; the full transfer depends on that pilot passing.
A subsequent sampled lookup opens the full release and measures its index
startup and loading cost. The two GPU jobs run sequentially.

Lookup and upload timings are not training throughput measurements. The OS
page cache is not evicted, and the small pilot is not representative of
full-dataset random I/O. No training job is started by this workflow.

Artifacts and scheduler/transfer status are under
``_runtime/jax-Ka/anm-pilot-v1/masif-native-release-v1`` on Coulson. The remote
release is at the corresponding directory under
``/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/anm-pilot-v1`` on Isambard.
