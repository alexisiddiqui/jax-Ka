Targeted native pKPDB recovery
=============================

``benchmarks/masif_pkpdb_recovery.py`` creates source-pinned native inventories
from the complete legacy-cache audit and the original failure diagnostics.
The normal annotation runner accepts them with ``--native-unit-inventory``.
Membership, split, chain selection, input hashes and unit identity are checked.
This path also supports contact-connected antibody-only assemblies: all
supplied chains of each selected component share a surface, with canonical
residue IDs preserved. It does not create an antigen or absent antibody chain.

The November 2026 recovery selection contains 191 confirmed defective cached
chains, 65 terminal-oxygen retries, 23 mesh/MSMS retries, and 48 intact antibody
surfaces from 47 parents. The 676 incomplete-input failures remain withheld.
Two large-grid APBS outliers use a separate inventory and resource allocation.
The original source structures, cache frames and training membership are
unchanged.

The APBS outlier run uses 32 GiB and an explicit
``--external-command-timeout 7200``. The default remains 1800 seconds. Both the
container command limit and host stage deadline account for this setting;
receipts record the command limit. This changes the resource budget, without
changing the PARSE/APBS numerical settings. The outlier job owns its GPU
consumer so a delayed PB calculation cannot outlive a separate consumer job.

Topology diagnostics
--------------------

``masif_topology_audit.py`` reads the flagged cached meshes, measures edge
incidence and the fraction of patch centers within 12 Angstrom of problematic
edges. The shortest-path graph mirrors the directed-edge selection in the
pinned author's ``geodesic_dists_graph.m``. This is the surface patch radius,
independent of ANM displacement. Affected-neighbourhood fractions are locality
diagnostics, not estimates of incorrect fingerprints.

The replay panel crosses severity quartiles with residue-size bins and adds
extrema, healthy controls and highly fragmented terminal-preparation cases.
``masif_topology_replay.py`` compares full native reprocessing with the old
cache. A smaller panel repeats the exact cached meshes twice through the
author's MATLAB charts and frozen MaSIF model. The saved normalized chemical
channels are held fixed for those repeats. Curvature and normals are
recomputed on the saved float32 geometry; this rounding limitation is recorded.

Comparisons report matched fractions and descriptor cosine distributions
separately near and far from problematic edges. Original/current topology is
also reported. Current cleanup does not guarantee a manifold mesh, so two
similar outputs cannot certify the absence of a shared defect. The panel is
diagnostic and is not a population error-rate estimate. No model is trained.

Candidate validation and integration
------------------------------------

``masif_pkpdb_recovery_release.py`` independently exports every successful
native unit, rechecks source hashes and canonical maps, validates triangles,
raw MSMS coverage and native convex-hull extent, checks GPU/TF agreement, packs
zstd level 1 shards, then decodes and validates the actual packed output.
Failures and remaining topology review flags stay explicit.

Outputs are recovery candidates. ``integration-plan.json`` lists which chain
records each validated candidate would replace. It does not apply the overlay
or relabel old preparation profiles. Bulk legacy release remains held pending
topology and preparation review. Only compact candidate release directories
are intended for eventual transfer, rather than compute scratch or snapshots.

Active run artifacts live under
``_runtime/jax-Ka/anm-pilot-v1/masif-pkpdb-recovery-v1``; ``jobs.json`` and
``status.json`` identify the scheduled jobs. ``report.rst`` is written after
the independent candidate QA and topology comparisons. Main and outlier GPU
consumers stop automatically after their dependent work.
