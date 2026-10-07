# Temporary pKPDB 5,000-structure pilot

2026-10-06 user scope: keep the next pilot at **5,000** structures; the dataset
will be redesigned later. Exclude ≥90% sequence-identity matches from training
to the frozen validation/test sets. Do not expand to 10k or change those splits.

## Leakage rules

Reserve all chains of every frozen validation/test assignment, including antibody
chains and antigens and held-out candidates outside the final usable benchmark.
A candidate PDB entry is excluded when any protein chain has ≥90% alignment
identity over ≥80% of the shorter sequence. This covers close fragments as well
as full-length matches. Exclude exact held-out PDB IDs independently.

Experimental reservations retain the existing stricter ≥30% identity / ≥80%
bidirectional-coverage rule. Collect the prior reference inventories, their parent
sequence proxies, the official delta systems, and every valid PDB/alternative PDB
listed in the local complete PKAD-R inventory; reserve all their protein chains.
Unresolved experimental sequences block release rather than silently passing.

Use the existing MMseqs2 binary with exhaustive search, permissive E-value
reporting, alignment identity calculation and coverage post-filtering. Persist
reference provenance, query/target FASTA hashes, commands, raw hits and the
per-PDB exclusion evidence. The 90% rule is a cross-split rule; no claim of
90%-clustered training diversity is made. Training component IDs initially group
exact sequence sets only.

## Mapping and cleaning

Select entries from the complete 121,294-PDB download in a deterministic shuffled
order (seed 20261006), admitting the first 5,000 that pass screening and have at
least one clean training site. Single deposited asymmetric units; protein-only
polymers; 30–1,500 declared protein residues. This is a bounded pilot cohort.

Keep historical scalar labels exactly as deposited. Match author chain,
residue number, insertion code where explicitly available, and residue type.
Do not realign/renumber labels to force a match. Ambiguous insertion codes,
duplicate label keys, nonfinite labels and unmatched residues are counted and
excluded from both arms. Do not infer historical prepared coordinates or rerun
PypKa. Multiple structural models are rejected; choose the first positive-
occupancy residue-coherent alternative, preserving conformer provenance.

Inputs remain strict backbone-only: residue identity, N–CA–C frames, Cα radius
edges, terminal and frame-validity flags; no side-chain-derived disulfide flag.
The input needs N, CA and C coordinates; entries lacking these are reported as
incomplete-backbone rejections. Missing side-chain atoms do not reject an entry.
Noncanonical peptide residues are omitted from the model and treated as sequence
gaps. No new coordinates are generated. Stored graph features never contain pKas
or native teacher energies.

Reuse the existing gap policy: calibrated visible anchors for short gaps;
uncalibrated long-gap influence stays flagged, without imputing a pKa. Retain
clean/uncertain tiers. Component exclusion radii (train/evaluation): ligands
15/25 Å, buffers 15/20 Å, glycans 20/25 Å, eligible exposed metals 25/25 Å.
Buried/coordinated metals and unsupported covalent components reject the entry.
The existing glycan and component chemistry gates are preserved.

## Artifacts and release

`pretraining/pkpdb-5k-v1` contains the label SQLite index, reference inventory,
sequence-hit evidence, per-entry backbone graphs, site masks, alternate-conformer
and gap/component provenance, audit histogram, final pilot manifest and checks.
The raw arm uses every unambiguously mapped finite label. The cleaned arm uses
the same 5,000 structures and identical inputs with per-site training masks.
Selection requires at least one clean site, so the comparison is conditional on
this cohort, not all of pKPDB. Both arms evaluate against the same existing clean
validation set. Teacher lineage/version differences remain a limitation.

The build stops on unexpected pipeline errors or unresolved references. It does
not relax exclusions to reach 5,000, and does not launch training. A 40-entry
smoke precedes production. Slurm requests 32 CPUs / 64 GB, excludes comp1400,
uses up to 16 preparation workers, and stays within the user-wide 400-core cap.

Recovery: the first full build stopped after 12,000 candidates and 2,399 accepted
structures because entry 3qza contained a component with no heavy atoms available
for distance/SASA screening. Such entries now receive the explicit conservative
rejection `component_without_heavy_atoms`. The resumed output is
`pretraining/pkpdb-5k-v2`; the original run remains preserved. Completed receipts
and sequence evidence are reused with a recorded parent protocol hash. No
sequence or uncertainty thresholds are relaxed.

## mask-all-v1 rebuild path (added 2026-10-06; not yet regenerated)

`src/pkabench/pkpdb_mask_all.py` rebuilds this pilot under the `mask-all-v1` component policy
(`component_mask_policy.py`; see the 2026-10-06 decision in `00_shared.md`). Cohort rules, leakage screening,
label mapping, gap tiers and inputs are unchanged; components never reject, and `sites.json` adds
`nearest_component_by_class`. It is a separate module so the `pkpdb-5k-v2` protocol hashes stay valid.
Threshold test (`audits/pkpdb-threshold-test-v1`, same 19,277 screened entries): entries with ≥1 training
site 5,222 → 9,312; training sites 216,111 → 290,778 (evaluation-mask sites 182,017). Recomputing the
pilot radii reproduces the pilot's clean-site counts for 5,191/5,222 entries; the other 31 differ by one
boundary site from 0.01 Å distance rounding in the test records. Regenerate into a new directory, e.g.
`python -m pkabench.pkpdb_mask_all $PKABENCH_RUNTIME/pretraining/pkpdb-5k-v3`.
