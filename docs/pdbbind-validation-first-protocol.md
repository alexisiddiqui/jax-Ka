# PDBbind validation first for ATLAS + PINDER + pKPDB

Decision recorded 2026-10-12. This amendment applies to the next combined
ATLAS + PINDER + pKPDB training release. Existing dataset contents, membership,
splits, nested portions, loaders and release receipts remain frozen for initial
testing. This document does not activate a new split or objective.

## Selection order

1. Curate high-quality PDBbind protein–protein measurements with both an
   experimental Kd and a numerical binding-assay pH.
2. Reserve a portion of those measurements for validation, stratified jointly
   by assay pH and binding affinity, with related examples kept together.
3. Freeze that validation membership and its protected sequence components.
4. Build the subsequent ATLAS, PINDER and pKPDB splits against that shared
   protection set, together with existing independent test/reference exclusions.
5. Freeze the combined release before training, then derive nested training
   portions from its training membership. Every portion uses the same validation.

PDBbind supplies the experimental validation anchor; ATLAS, PINDER and pKPDB
remain the three constituent training datasets. Additional validation examples
from those three sources are selected after the PDBbind anchor, subject to the
same global separation rules. Dataset source is not a boundary against leakage.

## High-quality paired measurement eligibility

Each eligible measurement must have:

- A traceable primary measurement reference and a numerical equilibrium Kd
  with original units, value and any qualifier preserved.
- A numerical **assay pH** explicitly linked to that Kd measurement. Do not
  substitute crystallisation, structure-deposition or unrelated experiment pH.
- Verified receptor/partner chains, measured constructs, sequence and mutations;
  the structure must represent the measured interaction sufficiently for the
  intended model. Parent PDB identity alone does not establish this match.
- Preparation and feature eligibility under the release's declared QC rules.
- No unresolved affinity/condition conflict or construct mismatch. Multiple pH
  measurements are separate records only when their individual Kd values and
  conditions can be linked unambiguously.

Record temperature, assay method and uncertainty when available, including a
missingness flag. Do not invent missing conditions or convert qualitative
"neutral pH" to a numerical target. For the initial high-quality anchor, reserve
approximate or inequality-valued Kd for separate review rather than treating
them as exact regression targets. Ki and IC50 are not Kd labels for this anchor.

The current v2020 index audit found 32 parent PDB IDs with numerical pH in
free-text notes: 30 affinity/assay-note candidates and two crystallisation notes.
Only 28 of the 30 affinity/assay-note candidates have an exact-form Kd field;
these are candidates, **not 28 verified high-quality measurements**. Some notes
have conflicting affinity values or multiple conditions. Primary-reference
curation may both reject candidates and recover additional paired measurements.
The broad pH metadata annotations already collected for PINDER/pKPDB do not by
themselves establish binding-assay conditions for PDBbind affinity labels.

## Joint stratification and leakage protection

Express affinity as `pKd = -log10(Kd / 1 M)` after validated unit conversion.
Use joint strata of numerical assay pH and pKd, rather than linear Kd. Declare
the validation fraction or target group count, bin edges, sampling seed and
sparse-stratum handling in the selection manifest **before sampling**. No
fraction or bin edges have been selected by this amendment.

Select whole leakage groups, not individual rows. Repeated measurements,
conditions, mutations and equivalent structures of the same interaction stay
together; homologous examples connected by the release's sequence rules also
stay in one component. Group assignment precedes stratified sampling. Report
the achieved distributions and group counts; a sparse eligible population may
not support every pH × pKd stratum or a precise requested row fraction. Never
split a component to fill a bin, relax quality to fill a quota, or choose bins
using model performance.

Protect the selected PDB IDs, constructs, chain pairs and sequence components
across **all three** datasets. Exact parent/construct/chain matches and homology
matches must be audited, including fragment and antibody cases. The current
ATLAS sequence rule connects chains at >=70% identity with >=80% coverage of
both chains, or >=90% identity with >=80% coverage of either chain. Preserve
that conservative rule in the future combined release unless a separately
recorded amendment changes it; retain the existing experimental-reference,
fragment and antibody/CDR protections as additional exclusions.

Candidates excluded by an existing protected test/reference set cannot be
reassigned to this validation anchor. After freezing the anchor, allocate any
additional validation groups and then training groups globally across ATLAS,
PINDER and pKPDB. No validation component may contribute a training example,
teacher-supervised training example or auxiliary training label in any source.
Unused PDBbind measurements are not automatically training data: their use
requires eligibility and membership in the permitted training partition.

## Activation and evaluation

Publish a new combined split version; do not overwrite existing release files.
Its manifest must include source versions/hashes, measurement evidence, quality
decisions, sampling configuration, component assignments, membership hashes and
per-source exclusion reasons. Verify zero train/validation/test identity and
component overlap, report pH/pKd coverage, and check that all nested portions
share the frozen validation membership. Preparation failures remain documented
as exclusions; replacements must be deterministic and completed before freezing.

The current splits remain usable for initial testing. A future checkpoint
trained on earlier membership may already have seen the new validation structures
or homologues, including through teacher supervision. Audit that exposure;
repartitioning alone does not remove it. Use training from unexposed initial
weights for claims of independent evaluation, or disclose the exposure and
report the result separately. Keep independent test sets out of selection.

Experimental Kd validation and teacher pKa validation measure different tasks.
The existing pKa heads and losses are not changed by this amendment. Before
using the anchor for model selection, define a supported affinity readout or
head, its pH conditioning, validation metric and selection rule. Report pKd
error separately from residue pKa and paired-state errors. An affinity head or
auxiliary loss is a subsequent implementation, not enabled here.

## Required work before the new combined release

Curate paired assay-pH/Kd evidence; freeze the group-level stratified anchor;
audit cross-source sequences and prior checkpoint exposure; construct and verify
the new ATLAS + PINDER + pKPDB membership; then activate the versioned release
and its explicit evaluation contract. No current dataset or split migration is
authorized by this protocol-only update.

## Separate pilot reservation — 2026-10-12

The user selected reservation of all eligible candidates for a small pilot,
rather than a 20% sample of the sparse index-note population. The separate audit
artifact is `pdbbind-validation-pilot-v1`; it is not an active training split.
Initial index screening reduced 28 numerical pH/Kd candidates to 21. Partial
primary-reference review holds two more out (unverified assay pH versus crystal
pH, and a measured-construct mismatch), leaving 19 provisional reservations.
None is yet released as a fully verified high-quality evaluation record.

Coverage reporting uses pH boundaries 6.5/8.0 and pKd boundaries 7.0/9.0:
five candidates are at pH <6.5 and fourteen at pH 6.5–8.0; none is above 8.0.
All candidates are reserved, so these bins do not drive row sampling. A
conservative local all-polymer-chain alignment gives ten provisional components;
verified partner and antibody/CDR mapping remains required. Five candidates
already occur by parent PDB ID in current pool-v4 training membership, and one
occurs in PINDER validation. Those facts do not modify existing membership.
Primary-reference/construct curation, cross-source sequence exposure and feature
eligibility must be completed before evaluation readiness is asserted. This
pilot can support preparation work now; a broadly representative independent
validation set requires additional verified measurements and the future combined
release protections.

## Companion Kd-only cohort and validation pH metadata

`pdbbind-kd-only-v2020` is a separate index-derived evaluation cohort containing
all 2,618 original protein–protein Kd entries, with no pH requirement. It
includes the paired pilot as an overlapping subset. Retain the original value,
unit, comparator and notes alongside molar Kd and pKd. The 2,524 exact-form
records, 64 approximate records and 30 one-sided bounds must not be scored as
one undifferentiated point-regression target. Inequalities reverse under the
negative logarithm. An exact-form record is not automatically a verified
high-quality measurement; known note conflicts and measured-construct mismatches
are flagged separately. This cohort does not establish independence from the
current training pools and does not create a new train/validation partition.

Attach pH to existing validation membership through versioned
`annotations/validation-ph-v1` sidecars and annotated table copies, without
editing original split files, features or labels. Current pool-v4 PINDER and
pKPDB validation must be covered explicitly; the older experimental-ph-v1
annotation release covered pool-v3 and the retired PINDER 400 cohort. Include
the current ATLAS validation and available frozen benchmark validation views.
Every sidecar row retains its source membership ID and explicit target/mask,
quality, provenance and missingness fields.

Deposited crystallisation, NMR-sample or EM-buffer pH is **structure-condition
pH**, not the binding-assay pH of a Kd measurement. Keep those target kinds
separate. ATLAS predicted structures have no measured structure or assay pH
in this release: preserve their rows with null pH targets and false masks.
Missing, conflicting, ranges-only and notes-only evidence remains masked under
the existing strict experimental-ph-v1 policy. Metadata availability does not
enable a new auxiliary loss or alter validation membership.

The first validation-ph-v1 build completed in Slurm job 7343141. Strict
structure-pH eligibility is 605/800 current PINDER validation rows, 578/800
current pKPDB validation rows, 1,840/2,618 Kd-only PDBbind rows, 181/284 full
benchmark validation rows, 102/142 active benchmark rows and 303/400 retired
PINDER rows. ATLAS retains all 2,000 frozen validation rows with missing pH and
false masks. Every original membership/source hash was checked unchanged during
the build; exported table order, target/mask consistency and artifact digests
were checked independently after transfer. These counts describe deposited
structure conditions and do not enlarge the verified paired assay-pH/Kd set.
