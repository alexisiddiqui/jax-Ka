# Stripped protein-only reference policy — 2026-10-04

**Latest amendment (2026-10-04): buffers use 15 Å training / 20 Å evaluation**, approved after the corrected PROPKA pilot. Glycans remain 20/25 Å; other ligands 15/25 Å; eligible metals 25/25 Å. See [buffer mask migration](15_buffer_mask_revision.md). Earlier 20/25 buffer decisions below are historical.

## Accepted glycan/buffer revision

Following the [glycan sensitivity pilot](12_glycan_sensitivity.md), the user
approved **20 Å training / 25 Å evaluation** exclusions for resolved neutral
glycans and eligible buffer/additive candidates. These are operational
uncertainty masks, not validated physical error bounds. Buffer eligibility still
requires the existing ligand chemical/contact screening; the radius does not authorize
removing covalently attached or metal-coordinating buffers.
Exposure and partner bridging are annotations: >=70% exposed and nonbridging
identifies the low-concern subset. They do not introduce new whole-entry
rejections for previously ligand-eligible buffers. All such buffer candidates
receive 20/25 Å, including buried or bridging instances. This preserves the
previously approved ligand-stripping scope while increasing their training mask.

For glycans, remove whole resolved trees and measure distance from all removed
heavy atoms. The pilot supports neutral sugars with checked N/O attachments;
charged, unsupported or ambiguous glycan chemistry is not automatically admitted.
Protein attachment residues must remain chemically valid after stripping.
Existing missing-region masks still apply. Other ligands retain 15/25 Å and
eligible monatomic metals/ions retain 25/25 Å; buried/coordinated metals remain
excluded.

**Implementation status:** completed in `glycan_buffer_policy.py` and new
`glycan-buffer-v2-pool1` / `glycan-buffer-v2-pool5000` campaigns. The existing
`stripped-components-v1` / `stripped-components-v2-5000` outputs and original
split proposal remain unchanged. The new final mask tables implement 20/25 Å
for glycans/buffers and give 689/151/512 structurally usable interface pairs
under the same train/validation/test assignments. Consumers must select the new
campaigns and read their final `site_masks.parquet`; no teacher runs or training
were launched. Execution and validation are recorded in
[the v2 recount](13_glycan_buffer_recount.md).

## Previous campaign policy

User-approved scope: remove eligible noncovalent ligands and exposed,
uncoordinated monatomic metals/ions; retain uncertainty masks and original
component coordinates. Labels describe the stripped protein-only system.
These operational radii are not validated bounds on ligand/metal-induced error.

| Removed species | Training exclusion | Evaluation exclusion |
| --- | ---: | ---: |
| Ligand | distance <15 Å | distance <25 Å |
| Eligible metal/ion | distance <25 Å | distance <25 Å |

Distance is the nearest native titratable functional atom to any original
removed heavy atom. All removed species contribute; passing one component does
not override another. Equal-to-cutoff sites pass. Original species identity,
coordinates, source hash and per-species radii are retained. No pKa imputation.

## Explicit screening choices

Only monatomic metal/ion species can pass. Require at least 70% exposed SASA
relative to the same isolated ion, using full nonwater assembly context,
1.4 Å probe, 1,000 points and element-specific radii. Require no declared
covalent/metal connection and no nonwater N/O/S/Se donor within 3 Å. Missing
connection records do not establish absence: the geometry screen still applies.
Multiatom metal complexes such as heme remain excluded. Unknown SASA excludes
the ion; buried/coordinated species remain excluded. Na/K/Cl are explicitly
inventoried instead of being silently discarded by the legacy cleaner.

Ligands with declared covalent/metal connections or protein contacts <1.9 Å
remain excluded. Other ligand stripping uses the agreed uncertainty masks,
including species ignored by PROPKA: they are not labelled validated by PROPKA.
Glycans and nonstandard residues remain separate exclusions pending a specific
conversion/deletion policy. Existing area, interface size and 1,500-residue gates
remain. These burial/contact thresholds are implementation choices stated here,
not additional empirical calibration findings.

## Masks and provenance

New campaigns use `stripped-policy init/scan/collect`; previous campaigns are
immutable. `removed_components.json` and `component_masks.json` live beside
each prepared structure. Final `site_masks.parquet` intersects component masks
with the existing provisional natural-gap clean+uncertain tiers. Natural-gap
tiers are retained as a column for separate evaluation. Long natural gaps remain
uncalibrated and use the existing conservative fallback. Incomplete functional
sites, artificial termini and unknown sequence coverage are not admitted.
Teacher coverage and sequence-disjoint split assignment remain separate checks.
Consumers must use the final mask table; the legacy `sites.supervision_mask`
alone is not this combined policy. No production split or training run is implied.

## Execution

- Current 1,450-pair pool: `campaigns/stripped-components-v1`, initializer 730443,
  preparation 730444–730459, collector 730461.
- Long-tail anchor reanalysis: job 730460 completed,
  `audits/radial-long-tails-v1/anchor-analysis-v2/`. Lengths 20/30, six complexes;
  no distance bands meet the previous ten-complex support requirement. No new
  long-tail cutoff is selected.
- Next 5,000 assemblies: `universe/coordinate-pool-v2-5000`, excluding the prior
  1,000 and preserving 3,750 general / 1,250 SAbDab sampling. Initializer 730434;
  download 730435–730442; index 730462–730477; index merge 730478.
- New-batch preparation: `campaigns/stripped-components-v2-5000`, initialization
  730479; preparation 730484–730499; collection 730500, all dependency-linked.

Compute-only jobs request 2 CPUs / 4 GB, exclude comp1400 and use the locked
submission wrapper enforcing <=400 queued+running user CPUs. Download failures
stay in the denominator. Candidate indexing checks the assembly size cap before
enumerating chain contacts. Clustering and a frozen combined split follow the
new preparation report; they are not yet completed.

## Combined split preparation

Both preparation campaigns completed without pipeline errors. The original pool
accepted 555 pairs (309 with training sites); the new pool accepted 2,693 pairs
(1,567 with training sites). These counts precede sequence grouping and teacher
coverage.

Job 730501 runs `split-audit` on both complete candidate manifests (8,683 pairs),
including rejected candidates that can connect sequence groups. Output:
`universe/combined-split-v1/`. It verifies mask hashes, unique site keys,
evaluation-mask containment and eligible natural-gap tiers, then runs MMseqs2
and reports full-chain versus CDR-mode connected components and usable sites.
The job requests 2 CPUs / 4 GB and excludes comp1400.

A production split is not yet frozen: the shared specification's maximum
component size (5%) must be checked and the forced-test experimental/PKAD-3
sequence inventory must be finalized. Missing antibody CDR annotations are
reported explicitly. No arbitrary reassignment of members of a connected
component is permitted to achieve target split sizes.

### Large-component diagnosis

Clustering job 730501 completed in 22m24s: 2,098 CDR-mode connected components,
with 2,603/8,678 assigned candidates in the largest (30%). Five candidates lack
CDR annotations; none has training-eligible sites. Diagnostic job 730502 reused
existing matches and reproduced the original largest-component count exactly.

| Graph included | Pairs | Largest component |
| --- | ---: | ---: |
| Whole assigned candidate pool | 8,678 | 2,603 |
| Accepted only | 3,247 | 881 |
| Training-usable only | 1,876 | 491 |
| General stratum only | 7,223 | 479 |

The giant component contains 1,310 antibody-stratum and 1,293 general-stratum
pairs. A sequence family of concatenated antibody CDRs touches 952 pairs within
it; another family containing antibody/TCR descriptions touches 279. Shared
antigens also contribute. The general stratum is therefore not an antibody-free
control. Removing rejected bridges does not eliminate the giant component.
These are diagnostic counterfactuals, not changes to the frozen-split policy.

Job 730503 additionally extracts PDB/chain/UniProt identifiers from the locally
installed KaML-CBtree train/test tables, retaining source hashes, into
`experimental-reservation-inventory.json`. This is a seed inventory, not proof
of complete PKAD-3 coverage or sequence decontamination. Reference-chain sequence
mapping and external-reference similarity searches remain necessary. The
[KaML repository](https://github.com/JanaShenLab/KaMLs/) documents its released
data splits. The PKAD-3 web endpoint returned a connection error during this
inspection.

[Barnase–barstar binding experiments](https://pubmed.ncbi.nlm.nih.gov/8494892/)
provide a literature-supported set-2 seed. Construct, condition and sequence
mapping still require curation; absence of a name match in candidate descriptions
does not establish absence of homologues. The complete set-2 inventory remains
unfinished. No production split has been frozen and no connected group has been
cut across train/test to satisfy quotas.

### Antigen-separated proposal (user approved feasibility assessment)

Job 730504 completed the proposal in 29 seconds using existing sequence matches.
Artifacts: `universe/combined-split-v1/antigen-proposal-v1/`. No pairs discarded;
no production split frozen. Annotated antibody–antigen pairs contribute only
antigen chains to the split graph. General pairs retain both partners; their
homologues share graph nodes with antigens. Antibody CDR grouping is separate,
used for novelty reporting at 30/50/70/90% identity with 80% bidirectional coverage.
The assignment targets 80/10/10 by candidate count, not usable-site count.

| Proposed split | Candidate pairs | Eligible pairs | Pairs with eligible interface sites |
| --- | ---: | ---: | ---: |
| Train | 6,947 | 1,539 | 1,195 |
| Validation | 868 | 153 | 92 |
| Test | 868 | 148 | 96 |

Train counts use training masks; validation/test use evaluation masks. There are
2,428 components; the largest falls from 2,603 to 541 candidates (6.2%, still
above the previous 5% target). Protein-node and detected sequence-edge separation
checks passed. The 500-pair test target is not met by this proposal.

Name screening flags 291 general-stratum pairs for antibody review. This is not
sequence-based annotation and cannot establish which chains are antigens. These
pairs conservatively retain full partner grouping. Their unresolved antibody
identities can hide antibody overlap in the provisional novelty classifications.

Among annotated antibody–antigen test candidates, 17 pairs have evaluation sites.
The number provisionally classified both-unseen is 2/5/15/16 at CDR identity
thresholds 30/50/70/90%; with evaluation interface sites, 1/2/6/6. These thresholds
are sensitivity analyses, not a chosen antibody family definition. The stricter
subset is too small for the planned standalone benchmark.

Only direct PDB matches to the experimental seed inventory are forced to test
(one component); external reference sequence matching and complete set-2 curation
are still required. Before freeze: resolve antibody chain annotations, choose
and document the novelty criterion, and address usable test-set size. Changing
split proportions or deliberately selecting large test groups requires an
explicitly recorded policy revision; this proposal has not done so.

Serialization verification initially caught missing nullable novelty columns in
the Parquet artifact. The first artifact is preserved in
`antigen-proposal-v1-incomplete-columns/`; job 730506 regenerated it with explicit
columns, and verification job 730507 passed for all 8,683 unique pairs and all
four novelty classifications. Final artifacts remain in `antigen-proposal-v1/`.
Reported split counts are unchanged.
