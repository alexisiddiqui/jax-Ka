# Structural dataset freeze v1

The user authorized resolving the protein G–Fc experimental reservation and freezing the current structural dataset. This freeze pins assignments, inputs, uncertainty masks and the current reference inventory. Method coverage, the production scorer and the experimental Set 2 labels remain separate work.

## Reservation and split rules

Use the completed buffer-15-20-v3 recount as the source. Reserve every sequence group with an existing protein G–Fc reference match in test, including all candidates in a group, regardless of current site usability. Existing test groups stay in test. This moves the previously training-assigned 41-pair group (29 direct reference matches) `c4293009afb208f2d2b0f`; its one usable training interface pair has no evaluation interface sites. The move therefore costs one usable training interface pair and adds no usable test interface pair. No balancing or reclustering is performed.

Recheck all existing sequence edges at ≥30% identity and ≥80% bidirectional coverage. General pairs use both partners; antibody–antigen pairs use the annotated antigen for the split graph. Recompute antibody CDR novelty at 30/50/70/90% thresholds against the revised training assignments. These are distinct annotations, not a claim that all antibody sequences are held out.

All available scoped reference matches and the official PKAD supplemental matches must remain outside training and validation. The accepted 1AXT H/P01865 exception to independent experimental scoring is preserved. Experimental Set 2 curation is incomplete: **new reference IDs fail closed for independent scoring** until checked against this frozen training set. A later overlapping system must be excluded from independent claims or require a new split version before training.

## Frozen inputs and consumer contract

Bundle: `_runtime/jax-Ka/pkabench/universe/structural-freeze-v1`.

- `assignments.parquet`: full candidate universe, fixed groups/splits, experimental reservations and refreshed CDR novelty.
- `site_masks.parquet`: current component/gap masks with split-aware `training_eligible` and `evaluation_eligible` columns. Apply the `interface` column additionally for interface-only scoring. Do not use legacy `sites.supervision_mask`.
- `structures.parquet` and `input-files.json`: prepared structures, actual AB/A/B hashes verified against preparation content hashes, plus student-input, site-annotation, removal, missing-atom and provenance hashes. Geometry stays in the existing campaigns; consumers must verify hashes before use.
- Campaign and reference snapshots, `report.json`, `verification.json`, and a manifest containing artifact/source hashes.

The approved buffer 15/20 Å, glycan 20/25 Å, other-ligand 15/25 Å and metal 25/25 Å policies are pinned, together with current gap masks. Existing campaign artifacts and old proposals are preserved.

The bundle is written in compute job 730736 and checked independently by job 730737, each 2 CPUs/4 GB, excluding comp1400. The independent verification receipt is stored beside the bundle to avoid modifying its hashed contents.

## Status

The freeze job completed. Frozen usable interface pairs are **778 train / 151 validation / 523 test**. The 41-candidate training group moved intact; only one member had usable training interface sites. A second matching group was already in test and is now explicitly tagged with the protein G–Fc reservation.

| Split | Usable interface pairs | Interface sites | All eligible sites | Usable groups |
|---|---:|---:|---:|---:|
| Train | 778 | 7,171 | 62,373 | 277 |
| Validation | 151 | 1,823 | 10,903 | 83 |
| Test | 523 | 6,158 | 43,093 | 205 |

Internal verification passed for 87,590 sequence edges, 625 scoped reference-match records, 3,660 freshly hashed protein preparations and 511,205 site masks. Group separation and mask/count consistency passed. CDR novelty was recomputed; at 30/50/70/90% thresholds the 126 usable antibody test pairs divide into both-unseen versus seen-antibody counts of 24/102, 32/94, 122/4 and 125/1 respectively.

Independent saved-bundle verification **passed**: 18 hashed artifacts and all 511,205 site masks checked; the two reserved groups contain 582 candidates and **zero training-eligible sites**. Of these, 541 candidates were already in test and 41 were moved from training. Manifest SHA256: `c458234d3fde5774eb853d0706a249e7b6a853036530b29e01e065086838a2e4`. Production remains disabled until the reproducible-preparation gate revision, ~50-pair method/mask smoke benchmark, and production scoring implementation are complete. Frozen structural eligibility does not promise that every method will return a label for every site; failures and coverage must be reported without altering assignments.
