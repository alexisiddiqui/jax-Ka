# Buffer uncertainty masks: 15 Å training / 20 Å evaluation

On 2026-10-04 the user approved “15/20 seems fine for buffers then” after the [corrected PROPKA sensitivity pilot](14_buffer_sensitivity.md).

Apply these radii to all chemically eligible buffer/additive candidates in the existing buffer list. Neutral alcohol/ether additives were tested; other buffer chemistries remain an explicit extrapolation. Simultaneously removing all tested buffers gave maximum bound-pKa / ΔpKa changes of 0.0290 / 0.0385 beyond 15 Å and 0.00208 / 0.00208 beyond 20 Å. Individual deletions had outliers beyond 15 Å, so this is an operational uncertainty policy rather than a guaranteed error bound. Exposure alone does not establish safety.

Glycans remain 20/25 Å, other ligands 15/25 Å and eligible exposed/uncoordinated monatomic metals 25/25 Å. Buried/coordinated metals remain excluded. Structural acceptance, chemistry gates, natural-gap tiers, protein coordinates, predictions, sequence groups and split assignments are preserved.

## Implementation

`buffer_mask_revision.py` creates new `buffer-15-20-v3-pool1` and `buffer-15-20-v3-pool5000` campaigns under `_runtime/jax-Ka/pkabench/campaigns`. Existing geometry and site annotations are linked immutably. Only buffer-bearing structures have component distances/masks recomputed. New removal and provenance records store the 15/20 Å policy. The authoritative output is each campaign's `site_masks.parquet`.

All computation runs in Slurm with 2 CPUs / 4 GB, excluding comp1400. Original campaign masks remain unchanged. Assertions verify identical site identities, no loss of previously retained sites, evaluation masks contained within training masks, unchanged gap/interface annotations and reused protein geometry. The subsequent recount checks unchanged structural acceptance and split/group/role assignments, and reevaluates prospective experimental reservations.

## Results

Completed successfully (migration jobs 730733/730734, recount 730735). All 3,660 accepted protein preparations were reused. Buffer masks were recomputed for 1,000 pairs, recovering 9,359 training-mask sites and 5,680 evaluation-mask sites across all prepared candidates before split/role filtering. No previously retained sites were lost.

| Split | Interface pairs before → after | Interface sites before → after | All eligible sites before → after | Usable groups before → after |
|---|---:|---:|---:|---:|
| Train | 689 → **779** | 6,391 → 7,176 | 56,891 → 62,502 | 246 → 278 |
| Validation | 151 → **151** | 1,769 → 1,823 | 10,636 → 10,903 | 83 → 83 |
| Test | 512 → **523** | 5,912 → 6,158 | 40,882 → 43,015 | 205 → 205 |

Recount verification passed: structural acceptance, group/split/role assignments and original proposal hash are unchanged; every retention count is nondecreasing. Counts precede teacher coverage and the split is not frozen.

**Experimental reservation review:** one newly usable training pair, `9aec54435ca500d1`, belongs to prospective protein G–Fc family `c4293009afb208f2d2b0f`. It has 5 training interface sites (zero evaluation interface sites). This family was previously inactive. Resolve its experimental reservation before production training/freeze; the migration does not silently move groups or claim Set 2 independence. Both new campaign reports retain `production_allowed: false`.

Artifacts: `_runtime/jax-Ka/pkabench/universe/buffer-15-20-v3-recount/report.json`, `eligibility-with-existing-assignments.parquet`, and `prospective-reservation-review.json`. The `versus_buffer20_25` report section is the direct comparison above; the older `by_split` section compares with the original pre-glycan proposal.


**Subsequent freeze:** the protein G–Fc reservation is resolved in [structural freeze v1](16_structural_split_freeze.md). The complete 41-candidate group moved to test (29 were direct matches), yielding 778/151/523 usable interface pairs. The recount above remains the pre-reservation result.
