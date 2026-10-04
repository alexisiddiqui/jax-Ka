# Glycan/buffer policy v2 — preparation and eligibility recount

**Subsequent buffer amendment:** buffers now use **15 Å training / 20 Å evaluation**; glycans retain 20/25 Å. The results below describe the earlier policy. See [the current buffer migration and recount](15_buffer_mask_revision.md).

User authorized implementation and recount after accepting 20 Å training /
25 Å evaluation for glycans and buffers. Source candidate universe remains the
existing 8,683 pairs. No downloads, reclustering, teacher predictions or split
reassignment are part of this update.

## Implementation

`pkabench.glycan_buffer_policy` writes new versioned campaigns. Previously
accepted protein preparations are reused through immutable file links; changed
component masks and provenance are new files. Glycan-rejected candidates are
reprocessed with whole-tree removal and the existing PDB2PQR/structural gates.
Unchanged prior rejections remain recorded in the denominator.

Glycans require neutral whitelisted identities, complete non-leaving CCD heavy
atoms, a checked intra-sugar bond graph and one ASN/SER/THR N/O attachment per
resolved tree. Every removed atom contributes to the distance mask. Unsupported
chemistry is recorded as a rejection rather than silently stripped.

Known buffer/additive candidates use 20/25 Å. Their previous ligand chemical
eligibility is preserved: covalent/metal-linked or <1.9 Å protein contacts stay
excluded. Exposure >=70% and no contact <=4 Å with both partners are annotations
only. A first smoke implementation inadvertently made these new entry gates;
it was corrected before full processing because the authorized change concerns
masking, not additional buffer exclusions. The first diagnostic is retained in
`campaigns/glycan-buffer-v2-smoke`; the corrected smoke is `...-smoke-r2`.

Other ligands retain 15/25 Å. Metals retain the existing 25/25 Å and
buried/coordinated-metal exclusions. Natural-gap masks, alternate-conformer
selection and protein partner definitions remain unchanged.

## Verification and execution

Three targeted tests passed (730618): radius boundaries/all-component masking,
glycan attachment and missing non-leaving atoms, and buffer bridging annotation.
Corrected smoke: 32 pairs, jobs 730619–730623 followed by collection.
Collection 730626 passed with no pipeline errors: 16/16 previous preparations
retained, 11/12 glycan-rejected candidates prepared, one still excluded for a
coordinated metal. Four unrelated prior rejections remain unchanged.

New campaigns:

- `campaigns/glycan-buffer-v2-pool1`, from `stripped-components-v1` (1,450 pairs).
- `campaigns/glycan-buffer-v2-pool5000`, from `stripped-components-v2-5000`
  (7,233 pairs).

Initialization: 730624/730625. Preparation: 730627–730642 (16 shards for pool 1)
and 730643–730690 (48 shards for pool 5000). Collections: 730691/730692.
Combined recount: 730693, output `universe/glycan-buffer-v2-recount`.

All scientific work runs through Slurm with 2 CPUs / 4 GB per job, excluding
comp1400 and respecting the shared 400 queued/running CPU cap. The final recount
will compare usable sites/pairs under the unchanged sequence-group assignments,
and flag newly activated prospective experimental reservations before any use
for training. Previous masks and split artifacts remain unchanged.

## Results

Completed without pipeline errors. All **3,248** previously accepted protein
preparations remain accepted and unchanged. **412** glycan-rejected candidates
now prepare successfully, bringing the total to **3,660**.

| Existing assignment | Usable interface pairs before | After | Interface sites before | After |
| --- | ---: | ---: | ---: | ---: |
| Train | 664 | 689 | 6,085 | 6,391 |
| Validation | 150 | 151 | 1,767 | 1,769 |
| Test | 500 | 512 | 5,844 | 5,912 |

Training gains **108** newly usable glycan pairs and loses **83** previously
usable pairs whose last interface sites fall inside the increased buffer radius.
The net gain is **25** training pairs. Validation gains one and test gains twelve;
neither loses a previously usable pair. Usable training sequence groups change
from **256 to 246** despite the pair-count increase: recovered examples and
lost examples are distributed differently across the existing groups. Validation
remains at 83 usable groups; test increases from 204 to 205. No rebalancing or
group reassignment was performed.

Across all prepared candidates, before benchmark-role/split filtering, glycan
rescue adds **12,014** usable training sites and the stricter buffer radius
removes **8,854**, a net gain of **3,160**. This broader count is distinct from
the split-specific interface counts above. Existing natural-gap masks continue
to limit how much of each newly prepared structure is usable.

The previously inactive prospective protein G–Fc reservation still contributes
**zero usable non-test interface pairs**. It does not block this update.

Independent verification job **730694** passed:

- All 3,248 reused protein content hashes and original geometry links agree.
- Evaluation masks for all **444,065** existing site records are unchanged.
- Existing training masks only become stricter, and every change occurs in a
  buffer-containing pair. Natural-gap tiers and interface annotations are unchanged.
- All 83 lost training interface pairs are due to buffer-mask changes; all
  108 newly usable training interface pairs come from glycan preparations.
- Component radii, unique site identities, evaluation-within-training mask
  containment, sequence groups and benchmark-role assignments pass checks.
- Original proposal hash remains
  `4a51bc4e8d8b21ef62bed308d99b8ae9761ce7102745ed2d81e79bb16472e212`.

These are structural eligibility counts before teacher coverage. The split is
not frozen and production label generation/training has not been started.

## Outputs

- [Combined recount](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/universe/glycan-buffer-v2-recount/report.json)
- [Independent verification](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/universe/glycan-buffer-v2-recount/verification.json)
- [Eligibility under unchanged assignments](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/universe/glycan-buffer-v2-recount/eligibility-with-existing-assignments.parquet)
- [Original-pool preparation report](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/glycan-buffer-v2-pool1/report.json)
- [Expansion-pool preparation report](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/glycan-buffer-v2-pool5000/report.json)

Each new campaign contains `structures.parquet`, `sites.parquet`, and the final
combined `site_masks.parquet`, with per-structure source/removal provenance.
Consumers must use the final mask table; legacy `sites.supervision_mask` alone
does not apply the revised glycan/buffer and natural-gap policy.
