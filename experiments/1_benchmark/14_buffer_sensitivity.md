# Corrected buffer/additive removal sensitivity

The pilot evaluated the then-current provisional 20 Å training / 25 Å evaluation buffer mask. The user subsequently approved **15/20 Å**; see [the migration](15_buffer_mask_revision.md). The previous generic component pilot used incorrectly aligned PDB atom names, which PROPKA uses to infer elements. Its buffer results cannot establish that this radius is necessary or sufficient.

## Authorized experiment

Run 30 distinct PDB assemblies in parallel, each requesting 2 CPUs and 4 GB on compute nodes, excluding comp1400. The existing runtime guard limits scientific execution to one allocated core; the second core remains reserved headroom. The submission wrapper enforces the user-wide 400-core ceiling.

Runtime: `/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/audits/buffer-propka-v2`.

Use existing prepared protein coordinates without relaxation or structure prediction. Correct PDB atom-name alignment, verify element parsing, require complete CCD heavy atoms, compare geometry-derived connectivity with CCD, and require neutral interacting groups and expected native atom types. Repeat the unmodified reference to detect numerical drift. Predict AB, A and B with buffers retained, with each buffer removed separately, and with all supported buffers removed together. Assign free-state ownership to the nearest partner and retain the bridging annotation.

The native PROPKA configuration ignores SO4, PO4, PEG and TRS. These species remain unsupported and are not counted as zero-effect observations. This bounded pilot covers complete neutral C/O alcohols and ethers: GOL, EDO, MPD, 1PE, PGE and PG4. Charged buffers and other chemistries require separate validated treatment. Other stripped components remain absent, so the comparison is conditional on the prepared protein context.

The inventory contains 518 candidate pairs with at least one in-scope buffer. Select a diagnostic sample by round-robin chemical identity, unique PDB assembly, prioritizing affected training pairs. Of 30 selected pairs, 18 lost all usable training interface sites when the buffer training mask increased from 15 to 20 Å. This convenience sample is not a prevalence estimate or an independent benchmark test set.

## Measurements

- Absolute bound pKa and bound-minus-free ΔpKa changes against nearest titratable functional-atom distance to removed heavy atoms.
- Bound buffer SASA and exposed fraction (bound/isolated), using fixed coordinates and the selected proteins as the environment.
- Sensitivity beyond 10, 15, 20 and 25 Å, with descriptive exposure bins.
- Sites and interface pairs retained as the exclusion radius increases, using the all-buffer deletion and the shared interface definition (residue ΔSASA >10 Å²). These are buffer-only counts; natural-gap and other-component masks are not applied.

PROPKA's finite Coulomb/burial/desolvation cutoffs (10/15/20 Å) constrain interpretation. These are model-sensitivity measurements, not physical error bounds. Repeated sites and components are correlated. Failure or ignored chemistry must never be interpreted as zero error. No production mask, split or label changes are authorized by the pilot results alone.

## Execution

Initialization: 730695. Chemistry smoke: 730696, passed. Parallel workers: 730697–730725. Collection and plots are dependency-controlled; final results will be recorded below after verification.

## Completed results

26/30 complexes passed chemistry checks: 121 buffer instances (50 GOL, 36 EDO, 14 MPD, 9 PGE, 7 PG4, 5 1PE), 2,742 unique matched native-eligible sites, and 15,317 observations including individual and simultaneous deletions. Seventeen successful complexes came from the lost-training-interface subset. Four were unsupported at the structure level: three incomplete PGE/PG4 instances and one EDO geometry/CCD connectivity disagreement. They contribute no zero-effect observations.

All tested buffers removed together, clearance measured to the nearest of those buffers:

| Exclusion radius | Retained sites | Interface sites | Pairs with interface sites | Maximum bound-pKa change | Maximum ΔpKa change |
|---|---:|---:|---:|---:|---:|
| 0 Å | 2,742 | 509 | 26 | 4.359 | 3.887 |
| 10 Å | 2,326 | 399 | 26 | 1.077 | 0.582 |
| 15 Å | 1,827 | 290 | 25 | 0.0290 | 0.0385 |
| 20 Å | 1,276 | 179 | 21 | 0.00208 | 0.00208 |
| 25 Å | 884 | 112 | 12 | 0 | 0 |

Distance is associated with a strong decline in model sensitivity. In this selected sample, increasing 15 to 20 Å removes 551 additional sites and 111 interface sites. This quantifies a tradeoff, not a recommendation to change all buffer classes. Zero beyond 25 Å is constrained by PROPKA's finite-range model.

For individual-buffer deletions beyond 10 Å, the pooled 95th percentile of bound-pKa change was 0.0211 for <30% exposed buffers, 0.00478 for 30–70% exposed buffers, and zero for ≥70% exposed buffers. Support was 29/49/43 buffer instances across 15/19/13 complexes. Rare >0.1 changes still occur in the highly exposed group, so exposure alone does not establish safety. SASA association is descriptive, with distance, chemistry and repeated sites confounded.

Individual deletions differ from simultaneous stripping: beyond 15 Å there were ten observations >0.1; beyond 20 Å four; beyond 25 Å one (maximum 0.12355). These must be assessed against the distance to *all* buffers before translating into an operational all-component mask. Both experiments are retained in the figures/data.

Unmodified reference repeats matched exactly. Verification checked 398 prediction input files for identical protein ATOM records, native source hashes, unique site/variant keys, monotonic retention, figure output, and the unchanged split proposal hash. Whole-complex bootstrap intervals (400 replicates) are in `verification.json`; they do not address sequence-family correlation or selection bias.

The combined `site_changes.csv` is authoritative for interface annotations, joined from the shared residue ΔSASA >10 Å² definition. Raw worker CSV interface placeholders from the initial run are superseded by this join. Predictions and distances were unaffected.

Figures: [individual deletion distance](../../../_runtime/jax-Ka/pkabench/audits/buffer-propka-v2/plots/error_vs_distance.png), [all-buffer distance](../../../_runtime/jax-Ka/pkabench/audits/buffer-propka-v2/plots/all_buffers_error_vs_distance.png), [SASA](../../../_runtime/jax-Ka/pkabench/audits/buffer-propka-v2/plots/error_vs_sasa.png), [exposed fraction](../../../_runtime/jax-Ka/pkabench/audits/buffer-propka-v2/plots/error_vs_exposure.png), [retained sites](../../../_runtime/jax-Ka/pkabench/audits/buffer-propka-v2/plots/retained_sites_vs_radius.png). PNG/PDF/SVG exports are provided. Runtime artifacts also include `inventory.json`, `manifest.json`, `report.json`, `verification.json`, and `individual_outliers_beyond15.json`.

At pilot completion the 20/25 Å buffer flags were unchanged; the subsequent user-approved 15/20 Å migration is recorded separately. Neutral alcohol/ether results do not validate sulfate, phosphate, TRIS or other unsupported buffer chemistries.

The outlier audit is complete: all ten individual-deletion observations >0.1 beyond 15 Å are within 20 Å of another tested buffer (nearest-any-buffer distances 2.65–17.07 Å), so the current union-of-buffers training mask excludes them. This includes the 28.63 Å individual-distance outlier, whose site is 17.01 Å from another MPD instance. The simultaneous-removal change at that site is −0.00319. Differences between single and joint removal demonstrate why the experiments must be distinguished; no mechanism for the remote individual effects was established here.

Collection completed as job 730726. Final figures completed as 730731, and final verification passed as 730732. Source protein geometry and split assignments remain unchanged. The pilot is complete; unsupported buffer classes remain an explicit limit.
