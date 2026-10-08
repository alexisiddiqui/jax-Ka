# Glycan stripping sensitivity pilot

**Subsequent buffer amendment:** buffers now use **15 Å training / 20 Å evaluation**; glycans retain 20/25 Å. The results below describe the earlier policy. See [the current buffer migration and recount](15_buffer_mask_revision.md).

User authorized a small PROPKA pilot on 2026-10-04, with error versus distance,
glycan SASA/exposure, and retained-site plots before changing the dataset policy.
Runtime: `_runtime/jax-Ka/pkabench/audits/glycan-propka-v1`.

## Scope and measurement

- Target 30 distinct deposited assemblies selected across observed glycan exposure
  from a bounded inventory of existing glycan-rejected candidates. This is a
  convenience sensitivity sample, not an estimate of dataset prevalence.
- Restrict the initial experiment to resolved neutral N/O-linked sugars with one
  unambiguous protein attachment per connected glycan tree. Exclude mixed
  nonprotein chemistry and ambiguous attachments; keep exclusion records.
- Prepare the protein using the existing PDB2PQR pipeline. Preserve its heavy
  coordinates across all predictions. Remove each complete resolved glycan tree
  separately, and all trees together. No structure prediction or relaxation.
- In separated A/B states, each glycan stays with its covalently attached protein
  partner. Compare both bound-state pKa and binding delta-pKa at matched sites.
- Verify PROPKA retains every sugar heavy atom, assigns interacting groups with
  neutral charges, and recognizes each glycan–protein bond. Repeat an unchanged
  reference to measure numerical reproducibility. Unsupported chemistry is not
  counted as a zero perturbation.
- Distance is the minimum from titratable functional atoms to any removed glycan
  heavy atom. SASA is calculated for the resolved whole tree in the selected
  protein/glycan context and in isolation, with a 1.4 Å probe and 1000 points.
- Report absolute SASA and exposed fraction. Use single-tree deletions for the
  exposure comparison; use simultaneous removal for unique-site retention and
  the main distance plot. Report 15/20/25 Å thresholds.
- Retention has two curves: glycan mask alone and glycan mask intersected with
  existing natural-gap masks. Counts require matched predictions, functional
  atom completeness and exclusion of artificial termini. Interface definition
  remains residue delta-SASA >10 Å².
- Bootstrap complexes, not individual observations, for descriptive p95 bands.
  Sequence-related complexes may still be correlated.

## Interpretation

These are **PROPKA sensitivity estimates**, not experimental errors or validated
physical bounds. PROPKA's finite interaction cutoffs constrain the distance
curves; zero changes beyond a radius cannot independently certify safety there.
Unresolved sugars and glycan-induced conformational changes are not modeled.
Production masks, split assignments and glycan rejection policy remain unchanged.

Execution uses Slurm compute nodes, excludes comp1400, requests 2 GB per CPU,
and uses the shared submission wrapper's 400-CPU budget check.

## Execution

Inventory job: 730548. Initial low-/high-exposure gate jobs: 730549/730550.
Completed results and plots are recorded below.

The first gate caught an atom-name alignment defect in the reused HETATM export:
PROPKA infers elements from the atom-name columns rather than the separate
element field. The new pilot corrects the formatting and checks every parsed
element. Earlier `propka-components-v1` results therefore need a separate typing
review before being used as ligand/metal cutoff validation; historical files
and operational mask choices have not been overwritten.

The native PROPKA control completed 29/30 structures (one PDB2PQR failure), jobs
730553–730560, collected by 730563. Inspection also found that native generic
typing can assign an amide-related sp2 type to a glycan ring atom at an N-linked
attachment. A matched arm in `audits/glycan-propka-ccd-v1` uses installed Biotite
CCD bond orders for sugar atom hybridization before PROPKA group extraction.
This is explicitly PROPKA with CCD sugar typing, not an unmodified PROPKA run.
It requires complete non-leaving sugar heavy atoms and agreement between the
observed intra-residue bond graph and the CCD. It does not change protein
coordinates, PROPKA energy parameters or experimental labels.

## Completed results

Both arms completed **29/30 complexes**; 1H3U failed PDB2PQR preparation in both.
The CCD arm contains **66 resolved glycan trees and 4,310 unique native eligible
matched sites**. Unmodified-reference repeats had maximum change **0.0**.
The native typing disagreed with CCD typing at 66 interacting atoms across all
29 complexes. Protein coordinate hashes agree between arms. CCD jobs
730566–730573, collection 730574, plots 730577.
Verification job 730580 matched 13,172 site–deletion observations between arms:
the typing corrections changed neither the measured pKa sensitivities nor the
retention counts in this sample. All 12 plot artifacts passed output checks.

All-glycan removal, before applying the existing missing-region masks:

| Minimum distance | Sites retained | Interface sites | Pairs with interface sites | Maximum bound pKa change | Maximum binding delta-pKa change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 Å | 4,310 | 443 | 29 | 1.87448 | 1.87448 |
| 15 Å | 3,607 | 374 | 29 | 1.87448 | 1.87448 |
| 20 Å | 3,105 | 315 | 28 | 0.05883 | 0.05883 |
| 25 Å | 2,638 | 244 | 25 | 0.02436 | 0.02142 |

The pooled 95th percentile of bound pKa change beyond 15 Å is 0.000829, but
the maximum demonstrates why that percentile alone should not set the mask.
There are six sites beyond 15 Å with bound-state or binding-shift changes above
0.1. All six are already excluded by the existing uncalibrated-gap mask. The
largest changes occur at TYR B31 (16.38 Å) and TYR C438 (16.20 Å) in 7WR9.
Thus they inform a stand-alone glycan rule, but are not current retained-site
errors. Details are in `outliers_beyond_15A.csv`.
Beyond 20/25 Å the pooled p95 is zero; native finite interaction cutoffs make
this unsuitable as independent evidence of physical safety.

With existing missing-region masks also applied:

| Glycan radius | Sites retained | Interface sites | Pairs with interface sites |
| --- | ---: | ---: | ---: |
| 0 Å | 1,000 | 101 | 9 |
| 15 Å | 864 | 83 | 9 |
| 20 Å | 755 | 69 | 7 |
| 25 Å | 655 | 55 | 7 |

The gap masks already exclude 312 interface sites as uncalibrated and 30 as
near-gap before any glycan radius is applied. Therefore glycan rescue alone
does not translate into retaining every candidate pair. These counts describe
this pilot and matched PROPKA coverage, not a production dataset expansion.

SASA plots use single-tree deletion while other glycans remain. They show
absolute bound SASA and exposed fraction separately at 15/20/25 Å. The
low-exposure (<30%) bin contains only one complex; exposure does not currently
support a reliable separate cutoff. Large-SASA bins also have limited support.

User accepted **20 Å provisional training / 25 Å evaluation** for resolved
neutral glycans and eligible buffers after reviewing the pilot. The implemented
buffer extension preserves existing ligand chemical eligibility; exposure and
bridging are annotations rather than new entry gates. Existing gap masks remain.
The 15 Å outliers argue against automatically
inheriting the ligand training radius for glycans. The buffer extension is an
operational choice, not a new buffer validation result. See the
[accepted policy and implementation status](07_stripped_reference_policy.md).
Existing production preparation outputs and split assignments remain unchanged.

## Artifacts

Primary plots use PROPKA 3.5.1 with CCD sugar typing:

- [Error versus distance](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/audits/glycan-propka-ccd-v1/plots/error_vs_distance.png)
- [Error versus absolute SASA](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/audits/glycan-propka-ccd-v1/plots/error_vs_sasa.png)
- [Error versus exposed fraction](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/audits/glycan-propka-ccd-v1/plots/error_vs_exposure.png)
- [Retained sites and pairs versus radius](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/audits/glycan-propka-ccd-v1/plots/retained_sites_vs_radius.png)

PDF and SVG versions sit beside the PNGs. The same directory contains
`bootstrap_summary.json` (400 complex-bootstrap replicates, seed 20261004,
minimum five complexes for intervals). Its parent contains `site_changes.csv`,
`distance_summary.csv`, `retention.csv`, `native_typing_disagreements.csv`,
per-state chemistry checks, and `report.json`. Native PROPKA control artifacts
remain in `audits/glycan-propka-v1`.
