# Missing-residue supervision and pKPDB coverage audit

User-approved decision, 2026-10-03. This supersedes the earlier no-radius policy
for downstream training/evaluation selection, while preserving inclusive
PDB2PQR preparation and the completed diagnostic campaigns.

- Training targets must be at least 10 Å from missing-residue regions.
- Evaluation targets must be at least 20 Å from missing-residue regions.
- Apply the same thresholds to terminal and internal missing residues. Do not
  apply the earlier exposure-dependent radius reduction automatically.
- Keep successfully repaired atoms eligible. Missing whole residues receive no
  invented labels; artificial termini remain excluded from native targets.
- These are provisional empirical thresholds, not guarantees of zero error.
  The deletion experiment measured distance to known deleted atoms. For naturally
  absent residues, record the uncertain region and distance method explicitly.
  Initially use the existing conservative contour-reach envelope, considering
  only whole-residue gaps, and report its supervision cost separately. Do not
  describe envelope clearance as measured atom distance.
- Preserve frozen teacher inputs/predictions. Write downstream mask overlays;
  do not recompute historical predictions merely to change selection masks.

## pKPDB–PDB–AFDB audit

Compare original PDB residue coverage with the corresponding pKPDB-prepared
coordinates. Match AFDB entries through UniProt and sequence as a surrogate for
identifying absent sequence positions. Original-versus-prepared coverage is the
evidence for whether missing residues were rebuilt; AFDB agreement alone is not.
Distinguish missing residues inside the experimental construct from sequence
outside the construct, including engineered deletions, tags and isoforms.

Start with a bounded, reproducible sample of overlapping entries. Preserve URLs,
download hashes, sequence/chain mappings, missing positions, atom-only repairs,
whole-residue additions and failures/ambiguous cases. Verify that retrieved data
are precomputed pKPDB records; avoid endpoints that silently launch new predictions.
If prepared coordinates cannot be retrieved, report that limitation rather than
claiming a residue-level audit from the methods description.

No structure reprediction, loop reconstruction, AFDB coordinate substitution,
or label transfer. AFDB structures/data are deferred to backbone-only tuning.

All downloads, parsing, analysis and tests run in Slurm compute allocations.
Exclude comp1400; cap all user pending/running CPU requests at 400; request 2 GB
per CPU. Runtime artifacts and environments live under `_runtime`.

## Implementation and initial retention

`pkabench mask-policy --campaign CAMPAIGN --out NEW_OVERLAY` writes a hashed
`site_masks.parquet` and a per-complex retention report. The downstream
`pkabench.mask_policy.select_sites(campaign, overlay, split)` selects `train_mask`
for training and `eval_mask` for validation/test, validating source hashes first.
Split membership and successful paired teacher coverage must also be applied;
these masks do not define train/test splits or fabricate missing teacher outputs.

On the 296 accepted new-pool pairs, conservative envelope clearance retains
8,803 training sites (139 complexes) and 6,831 evaluation sites (104 complexes),
from 35,375 eligible native sites. Corresponding interface counts are 1,111 and
829. On the older 67-pair pool, retention is 1,411/937 sites in 29/22 complexes.
These substantial exclusions come from uncertain missing-region envelopes as
well as the radius; they are not equivalent to the known-deleted-coordinate
retention curves. Do not claim the experiment validated these envelopes.

Mask/mapping and existing supervision tests: nine passed on a compute node.

## Initial coverage audit result

`audits/pkpdb-afdb-coverage-v2/` contains the 12-entry bounded audit, download
receipts, original PDB coordinates, SIFTS mappings, AFDB metadata/coordinates and
per-entry outcomes. Eight entries returned precomputed pKPDB data, three returned
no precomputed data, and 1p9i returned HTTP 500. Five entries have sequence-verified
AFDB coordinates covering mapped missing positions. Assembly-copy names were
checked against original author chains and matching construct sequences;
unmapped positions and ambiguous matches remain explicit.

The inspected server `routes_direct.py` exposes a query-only `/query/<idcode>`
endpoint. Its `/pkas/<idcode>` response links to original RCSB coordinates and
can launch pKAI for missing entries, so this audit did not use that endpoint.
The prepared-output endpoint requires a submission identifier; it is not a
historical pKPDB coordinate export. No historical prepared coordinates were
retrieved. Therefore the whole-residue repair question remains unresolved at
entry level. Completing it requires the original/prepared coordinate pairs from
the pKPDB archive (the deposited database code has `pdb_file`/`pdb_file_hs` fields)
or an equivalent documented read-only export. No new prediction was requested.

## Crystal comparison before further mask decisions

The next comparison uses all eight confirmed pKPDB entries from the initial
audit: 1ciq, 1ied, 1jk6, 1k04, 1kql, 1mbv, 1ocw and 1or7. Run fresh PROPKA
and PypKa on the associated original crystal coordinates, with shared PDB2PQR
atom completion and no whole-residue reconstruction. Match stored and fresh
absolute pKa by author chain, residue number, insertion code and group.
Retain all matched sites: do not apply the 10/20 Å masks to this comparison.
Missing-region envelope clearance is only descriptive; the known-deletion
experiment does not validate these conservative envelopes.

Outputs: `_runtime/jax-Ka/pkabench/audits/pkpdb-crystal-compare-v1/`, including
per-entry preparation/method outcomes, site comparisons and aggregate errors.
Jobs 730330–730337 run the eight entries; dependent collector 730338 writes
the comparison summary. Each requests two CPUs and 4 GB, excluding comp1400.
This is an initial eight-entry audit, not the full pKPDB. Historical preparation,
method versions/settings, omitted nonprotein context and current PypKa's
−2 to 16 pH range remain confounders; agreement alone cannot establish whether
historical missing residues were reconstructed.

## Observable-anchor calibration (next analysis)

Replace worst-case envelope reasoning with measured diagnostics before changing
production masks. `expanded-radial anchors` reuses the completed deletion study:
distance is the nearest target functional-group atom to backbone N/CA/C/O atoms
of the visible residue beside a terminal deletion, or either visible flank for
an internal deletion. Terminal plots separate N/C ends and lengths 1/3/5/10.
Identical deletion sets are deduplicated; all observation rows retain exposure,
length, status and original known-deleted-atom distance. No new pKa runs.

The retained-site p95 error is evaluated on a 2 Å radius grid with 1,000 whole-
complex bootstrap resamples (seed 20261003), preserving repeated observations.
Report pointwise upper 95% bounds, observation counts and contributing complexes.
Exploratory candidates require at least 10 complexes and 20 observations and
upper bounds below 0.1 pKa at all larger supported radii. These are in-sample
diagnostics, not independently validated or simultaneous confidence guarantees.
Job 730339 runs this analysis with two CPUs / 4 GB, excluding comp1400.

Longer tails and internal loops need separate calibration. A proposed polymer
contact-probability model and 5% threshold remain hypotheses, not a calibrated
replacement; no Flory prefactor is assumed here. Likewise, ordered-deletion
calibration being conservative for disordered tails is untested. Existing
same-seed intact repeats were exactly reproducible, so the distant error floor
has not been established as solver noise. Crystal-versus-pKPDB agreement tests
reproducibility of stored labels, not the missing-residue counterfactual.
Clean/uncertain/near-gap tiers and retention counts can follow this calibration;
existing production masks remain unchanged during analysis.

Completed reanalysis: job 730340, `anchor-analysis-v2/` under the radial study.
The pooled retained-site p95 passes even at radius zero: distant observations
dilute near-gap errors. Therefore it must not select the mask on its own.
Added 5 Å local bands with whole-complex bootstrap upper bounds (black marks
in the error plot). The last supported band failing the 0.1 threshold ends at
10 Å for one-residue deletions, 15 Å for three/five, and 20 Å for ten.
All later supported bands pass; sparse bands remain unknown. These exploratory
boundaries pool N/C ends for confidence calculations while plotting each end
separately, and concern paired ΔpKa rather than absolute pKa. They are not yet
production cutoffs. `observations.csv`, `shell_summary.csv`,
`radius_summary.csv`, `anchor_error.png/pdf` and `anchor_retention.png` retain
the geometry, diagnostics and counts. No scientific calculations ran on login
nodes, and no pKa predictions were repeated.

## Provisional anchor-tier retention audit

`anchor-tiers --campaign CAMPAIGN --out NEW_OVERLAY` writes a separate
`site_tiers.parquet`, hashed source manifest and per-complex retention report.
Jobs 730344/730345 audit the 296-pair and 67-pair pools, respectively.
Production masks and frozen teacher inputs are unchanged.

- Clean: eligible and at least 20 Å of conservative envelope clearance from
  every missing whole-residue region.
- Uncertain: passes every applicable short-tail anchor rule, but fails clean;
  any uncalibrated gap must still have at least 20 Å envelope clearance.
- Near gap: fails at least one calibrated terminal anchor rule.
- Uncalibrated: otherwise affected by an internal gap, tail longer than ten
  residues, or unavailable complete observed backbone anchor.
- Ineligible: unknown sequence coverage, incomplete target functional atoms,
  or artificial terminal target.

Near-gap classification takes precedence over uncalibrated, but an independent
flag preserves overlapping uncalibrated influence. Anchor atoms must be observed
N/CA/C/O, not PDB2PQR repairs. Radii are 10 Å for length one, 15 Å for lengths
two through five, and 20 Å for six through ten. Untested intermediate lengths
use the next larger tested dose: an audit assumption, not new calibration.
Distances equal to the radius pass. Provisional retention is clean + uncertain;
it is not a new train/test selector. Report comparison with both old masks,
including sites gained and lost, interface counts and complexes represented.
Counts precede teacher coverage and sequence-disjoint split assignment.

Completed results (both jobs successful):

| Pool | Eligible | Clean | Uncertain | Near gap | Uncalibrated | Provisional retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 296 pairs | 35,375 | 6,831 | 5,477 | 4,974 | 18,093 | 12,308 |
| 67 pairs | 6,037 | 937 | 1,673 | 1,308 | 2,119 | 2,610 |

The new-pool overlay gains 4,053 and loses 548 sites relative to the old training
mask (8,803), net +3,505; retained sites span 171 complexes and include 1,841
interface sites. The older pool gains 1,241 and loses 42, net +1,199 relative to
1,411; retained sites span 43 complexes and include 423 interface sites.
Losses are possible because the new provisional policy requires 20 Å envelope
clearance for uncalibrated gaps, whereas the old training mask used 10 Å, and
the calibrated anchor rules are a different geometric criterion. Recomputed
old train/evaluation totals exactly match the prior audit in both pools.
Long tails/internal gaps remain the dominant unresolved category; these results
do not justify discarding the uncalibrated sites permanently.

## Internal visible-flank reanalysis

`expanded-radial internal-anchors` reuses the verified anchor observations and
completed pKa runs. Job 730346 (2 CPUs, 4 GB, excluding comp1400) writes
`internal-anchor-analysis-v1/` under the radial study. Separate strata are
buried1, buried_charged1, buried3, internal_near3 and internal_remote3.
Distance is the minimum from the target functional atoms to either retained
flank's backbone N/CA/C/O. Near/remote indicate the deleted region's distance
to the partner (<=5 / >=15 Å), not the target site's interface status.

Manifest membership restores overlapping categories after geometry-table
deduplication; a perturbation is counted once per category. Five-Å local bands
have 1,000 whole-complex bootstrap replicates, seed 20261004, with pointwise
upper 95% bounds where at least ten complexes and twenty observations remain.
Sparse bands remain explicitly unsupported. Retention curves include both
structurally eligible and valid paired-prediction observations. Results concern
paired ΔpKa for one- and three-residue deletions only, not arbitrary loop lengths.
No new pKa predictions, long-tail extrapolation or production mask changes.

Completed in eight seconds. Buried1 (23 complexes), buried_charged1 (21),
buried3 (20), and internal_near3 (23) all have their last supported failing
5 Å band end at 15 Å; every subsequent supported band passes the 0.1 ΔpKa
upper-bound criterion. Internal_remote3 (19 complexes) has no failing supported
band, but its 0–5 Å band is unsupported: this does not establish a zero-radius
rule. Distant sparse bands also remain unsupported. The observations support
testing a provisional 15 Å nearest-flank radius for the measured one/three-
residue internal deletions, not extending it to arbitrary internal-gap lengths.
Two-residue gaps would require an explicit interpolation assumption. Differences
in absolute pKa may cancel in paired ΔpKa, especially away from the interface;
these results should not be used as absolute-pKa error guarantees.
Artifacts: `internal_anchor_error.png/pdf`, `shell_summary.csv`, `retention.csv`,
`observations.csv`, `report.json` and hashed `provenance.json`.

## Short internal-gap overlay and long-tail pilot

Jobs 730349/730350 write `anchor-tiers-v2` in both pools. Terminal rules stay
the same; internal lengths 1–3 now use 15 Å from the nearest visible flank,
requiring both flanks to have observed N/CA/C/O. Length two uses the tested
length-three rule as an explicit interpolation assumption. Longer internal
gaps remain uncalibrated. Version-one overlays are preserved; production
masks remain unchanged. Older-pool retention rises from 2,610 to 2,724 sites.
New-pool retention rises from 12,308 to 12,681 sites across 173 complexes
(6,831 clean + 5,850 uncertain, including 1,907 interface sites). Near-gap and
uncalibrated tiers contain 5,938 and 16,756 eligible sites, respectively.
The older pool has 937 clean + 1,787 uncertain, including 449 interface sites;
near-gap and uncalibrated tiers contain 1,468 and 1,845 sites.

Job 730351 prepares and controls `radial-long-tails-v1`: up to two references
per homomer/heteromer/antibody class from the existing near-complete study,
complete and smaller references first. Choose the first sorted chain with
at least 80 observed residues, delete N20/N30/C20/C30, and run fresh intact
baseline and same-seed repeat. At most 36 variant jobs, each computing bound
and free states, use the existing dispatcher with the global 400-CPU cap,
2 GB per CPU and comp1400 excluded. Preparation failures remain explicit;
the controller collects results automatically. This six-reference pilot tests
ordered-coordinate deletion sensitivity, not the conformations of naturally
disordered tails, and is too small for the prior ten-complex bootstrap-support
criterion. Its results guide expansion rather than establish a production
long-tail cutoff. No structure prediction or missing-residue reconstruction.

## Ligand rejection inventory

Job 730398 runs `ligand-audit` on the 555 ligand-rejected candidate pairs in
the 1,000-assembly sample. It reads frozen resolved coordinates, inventories all
nonprotein components (including chemistry hidden behind the first rejection),
measures nearest heavy-atom distance to each selected protein partner, and
records covalent/metal connections, <1.9 Å contacts and <=4 Å bridging.
Existing GOL/EDO/PEG removal protection is reproduced at chain/component-name
level. The audit reports pair-deduplicated component identities and potential
remote-only/additive-only counterfactuals without changing chemistry policy.

`audits/ligand-inventory-v1/` contains components.csv, pairs.json, report.json,
source hashes and progress. A >10 Å distance is descriptive, not evidence of
electrostatic independence. The wider additive candidate list includes buffers
and ions and does not authorize their removal. Counterfactual categories overlap,
and preparation has not been rerun, so these counts are not recovered examples.

Completed: all 555 pairs across 355 assemblies audited, zero errors. Twenty
pairs have only unprotected remaining ligands >10 Å from both partners; thirty
have only unprotected candidate additives; their broader remote-or-additive
category covers 48 pairs (categories overlap). Another 100 pairs contain other
forbidden chemistry, and 288 have a protected component (declared connection,
short contact or partner bridge; these sets also overlap). Most frequent component
identities by distinct pair include SO4 101, GOL 54, EDO 51, MG 36, PO4 31,
ACT 29, ACE 27, PEG 21 and NAD 20. These are all observed components, including
already removable additives and other chemistry, not solely the causal rejection.
Only 48/555 pairs meet this simple review screen; do not describe the majority
as recoverable by additive cleanup. No recovery or policy change is established.

## Candidate-ligand removal preparation diagnostic

Jobs 730399–730406 test the 48 remote-or-additive-only pairs in eight shards;
the dependent collector writes `audits/ligand-review-v1/report.json`. Original
sources remain unchanged. Each pair records the removed chain/component names,
distances, CCD name/type/formal charge where present, atom count and source/
modified hashes, then reruns normal inclusive preparation and geometry gates.
Every instance sharing a removed chain/component name must be unprotected.
Existing approved additive removal remains part of preparation.

Reports distinguish neutral-additive candidates (GOL/EDO/PEG and related
glycols, MPD/DMS), buffers/ions/other review candidates and remote other ligands.
Passing is only preparation feasibility: no pKa equivalence has been tested,
and neither production admission nor wider removal rules are changed.

All 48 attempts completed with zero pipeline errors: 39 prepared, nine rejected
(four buried-area, two PDB2PQR, one interpartner disulfide, two remaining PEG).
Successful review classes: three neutral-additive-only, eighteen buffer/ion/other,
one mixed neutral-additive + buffer/ion, seventeen remote-other-ligand pairs.
The two PEG rejections show the heavy-atom inventory screen does not guarantee
the exact production removal rule will pass; preserve those failures rather than
automatically widening removal. Four area rejections have half-sum areas about
301, 314, 384 and 405 Å², all with at least eleven interface residues.
The diagnostic prepared count is not a production dataset increase.

## Buffer-candidate exposure

Job 730410 computed component SASA for buffer/ion review candidates within the
48-pair diagnostic, including pairs that failed preparation. Artifacts are in
`ligand-review-v1/buffer-sasa/`. Heavy atoms, 1.4 Å probe, 1,000 Fibonacci points,
element-specific Single radii and `ignore_ions=False` are used consistently for
the isolated component and component plus selected proteins. Water, other
components and omitted proteins are excluded; this is selected-partner burial,
not total crystal-environment exposure. Observations can repeat across pairs.
Median exposed area / isolated-area fraction: ACT 92 Å² / 51%, FMT 35 / 22%,
MES 222 / 68%, PO4 84 / 42%, SO4 120 / 61%, TRS 153 / 60%. Sulfate exposure
ranges from about 0.1% to 100%; blanket solvent-exposed classification is false.

Installed pKAI `protein.py:47` reads only ATOM records, ignoring normal HETATM
buffer/ligand records. Identical with/without predictions therefore cannot
establish harmless removal. A perturbation test requires a method that explicitly
represents the component with appropriate parameters; SASA alone is not a pKa
error estimate. No removal policy changed.

## Buffer exposure/distance threshold grid

Job 730411 completed `ligand-review grid`; results in
`ligand-review-v1/buffer-threshold-grid-v1/`. Exposure fractions 0.70/0.80/0.90/
0.95, minimum distances to native titratable functional atoms 5/10/15/20 Å,
and uncertainty masks 10/20 Å were evaluated. Every newly removed component
must pass; mixed buffer + other ligand/additive removals are not silently allowed.
Prepared atom completion is included in distances; ARG and native termini count,
artificial termini do not. Missing residues have no coordinates. Counts precede
missing-region masks, teacher coverage and sequence split assignment.

All tested combinations yield the same one buffer-only pair, complex
531aa2564d439359, with 94 native sites / 15 interface sites retained under either
mask. At 70% exposure and 5 Å distance, twelve component observations pass but
only one whole pair does; at >=10 Å only one component observation passes.
The distinction matters: one exposed component does not justify removing other
buried or close components in the same structure. This screen does not recover
many examples and does not calibrate ligand-induced pKa error. No production
rule or labels changed; components.csv and site_masks.csv retain per-site data.

## Component-removal reference feasibility and SASA/error request

User requested ligand/metal deletion-error distributions at 10/20 Å, including
the effect of ligand SASA. Jobs 730412–730419 inventoried all 419 pairs whose
first rejection was metal (208) or nonstandard residue (211); collector 730420
completed with zero errors. `audits/component-support-v1/` retains component
identities, CCD metadata when supplied, source hashes and supplied covalent/metal
connections. No such connections were recovered from these input records; that
must not be interpreted as evidence that coordination/covalent links are absent.
Component frequencies overlap across pairs and rejection categories: ZN 67,
MG 48, CA 43, HEM 39, FE 17, NI 17; MSE 76, TPO 21, SEP 13, LLP/CSO/PCA 8 each.

Installed PypKa 2.10 `constants.py` lists only NA+/CL- for `keep_ions`;
`clean/cleaning.py:add_non_protein` requires ATOM records and matching ion
atom/residue names. G54A7 supplies +1/-1 charges and 1.097/1.820 Å radii.
No verified arbitrary-ligand or Zn/Mg/Ca retention/parameter path currently
exists in this workflow. The ordinary cleaner already strips single-atom
Na/K/Cl, so the forbidden-component inventory is not a sodium candidate survey.
Job 730421 is an isolated synthetic-sodium preprocessing gate against the
actual DelPhi coordinate/charge/radius arrays, not a biological calibration.

The requested ligand SASA/error plot is still pending a represented reference:
first validate component atom retention and charge/radius assignment, then
compare retained versus removed species with identical protein coordinates.
For each component preserve absolute SASA, isolated SASA/exposed fraction,
charge/protonation model, declared/geometry-derived coordination, AB and free
assignment. Analyse absolute AB/free and paired ΔpKa errors in local distance
bands and beyond 10/20 Å, with exposure strata and complex-level bootstrap.
State where both SASA and distance have overlapping support to avoid attributing
distance effects to exposure. Protein-only or ignored-ligand predictions cannot
provide this comparison. No 10/20 Å ligand/metal cutoff has been validated.
Nonstandard residues remain identity-specific: the original residue's atom
positions can define a removal mask, but deletion/conversion is not assumed
chemically equivalent and artificial termini must remain excluded as targets.

## PROPKA component removal sensitivity

Jobs 730423–730433 completed `audits/propka-components-v1/`: 39 prepared
ligand-review pairs plus twelve supported-ion metal candidates; 28 pairs
completed (17 ligand, 11 metal), 23 recorded failures/unsupported cases.
Component atom retention and metal group charge were checked in native PROPKA
objects. Default PROPKA ignores SO4/PO4/PEG/EPE/TRS; these cases fail retention
checks rather than contributing false zero effects. Five-character CCD names
also require a future explicit PDB alias mapping and were not truncated.

One component is removed at a time (up to four per pair), retaining other
selected components and identical protein coordinates. Free-state components
are assigned to their nearest partner, with bridging cases flagged; this is a
conditional state convention. Actual SASA and isolated-area fraction accompany
each site-component observation. Data cover 27 ligand instances and 34 metal
instances, with repeated sites, in `site_changes.csv`. Native typing is saved
per state/calculation. PROPKA ligand retention does not validate chemical typing.

Maximum absolute protein pKa changes at distances >=10/15/20/25 Å are
0.111/0.040/0.00166/0 for represented ligands and
1.376/1.376/1.375/0 for represented metals. Thus the 15 Å metal training radius
does not bound every observed change below 0.1. At 25 Å all sampled changes are
zero, but model cutoffs (Coulomb 10, burial 15, desolvation 20 Å) prevent treating
that result as a physical guarantee. SASA-stratified scatter and pooled p95 are
in `plots/sasa_vs_change.png/pdf`; support/summary in `plots/summary.json`.
Exposure comparisons are descriptive and confounded by identity and distance;
there is no fitted causal SASA effect or population-safe radius claim. Production
15/25 flags remain provisional, with metal exceptions needing review.

## Evidence

- Completed radial experiment: 23 references, 336 calculations; artifacts in
  `_runtime/jax-Ka/pkabench/audits/radial-error-v2-nearcomplete/`.
- PypKa server methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC11223823/
- Deposited pKPDB code: https://github.com/mms-fcul/pKPDB
- AFDB is used for coverage evidence in this audit, not assumed experimental truth.
