# Split representation review — 2026-10-04

Job 730517 completed in one minute on a compute node (2 CPUs / 4 GB,
comp1400 excluded). Dataset, masks and assignments were unchanged. Input hashes,
quota reconciliation, native-site mask denominators and source provenance were
checked. Artifacts are under
`universe/combined-split-v1/usable-proposal-v2/representation-review/`:
`report.json` and `pairs.csv`.

## Eligible-interface cohort

| Measure | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Pairs | 664 | 150 | 500 |
| Distinct assemblies | 433 | 96 | 363 |
| Sequence components with usable pairs | 256 | 83 | 204 |
| Homomer pairs | 353 | 68 | 152 |
| Other heteromer pairs | 226 | 60 | 231 |
| Antibody–antigen pairs | 85 | 22 | 117 |
| Median protein residues | 462 | 397.5 | 363 |
| Median half-sum buried area (Å²) | 1,361 | 1,197 | 992 |
| Median retained interface sites | 7 | 9 | 11 |
| Pairs with only 1–2 retained interface sites | 124 | 28 | 67 |
| Median retained fraction of native interface sites | 39.0% | 65.2% | 82.4% |

Train uses training masks; validation/test use evaluation masks. Native denotes
complete functional atoms and no artificial terminus. General homomer/heteromer
classification uses the existing prepared-partner sequence equality flag, not
an independent homology definition. Counts describe prepared supervision before
teacher coverage.

Under the same evaluation mask, 294 training pairs retain any interface sites
(median 10 sites), versus 150 validation (median 9) and 500 test (median 11).
This does not invalidate the 664 training-mask-eligible pairs. It shows that
meeting held-out quotas preferentially assigned evaluation-eligible structures
away from training; the resulting cohorts differ in missing-region/component
mask burden. Median retention comparisons also reflect different complexes.

## Family concentration and class imbalance

The largest test sequence component contributes 107/500 pairs (21.4%). Its
frequent descriptions include MHC class I and beta-2-microglobulin. The next
largest contributes 24 pairs and includes lysozyme systems. Antibody–antigen
pairs account for 23.4% of test versus 12.8% of training. Test proteins and buried
interfaces are smaller at the median than training proteins/interfaces.

The reciprocal sum of squared component pair shares is 19.0 for test, compared
with 72.2 for training. This is a concentration measure, not a claim that the
204 test components represent only 19 independent observations.

## Recommendation

Keep whole groups intact. Report equal-weighted component scores alongside
pair-pooled scores, with bootstrap resampling by component. Also report
antibody–antigen, other heteromer and homomer results separately, plus a
sensitivity analysis excluding the largest test component. Report missing-region
tiers and supervision coverage. These additional reporting choices are
recommendations, not implemented scoring changes.

The allocation is feasible but not a representative random sample: the allocator
optimized usable quotas and retention of training-interface pairs. No prediction
outcome was used. Do not describe the test as balanced or general-PDB-unbiased.
The dataset remains fixed and the proposal remains unfrozen. Experimental
set-2/PKAD-3 completeness checks remain unfinished; unresolved antibody roles
are excluded from benchmark quotas. No new pKa predictions were launched by
this representation review.

## Accepted decision

The user accepted retaining the current split despite its imbalance. No further
balancing or broad download is requested. Primary reporting will average
per-complex scores within sequence groups and then weight groups equally.
Secondary reporting includes pooled scores and antibody/general breakdowns;
confidence intervals resample sequence groups. This reporting policy is recorded
in the shared specification; it is not a claim that the current smoke scorer
already implements production aggregation.

Job 730518 checks experimental matches against *all* candidate chains, including
antibody chains that are deliberately absent from the antigen-only split graph.
The available inventory's exact-structure sequences and parent-sequence proxies
are accounted for separately. This check does not certify completeness of the
uncurated experimental set 2.

### All-chain experimental reservation audit

Jobs 730518/730519 found a conflict between the older all-chain PKAD reservation
rule and the accepted antigen-held-out design. All 601 non-test candidate matches
are antibody–antigen pairs and are attributable to PKAD reference 1AXT chain H
(UniProt P01865). Among these candidates, 44 training and 14 validation pairs have
usable interface sites. These are full-chain homology flags at 30% identity and
80% bidirectional coverage, not proof of shared experimental labels or antibody
CDR identity. All available identifiers are represented by 305 exact deposited
chain sequences or 75 parent-sequence proxies; inventory completeness remains
unverified.

Strictly moving every flagged whole component to test would move 210 components,
leaving 569 training and 131 validation interface-eligible pairs. This also moves
unflagged partners connected to the flagged candidates. The counterfactual was
not applied: the user accepted retaining the current split.

No production freeze can be called compliant with the old all-chain reservation
rule yet. Proposed resolution: retain the antigen-held-out split and explicitly
exclude the overlapping antibody reference family from claims of independent
PKAD experimental evaluation, or revise the assignments to enforce the older
rule. This experimental-evaluation scope choice has not been applied. The main
antigen-held-out split itself is unchanged.

### Approved experimental exception

The user approved retaining the antigen-held-out split and excluding the
1AXT H/P01865 reference family from independent PKAD evaluation claims.
Job 730520 applies this as a versioned `independent-experimental-scope.json`
manifest and reruns the all-chain audit into `leakage-audit-scoped.json`.
The unscoped audit remains available. The split hash must remain unchanged.

The current-inventory exclusion matches 1AXT H, P01865 accession or an identical
reference sequence. Newly introduced related antibody references require family
review; unrecognized reference IDs cannot silently enter independent experimental
scoring. Excluded references remain available for explicitly non-independent
diagnostics. This is a change to experimental evaluation scope, not removal of
training complexes or a claim of full PKAD-3/set-2 coverage.

The scoped audit passed: all 606 candidate pairs matching retained experimental
references are assigned test; no train/validation conflicts remain in the
available inventory. One reference ID (1AXT H) is excluded. Proposal hash remains
`4a51bc4e8d8b21ef62bed308d99b8ae9761ce7102745ed2d81e79bb16472e212`;
664/150/500 interface-eligible train/validation/test counts are unchanged.
Complete PKAD-3/set-2 inventory coverage is still not certified; the proposal is
not marked frozen by this audit.
