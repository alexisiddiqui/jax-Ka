# Antigen-containing SAbDab2 audit for a prospective v7 pool

The current task is an overlap audit against abdata-v6c, not activation of a
new split. Current pool files, structures, teacher labels and augmentations
remain immutable. The user selected antigen-containing SAbDab2 instances only.
Apo/alternative structures retain their existing augmentation role and do not
receive new pKAI annotations through this work.

The agreed **v2** policy maps published SAbDab train to our training and
published test to our validation, with a separate external hard/novel test.
See [the revised hierarchy](#agreed-revised-role-hierarchy) and
[the inactive implementation](#v2-metadata-role-engine-inactive), followed by
[the completed conflict adjudication](#conflict-ledger-and-inactive-counterfactuals). Earlier v1
exclusion tables below are historical evidence, not v2 eligibility decisions.

## Inputs and scope

Use the pinned SAbDab2 ML v0.3.0 archive from
<https://zenodo.org/records/23343734>, with its `abag_split.csv` and
`ab_ag_split` column. This is the antigen-aware published train/test split.
Its `ab_ag_cluster` is a **sequence-connected component**, not a geometric
interface cluster. Retain the original split and component labels on every
audit record, including records excluded from future training.

Eligibility for this first audit is nonempty retained `agchains`, rather than
the original `holo` flag: the curators can remove all antigens from a structure
that was deposited as holo. The archive contains 16,415 instances; this filter
selects 10,878 and excludes 5,537 without a retained antigen. Protein, peptide,
nucleic-acid and other retained antigen types remain explicit metadata. Being
in this audit does not establish eligibility for protein pKAI preparation.

The baseline is `_runtime/abdata-v6c`, including both PINDER and pKPDB. Distinguish
the current validation membership from the larger pinned validation protection
set: the latter deliberately retains reference sequences from candidates later
excluded for quality. Keep external held-out references as a separate scope.

## Three independent overlap measurements

1. **Variable-domain sequences.** Compare H and L at shared IMGT positions,
   with exact matches divided by the longer nongap domain length. Use positions
   5–123 for the variable-core summary so constant domains do not distort an
   Fv comparison. Report H and L identities and a length-weighted paired
   identity to the **same verified H/L reference**. Report exact variable-pair
   matches separately. These identities describe overlap; high framework
   identity alone does not create an additional exclusion threshold.

2. **CDRs and published sequence components.** Reproduce the existing v6c
   `heavy-or-paired-or-exact-v1` CDR gates: heavy-chain CDR3 identity >=0.90 or
   concatenated CDR identity >=0.80 (sum the longer length of each CDR in the
   denominator, rather than taking the longer total CDR length); H/L pairing
   must use one verified reference;
   light similarity alone is not that legacy policy's rejection rule. Preserve
   its separate held-out concatenated-CDR MMseqs2 gate, 70% identity and 80%
   coverage of both sequences. Report the original exact full-chain gate too.
   Separately anchor published SAbDab components through verified chain aliases
   or the published CDR base-cluster thresholds (H3 95%, H123 100%, L123 100%).
   This **additional** published-component analysis can connect light-chain
   relatives that the legacy rule permits; it must not be presented as the
   unchanged legacy policy. Connected components also propagate through the
   official antigen clusters. External validation antigens absent from the
   archive require their own additional homology audit before a release.

3. **PINDER interface clusters.** Match a deposited antibody–antigen author
   chain pair to the existing PINDER index/metadata and retain every matched
   `cluster_id`. Compare those IDs with all source groups of current Ab/Ag
   validation units, including replaced half-Fv aliases. Match both directions
   of the chain pair. PDB identity alone does not establish a mapping. An
   unmapped instance has **unknown interface-cluster overlap**, not a negative
   result. This lookup does not perform fresh structural clustering or the
   extra transitive/iAlign de-leaking steps of the PINDER release procedure.

VNAR numbering is outside the current HKL reference-comparison implementation;
retain those records with an explicit unsupported flag rather than claiming a
passing CDR screen. Legacy CDR-only reference caches can be enriched with full
ANARCI numbering in an audit sidecar only after exact reproduction of all their
cached CDRs and domain boundaries. Preserve insertion-code order when forming
biological CDR strings; the older concat-MMseqs rule is reproduced separately.

## Interpreting the result

Report SAbDab training overlap with current validation/protected references,
and SAbDab test overlap with existing training. The initial paired-sequence
training comparison covers the reconstructed whole-Fv subset; source PDB and
PINDER cluster comparisons additionally cover the wider pools. Full sequence
exposure checks on the remaining legacy and pKPDB training records are a
required subsequent release check, not implied by the paired-subset result.

Overlapping evidence counts are not additive. A structure can share both an
exact Fv sequence and a CDR/component/interface cluster. Counts of instances,
unique antibodies, current pool units and connected components are different
quantities and must be labelled explicitly.

No record is accepted for v7 just because it lacks a detected overlap. Missing
interface coverage, unsupported numbering and unresolved antigen homology are
separate review states. Geometry, chain-role assignment, preparation, teacher
quality and cached-feature compatibility still apply.

## Proposed minimal integration policy, pending audit review

- Keep published SAbDab train/test labels unchanged. Add eligible training
  instances only from its training partition after applying shared protection
  and preparation rules across PINDER, pKPDB and SAbDab sources.
- Protect the official SAbDab test from existing training exposure in the new
  release. Any conflicting current-training records must be excluded in the
  new manifest; filtering only newly added SAbDab rows is insufficient.
- If the official test overlaps current validation, make that conflict explicit
  before selecting the new validation membership. The same component cannot
  be both model-selection validation and an independent reported test. Existing
  pool files stay frozen; any approved changes belong in the v7 manifest.
- Use SAbDab instance/chain/residue annotations for provenance and selection.
  Its ML structures crop antibodies to variable domains, whereas current Fv
  inputs can retain constant domains. Choose one declared preparation policy
  before building new labels. Reconstructing full observed source chains with
  the existing pipeline is a candidate that preserves more existing caches;
  directly mixing cropped and full constructs is not a silent equivalence.
- Reuse a current prepared unit only when chain roles, retained antigen chains,
  residue selection, coordinates, preparation policy and teacher context agree.
  Otherwise prepare/label the new canonical unit through the same pipeline.
  A matching PDB ID or variable sequence is insufficient for cache reuse.
- Deduplicate by canonical prepared structure/context, retaining source aliases
  and split provenance. Do not deduplicate different observed conformations
  solely because their antibody sequences agree. Keep all H/L chains on the
  antibody side and all selected antigen chains on the antigen side.
- Keep apo/alternative geometry in the existing augmentation path. The user
  explicitly excluded new pKAI annotation of apo structures.

The separate PDBbind validation-first protocol remains an additional future
combined-release requirement; this audit neither replaces it nor treats its
provisional measurements as a completed experimental validation anchor.
New partitioning cannot remove exposure from a checkpoint already trained on
older pool membership.

## Execution and artifacts

`benchmarks/sabdab_pool_overlap.py` runs in a real Slurm CPU allocation with
`--mem-per-cpu=2G`. It takes explicit baseline/runtime/audit paths and an ANARCI
Python executable. `sabdab_reference_numbering.py` supplies the optional
numbering sidecar using the existing ANARCI HKL/IMGT configuration.

Artifacts are under `_runtime/sabdab-v7-overlap-v1/results`: per-instance
decisions, paired identities, raw nearest-CDR evidence, exact PINDER chain-pair
matches, official-component anchors, input/code hashes and a summary. These
artifacts are audit evidence, not a released training manifest. No structure
generation, teacher run, training, pool mutation or dataset transfer is enabled
by either audit entry point.

## First measured audit, 2026-11-22

The corrected audit completed in Slurm job 807730. Independent verification in
807731 reproduced the pinned legacy CDR function on 2,000 domain pairs,
checked 19,429 biological CDR strings against SAbDab metadata (including 6,677
domains with position-112 insertions), verified input/code hashes, and confirmed
zero overlap of the **official train/test component labels**. That last check
does not mean the current pools are independent of those partitions.

For compatibility with the current protein-pair pipeline, report the
protein/peptide-antigen subset separately: 7,684 instances, comprising 6,377
published training and 1,307 test instances. Another 3,194 retained-antigen
instances have only other antigen types. The full audit includes those too.

| Protein/peptide subset, instance counts | Published train | Published test |
|---|---:|---:|
| Exact variable H/L pair to current 67-Fv validation | 90 | 33 |
| Distinct current validation Fvs reached by exact pairs | 31 | 11 |
| Heavy-CDR gate against current 67-Fv validation | 117 | 49 |
| Existing CDR/exact gate against all pinned validation references | 296 | 147 |
| Published components anchored to current Fv validation | 2,499 | 342 |
| Shared PINDER cluster with current 112-unit Ab/Ag validation | 182 | 45 |
| Unavailable PINDER interface mapping | 3,745 | 816 |
| Exact variable H/L pair to current 225-Fv training subset | 253 | 35 |

The 35 test instances in the last row reach 18 existing training Fvs. Exact
Fv-unit matches to the official train and test subsets reach disjoint current
Fv-unit IDs. At least 11 current validation Fvs require reconciliation with the
official test; the exact-pair count is a lower bound on component-level overlap.
The existing validation cannot be kept intact while claiming the complete
official SAbDab test is independent of model selection.

There are 1,559 supported protein/peptide training instances outside the union
of the detected protected validation, external-reference and mapped-interface
hits. This is a screening remainder, **not an approved addition count**. It
still includes interfaces without a PINDER mapping and needs the remaining
cross-source exposure, antigen-homology, preparation and context checks.

The additional author-chain context diagnostic covered 225 current training and
67 validation Fvs. It found a SAbDab H/L author-chain instance match for 155 and
42 respectively. Of these, 53 training and 12 validation units have a source
antigen chain absent from the corresponding curated SAbDab instance; 44 of the
65 have no retained antigen at all and 21 retain a different antigen. These are
review flags, not confirmed preparation errors: antigen definitions, antibody
antigens, filtering and construct selection can differ. Only 59 training and
23 validation units have a matched H/L instance with the same sole retained
antigen chain; even these require coordinate/residue/context checks for reuse.

Detailed results are in `results/REPORT.md`, `stratified-summary.json` and
`context-diagnostics.json` under the audit root. Superseded preliminary outputs
are explicitly marked invalid and retained separately. No pool was modified.

## Full source inventory and policy implementation

The next stage is implemented in `benchmarks/sabdab_v7_exposure.py`,
`sabdab_v7_screen.py`, `sabdab_v7_antigen.py`, `sabdab_v7_report.py` and
`sabdab_v7_verify.py`. The versioned switch is
`config/v7-source-policy.json`; `pkabench.source_policy.SourcePolicy` validates
the configuration, ranks sources, selects eligible owners only after prepared
identity/context is established, and keeps evaluation protection separate.
The default precedence is pKPDB > SAbDab > PINDER. SAbDab has its own native
source IDs, official labels and first-party source catalogue.

The audit root is `_runtime/sabdab-v7-exposure-v1`. The final outputs are in
`report-v3`; prior stage snapshots are retained. These are inactive proposed
memberships, not a new training release or loader activation.

- Inventory: all 86,044 existing PINDER/pKPDB train/validation entries and
  10,878 retained-antigen SAbDab instances; 77,704 unique sequences.
- Cached numbering was reused and 11,065 sequences were numbered/enriched.
  Every existing-pool sequence has an attempted numbering status. A no-HKL
  result is not represented as proof that a chain cannot be antibody related.
- One identical variable sequence has alternative published IMGT framework
  positions. Both annotations are preserved. Framework position identity is
  not treated as sequence identity or used as a new exclusion threshold.
- The exact/global antigen audit verified 19,207,238 length-compatible test
  comparisons, 17,306,331 external-reference comparisons and 243,178
  validation-to-SAbDab comparisons. Only a safe sequence-length bound pruned
  comparisons; no approximate search was used to certify a negative result.
- Antigen checks use the union of resolved and expected sequence metadata,
  conservatively accounting for the proposed full-source-chain input policy.
  This is additional protection, not a claim that the cropped ML structures
  expose every expected residue. Published component IDs remain unchanged.
- Apo-only official-test antibody metadata is included in protection checks;
  no apo structure receives new labels, features or supervised membership.

Under the configured full-official-test protection policy:

| Source/scope | Starting training entries | Proposed exclusions | Remainder |
|---|---:|---:|---:|
| pKPDB | 56,191 | 1,040 | 55,151 |
| PINDER | 28,270 | 443 | 27,827 |
| SAbDab protein/peptide | 6,377 | 5,350 | 1,027 |

The 1,027 SAbDab instances remain **candidates**, not approved additions. Of
these, 713 have no PINDER interface mapping. Parent-PDB overlap occurs for
16 candidates with pKPDB and 213 with PINDER; this is a reuse-review index,
not proof of duplicate prepared inputs. Prepared keys remain null until
coordinate, residue, role and teacher-context compatibility is established.
Existing half-Fv aliases remain lineage rather than identity evidence.

The explicit preserve-pKPDB counterfactual keeps 760 additional pKPDB training
entries, but excludes 631 protein/peptide SAbDab test instances from clean
evaluation. It is recorded separately and is not the configured default.
Source precedence never silently changes that evaluation contract.

Per-instance historical flags record **v6c membership overlap**, not proof that
a particular checkpoint used all eligible records. Any checkpoint exposure
claim requires its actual run membership. PDBbind's future validation-anchor
protocol remains unactivated. Current pool and label files remain unchanged.

Validation: 22 policy/cache/antigen-boundary tests pass. Independent verification
checks all frozen input hashes, 3,600 pairs against the pinned legacy CDR
function, 384 explicit global-alignment tracebacks, 300 exact antigen matches,
and all exhaustive comparison-count receipts. There are zero numbering
reproduction mismatches. The verification receipt explicitly says
`release_ready=false`: structural interface checks, context adjudication and
prepared-unit/cache qualification remain ahead of native teacher/feature builds.

## Agreed revised role hierarchy

The user subsequently selected SAbDab's published training partition for our
training and its published test partition for our **validation**, with a
separate strictly novel test cohort. This supersedes the earlier proposal to
carve validation from published training while retaining the published test as
our independent test. The agreed design is recorded in
`config/v7-split-authority-v2.json`; it is not an activated release or an input
to the historical v1 audit runner.

Original `ab_ag_split` labels remain unchanged as provenance. Operational
roles are stored separately. Other-source units inherit roles through the
published antibody/antigen components at the actual declared input scope.
Units bridging training and validation are quarantined or reconstructed into
nonbridging inputs; they cannot drag authoritative components across roles.
Unmapped general proteins retain a declared family/group split after common
holdout checks. Source precedence pKPDB > SAbDab > PINDER selects eligible
equivalent inputs, not evaluation roles.

The stricter checks select the hard/novel test **against both training and
validation**, rather than deleting published SAbDab training to preserve every
legacy cohort. Legacy cases overlapping the development data remain labelled
diagnostics. Experimental references qualify for independent claims only when
their actual measured scopes are verified clean under the final roles.
Unknown novelty does not qualify as hard-test independence, and feasibility
must be measured without weakening criteria to meet a quota.

If any future hard-test component is taken from SAbDab's published test pool,
it must first be removed from operational validation before model selection.
The default instead seeks a separate external cohort. Later nanobody sources
use the same role inheritance and novelty machinery, with versioned source
registration and frozen-test checks. Apo geometry retains its augmentation
role without new pKAI labels and cannot cross native split boundaries.

Earlier experiments retain their actual split versions and run memberships.
Their conceptual train/validation/novelty roles can be explained consistently;
they are not retroactively relabelled or declared independent under v2. The
earlier 1,027 candidate remainder is a v1 audit result, not the target v2 size.

## v2 metadata role engine (inactive)

`pkabench.split_authority.SplitAuthority` now parses and hashes the agreed v2
policy. It deliberately accepts inactive audits only. The historical schema-1
`SourcePolicy` and its runners are unchanged. `SourceRegistry` records source
identity separately from paired/site-only data kind and allows later sources
to register without changing the role-assignment algorithm. These declarations
do not yet add a production training loader.

`pkabench.role_engine.assign` is a pure decision function over declared input
scopes and published-profile evidence. It preserves native source splits,
records operational roles separately, quarantines bridges and keeps incomplete
mapping pending. Strict-profile evidence cannot veto an authoritative training
or validation assignment. Source precedence is used only by compatible-record
ownership, requiring established equal prepared keys, teacher contexts and
operational roles. Parent PDB identity or crop containment is insufficient.
When removal of PINDER records alone resolves a group conflict, compatible
pKPDB records are retained. Known antibody chains without a usable CDR3 stay
pending rather than being declared clean from partial numbering.

`benchmarks/sabdab_v7_roles.py` runs three stages in real Slurm allocations
with `--mem-per-cpu=2G`:

1. `prepare --inventory ... --screen ... --authority ... --registry ...
   --output ... --workers N` verifies frozen inputs and constructs a new audit
   inventory. Authoritative references include the complete published component
   catalogue; supervised SAbDab candidates require protein/peptide antigen.
2. `align --root ... --scope union --shard I --shards K --workers N` computes
   exhaustive length-compatible global antigen alignments against both roles.
   Union targets include resolved and expected antigens, but the report filters
   target identities and component memberships back to the resolved scope.
3. `report --root ... --workers N` computes published CDR-component links,
   proposes operational roles, checks family/exact-input boundaries and applies
   the existing 70% identity/80% coverage guard to unmapped general proteins.
   It also reports the effect of expected-antigen reference scope separately.

SAbDab input scope here is the published variable domain plus resolved antigen;
existing Fv units use their observed coordinate sequences. Other existing units
still use the pinned source-sequence inventories, so native preparation must
confirm those scopes before release activation. Expected or numbering-reference
sequences are never silently substituted for observed inputs.

Outputs are compressed Parquet assignments and evidence with authority,
registry and evidence hashes, plus a JSON receipt. Candidate roles do not prove
prepared-context equivalence, duplicate ownership, feature compatibility or
augmentation-donor safety. None of these audit outputs is a production pool.

External feasibility reports aggregate known chain overlaps to legacy complex
IDs. Remaining complexes stay pending: complete strict CDR and generic-protein
checks against final development inputs, interface evidence and prior-use
history are still required. The audit does not declare a hard-test size or
freeze test membership from incomplete evidence. It does not train a model,
build teacher labels, annotate apo structures or change existing releases.

## Completed v2 metadata audit

Audit root: `_runtime/sabdab-v7-roles-v2/audit`. The report and independent
verification both completed successfully. These counts describe candidate
metadata roles before native preparation and duplicate-context qualification:

| Source | Train candidates | Validation candidates | Quarantine | Pending |
|---|---:|---:|---:|---:|
| pKPDB | 55,119 | 1,034 | 817 | 21 |
| SAbDab protein/peptide | 6,377 | 1,307 | 0 | 0 |
| PINDER | 28,039 | 787 | 186 | 41 |
| Total | 89,535 | 3,128 | 1,003 | 62 |

Another 3,194 SAbDab instances have only other antigen types and remain
diagnostic catalogue entries. No native source roles were overwritten:
proposed operational roles are separate fields. Among existing validation
entries, 30 pKPDB and 122 PINDER records become training candidates; 265 pKPDB
and 162 PINDER training entries become validation candidates. Conflict and
pending decisions are recorded separately. The old 1,027 SAbDab remainder
does not apply under this authority policy.

The 1,003 quarantines comprise 46 direct train/validation component bridges,
891 declared family/exact-input group conflicts and 66 general-protein holdout
conflicts. For pKPDB specifically these are 16, 794 and 7, respectively. These
are split-consistency decisions, not structural-quality diagnoses or evidence
that all affected records need rebuilding. All 62 pending records are unmapped
antibody/unsupported-source-split cases (21 pKPDB, 41 PINDER); they require an
explicit disposition before entering a release. The detailed reason sidecar is
`report-diagnostics.json` under the audit root.

The exhaustive antigen stage evaluated 144,168,816 length-compatible pairs
for 52,171 queries against 5,132 resolved/expected targets. Role assignment
uses the 3,304 resolved targets; hard-test overlap screening uses the broader
union. Reference-scope sensitivity changes 103 component/family-stage unit
proposals before the independent general-protein holdout guard; this is not a
second activated pool or a final-release count.

Of 2,744 legacy external candidate complexes, 1,198 already have known
component or exact-chain overlap with the proposed development data. The
remaining 1,546 require further novelty checks; they are not certified hard
tests. Structural evidence, full strict sequence checks against final native
inputs and prior-use provenance remain outstanding. No test registry is frozen.

Validation: 52 focused tests pass. Independent saved-artifact verification
confirms unchanged original input hashes, preserved source splits, consistent
published-component roles, zero declared family/exact-sequence cross-role
conflicts, source-priority invariance and 240 antigen-alignment tracebacks.
`report/verification.json` and `report/summary.json` both retain
`release_ready=false`. Native scope/reuse qualification, augmentation donor
checks and production-loader integration remain ahead of release activation.

## Conflict ledger and inactive counterfactuals

The next read-only audit is under `_runtime/sabdab-v7-adjudication-v1`.
`audit/verification.json` independently verifies all five saved counterfactuals
and confirms unchanged historical inputs. No policy was activated. The original
v2 assignment Parquet was reproduced exactly, including its SHA256.

The 891 group quarantines now have explicit causes:

| Group cause | Records |
|---|---:|
| Legacy family only | 799 |
| Exact input sequence only | 84 |
| Both family and exact sequence | 8 |

`unit-conflict-ledger.parquet` links affected records to their causing groups;
`conflict-group-membership.parquet` gives each group's opposing members and
removal rule. `focus-sequence-membership.parquet` supplies chain/domain input
scope. `sequence-component-edges.parquet` reconstructs 1,756 scored antibody
and antigen edges, verifying 2,146 native unit/axis component assertions. Only
99 positive CDR queries required targeted reconstruction; the full antigen
alignment scan was reused. Identity edges do not establish prepared-context
equivalence or permission to reuse teacher labels.

`role_engine.quarantine_group_conflicts(explain=True)` preserves historical
behavior and returns its decision log. The new pure functions in
`pkabench.group_adjudication` implement explicitly inactive alternatives.
`config/v7-group-adjudication-v1.json` configures them separately from split
authority. `benchmarks/sabdab_v7_adjudicate.py` computes the proposals;
`sabdab_v7_adjudication_verify.py` independently verifies their saved source
component matches, retained boundaries, group causes and membership totals.

| Counterfactual | Train candidates | Validation candidates | Quarantine | Pending |
|---|---:|---:|---:|---:|
| CF0: reproduce v2 | 89,535 | 3,128 | 1,003 | 62 |
| CF1: re-evaluate after anchored/PINDER-first removals | 89,535 | 3,135 | 996 | 62 |
| CF2: published components govern mapped records | 89,829 | 3,619 | 218 | 62 |
| CF3: additionally propagate agreeing family anchors | 89,487 | 3,821 | 358 | 62 |
| CF4: CF3 with resolved/expected reference union | 89,458 | 3,770 | 440 | 60 |

Every case retains the 3,194 other-antigen SAbDab diagnostic records separately.
SAbDab's 6,377 training and 1,307 validation assignments stay fixed throughout.
Each counterfactual reruns the generic holdout search for its actual candidate
queries and targets. No recovery count is inferred merely from dropping a rule.

CF1 fixes avoidable collateral removals without changing which families are
protected; it recovers seven PINDER validation candidates. The staged algorithm
is deterministic and conservative, not a claim of globally minimal graph cuts.

CF2 exempts fully assessed, unambiguously component-mapped native records from
conflicting legacy family restrictions. General proteins without a published
match retain their declared fallback grouping, and exact input-sequence and
generic protein checks remain. It recovers 793 old quarantines (296 to train,
497 to validation), while the rerun generic guard newly withholds eight
previously active records: **net recovery 785**.

CF2 is the recommended candidate policy for review. It most directly follows
the agreed hierarchy while avoiding additional role inference for general
proteins. CF3 propagates a role to assessed, unmapped general members only when
all relevant family anchors agree; mixed anchors remain withheld. That extra
inference leaves more quarantines, so it is retained as a counterfactual rather
than recommended for activation. These choices are versioned methodological
alternatives, not retroactive claims that v2 violated its own boundaries.

The CF2 source counts are:

| Source | Train candidates | Validation candidates | Quarantine | Pending |
|---|---:|---:|---:|---:|
| pKPDB | 55,401 | 1,516 | 53 | 21 |
| PINDER | 28,051 | 796 | 165 | 41 |
| SAbDab protein/peptide | 6,377 | 1,307 | 0 | 0 |

The old legacy group remains provenance even where CF2 supersedes it. A future
production release must define operational/sampling groups separately and
regenerate versioned nested `min_fraction` values using the existing method.
These audit tables do not modify the current sampler, weights, objective or
production loader and are not drop-in production pool files.

## Native scope checks following adjudication

`sabdab_v7_native_scope.py` checked 266 native inputs: the 46 direct bridges,
66 original generic quarantines, 62 pending antibodies, their relevant native
overlap counterparts and a small sample of recovered/general/Fv contexts.
All 266 were read successfully. Source checksums, canonical residue selection
and conformer replay were checked against the existing pKPDB graph receipts.
PINDER bound/free coordinates were checked directly; whole-Fv records instead
use their actual curated `labels-v3` inputs, pinned teacher outputs and declared
joint H+L versus antigen side partition. Separate A/B files are not required
for that representation.

The final receipt is `native-scope-v4/summary.json`. Of these selected inputs,
186 have observed sequence sets differing from their sequence inventories;
missing/unresolved residues can explain such differences. This is not a
whole-pool rate, a structural-quality failure or evidence that labels must be
rebuilt. The current geometry was never changed. Teacher-context equivalence
still requires its own qualification, particularly for historical pKPDB labels.

Targeted ANARCI numbering reused all available caches and numbered only the
new observed sequences. `sabdab_v7_native_remap.py` compared the observed native
chains against the same published component references (472,549 global antigen
pairs for 315 queries), without activating new assignments. The checks show:

- 45 of 46 direct bridges persist at observed coordinate scope. One has a
  single component role at that scope and requires a whole-pool rescreen.
- 62 of the original 66 generic-quarantined units still have a recorded
  overlap at observed scope. Four lose their recorded links and also require
  a whole-pool rescreen; absence of an old edge is not a completed new screen.
- All 62 previously unmapped antibodies remain pending. They are not forced
  into training, retired automatically or certified as novel test examples.

`native-remap/verification.json` verifies the recorded results and 1,257 native
input/provenance file hashes. Three native-scope regression tests cover typed
residue identity, chain breaks, context hashes and partially unassessed report
rows; the 62 policy/ledger tests also pass. Earlier failed scope-report attempts
are retained as diagnostics; the final validated receipts are the v4 scope and
native-remap paths above.

No inventory correction above automatically releases a record. A final
production manifest needs complete scope-qualified checks, compatible teacher
and feature reuse, duplicate ownership, native augmentation donor checks and
actual loader membership verification. The SAbDab native annotation/timing
pilot and hard-test freeze remain subsequent stages. All current releases,
structures, teacher labels and features remain unchanged.

## Native preparation and source-aware loader pilot

The inactive preparation pilot selected 25 antigen-containing SAbDab ML
instances, stratified across published train/test, size, protein/peptide,
paired/single-domain antibodies, scFv and multichain-antigen contexts. The
selection included 15 published training and 10 published test instances;
it is a diagnostic sample, not a yield estimate for the full catalogue.
`_runtime/sabdab-v7-preparation-pilot-v1/pilot-v4/summary.json` records
24 prepared assemblies and one withheld SD-H/peptide instance containing MSE.
MSE is not silently changed to methionine or dropped from the native polymer.

`pkabench.sabdab_units` selects polypeptide entities, resolves conformers with
the existing policy and retains observed canonical heavy atoms. Heavy and
light chains form one antibody side; all declared protein/peptide antigen
chains form the other. Distinct author-domain aliases of an scFv can share
one physical label chain. The canonical residue inverse map preserves those
aliases and source numbering; observed gaps are retained. No atoms are added.
Four preparation tests cover role partitioning, free-amino-acid exclusion,
shared chains, scFv aliases and ambiguous overlapping roles.

The capped parent-context screen is recorded in `reuse-v2/reuse-screen.json`.
Its 10 same-PDB candidate pairs include four differing observed assembly
scopes and six matching chain-sequence multisets. The six matches still need
atom correspondence, biological partition and teacher-protocol checks; none
is certified for wholesale reuse. Historical pKPDB site labels also do not
supply the new paired AB/A/B pKAI targets merely because coordinates overlap.
These are candidate pairs, not ten distinct SAbDab instances.

The loader now has an opt-in source-aware manifest path. An explicit
`data_kind="paired"` permits SAbDab records to use the existing paired tensor
loader while retaining `dataset="sabdab"`, native ID, published split,
component, operational role and input scope. It requires an explicit nested
training pool and exact provenance coverage, rejects role mismatches and
held-out IDs in that pool, and enforces published SAbDab train/test as our
train/validation. Source priority remains separate from split authority.
Packing supports explicitly selected zstd level 1 and the existing bounded
five-file store layout; default legacy compression and routing are unchanged.

The final loader job (809038) passed 24 regression tests with one optional
Torch test skipped, then all 10 source-aware checks under the default v2
format. The first run uses v1 to exercise the existing legacy pool fixtures.
New loader checks use explicitly synthetic tensor fixtures: they verify
packing, source identity, target/mask preservation, batch padding, nested
selection and split guards. They are not an end-to-end native-label test.

No native pKAI labels or MaSIF features have been generated in this stage,
and no production recipe has switched to v7. The next stage is a small native
annotation pilot, followed by loading its real packed records, checking
termini/curation and augmentation provenance, and measuring annotation cost.
Only after that qualification should the full build, final ownership and
versioned pool manifests be produced. The separate novel test is still not
frozen, and unresolved reuse candidates remain unapproved.

## Native annotation pilot: completed

The five-complex native pilot is recorded in
`_runtime/sabdab-v7-native-annotation-pilot-v1/pilot-report-v2.json`.
It covers two conventional Fv crops (original structures annotated FAB), a
single-domain antibody, an scFv and a single-domain antibody with two antigen
chains, using both published training and test/operational-validation examples.
All five have fixed-coordinate native pKAI and pKAI+ labels for AB/A/B, and all
15 AB/A/B surfaces passed the verified original-paper MaSIF pipeline. No apo
labels, ANM views or model training were added.

The label pilot took 24.2 seconds on five CPU workers, including a separate
cold teacher interpreter per complex. MaSIF took 814.6 seconds (13.6 minutes)
with four preprocessing workers, 20 allocated CPU cores and one A40. This run
used batch size one, including repeated container, model and MATLAB startup.
Five deliberately selected 164–545-residue complexes do not establish a
whole-catalogue yield or a reliable full-run ETA. Full-build startup
amortisation and larger antigen tails remain unmeasured.

The MaSIF cache contains all 15 frames in five reader files, uses zstd level 1
and occupies 21,025,543 frame bytes. The existing `MaSIFStore` loaded all five
complete, disjoint Ab/Ag partitions and the bound frames independently map to
the same canonical residue indices. Every surface checked three real patches
against TensorFlow reference inference (maximum permitted error 1e-4), along
with finite descriptors, triangle geometry and surface-to-residue ownership.
Native geometry checks on all five assemblies retained H–L edges (5,010 and
5,018 directed residue edges in the two conventional examples), retained
joint antigen-chain contacts and removed all opposite-side free-branch edges.

Source ML CIFs omit `chem_comp` types. The final curation now recovers missing
types from the installed, content-pinned PDB CCD before applying the existing
`mask-all-v1` classifier and radii. Unknown types remain conservative at 30 A;
canonical nonpolymer amino acids remain excluded from the protein model but
are included in omitted-context masks. The initial untyped-component screen
is retained for provenance. Final `labels-v2` reused the teacher files
byte-for-byte, changing curation evidence and masks only; no teacher or surface
recalculation was needed. Eight source/terminus/CCD tests and all 26 graph,
source-loading and legacy regression checks passed.

The strict existing full-deposited-sequence gap rules retained only two
complexes for the paired loader: one training example with 10 sites and one
validation example with 33 sites (3 and 8 interface sites respectively).
Those real records were packed and loaded with exact target, node, mask and
padding checks. The graph cache uses five reader files, zstd level 1 and
4,773,995 compressed frame bytes. Its all-inclusion smoke pool is explicitly
pilot-only, not a regenerated production `min_fraction` schedule. Native
published SAbDab `test` remains operational validation; no operational novel
test cohort is included or certified by these checks.

The boundary sensitivity audit isolates the policy issue:

| Example | Operational role | Strict paired/interface sites | Relax Ab boundary regions only | Relax all declared-unit boundary regions |
|---|---|---:|---:|---:|
| 1tjh H/L | train | 0 / 0 | 21 / 10 | 23 / 12 |
| 3mly H/L | validation | 0 / 0 | 17 / 0 | 45 / 20 |
| 8hxq single domain | train | 10 / 3 | 25 / 10 | 32 / 16 |
| 8pmz scFv | validation | 0 / 0 | 0 / 0 | 61 / 21 |
| 8iee multichain Ag | validation | 33 / 8 | 33 / 8 | 92 / 25 |

Both relaxed columns are diagnostic only. They omit terminal gap regions
(Ab only or both biological partners) and the omitted interdomain linker
between distinct scFv author aliases, while retaining other internal gaps,
incomplete-atom/component masks and source-based true-terminus evidence.
They do not prove that every omitted boundary is an intentional crop or that
its influence can safely be ignored. Stored native supervision remains the
strict result, and the 5/5 diagnostic retention is not a catalogue-wide rate.

The native annotation and reader implementation therefore works, but a full
release should wait for a consistent, versioned definition of the prepared
unit and its crop boundaries across SAbDab and the existing sources. Intentional
domain excision must not silently become an unresolved full-length tail, nor
should unresolved residues be exempted merely because an ML crop removed them.
Final ownership/reuse qualification, full-pool manifests, the independent novel
test freeze and production recipe activation remain separate release steps.

## Correction: published antibody unit scope

The antibody-boundary issue above is now corrected in the pipeline, with
verification in `_runtime/sabdab-v7-antibody-scope-fix-v1/report.json`.
`pkabench.sabdab_scope` reads the published `VH_imgt_keep_interval` and
`VL_imgt_keep_interval` together with their numbered sequences. It validates
every retained IMGT key and residue identity against the prepared inverse map,
then projects the declared domain endpoints into deposited polymer positions.
Unobserved, ambiguous or inconsistent declared endpoints fail validation;
observed extrema are never substituted for missing declared boundaries.

This defines two different kinds of absence:

- Positions outside the verified antibody intervals are intentional omissions
  from the prepared unit. They are recorded as provenance, not gap defects.
  This includes omitted constant domains and the source-excluded linker
  between the two declared domains of a single physical scFv chain.
- Missing positions inside those intervals remain unresolved sequence defects
  and use the existing `long-gap-v1` rules. Antigen/full-chain gaps, incomplete
  atoms, disulfide eligibility and component masks keep their existing rules.

Full deposited-sequence evidence still determines natural termini. The crop
correction does not create NTR/CTR sites, change exported teacher atoms or
coordinates, or alter labels. New SAbDab preparation records embed the declared
crop contract and verification; old prepared records can be re-curated against
the content-pinned published metadata. Missing crop metadata retains explicit
conservative full-sequence handling rather than authorizing an inferred crop.
Graph manifests record the supervision-scope policy separately from geometry
preparation and source/split identity.

The five-complex check reused all teacher bytes and terminal evidence exactly.
It independently verified that every removed gap was on the antibody side and
entirely outside the published retained intervals. Antigen gaps, missing-atom
and component evidence were unchanged, and no previously eligible sites were
lost. The resulting operational paired/interface counts are:

| Example | Previous | Corrected |
|---|---:|---:|
| 1tjh H/L, train | 0 / 0 | 21 / 10 |
| 3mly H/L, validation | 0 / 0 | 17 / 0 |
| 8hxq single domain, train | 10 / 3 | 25 / 10 |
| 8pmz scFv, validation | 0 / 0 | 0 / 0 |
| 8iee multichain Ag, validation | 33 / 8 | 33 / 8 |

The paired loader now accepts three examples (two train, one validation).
3mly and 8pmz still lack qualifying paired interface queries because antigen
gap masks remain active. This is intentionally different from the earlier
diagnostic that relaxed all unit boundaries; no antigen exemption was added.
The new real-record cache passes label/node/mask/padding checks and remains
five files at zstd level 1 (6,580,051 compressed frame bytes).

Sixteen preparation/scope/terminus/CCD tests and all 34 regression checks
passed. Dedicated cases cover real missing residues inside a declared domain,
unproved boundaries, retained-residue mismatches, scFv linker intervals and
unchanged antigen tails. Fresh preparation of the original 25-example panel
again produced 24 valid assemblies and the same MSE withholding; all 24 have
verified published scope with identical native coordinates, residue maps and
biological-side partitions to the previous preparation. The correction needs
no teacher or MaSIF recalculation and does not activate the production release.

## Deposited-coordinate audit of the two antigen exclusions

`_runtime/sabdab-v7-antigen-scope-audit-v1/report-v2.json` compares the ML
antigens with their original deposited coordinates. 3mly uses the pinned
local original mmCIF; 8pmz was downloaded from the public RCSB coordinate
service with its URL and byte hash recorded. Both source sequences match
the published SAbDab expected antigen sequence. The retained author residue
indices and sequences match the published `ag_index_list`/`agresolvedseqs`,
and all retained heavy-atom identities and coordinates match the originals
(maximum unaligned displacement below 0.000003 A).

| Antigen | Deposited polymer length | Residues with deposited coordinates | Retained by SAbDab ML | Observed residues removed by ML | Explicitly unobserved deposited positions |
|---|---:|---:|---:|---:|---|
| 3mly, author chain P | 23 | 12 | 12 | 0 | 1–5 and 18–23 |
| 8pmz, author chain C | 211 | 151 | 151 | 0 | 1–9 and 161–211 |

The original `_pdbx_unobs_or_zero_occ_residues` annotations independently
cover exactly these missing positions: 11 for 3mly and 60 for 8pmz. Thus the
remaining antigen gaps are genuinely unobserved deposited regions, not
observed portions removed by the ML preparation. These cases do not justify
an additional antigen crop exemption. Existing antigen gap masks remain
unchanged, including the calibrated-length limit on the 51-residue 8pmz tail.

This is an eligibility conclusion for the current paired pKa objective,
which requires at least one unmasked interface query. It is not a structure
processing failure or a rejection of the existing MaSIF surfaces. 3mly still
has 17 eligible non-interface pKa sites; all five pilot complexes retain
their successful native labels and AB/A/B surface annotations. No teacher,
surface, training membership or preparation policy was changed by this audit.

## Whole-pool native preparation and build audit

The full audit is in `_runtime/sabdab-v7-build-audit-v1`; its current build
proposal is `build-manifest-v2/report.json`, with per-unit actions in
`build-manifest-v2/units.parquet`. It covers all 96,922 catalogue records and
remains inactive. The earlier `build-manifest` is superseded: v2 also includes
the verified paired cross-source teacher comparison and ownership decision.

All 7,684 protein/peptide-antigen SAbDab candidates were attempted using the
corrected published antibody boundaries. Preparation succeeded for 7,480.
The prepared coordinates, original source payload, inverse maps, sites and
evidence occupy 1,311,774,924 compressed bytes in one zstd-level-1 shard and
three reader metadata files. No permanent per-complex directory tree was
created. The complete eligibility accounting is:

| Outcome | Candidates |
|---|---:|
| Prepared, verified published scope, potential paired interface supervision | 4,926 |
| Prepared, verified scope, no qualifying interface query | 2,454 |
| Preparation held | 204 |
| Above the current 1,536-residue capacity | 78 |
| Prepared antigen sequence differs from published retained sequence | 22 |
| Total | 7,684 |

Before the independent retained-antigen scope check, 4,937 candidates had
potential interface supervision; eleven of the 22 scope holds belonged to
that group. The other eleven lacked qualifying interface queries already.
These holds do not silently acquire an antigen crop exemption. Preparation
failures comprise 155 incomplete native backbones, 17 selected MSE cases,
15 missing published antibody residues, 14 ambiguous disulfide geometries
and three missing biological partners. These are explicit audit outcomes,
not reasons to change the established masking or chemistry rules.

Potential supervision is not yet final teacher availability. Three qualified
pilot complexes reuse their verified native teachers; a fourth can reuse a
compatible PINDER teacher as described below. Up to 4,922 new native
pKAI/pKAI+ AB/A/B predictions remain. Final graph admission still requires
finite, corresponding predictions and at least one qualified interface query.
The 4,926 candidates comprise 4,177 train and 749 validation records before
the remaining duplicate-context review. No apo teacher predictions are planned.

### Existing native contexts and observed-scope roles

All 56,991 pKPDB and 29,053 PINDER native inputs passed their source-receipt,
conformer/topology and coordinate checks. PINDER free partners were verified
as removal of the other biological partner, preserving joint H/L where
present. No native geometry was rebuilt. In 71,759 units the actual observed
sequence set differs from the earlier inventory sequence set; this is a
scope distinction, not evidence that the coordinate files changed.

The split recheck therefore used actual observed sequences, enriched numbering
for 70,709 queries, reused 11,939 already assessed antigen queries and checked
70,664 new ones. An optional exact longest-common-subsequence upper bound
rejects only pairs that cannot reach the published identity threshold in any
alignment. Every surviving pair still uses the original global affine-gap
aligner and thresholds. Multiword boundary tests passed; a length-spread
check including actual queries and edited positive controls returned exactly
the same hits for 47,327 candidate pairs, approximately 75 times faster.
The complete new screen rejected 127,077,756 impossible pairs and aligned
40,799 survivors. This changes computation cost, not the accepted split rule.

The observed-scope proposal changes 229 previous candidate roles. SAbDab's
published antigen-aware train/test mapping remains authoritative throughout:

| Source | Train | Validation | Quarantine | Pending | Diagnostic |
|---|---:|---:|---:|---:|---:|
| pKPDB | 55,367 | 1,498 | 104 | 22 | 0 |
| PINDER | 28,023 | 756 | 231 | 43 | 0 |
| SAbDab | 6,377 | 1,307 | 0 | 0 | 3,194 |

These are role assignments before preparation/teacher eligibility and source
ownership. The compatible PINDER alias below removes one further training
record's sampling weight, leaving 28,022 unique PINDER training owners in
this proposal. No current pool membership or training recipe has changed.

### Reuse and ownership

Sequence/atom-count pruning found seven cross-source pKPDB matches and one
PINDER match. All eight have matching native heavy-atom geometry. The seven
pKPDB historical site-only label contexts do not substitute for paired pKAI
targets. For `sabdab:pdb_00001mvu_B_A` versus
`pinder:1mvu__fv__0986fbf745bf2ff9`, the audit additionally verified the pinned
export archive, all six teacher package/model files, true termini, and exact
AB/A/B exported chemical atom coordinates after residue remapping. The reused
labels provide 59 qualified queries, including 23 interface queries, under
SAbDab's own preparation masks.

The configurable precedence `pkpdb,sabdab,pinder` consequently selects SAbDab
for that compatible paired context; PINDER is retained as provenance and an
alias with no additional sampling weight. Changing `--precedence` changes
ownership preference, independently of authoritative split assignment. A PDB
identifier, sequence match or near-identical geometry alone cannot invoke it.

The geometry screen also found 2,320 near-identical pairs within SAbDab,
often symmetry-related copies at approximately coordinate-file precision.
These are candidate relationships, not 2,320 independently removable records.
There are 379 SAbDab records requiring further context review, 280 of which
otherwise qualify for the proposed build. Their `ownership_ready` field is
false: teacher-context identity and canonical membership must be settled
before activation. The budget includes them conservatively and can fall after
adjudication. No 1/n weighting or inferred source ownership was introduced.

The existing qualified native MaSIF release can supply 100,141 frames
(168.6 GB) for retained contexts. Its prior complete source/geometry/field
audit and explicit mixed preparation profiles remain the reuse evidence;
this audit did not repeat a full 170 GB payload scan. Role/group changes
require metadata repacking even when the numerical descriptors are reused.
The local v6c PINDER graph metadata describes an 81.1 GB payload, but the
payload itself is not local. Isambard authentication currently fails, so
remote payload location/current readability remains a release preflight.

### Build and storage envelope

The paired assembly reader requires separate biological A and B surfaces:
joint antibody versus joint antigen, including all physical H/L or antigen
chains belonging to each side. AB is an optional bound-complex diagnostic,
not an overlapping third member of the assembly partition. The existing
approved native surface policy is retained for general site-only proteins;
known pKPDB antibody cases use their pinned assembly-role inventory.

| Missing required surfaces | Views |
|---|---:|
| pKPDB | 3,350 |
| PINDER | 57,054 |
| SAbDab | 9,846 |
| Total | 70,250 |

Size-binned measurements from 101,006 successful historical surface IDs give
about 16.6--20.4 GB for the new required SAbDab surfaces and 116--146 GB for
all missing required surfaces. These are sums of size-bin medians/p90s, not
confidence intervals. Only two successful historical surfaces exceed 1,024
residues; the 259 new SAbDab views in that size range need a tail-cost check
before treating the estimate as a launch commitment.

The SAbDab graph point estimate is 17.2 GB, with a 10.3--34.4 GB planning
range, scaled by residues from the three real paired pilot graphs. Teacher
yield and query/neighbor density can change this. Its existing packed graph
reader needs five files, including one `records.bin`; prepared and surface
releases use bounded 2 GiB shards. All new packing uses zstd level 1. Original
teacher interchange files remain bounded worker scratch; they must not become
a permanent per-structure inode tree. Exact label-package bytes and the final
combined inode budget must be checked when materialising the build.

For SAbDab required A/B surfaces, the planning scenarios are about 28--49 h
at 16 workers or 19--34 h at 23 workers. The corresponding full-pool missing
surface scenarios are roughly 8--14 days or 5--10 days. These include explicit
cold-start/retry allowances over historical warm measurements. The projection
assumes two allocated CPUs per worker; concurrent GPU contention at those
worker counts has not been measured in this audit. Optional AB surfaces are
budgeted separately and are excluded from these figures.

The five-example cold-process teacher timing panel gives a deliberately broad
2--25 h estimate at 16 CPU workers for the 4,922 missing SAbDab labels. It is
not a catalogue-scale throughput measurement. Teacher/graph qualification
should precede expensive surface construction to avoid annotating units that
cannot supply the current paired objective. Source-owner review, versioned
min_fraction regeneration, the independent novel-test freeze and production
recipe integration remain activation gates. No full annotation build,
transfer, training run, commit or publication was performed by this audit.

Final verification passed for all 96,922 manifest rows, source totals, input
and output hashes, unique surface tasks, build eligibility and the compatible
owner/alias role agreement. All twelve final scope/reuse/exact-bound/teacher
mapping regression tests passed, in addition to the seventeen preparation,
source-evidence and boundary checks run before the full SAbDab preparation.
The final receipt is `_runtime/sabdab-v7-build-audit-v1/stage-receipt.json`.

## Native pKAI build, with MaSIF deferred

The subsequent user instruction resumes pKAI annotation only. The build lives
in `_runtime/sabdab-v7-pkai-build-v1` and uses CPU partitions exclusively;
comp1400/GPU allocation, new MaSIF surfaces and apo labels are excluded.
`benchmarks/sabdab_v7_pkai_build.py` consumes the frozen packed native
preparations and the inactive v2 build manifest. It does not re-prepare crops,
relax gap/component masks, add molecular repair or change operational roles.

Both released teachers, pKAI and pKAI+, run for native AB, joint biological A
and joint biological B, with the same six pinned package/model files and
true-terminal export protocol as the existing whole-Fv labels. Prediction
interchange files and logs exist only in bounded worker scratch. Labels,
post-teacher site masks, remapping requests and the actual exported teacher
PDB bytes are packed into independent zstd-level-1 frames. Source coordinates,
inverse maps and source evidence remain referenced through the verified
native preparation frame rather than duplicated in a permanent directory tree.

The size/role-spread canary admitted all 16 complexes: twelve new predictions,
three verified native pilot reuses and the compatible PINDER reuse. It took
20.7 seconds with six workers and produced 2,131,360 compressed bytes in four
reader files. All frames passed a separate verification of both teacher
models, target identities, original versus post-teacher masks, natural termini,
source/role identity and exact chemical AB = disjoint A union B. The full run
reuses these successful canary frames rather than predicting them again.

The full candidate set remains 4,926 complexes, including 280 whose potential
duplicate context must be settled before training activation. Computing their
labels does not grant duplicate sampling weight or override that hold.
The build and independent verifier retain this flag explicitly. The same
verifier checks the complete final file/frame hashes, model/package/code pins,
cohort accounting and every qualified record. Separate source-native masks
remain authoritative when labels are reused. Training graph materialisation,
min_fraction regeneration and recipe activation remain separate steps.

The complete annotation build admitted all 4,926 candidates (4,177 train and
749 validation), with no teacher failures or loss of the required interface
queries. It took 500.3 seconds with 32 workers on comp0652. Including the
canary, 4,922 complexes received new native predictions; four reused their
previously qualified teachers. The full build reused all sixteen canary
outputs and therefore ran only 4,910 additional predictions. The final label
store is 550,176,682 compressed bytes in four reader files; progress, input
pins and per-unit reports are separate build artifacts. These measured build
timings supersede the earlier deliberately broad teacher-cost projection for
this prepared candidate set. They do not include the preceding full preparation
and source/split audit. The 280 duplicate-context review flags remain active,
and no training membership, weights, loss or production recipe changed.

The independent full verification also passed for all 4,926 records. It
re-read the packed file/frame hashes, checked both models' finite values and
identical target identities, independently re-applied the frozen pre-teacher
site masks, retained the original terminal evidence and verified exact chemical
AB = disjoint A union B for every qualified assembly. Frozen input, model and
code pins passed. Its receipt is
`_runtime/sabdab-v7-pkai-build-v1/full/independent-verification.json`.

## Native graph candidate and loader checks

The next stage is complete in `_runtime/sabdab-v7-native-training-v1`.
Its inactive candidate index is `candidate-manifests-v2/candidate.json`.
The earlier `graphs/`, `native-patches/` and `native-routes/` directories are
superseded by their `-v2` counterparts and are not selected by this index.
No current training membership or production recipe has been activated.

All 280 pending native-copy flags were resolved. Of the 4,926 labelled SAbDab
candidates, 4,719 supply native sampling representatives, five are exact
context/teacher aliases and 202 additional numerical copies are withheld
from sampling. Fourteen reviewed pairs retain distinct contexts. Numerical
copy grouping requires the same parent PDB, published component and role,
whole-assembly atom inventory, biological sides, true termini, query masks
and target identities, with full-native heavy-atom RMSD at most 0.002 Å and
maximum atom error at most 0.005 Å. Groups use complete-link checks and a
deterministic lexicographic representative; tolerance matches do not chain.

This sampling reduction does not authorize teacher substitution. Exact
aliases additionally require congruent exported teacher atoms below 1e-8 Å
and identical values from both teachers. Some coordinate-near pairs in the
review differ by up to 0.74 pKa units. Each selected representative retains
its own labels; labels are neither averaged nor used to choose that
representative. The policy and pair-level evidence are in `ownership/`.

The admitted source totals are:

| Source | Train | Validation |
|---|---:|---:|
| pKPDB | 55,367 | 1,498 |
| PINDER | 28,022 | 714 |
| SAbDab | 4,146 | 573 |
| Total | 87,535 | 2,785 |

SAbDab's published antigen-aware train/test roles remain train/validation.
Existing native source roles follow the frozen observed-scope adjudication.
Among 298 PINDER records requiring query-mask changes for their new roles,
256 have packed replacement records. The remaining 42 have no qualified
interface query under the new validation mask and remain excluded; their
masks were not relaxed and they were not returned to training. The compatible
cross-source PINDER alias also contributes no additional sampling weight.
Untouched pKPDB/PINDER payloads remain in their existing packed stores.

`pools-v2/` regenerates nested `min_fraction` values with the existing seeded
within-stratum rank algorithm and a versioned salt, `pool-v7-native-v1`.
Matched published components form shared operational groups across sources;
unmapped entries retain source-native family groups and original strata.
Training and validation groups are disjoint. There are no new inverse-group
weights or loss changes. The verifier independently recomputed these ranks
and checked all 90,320 catalogue memberships. Paired weight normalization
uses selected training complexes only, with separate recorded values for
each source and fraction; this does not prescribe a new source-mixing recipe.

The final SAbDab graph store contains 4,719 complexes and 302,339 paired
queries in **17,324,994,307 compressed bytes**, with five reader files and
zstd level 1. Its 16-worker build took 402.4 seconds. PINDER's 256 replacement
records occupy **771,291,656 compressed bytes** in another five reader
files. Thus the incremental graph payload is about 18.1 GB plus small pool
and manifest files. No permanent per-complex graph directory tree is needed.
Preparation and teacher-label stores remain separate provenance artifacts.

The real-cache controls caught a mismatch in the initial new builder:
existing student graphs encode observed topology/segment endpoints, whereas
the first SAbDab/Fv builder used chemical true termini for those node flags.
The final builders preserve the existing student convention, while teacher
export and terminal-query eligibility continue to require source evidence
for true termini. An intentional Fv crop boundary consequently gains no
new terminal teacher target. Query values and scope masks remain unchanged.

`production_loading.source_from_manifest` routes the explicit catalogue and
tensor kind: pKPDB remains site-only; SAbDab is its own paired source rather
than a renamed PINDER entry. Manifest-pinned record overrides route changed
PINDER query masks without copying the full base cache. Registered legacy
native stores without a `data_kind` field require their metadata pin and
known source/field contract. New unknown catalogues must declare a kind.
The existing pKPDB validation convention still scores `train_mask`.

Joint antibody H/L graphs retain H--L edges in the biological A free branch;
only edges to biological B are removed. This also applies to opted-in v7
alternative geometry, using an explicit residue-to-biological-side map.
The v7 augmentation guard checks recipient native context and all donor
roles before using a view. Unknown or conflicting donors fall back to native
geometry. The candidate currently has no qualified augmentation routing
receipts, so its alternative views remain disabled until that provenance is
audited. Existing active recipes retain their previous behaviour. Apo,
predicted-apo and ANM structures receive no new pKAI labels.

Final independent checks passed:

- Every final SAbDab frame's tensor digest, native/preparation/teacher
  correspondence, published role, native sequence, biological side, qualified
  query identity, true-terminal eligibility, AB/free targets, weights and
  free-branch masks; all 16 canary frames match their full-build counterparts.
- Bounded real-cache probes copied 15 pKPDB and 34 PINDER compressed frames
  from Isambard, about 196 MB in total. Sixteen sampled PINDER replacement
  records preserve native geometry and all shared teacher values. The full
  native payloads were not copied or independently rehashed in this stage.
- Actual source loading covered those 49 native records and 20 size/role-spread
  SAbDab records, including populated batches and padding at candidate
  capacities. Labels, eligibility masks, branch masks and padding passed.
- Twenty targeted loader, donor-role, joint-antibody, endpoint and source
  contract tests passed in CPU allocations.

With four assembly workers, repeated one-record loads into full candidate
batch capacities had median host times of 46.8 ms for pKPDB, 89.9 ms for
PINDER and 58.9 ms for SAbDab (p95: 75.1, 136.7 and 103.8 ms respectively).
These are local host-loading checks, not model throughput or Isambard storage
measurements; the OS cache was not evicted. The frozen reports are
`semantic-verification.json`, `loader-verification.json` and
`cache-smoke-v2/summary.json`; `stage-receipt.json` pins the final evidence.

Production activation still requires target relocation/verification, an
explicit source-aware training recipe (including any required auxiliary
tables), and the independent novel-test registry freeze. Alternative views
additionally require qualified donor receipts. This stage did not run model
training, GPU jobs, MaSIF annotation, a full release transfer, or a commit.
