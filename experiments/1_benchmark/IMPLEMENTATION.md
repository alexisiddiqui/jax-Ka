# Smoke pipeline operations

The `pkabench` package implements fixed-manifest preparation, three-state
subprocess adapters, immutable job shards, strict merging, smoke scoring and
gate evidence. Metadata and a bounded coordinate sample are now downloaded for
candidate-pool planning; no production train/test split has been created.

## Execution

All commands below are submitted through Slurm. Do not run Python, dependency
installation, tests, archive inspection, prep or analysis on a login node.
The Python CLI and shell entrypoints reject execution outside an allocation,
on comp1400, or without `--mem-per-cpu=2G`.

Submission wrapper:

```bash
submit=/home/coulson/oc/lina4225/_HPC/submission/jax-Ka/pkabench/submit.sh
campaign=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/campaigns/smoke-v3
bash "$submit" install runner
pypka_install=$(bash "$submit" install pypka)
PKABENCH_AFTEROK="$pypka_install" bash "$submit" install fortran
bash "$submit" install pkai
```

Wait for installation receipts before submitting dependent stages. Environments
are stored under `_runtime/jax-Ka/pkabench/envs`; package locks, source/weight
hashes, logs and receipts are adjacent. Validated environments are immutable.
PypKa needs Python 3.10, NumPy 1.26 and the legacy `libgfortran.so.4` runtime
shipped in the isolated `fortran` prefix. The runner uses Python 3.11.
The exact tested Python patch versions, package resolutions and Fortran package
builds are pinned in the installer and adjacent requirements/explicit files.

```bash
bash "$submit" work validate --tests tests/test_pkabench.py tests/test_interface_exposure.py tests/test_reference_parser.py tests/test_reference_termini.py tests/test_biotite_integration.py
bash "$submit" work discover
bash "$submit" work prepare \
  --manifest /home/coulson/oc/lina4225/_HPC/install/jax-Ka/foldbench-protein-protein.csv \
  --archive /home/coulson/oc/lina4225/jax-Ka/ground_truth_1522.tar \
  --out "$campaign" --count 50 --seed 20261003 \
  --completion-executable /home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/pypka/bin/pdb2pqr30
bash "$submit" work inspect --campaign "$campaign"
bash "$submit" work preflight --campaign "$campaign" --methods propka jaxka pkai pkai_plus kaml
bash "$submit" work gates --campaign "$campaign"
```

`prepare` refuses to overwrite a frozen manifest. Use a new campaign directory
after changing prep. FoldBench input now requires exactly two observed protein
chains matching the selected partners. The earlier smoke-v2 run could extract
a pair from a larger assembly; it remains an immutable diagnostic record, not
a conforming benchmark. SAbDab H+L/antigen selection is not implemented by this
FoldBench entrypoint. Assembly-wide ligands/metals/noncanonical components are
rejected, including components outside the selected chains. The rejection audit
below supports retaining these chemical exclusions. Coordinates are canonicalized
to 0.001 Å before splitting states to match PDB-only methods.

The submission wrapper locks scheduler bookkeeping, expands pending array elements,
counts all this user's live CPU requests, and admits work only below 400 CPUs.
Both responsive CPU partitions are eligible; unavailable nodes and comp1400 are
excluded on each submission. Work jobs reserve 2 CPUs/4 GB; installation jobs
reserve 4 CPUs/8 GB. Predictors remain single-threaded. No exclusive/GPU resources
are requested. External submissions outside this wrapper cannot share its lock;
the wrapper rechecks live demand before each admission.

## Gates and diagnostic runs

`gates` runs the smallest accepted pair through the teacher and records G1/G1b,
the acceptance histogram, and the latest `preflight` evidence for G3–G5.
PypKa uses native `getTitrationCurve()` output, never midpoint-reconstructed
`getAverageProt()`. Raw Monte Carlo energies include tautomer-state interactions;
these are retained without inventing a scalar site-pair reduction.

Historical G2 remains unresolved; current-label G2 is documented in
[the frozen smoke revision](17_frozen_smoke_and_scoring.md). The
[historical evidence amendment](18_historical_pkpdb_settings.md) records conflicts
between generation code and shared server constants. Official server `PKPDB_PARAMS` metadata and live precomputed
responses agree on dielectric 15/80, ionic strength 0.1, grid 81, G54A7, no ions,
nonperiodic boundaries and SER/THR titration off. The teacher now explicitly sets
`ser_thr_titration=False`. The metadata names PypKa 2.1.0; the installed teacher is
2.10.0. Temperature and hydrogen optimisation are absent. These shared server
constants are not independent evidence and are not a per-simulation `sim_settings`
export, so they cannot establish historical SER/THR settings or close historical G2.
The historical generation code also omits temperature/hydrogen optimisation when
saving its settings; an export may need accompanying version/configuration provenance.

When an actual deposit export is available, submit `work verify-deposit
--manifest /absolute/path/to/settings.json --source-url <deposit-source>`.
The JSON may contain settings directly or in `pypka_params`, `delphi_params` and `mc_params`.
The verifier records the source, content hash and per-parameter comparisons;
missing evidence stays unresolved and mismatches fail G2. The current installation
defaults are never used to fill missing historical settings.

KaML is excluded by its released single-chain-only interface. `preflight` records
this with source provenance; it does not synthesize a new multichain predictor.
The other G3 checks measure partner-order/label invariance and AB versus isolated-partner
response. PROPKA's native determinant graph is also inspected for explicit cross-chain
terms: this establishes joint-system eligibility while its measured order sensitivity
remains a separate caveat. Far-box control failures are reported independently. G4 executes a real
gradient/update step in memory for both released pKAI models without altering weights.
G5 deletes a heavy atom and validates the PDB2PQR reconstruction against preserved
identities and observed coordinates.

After G1 and retained-method G3 pass, the following completes diagnostic work on
the *already frozen* smoke set despite a low acceptance rate or unresolved G2:

```bash
bash "$submit" work dispatch --campaign "$campaign" --diagnostic \
  --methods pypka propka jaxka pkai pkai_plus null
```

This does not authorize production expansion. Without `--diagnostic`, acceptance
and G2 block dispatch. The diagnostic exception is limited to at most 50 frozen
candidates. `submission.json` records submitted job IDs; `unsent.json` records work
left outside the scheduler. Repeat `dispatch` after capacity becomes available.

After all expected jobs finish:

```bash
bash "$submit" work merge --campaign "$campaign" --methods pypka propka jaxka pkai pkai_plus null
bash "$submit" work score --campaign "$campaign" --methods propka jaxka pkai pkai_plus null
bash "$submit" work report --campaign "$campaign"
```

Merge includes failed-method rows as coverage failures but rejects missing jobs,
duplicate sites, incompatible inputs/configuration/code hashes and corrupted outputs.
Successful shards cannot be overwritten after code changes. Failed attempts are
archived before a corrected retry. Use a new campaign for scientific code changes.

Scores are explicitly smoke diagnostics labelled agreement with PB. They include
common-site metrics, separate coverage, structural zeros and linkage completeness.
No component-bootstrap confidence interval is manufactured for this unsplit set.
Teacher runs emitting groups outside the nine-group schema cannot yield a complete
linkage curve in the normalized output; raw data remains available for reconciliation.

The `report` stage retains candidate-versus-accepted size distributions, successful
job timing quantiles, provisional resource estimates, Slurm accounting, and the
largest observed queue admission count. These few accepted pairs do not establish
a population timeout rate. `PKABENCH_AFTEROK=<jobid[:jobid...]>` can be set on the
submission wrapper to schedule a dependent stage; its CPU request still counts
against the pending-plus-running cap.

`PKABENCH_AFTERANY` is available for final collection after both successful and
failed jobs. Submit `work finalize --campaign ... --methods propka jaxka pkai
pkai_plus null` to merge, score, refresh gate evidence and write the resource
report in one allocation. Scheduler-level OOM, timeout and node failures are
reconciled to explicit failed prediction rows; unfinished jobs still block merge.

## Rejection and teacher audit (2026-10-03)

Audit outputs are under `_runtime/jax-Ka/pkabench/audits/`:

- `prep-v1/audit.json`: all 50 frozen candidates, every forbidden component,
  distance to the pair, binary eligibility, and two diagnostic counterfactuals.
- `deposit-v1/`: commit-pinned official server sources and archive hashes.
- `teacher-v2/audit.json`: saved `/query/4lzt` and `/query/1ubq` responses and
  comparison with the corrected teacher settings. These endpoints only retrieve
  precomputed data; no remote predictions were requested.
- `findings-v2.json`: consolidated evidence, corrected curation and the native
  teacher validation in AB/A/B.

The original 7/50 acceptance becomes 21/50 only if all off-chain components are
dropped. Keeping forbidden components within 10 Å yields 8/50. This is a diagnostic
sensitivity check, not approval to drop distant components or a validated chemical
association rule. Most losses cannot be attributed solely to unrelated distant
components. Only 21/50 candidates are binary assemblies; five original accepted
pairs satisfy that rule. Corrected smoke-v3 uses the same 50 identities and accepts
5/50 (five of 21 binary candidates), with no pipeline errors. No filters were relaxed.
The sole candidate recovered by the 10 Å counterfactual is 8ih8 (a GOL component
25.95 Å from the pair); it is itself a nonbinary assembly. Thus this relaxation
does not rescue another conforming binary candidate in the sampled set.

The corrected teacher passed AB/A/B with native 73-point curves, SER/THR disabled,
and no unexpected output groups. Charge coverage is still incomplete: ARG17 in
each chain of 8tto is not reported. Installed PypKa `constants.py` excludes ARG
from `TITRABLETAUTOMERS`; this is a teacher capability limitation, not a failed
job. Do not invent an ARG midpoint or native titration curve. The current linkage
code conservatively flags the missing curves. Supporting cancellation of the
teacher's fixed charges requires explicit representation and validation before
claiming complete linkage. The corrected code passed 43 relevant tests in Slurm;
the retained-method preflight and G5 completion check passed again.

The recommendation at that strict-audit stage was to construct a curated binary protein-only candidate
pool before expanding the benchmark. Do not repeatedly resample until the acceptance
gate happens to pass. Preserve exclusions and report the selection funnel.

The source evidence is
[official server constants](https://github.com/mms-fcul/PypKa-Server-Back/blob/824709cbb158769d4403fb81c5a7a2b3aadcd8ec/server/const.py),
[the query route](https://github.com/mms-fcul/PypKa-Server-Back/blob/824709cbb158769d4403fb81c5a7a2b3aadcd8ec/server/routes_direct.py),
and the locally pinned pKPDB `src/fill.py`. The public CSV export contains labels,
not per-simulation configuration. `export_pkpdb_settings.sql` is a read-only query
for a database owner to export settings for two representative records; it has not
been executed against any external database. Historical provenance or an explicit
decision to treat the current teacher as distinct from pKPDB remains necessary.

The audit commands use the same compute-only entrypoint:

```bash
bash "$submit" work audit-prep --campaign /absolute/path/to/smoke-v2 --out /new/audit/directory
bash "$submit" work audit-sources --out /new/source/directory
bash "$submit" work audit-teacher --source-root /source/directory --out /new/teacher/audit
bash "$submit" work audit-findings --campaign "$campaign" --prep-audit /audit/directory/audit.json --out /new/findings.json
```

Scientific changes invalidate old completed shard hashes. Do not merge smoke-v2
predictions into smoke-v3 or treat the old scores as corrected results. Smoke-v3
currently validates preparation, preflight and one teacher pair; it is not a new
full method campaign.

## Full local protein–protein feasibility audit

The user requested checking the rest of the dataset before interpreting the smoke
losses as acceptable. All 279 unique general protein–protein FoldBench pairs from
239 assemblies were screened, including the 229 outside the original smoke sample.
No source CIFs were missing and no candidate/scenario had a pipeline error.
Sixteen independent Slurm jobs used 2 CPUs/4 GB each, excluding comp1400 and using
the existing admission cap. These jobs generated diagnostic prep only, not teacher
labels or a production split. The audit/benchmark test subset passed 23 tests.

| Diagnostic scenario | Passing pairs | Split components at 30% identity |
|---|---:|---:|
| Current binary-assembly and chemical filters | 28 | 27 |
| Remove listed neutral additive candidates | 32 | 31 |
| Also remove listed buffer/salt candidates | 35 | 34 |
| Same additive cleanup, allow selected pairs from larger assemblies | 62 | 55 |
| Selected chains only, discard all outside components (unsafe upper bound) | 118 | 103 |

The same geometry, interface-defect and completion filters run after each proposed
cleanup. The scenarios are not adopted production policies. Neutral candidates
are GOL/EDO/PEG/PGE/PG4/1PE/MPD/DMS; broader candidates include SO4/PO4/ACT/FMT/TRS/MES/HEP/BME.
Declared covalent/metal connections and sub-1.9 Å pair proximity protect candidates
from removal. Missing connection records do not establish absence of a bond.
Component identity and distance alone do not establish that removal preserves
binding or electrostatics. In particular, GOL in rescued 7x80 contacts both partners
within 4 Å. The other rescued binary structures contain EDO, PEG, GOL, SO4 or FMT;
these are a small, reviewable cleanup pilot rather than an unrestricted allowlist.

The 30% identity rule groups examples; it does not delete representatives. The audit
uses MMseqs2 18-8cc5c all-versus-all alignments, >=80% coverage of both chains, and
connected components of the chain-similarity and complex-partner graphs. The graph
includes all 279 candidates before prep, including rejected bridges. All 558 chain
occurrences had deposited canonical sequences (314 distinct sequences); no observed
sequence fallback was needed. At 50% and 90% identity, the strict pool still has 27
components, and the additive-cleanup binary pool still has 34. No CDR substitution,
experimental forced-test assignment, or final split was performed.

Selection is the major loss here: 139/279 candidates fail the two-chain assembly
rule before chemistry is considered. Of the 140 binary candidates, 28 pass unchanged
and 35 pass the broader diagnostic cleanup. Acceptance also decreases with size:
12/96 below 200 residues, 11/106 at 200–499, 5/67 at 500–999, and 0/10 at >=1000.
These are descriptive selection effects, not an estimate of performance bias.
The remaining 229 candidates yield 23 strict passes, matching the smoke's low yield.

Do not extrapolate the weak extra clustering loss to the PDB-wide training universe:
[FoldBench is already a low-homology benchmark](https://github.com/BEAM-Labs/FoldBench)
and its antibody–antigen, protein–peptide and protein–ligand collections are separate
tasks. This audit covers the general protein–protein manifest only. The local pool
cannot provide the planned 500 test complexes even before filtering.

Artifacts: `_runtime/jax-Ka/pkabench/audits/foldbench-full-v1/` contains `summary.json`,
`report.json`, `retention.csv`, `size_bias.csv`, per-candidate `rows/`, diagnostic
coordinates, source hashes, MMseqs2 release/binary receipts, input FASTA and hits.
Reproduce with `work dataset-audit index`, `scan --shard N --shards 16`,
`install-mmseqs`, `sequence`, and `summarize --shards 16`, all through the existing
Slurm submission wrapper. Every stage takes `--root`; `index` additionally takes
`--manifest`, `--archive` and the original `--smoke .../manifest.json`.

Recommendation: review the seven rescued binary structures to define additive
handling, and revisit whether multi-chain assemblies can supply labelled pairwise
examples with explicit omitted-context annotations. Keep the existing filters
until those definitions are settled. Small-molecule/drug/ligand training is deferred
to [section 05](../5_ligands/05_ligand_complexes.md), after backbone-only protein work.

The follow-up `work dataset-audit defects --root .../foldbench-full-v1` separately
inventories missing atoms and unobserved sequence positions in `defects.json`.
Its multi-chain histogram is 58 three-chain, 55 four-chain, one five-chain,
17 six-chain, five eight-chain, one twelve-chain and two eighteen-chain cases.
Of those 139 assemblies, 123 have <=1,500 observed canonical residues. These are
chain-count exclusions, not gap/atom-defect exclusions. Among the 28 strict passes,
24 already have unobserved terminal sequence positions and five have distal
missing atoms before prep. Missing segment locations are unknown; flank distances
are recorded as evidence, not as proof of the missing segment's spatial extent.
The revised site-supervision policy is specified in `00_shared.md` and implemented
by `curation-pilot`. The strict audit remains an immutable historical baseline.

## Historical exposure and alternate-conformation pilot

This section records the superseded `curation-v3` diagnostic. The current
PDB2PQR-inclusive policy below does not apply these radius or altloc masks.

`curation-v3` uses two explicit partner sets, optional narrow neutral-additive
cleanup, distal defect masking, and a separate observed student input. It records
omitted source protein context rather than claiming a full-assembly response.
The additive policy removes only eligible nonbridging GOL/EDO/PEG instances;
SO4/FMT and bridging GOL remain excluded. The seven reviewed binary structures
are recorded in the curation report. Geometry and interface-defect filters remain.

The `sites` schema version 2 contains both uniform `supervision_mask_10/15/20`
and `supervision_mask_adaptive_10/15/20` comparisons. The latter use 5/8/10 Å only
for short, exposed-anchor tail candidates defined in the shared contract. The
default provisional mask is the adaptive 15 Å policy (8 Å for that tail class).
Other defects retain the general radius and conservative uncertainty envelopes.
No missing-site labels are imputed; masked inputs and supervision remain distinct.
Partial supervision blocks whole-system linkage reporting.

`conformers.json` retains selected and alternate residue IDs, occupancies, atom
coordinates, missing-atom inventories and original source row indices. Selection
is coherent within each residue, with shared atoms retained and zero-occupancy
atoms removed. A selected conformation is prepared once and split into AB/A/B.
Ambiguous-residue targets and neighbouring uncertain regions are masked pending
validation. Repeated local alt letters are not treated as a global ensemble.
Multiple coordinate models are explicitly rejected by this pilot until separate
model-specific paired examples are implemented; they are not silently collapsed.

`curation-report` adds adaptive/uniform retention, residue-type/size/distance
strata, exposure classes and sequence-component counts. Sequence clustering still
groups split units rather than deleting examples. All 279 candidates contribute
to the graph, including rejected bridges. Full source files remain unchanged.

Use the existing Slurm wrapper for all stages:

```bash
bash "$submit" work curation-pilot init --audit /path/to/foldbench-full-v1 --campaign /new/campaign
bash "$submit" work curation-pilot scan --campaign /new/campaign --shard 0 --shards 16
# Submit each remaining shard, then collect after all finish.
bash "$submit" work curation-report --campaign /new/campaign
bash "$submit" work sensitivity init --campaign /new/campaign --out /new/defect-pilot
bash "$submit" work sensitivity init-altloc --campaign /new/campaign --out /new/altloc-pilot
```

Each sensitivity initialiser writes `tasks.tsv` listing frozen campaign paths and
complex IDs. Submit each through `work run --campaign PATH --complex-id ID
--method pypka`, then `work sensitivity collect --out /pilot/path`. The defect
pilot compares intact/repeat, distal side-chain deletion/reconstruction and
two-residue terminal truncation. The alternate-location pilot switches one
complete deposited local conformer at a time and includes a repeat reference.
These are within-configuration diagnostics, not evidence of historical pKPDB
equivalence. Three cases cannot establish a universal safe radius.

Metadata downloads are under `_runtime/jax-Ka/pkabench/universe/metadata-20261003`:
101,956 RCSB assembly-1 candidate IDs; SAbDab's current official CSV contains
22,548 instances from 11,667 entries, with 16,800 protein-antigen instances from
8,548 entries before assembly/chain/quality reconciliation. The legacy SAbDab TSV
URL returned HTML and was replaced with the current `/api/download/all-summary`
endpoint. Retain the original download evidence and validated CSV hash.
All 24 seeded RCSB coordinate samples downloaded successfully (3,921,299 bytes);
their mean suggests approximately 15.5 GiB compressed for the broad RCSB pool,
which is a rough storage estimate, not an accepted-example count or bulk download.

## Current policy: PDB2PQR-inclusive preparation and measured radial error

The user chose to include as many missing-coordinate structures as PDB2PQR can
prepare, and to choose any distance exclusions from measured error. The current
campaign is `campaigns/pdb2pqr-inclusive-v1`. It removes pre-rejection for interface
gaps and missing side chains, attempts repair before missing-backbone topology
checks, and runs PDB2PQR on every chemically eligible selected pair. Success must
preserve observed heavy-atom identities and coordinates. There is no silent
residue deletion to make a failed preparation pass. Scope/geometry/identity
checks remain; PDB2PQR success is not a claim that every predictor will succeed.

The full local rerun retained **67/279 pairs**, with no pipeline errors, 6,037
structurally eligible native sites and 1,143 interface sites across all 67 pairs.
The retained interface-bearing pool spans 61 sequence components at 30% identity.
These are preparation/eligibility counts before teacher coverage. Rejections were
82 ligand, 46 nonstandard residue, 41 metal, 17 size, 11 glycan, 11 buried-area,
two interpartner disulfide, one alternate residue identity and one PDB2PQR failure.
Most remaining losses therefore concern the current protein-only scope rather
than missing coordinates. The preparation report preserves every candidate.

First deposited positive-occupancy altloc per residue is used with shared atoms;
the same completed geometry supplies AB/A/B. Alternate locations add neither
exclusion masks nor extra teacher runs. The old alternate-conformer campaign and
superseded teacher pilots were stopped when this policy changed. Their partial
outputs are retained separately and will not be merged into the new experiment.

`supervision_mask` now includes parameterisable native sites without a
missing-coordinate distance cutoff, including reconstructed observed residues.
Absent residues have no targets. Artificial termini are not native targets.
The older uniform/adaptive masks are diagnostic columns only. Inputs retain
observed-atom masks and reconstruction/missing-sequence provenance.

`radial-error init` selects three small complete/near-complete references, allowing
at most two originally missing terminal residues and two originally incomplete
residues, with no internal gaps or additive removal. It records actual completeness
for each case. Whole-residue perturbations remove N-terminal 1/3, C-terminal 1/3,
one buried internal residue, and a buried three-residue segment where available.
Burial is measured before deletion using bound residue SASA divided by the same
isolated-residue SASA (<=0.1 for the centre; <=0.2 for all three segment residues).
Deleted residue identities, charge-bearing types, exposure and reference atom
coordinates are saved. Rejected perturbations remain explicit outcomes.

The initial `audits/radial-error-v1` selected 8upb (128 residues), 8g0p (241) and
8zlz (260), each with zero missing sequence positions and zero missing heavy
atoms in the selected partners. All 17 deletion variants prepared successfully.
8g0p has no eligible buried three-residue segment under the stated screen; this
unavailable variant is recorded rather than substituting an exposed segment.
Together with intact/repeat controls, the pilot contains 23 teacher jobs.

```bash
bash "$submit" work radial-error init --campaign /path/to/pdb2pqr-inclusive-v1 --out /new/radial-pilot
# Each tasks.tsv entry is submitted through the same capped wrapper:
bash "$submit" work run --campaign /pilot/complex/variant --complex-id ID --method pypka
bash "$submit" work radial-error collect --out /new/radial-pilot
```

Baseline and repeat calculations check reproducibility at the same settings and
default Monte Carlo seed (1234567); they do not estimate independent-seed sampling
variance. Perturbed inputs
receive the same preparation and teacher settings as the reference. Distances
are measured from retained functional-group atoms to the nearest **deleted atom
in the intact reference**, not to an inferred missing-region envelope. Collection
reports AB pKa error, free pKa error and paired ΔpKa error, per-site coverage,
0–5/5–10/10–15/15–20/20–30/>=30 Å bands and cumulative outside-radius statistics.
Terminal/buried deletion size and exposure strata are separate. `site_errors.csv`
retains individual signed errors and statuses; `report.json` includes median,
MAE, RMSE, p95, maxima and error-threshold fractions. No radius is automatically
selected from three correlated examples. The teacher remains a conditional PB
reference with unresolved historical G2 equivalence and missing ARG coverage.

### Completed radial pilot (2026-10-03)

All 23 teacher jobs (69 AB/A/B calculations) completed with no job-level teacher
failures. All 17 deletion preparations passed. The same-seed intact repeats were
identical at all 211 comparable site pairs across the three references. Missing
or out-of-range site predictions remain explicit in the per-site CSV; job success
does not imply complete coverage. The relevant regression suite passed 62 tests.

| Deletion | Reference complexes | Maximum absolute ΔpKa error at >=10 Å | At >=15 Å |
|---|---:|---:|---:|
| Terminal, 1 or 3 residues | 3 | 0.064 | 0.039 |
| Buried, 1 residue | 3 | 0.053 | 0.053 |
| Buried, 3-residue segment | 2 | 0.150 | 0.095 |

The overall maximum was 1.218 pKa units for an N-terminal three-residue deletion;
the buried three-residue maximum was 0.952. Large local outliers coexist with much
smaller distant errors. Absolute AB/free pKa errors are also retained, so paired
cancellation can be inspected rather than assumed. These few chosen references
and short deletions do not validate a general safe radius, long missing tails,
large loops, or coordinate relaxation. The inclusive policy therefore keeps
distance as metadata and does not impose a new cutoff from this pilot.

Final artifacts in `_runtime/jax-Ka/pkabench/audits/radial-error-v1/`:
`report.json`, `site_errors.csv`, `radial_shells.csv`, `outside_radius.csv`,
`radial_shells_by_exposure.csv`, `per_variant.csv`, frozen source/deletion manifests,
prepared coordinates, native teacher curves/energies and version/hash receipts.
The earlier two-reference report is clearly separated in `partial/`.

### Bounded PDB-wide sample and expanded deletion study (2026-10-03)

`candidate-pool` freezes a reproducible 1,000-assembly sample from the downloaded
RCSB metadata: 750 general entries and 250 entries with SAbDab protein-antigen
annotations. All 1,000 coordinate downloads succeeded. First-conformer contact
discovery produced 1,450 explicit partner pairs; three assemblies were rejected
during conformer resolution. General interfaces are capped at three per assembly,
antibody interfaces at two, ranked by contact count. Antibody H/L chains must
contact and map to annotated author chains; the antigen must physically contact
the selected antibody copy. This is a bounded stratified audit, not an unbiased
estimate over all PDB interfaces.

Inclusive preparation accepted 296 pairs in 210 assemblies (202 general, 94
antibody), with zero pipeline errors. Exclusions: 555 ligand, 211 nonstandard
residue, 208 metal, 130 glycan, 37 buried area, 10 PDB2PQR failures and three
interpartner disulfides. Missing-coordinate distance does not reject candidates
or mask active supervision. Chemistry remains protein-only for this round.

MMseqs2 at 30% identity/80% bidirectional coverage groups all candidate sequences
before preparation. Full-chain grouping gives 531 components (largest 368);
concatenated antibody CDR partner grouping gives 569 (largest 288), with one
missing-CDR case explicitly unassigned. Among accepted examples, these graphs
cover 296/295 pairs in 108/125 components, with largest components 121/95. These
are connected split groups, not deleted examples; the large component still
prevents declaring a balanced frozen production split. Rejected candidates remain
possible graph bridges in this conservative feasibility audit.
Rebuilding the graph using only accepted candidates gives 111 full-chain groups
(largest 119/296) or 130 CDR-based groups (largest 85/295). Thus rejected bridges
explain some, but not most, of the large-component issue. Both views are retained.

Artifacts: `_runtime/jax-Ka/pkabench/universe/coordinate-pool-v1/` contains the
frozen download manifest, compressed and unpacked source hashes, receipts,
interface index, sequence alignments and `report.json`. Preparation is in
`campaigns/coordinate-pool-inclusive-v1/`.

The expanded reference search targets 30 cases across interface type, size
(80–249/250–399/400–600 residues) and charge fraction. Complete references are
selected first; near-complete means at most two missing terminal residues and
two incomplete residues, with no internal gap or additive removal. Deposited
canonical sequences are required, and entries cannot repeat. The combined new
and local pools supply 23 eligible entries: 17 complete and six near-complete.
The shortfall is retained rather than relaxing these limits further. The strict
17-case selection is preserved in `audits/radial-error-v2/plan.json`; the active
study is `audits/radial-error-v2-nearcomplete/`.
The selected entries comprise eight homomers, eight heteromers and seven
antibody–antigen complexes; size bands contain seven, ten and six references,
respectively. All 336 baseline/repeat/deletion preparations passed.

Variants include N/C deletions of 1/3/5/10 residues, buried single/triple deletions,
a buried charged residue where available, and three-residue internal windows
near (<=5 Å) or remote (>=15 Å) from the partner. Unavailable variant classes and
preparation failures remain explicit. All burial and distance measurements use
the intact reference. Intact references pass normal interface geometry gates;
controlled perturbations bypass only those admission thresholds, because
discarding deletions that reduce interface area would censor the experiment.
Chemical, PDB2PQR and atom-identity checks still apply. There is no relaxation.

`expanded-radial plan/prepare/assemble/monitor` runs under the existing Slurm
wrapper. The monitor submits/resumes bounded batches from per-task ledgers,
counts all user pending/running cores against 400, excludes comp1400 and uses
2 CPUs/4 GB per job. Each teacher task has a 5,400-second total AB/A/B timeout.
Scheduler failures count as missing coverage. To resume an interrupted controller:

```bash
bash "$submit" work expanded-radial monitor --out /path/to/radial-error-v2-nearcomplete
```

Collection includes baseline teacher coverage, deletion-size/exposure strata,
reference completeness/interface-type strata, and both pooled and equal-complex
MAE. Repeats retain the same seed and test reproducibility only. No radius or
production release is selected automatically. The relevant regression suite
passes 67 tests across the full run and focused follow-ups, including bounded
sampling, the perturbation geometry rule, equal-complex error weighting and
duplicate-free dispatch under the global CPU cap. The controller uses one queue
snapshot per cycle and consults accounting only for tasks that left the queue.
