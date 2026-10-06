# Experimental pilot curation

The v1 release lives in `_runtime/jax-Ka/pkabench/experimental/pilot-v1/`.
`report.md` is the human-readable result; `release_manifest.json` records artifact
hashes and the generating Slurm job. This is a curation preflight, not a dataset
approved for model training.

Run these scripts only inside a Slurm allocation after sourcing
`_HPC/install/jax-Ka/pkabench/common.sh`, using the runtime `runner` environment:

1. `fetch_sources.py`: archive primary texts, metadata and initial candidate structures.
2. `inspect_sources.py`: extract primary tables and deposited structure metadata.
3. `fetch_pkadr.py`: archive the versioned PKAD-R secondary inventory.
4. `build_manifest.py`: extract verified measurements, preserve secondary candidates,
   check residue numbering and write validation receipts/report.

Use two CPUs and 2 GB per CPU; exclude comp1400 and observe the user-wide 400-core
queued/running cap under the existing submission lock. Source downloads and parsing
also belong on compute nodes. Existing downloaded sources are retained by the builder.

Outputs distinguish:

- `verified_measurements.json` / `.csv`: primary-text affinity and avidity records,
  with original observable, conditions, uncertainties, construct holds and provenance.
- `pkadr_candidate_inventory.json`: 1,024 secondary records, with raw fields intact.
- `lead_residue_candidates.json`: 42 barnase/protein G candidates. Censored and
  approximate measurements have no ordinary numeric point target.
- `residue_mapping_preflight.json`: author-number/name checks; these do not certify
  construct identity, completeness, terminal chemistry or quality masks.
- `reservation_proposal.json`: two connected evaluation-candidate blocks; this does
  not change the existing synthetic split.
- `validation.json`, `counts.json`, `release_manifest.json`: validation and provenance.

Training eligibility stays false until primary conditions, exact constructs,
structure preparation/masks and sequence/component leakage checks are resolved.
Absolute pKas, bound-minus-free residue shifts, binary affinity and avidity are
different observables and must not share an unqualified target column.

## v2 structural preflight and frozen baselines

The next release is in `_runtime/jax-Ka/pkabench/experimental/pilot-v2/`.
Its report, plot, per-observation CSV and JSON record provisional comparisons
against secondary labels; these are not certified independent evaluation results.

Compute-node sequence:

1. `prepare_baselines.py` uses the runner environment to prepare isolated chains,
   check components and apply the existing natural-gap overlay. It excludes
   zinc-containing 1A2P and checks the deposited, sequence-identical alternative 1BNI.
2. `baselines.sbatch` runs nine tasks: three structures × PROPKA/pKAI/JAX-Ka.
   Submit under the existing lock/cap checks, after preparation completes.
3. `audit_overlap.py` checks the actual frozen assignments and archives the pKAI
   paper's description of its upstream split. Use the runner environment.
4. `check_missing_outputs.py` records PROPKA suppression and pKAI terminal support.
5. `score_baselines.py`, in the finetune-v1 environment, validates receipts and
   generates common-site metrics and the plot. It never admits labels to training.

Completed prediction directories are immutable inputs. Use a new release directory
for a new run. The current scripts target their named release; do not resubmit an
already completed array or assume rerunning preparation refreshes an existing gap
overlay. The released metadata identifies the exact code used at execution.

## Frozen CatBoost–PypKa hybrid extension

`_runtime/jax-Ka/pkabench/experimental/hybrid-v1/` contains the separate hybrid
release. The original experimental pilot outputs are read-only inputs.

Run `hybrid_teacher.sbatch`, then `hybrid_inference.sbatch`, then
`hybrid_replay.sbatch`, then `hybrid_score.sbatch`, with `afterok` dependencies
and the same submission lock/resource checks. Each task requests two CPUs and
4 GB. The three structures run concurrently. Each structure's unmodified teacher
replay must pass before its three hybrid seeds run.

`hybrid_features.py` verifies its extracted feature implementation against a
saved native-intrinsic feature fixture. `hybrid_predict.py` checks and loads the
frozen models; it never calls fit. `hybrid_replay.py` reuses the existing energy
conversion and native-order Monte Carlo implementation. Artificial termini and
masked sites retain teacher intrinsics. The scorer restricts the comparison to
identical masked point labels with replaced intrinsic terms and valid outputs
from all methods. The report distinguishes teacher assistance from standalone
prediction and retains all experimental source-verification holds.

## Full PKAD-R expansion

The versioned structural audit is in
`_runtime/jax-Ka/pkabench/experimental/pkadr-full-v1/`. `full_inventory.py`
creates one immutable task per PDB/author-chain, prepares structures, and applies
the existing missing-region and removable-component policies. Its Slurm array is
followed by `collect_full_inventory.sbatch`, which forms sequence-family blocks
and checks the actual frozen candidate universe at 30% identity and 80%
bidirectional coverage. Alternative PDB identifiers are written to an explicit
recovery queue rather than silently substituted.

`full_baselines.py init` creates the frozen-method task list. The corresponding
array runs PROPKA, pKAI, pKAI+, JAX-Ka and PypKa sequentially for each prepared
structure and never fits a model. After every task has a complete receipt,
`score_full_baselines.sbatch` reports the exact common-method intersection.
Its headline metric is family-macro MAE, its uncertainty unit is a whole sequence
family, and its shifted subset requires an experimental deviation of at least
0.5 pKa from the residue-type null. Pooled site metrics are supporting data.
Censored records remain one-sided constraints. Primary-source and construct
verification still gate both training and independent headline evaluation.

The original full-baseline JAX-Ka adapter inherited the 64-step constructor
default. That release remains immutable. `full_jaxka_recovery.sbatch` creates a
JAX-Ka-only recovery release using the accepted frozen-production configuration
of 1,024 solver steps and the unchanged residual and midpoint-validity rules.
After its array completes, `score_full_baselines_v2.sbatch` substitutes only
those JAX-Ka predictions and records both configurations' coverage. It does not
fit a model or alter any other method output.

## Redox-state recovery diagnostic

`dsba_reduced_recovery.py` treats the reduced DsbA structure 1A2L as a new,
immutable structure task for PKAD-R record 142. Preparation requires an exact
deposited-sequence match to the original 1DSB mapping, a complete Cys30 thiol,
and no detected disulfide. It does not overwrite the oxidized 1DSB task.

The five-task method array runs frozen PROPKA, pKAI, pKAI+, JAX-Ka and PypKa
adapters. JAX-Ka uses the accepted 1,024-step configuration. The dependent
scorer compares the oxidized and reduced predictions to the experimental 3.5
pKa label and writes a separate anchor-aware natural-gap annotation. That
annotation uses the calibrated visible-flank rule for the two short terminal
gaps; the conservative missing-region envelope remains in the report as a
diagnostic rather than an automatic exclusion. This is a one-record state
correction check, not a method ranking or a fitted model.

## Experimental admission ledger and candidate folds

`build_experimental_admission.sbatch` builds the versioned admission release
from the immutable full structural inventory, corrected frozen baselines,
primary-source decisions and reduced-DsbA recovery. It writes all 1,024 archive
records to a ledger while keeping structural candidacy, primary verification,
fold assignment and final eligibility as separate fields.

The five candidate folds contain only locally train/validation-disjoint numeric
point labels. A whole sequence family is assigned to exactly one fold. The
deterministic assignment balances record count, residue type and null-relative
shift, then applies local moves and swaps to reduce imbalance caused by large
families. Receiving a fold never admits a label: both model-fit and headline
evaluation flags remain independent of the fold assignment. The v8 fit gate
admits only rows whose recorded primary-source, construct, state, condition and
structural checks pass; headline-evaluation eligibility remains false.

`primary_review_queue.csv` orders unchecked candidates by information content:
shifts of at least 2 pKa first, then shifts of at least 0.5, then near-null
records. Censored labels remain outside the point folds pending an interval-aware
objective. The first greedy-fold release is retained as v1; v2 adds the review
queue and optimized assignment. v3 incorporates the first prioritized
primary-source batch. v5 adds the second review batch and explicitly represents
alternate-structure recoveries. v6 records the prepared, state-correct human
thioredoxin replacements. v7 resolves the remaining six large-shift records and
adds machine-readable primary-condition corrections and label interpretations.
v8 applies the first fit gate to all 21 exact or recovered-exact candidates,
admits 12 verified point labels, and records explicit holds for the other nine.
Family membership and fold assignment are unchanged; v8 is the current
candidate freeze.

## First experimental JAX-Ka fit

`fit_experimental_jaxka.py` exercises the shared `LocalTerms` and implicit
solver path on the 12 v8 fit-eligible scalar labels. It changes only the three
global JAX-Ka physical scales and uses a Huber loss on
`effective_pka(pH=label) - label`; it does not synthesize titration curves.
Each of the five sequence families receives equal total weight.

The unconstrained v1 endpoint lowered the weighted scalar objective from
2.0999 to 0.5795, but reduced valid production midpoint coverage from 10/12
to 6/12. That endpoint fails the readout gate: a target-pH residual can improve
while the full titration curve becomes invalid elsewhere. The immutable v2
audit checks all 81 saved states and selects update 3, the lowest-loss state
that retains every initially valid midpoint. Its scales are 0.9208
desolvation, 1.0868 hydrogen-bond/reorganization and 1.0867 Coulomb.

On the ten-label common support, frozen versus selected training MAE is 2.526
versus 2.487 pKa and RMSE is 3.433 versus 3.349. Family-macro MAE changes from
1.870 to 1.916, so this fit does not establish an improvement across families.
These are all-data resubstitution diagnostics. Leave-one-family-out fitting is
required before estimating transfer, and the two initially invalid labels
(DsbA Cys30 and T4 lysozyme His31) remain invalid.
