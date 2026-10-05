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
