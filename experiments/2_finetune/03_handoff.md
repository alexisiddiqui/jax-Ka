# Experiment 02 data handoff

Submission record (2026-10-05): initialization 731899; feature shards 731900–731931.
Each shard requests two CPUs and 4 GB. The separate PypKa recovery array is 731801,
with collector 731864. All exclude comp1400. The consolidated report was generated
by job 731865 and its two figures were visually reviewed.

The active candidate is `/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/finetune/handoff-v2`.
Use it only after `verification.json` reports nonempty train and validation pKAI
support and successful frozen-prediction checks. These checks have passed: 64,114
frozen-state predictions checked; 23,958/8,157 train/validation pKAI pairs and
3,105/1,379 interface pairs. The experiment 02 diagnostic is complete;
see [its results](05_diagnostic_results.md).

The source is the verified full structural benchmark, with the fixed 500 training
pilot IDs and 151 frozen validation IDs. Test structures and labels are absent.
Targets use the authoritative split-specific uncertainty masks and full site keys
`complex_id, chain, resnum, icode, group`. Labels are bound minus free partner.

| Artifact | Meaning |
|---|---|
| manifest.json | Source hashes, code hashes, split and target counts |
| targets.parquet | Valid paired PypKa labels and available frozen baselines |
| structures.parquet | Frozen state paths and hashes |
| pkai_pairs.parquet | Paired AB/free feature-file paths and row indices |
| catboost_pairs.parquet | PROPKA residual targets joined to geometric features |
| geometry.parquet | Geometry-only features with explicit proxy definitions |
| verification.json | Alignment, frozen prediction checks, coverage and hashes |

pKAI feature archives contain `x` (4008 native input features per row) and
`absolute_pka` (unrounded frozen predictions). Match rows through the paired
index table, never by assuming identical AB/free ordering. Released pKAI supports
six side-chain types; unsupported groups are counted explicitly. Fit normalization
and weights on training rows only. Report interface results separately from the
broader shell. CatBoost uses PypKa ΔpKa minus PROPKA ΔpKa as its target.

The original `handoff-v1` is invalid: the extractor assumed NumPy was installed in
the minimal pKAI environment, and its first verifier incorrectly allowed zero
pKAI support. `INVALID.json` records this; never use that version for fitting.
Version 2 exports native torch values through JSON and compresses them in the
runner environment, preserving the validated pKAI environment unchanged. The
verifier now requires nonempty train/validation support and baseline checks.

Pilot timeout recovery is independent: 61 states across 26 training complexes are
retried with a three-hour per-state limit. It produces a separate label overlay.
Do not substitute those labels into an existing handoff without issuing and
verifying a new version. One complex's two preparation failures remain separate.
