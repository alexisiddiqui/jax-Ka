# jaxpb: batched JAX Poisson–Boltzmann + MC titration (PyPKA replacement)

Design: `docs/JAXPB.md`. Code: `src/jaxpb/`. Unit tests: `tests/test_jaxpb_{records,assemble,titrate,api}.py`.

This directory holds the build record from the standalone `jaxpb-pipeline` checkout (2026-11-19..21):

- `briefs/` — the build briefs per stage (BUILD-M0-M4, M5, M6, PROFILE, SPEED, BENCH).
- `results/` — closing-test numbers per milestone (`M*.json` = Coulson CPU / Isambard staging, `M*-a40.json` = A40 runs,
  `determinism.json`, `profile.json`, `speed.json`, and the energy-track / PB-track hand-over notes).
- `closing/` — the GPU closing-test scripts (argparse CLIs, not pytest): `pb_m1.py`, `pb_m2.py`, `pb_m5.py`, `sor_pallas.py`,
  `determinism.py`, `m6.py`. In the standalone checkout they were `tests/test_<name>.py`.
- `jobs/` — the Coulson / Isambard sbatch wrappers, stage scripts and benchmark drivers, kept as run. They were run from frozen
  copies of the standalone layout, so they refer to `tests/test_<name>.py` (now `closing/<name>.py`, or
  `tests/test_jaxpb_{assemble,titrate,api,records}.py` for the energy-track and driver tests) and `src/jaxpb/prep_pypka.py`
  (now `python -m jaxpb.prep_pypka` with `src` on `PYTHONPATH`). `jobs/a40site/sitecustomize.py` is the sm_86 Pallas shim for
  the Coulson A40 (on `PYTHONPATH` only there, never on Isambard).

## What each milestone tested

| M | closing test | result |
|---|---|---|
| M0 | goldens (5 v2 pilot outputs + 99 val states recorded through PyPKA on Coulson), compile counter, cache tarball | v2 numbers reproduced within 2e-6, sweeps 100%, Pallas executables served from the tarball (`M0-a40.json`) |
| M1 | records, size classes, argument-pure maps + PB (XLA lane pool) vs v2 | sweeps 100%, 0 compiles on pass 2; near miss: 1/2,329 tautomers at 1.05e-6 (GPU run-to-run f32 nondeterminism, fixed in M5) and 45 compiles vs budget 44 (`M1-a40.json`, `determinism.json`) |
| M2 | DelPhi spectral-radius estimator | sweeps equal to DelPhi 0.905 focused / 0.865 model; L2 solvation p99 4.3e-3 without logged ρ (`M2-a40.json`) |
| M3 | JAX assembly (calcBackEnergy, calcpKint, calcInteractionWith, pair rule) | g, W vs saved mc-energies on 99 states, worst 1.9e-6 kT (`M3-a40.json`) |
| M4 | MC titration in fixed Mp buckets, pKhalf | MC counts bitwise equal to v2 on 426/426 states at 50k and 200k sweeps, Pallas vs XLA bitwise (`M4-a40.json`) |
| M5 | Pallas SOR kernel + determinism | sweeps 100.00% equal to XLA (92,691 calls), energies bitwise, SOR 10.2×, PB 3.5× (`M5.json`) |
| M6 | driver (`jaxpb.api.run`): shards, resume, overflow retry | L3 on 104 states vs PyPKA: pKint MAE 0.0042, W p99 0.0049 kT, pKa MAE 0.0084 at 200k sweeps (floor 0.0041), 0 midpoint mismatches; compiles flat at 42 from 104 to 1,040 states; resume and fresh-process runs bitwise (`M6.json`) |
| SPEED | geometry + maps levers 1–4, 6, 7 | every lever bitwise on 104/104 states; 104-state mix 239 → 346 states/A40-h, labels bitwise equal to M6 (`speed.json`) |

M7 (converged PB mode) is not built.

## Running the driver

```
PYTHONPATH=src python -m jaxpb.api <out_dir> --inputs shard_00.npz shard_01.npz [--cache cc.tar] [--shard-size 256] \
    [--mc-sweeps 200000] [--backend auto|pallas|xla] [--max-shards N]
```

or `jaxpb <out_dir> ...` after `pip install -e .`. Inputs are `jaxpb.records` StateRecord shards written by the PyPKA producer
(`python -m jaxpb.prep_pypka prep|record ...` in the PyPKA env on Coulson). The driver writes `labels_<k>.npz`, `plan.json` and
`run.log.jsonl`; a restart with the same items and config skips finished shards. Importing `jaxpb.api` turns process-wide x64 on
(the PB path runs its float32 programs inside an x64-off context).

## Unit tests

```
PYTHONPATH=src JAX_PLATFORMS=cpu pytest tests/test_jaxpb_*.py
```

Optional data-backed checks: `JAXPB_GOLDENS` (glob of M0 golden records shards) and `JAXPB_PYPKA_STS`
(`<pypka>/G54A7/sts`); they skip when unset. `tests/test_jaxpb_titrate.py cmf|cpu` read `JAXPB_MFS_V2`.
