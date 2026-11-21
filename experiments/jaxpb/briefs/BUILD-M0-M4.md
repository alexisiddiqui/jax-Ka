# jaxpb-pipeline build, milestones M0–M4 — 2026-11-19 (user approved with the design's recommended defaults)

Design (read first, it is the spec): `work/jaxpb-pipeline/DESIGN.md`. Prototypes to reuse (read-only):
`work/jaxpb-pilot/{code,v2/code}` (+ reports), `work/mf-surrogate/{code,v2/code}` (+ v2/report.md, summary.json).
Defaults settled by the user: Pallas Triton kernels with XLA references (no jax-triton); PB mode "delphi" first; PyPKA producer on
Coulson; MC 200k sweeps default (configurable); store g and W per state; tolerance-based parity; compile cache as one tarball; f32
with f64 only in probe generation.

## Package
Local source of truth: `work/jaxpb-pipeline/src/jaxpb/` (package) + `work/jaxpb-pipeline/tests/` + `work/jaxpb-pipeline/jobs/`.
Modules (design §5): records.py, geometry.py, maps.py, pb.py, sor.py, assemble.py, titrate.py, api.py, prep_pypka.py, compile_budget.py.
Rules for code: every jitted function is module-level and argument-pure (no closures over arrays, no data-dependent static shapes);
all static shapes come from the bucket tables in records.py; every overflow sets a flag that the caller must handle (never silent);
each hot op has an XLA implementation; Pallas versions are optional behind `backend` and tested against XLA (interpret=True on CPU
where the local JAX supports it). No git repo/commits.

## Two tracks (run in parallel; strict file ownership — never edit the other track's files; agree interfaces via records.py
dataclasses, which the PB track owns; the energy track may request fields by message to the controller)
**PB track (M0, M1, M2)** owns: records.py, geometry.py, maps.py, pb.py, sor.py, prep_pypka.py, compile_budget.py, tests for those,
jobs/ for PB.
- M0: goldens. (a) v2 outputs of the 5 pilot states (Isambard `$R/audits/jaxpb-pilot-v2/out/`, local `work/jaxpb-pilot/v2/results/`).
  (b) ~100 validation states recorded with the pbrec hook on Coulson (DelPhi inputs + outputs + logged spectral radii + mc-energies +
  result), chosen from `~/_runtime/jax-Ka/pkabench/tierB/hybrid-mc-v1/requests` (val split only; spread of sizes; include ≥ 10 complete
  AB+A+B triples and Ab/Ag + general roles); include the extra fields assembly needs (pKmod, site type, chargediff, occ, PyPKA site
  order). Pack into few shards (one npz per ~50 states), relay Coulson→Mac→Isambard with sha256 on all sides. Compile-counter harness
  (count XLA compiles via jax monitoring/log hook) and the cache-tarball check (incl. that Pallas executables are cached).
- M1: records + size classes + argument-pure geometry/maps/PB with the XLA lane pool (design §1–2). Tests: energies/site potentials
  ≤ 1e-6 vs v2 on the 5 states, SOR sweep counts 100% equal, compile count ≤ budget on those states and 0 on a second pass, overflow
  margins raised (Ka 24, Kp 768) with automatic XL retry.
- M2: DelPhi spectral-radius estimator for focused/model runs (find DelPhi's method — delphi4py/DelPhi source in the pypka env on
  Coulson; logs in the goldens). Test: sweep counts vs DelPhi at ≥ v1 rates (focused 87–91%, model 83–88%) and L2 criteria
  (solvation median rel ≤ 1e-3, p99 ≤ 1e-2; site potentials p99 ≤ 0.02 kT/e) on the ~100 states WITHOUT logged ρ.
**Energy track (M3, M4)** owns: assemble.py, titrate.py, tests for those, jobs/ for titration.
- M3: assembly in JAX — PyPKA's calcBackEnergy, calcpKint, calcInteractionWith (sign/symmetrisation/direction averaging), float32
  casts, possible_states_g/occ, interactions + lookup, pair selection (couple_min 2.0) — read the PyPKA source on Coulson (report §1
  has file:line refs). Inputs = per-tautomer esolvM, esolvS, sitpotM (+ charges/metadata). Test on DelPhi's own recorded outputs
  (5 pilot states now, the ~100 M0 states when ready): g and W vs the saved mc-energies.json ≤ 1e-5 (abs, kT).
- M4: titration — gmc.chain kernel unchanged, fixed Mp buckets (64..2048), module-level jits, padded states free, curves_to_pka
  identical to PyPKA's pKhalf; exact-cluster MF (cmf) as deterministic fallback. Tests: the 426 saved-energy states (mf-surrogate v2
  pack on Isambard/Coulson) reproduce the v2 report rows at 50k and 200k sweeps; reference vs Pallas bitwise; compile count = number of
  Mp buckets touched, 0 on a second pass.

## Compute / paths / rules
- Isambard: dev code synced to `$S/_submission/jaxpb-pipeline-dev/` (rsync from local; this dev folder may be updated); any run whose
  outputs are kept is launched from a frozen copy `$S/_submission/jaxpb-pipeline-<milestone>/` (never modified after use). Outputs
  `$R/audits/jaxpb-pipeline/<milestone>/`. Env: `source $S/_submission/jaxka_env.sh; $JAXKA_REPO/.venv/bin/python` (JAX 0.11.2) — no
  new envs, no installs, no downloads. Every job 1 GPU + 72 CPUs, `--mem-per-cpu=2G`; nothing heavier than ls/rsync/sbatch on login
  nodes. Inode quota tight (~951k/1,024k): few files, npz shards, compile cache as a tarball.
- Coulson: CPU Slurm amd96, `--mem-per-cpu=2G`; outputs `~/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/` (new); never edit
  ~/jax-Ka's working tree or any env; never touch masif-*/pkai-scale-* jobs.
- Local Mac has JAX 0.4.38 (older than Isambard's 0.11.2): use it only for quick pure-Python/CPU unit tests that don't depend on
  version-specific APIs; authoritative tests run on Isambard.
- Never touch other sessions' jobs/folders (pool5-*, v5_*, maxent-*, quartet-*). If Isambard SSH fails with a certificate error, stop
  and report (only the user renews). No git commits.
- Reports: the subagent hook blocks .md files — put each milestone's results in `work/jaxpb-pipeline/results/<milestone>.json`
  (numbers + paths) and summarise in the final message; the controller writes the markdown.
