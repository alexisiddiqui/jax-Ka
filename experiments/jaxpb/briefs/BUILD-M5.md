# jaxpb-pipeline M5: determinism + SOR kernel — 2026-11-19 (user approved, on Coulson A40)

Spec: DESIGN.md §3 (SOR kernel) + BUILD-M0-M4.md rules. Current state: results/M0..M4.json and M*-a40.json (A40 closing tests:
M0/M2/M3/M4 pass; M1 near-miss — 1/2,329 tautomers at 1.05e-6 because the PB pipeline is NOT run-to-run deterministic on GPU
(two passes in one process differ up to 1.5e-5), compiles 45 vs budget 44 (SOR compiles 6×), SOR lane waste 2.5–3× vs v2 1.2×,
so M1 is slower than v2 on the same A40 (12.8 vs 15 s, 31.7 vs 55 s... see M1-a40.json timing)).

## Part A — determinism (do first)
1. Find every source of run-to-run variation in geometry/maps/pb/sor/post: float scatter-adds (`.at[].add` on GPU uses atomics —
   v2's probe-scatter map construction is the prime suspect), segment sums, reductions with nondeterministic order, `nonzero`
   ordering. Test with the XLA flag `--xla_gpu_deterministic_ops=true` first (and `XLA_FLAGS` exclusion of autotuning) to confirm
   which stage is responsible and what it costs.
2. Fix in code where possible (integer/boolean scatters are order-independent: e.g. build occupancy as integer counts or
   max/or-scatter, sort + segment_sum for float sums, fixed-order reductions) rather than relying on the global flag; keep the
   flag as a fallback only if a code fix costs > 10% time.
3. Closing test: 3 passes in one process + 1 fresh process give BITWISE identical energies/site potentials/sweeps on the 5 pilot
   states and 10 M0 states; parity vs v2 unchanged (sweeps 100%; energies ≤ 1e-6 on all 2,329 tautomers or, if a tautomer sits at
   1.0–1.1e-6 deterministically, report it as a fixed cross-implementation difference and compare like with like).

## Part B — SOR kernel (M5)
- Pallas Triton kernel per DESIGN §3: one program per lane (or the fallback per-colour half-sweep on (lane, z-slab) grid inside a
  fori_loop if one-program-per-lane is too slow/large), reads the uint8 operator code + rhs, red-black with DelPhi's ordering
  (even colour first), fixed ω per lane, max-change stop checked every 10 sweeps exactly as the XLA reference, 500 cap, early exit
  per lane, no host repacking. Deterministic (fixed reduction order for the max-change test).
- Keep the XLA reference; `backend` switch as titrate.py does; interpret=True CPU test on a small grid.
- Also fix the lane-pool waste (refill or early-exit so waste → ~1.0–1.2×) and make the SOR compile count match the budget (or
  update the budget with a justified number).
- Closing tests (DESIGN M5): sweep counts ≥ 99.9% equal to the XLA reference (report exact %), energies/site potentials ≤ 1e-5 vs
  XLA reference, end-to-end L2 criteria still pass on the 99 M0 states, ≥ 2× on SOR time vs the XLA path, total per-state time vs
  v2 and vs M1 on the A40; compile count; deterministic (bitwise across passes).

## Compute / rules
Coulson partition `oc`, node comp1400, `--gres=gpu:a40:1`, ≤ 12 CPUs, `--mem-per-cpu=2G`; env
`~/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9` with the A40 sm_86 shim (`jobs/a40site/sitecustomize.py`). Never touch the
queued pkai-scale-train array (747977), masif-* jobs, ~/jax-Ka's working tree; 1 GPU only, no exclusive mode. Frozen copies per
kept run under `~/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/a40-<tag>/` (never modified after use); outputs
`tierB/jaxpb-pipeline/a40/`. Local `work/jaxpb-pipeline/src/` stays the source of truth; re-run affected CPU tests locally /
on Coulson CPU after code changes. No installs, downloads, envs or git commits; few files. Results in `results/M5.json` and
`results/determinism.json` (harness blocks .md).
