# Investigate and profile PyPKA's PB (2026-11-24, user request; investigation only, no jaxpb code changes)

Why: jaxpb CPU PB is 351 s/state at 12 CPUs (463 before phase 4; see OPT4-REPORT.md, results/opt4.json) against PyPKA's 8.8 s PB
(~25-29 s total) on the same 5 OPT2 states (results/bench.json, results/profile2.json). Before choosing phase 5 we need to know WHY
PyPKA is that fast, per stage. The user's framing: high-compute, repetitive-arithmetic work is a good fit for JAX; work that is
memory-bound, or that is fast only because of in-place updates, may need a different approach (native kernel, different algorithm).
Classify every stage accordingly, with evidence.

## What to find out
1. Locate the PyPKA install and the PB engine it calls on Coulson (pkabench env/folders; check how pkabench runs it and which
   binary/library does the PB: DelPhi/Fortran/C++/Cython). Read its source where available (do not install or download anything).
2. Algorithm per stage, concretely: grid size and resolution, focusing (coarse-to-fine? how many levels, sub-grid size), how the
   surface/dielectric/ion maps are built (per-atom loops? cell lists?), the solver (SOR, multigrid, conjugate gradient?), the
   convergence criterion and typical iteration counts, how many PB solves per state (per site, per tautomer, model compounds),
   what is reused between solves, floating point precision, how boundary conditions are set, how the energies are extracted.
3. Measured profile on the 5 OPT2 states (the same states as jaxpb's 464 s), single core as pkabench pins it: wall time per stage
   (maps/surface, SOR or solver iterations, energy extraction, other), number of solves, number of grid points, solver iterations
   and the final residual, peak RSS. Use perf/valgrind/gprof/ltrace only if already available; otherwise use timers/instrumented
   runs, or in-process wall-clock hooks. State what you could not measure.
4. Hardware behaviour of its hot loops where measurable: working-set size per solve vs L2/L3 (32 MB L3 per CCD on amd96), whether it
   updates the grid in place, memory traffic per sweep (estimate from the code: bytes read/written per grid point), arithmetic
   intensity, and whether it is memory-bound or compute-bound. Use perf stat counters if the node allows them.
5. Work comparison with jaxpb on equal terms: total grid points x sweeps, atoms x queries per map, number of solves, per state.
   Is jaxpb doing more ALGORITHMIC work (more points, more sweeps, tighter convergence, more solves, padded lanes) or the same work
   with a worse implementation? Quantify: e.g. "PyPKA does X sweeps on N points, jaxpb does Y on M". Use jaxpb numbers from
   results/opt4.json / profile2.json; if something needed is not logged, say so rather than guess.
6. For each stage: classify as (a) compute-bound and repetitive (good for JAX/XLA), (b) memory-bound (needs fusion/tiling/native
   kernel), (c) fast because of in-place / sequential updates (red/black or Gauss-Seidel style, early exit) and what the JAX
   counterpart would be, or (d) fast because it does less algorithmic work. Give a recommended approach per stage and the expected
   ceiling, flagged as estimate vs measured.

## Rules (strict, unchanged)
Coulson only via `ssh Coulson-Chemistry`; CPU partition amd96, <= 12 CPUs, --mem-per-cpu=2G (pin to physical cores as pkabench does).
Env envs/ogqt-main-7c3e3e9 for jax; PyPKA via the pkabench setup. New output folder tierB/jaxpb-pipeline/pypkaprof/. Never modify
existing folders or touch others' jobs (pkai-scale-*, abag-fv-*, cov-*, masif-*, pool5-*) or ~/jax-Ka; no login-node compute,
installs, downloads, commits or pushes. Keep file counts low. Do not change src/jaxpb.

## Output
results/pypkaprof.json (raw numbers, job ids, paths) and PYPKAPROF-REPORT.md (<= 600 words): per-stage table (PyPKA s/state, what
it does, algorithm, bound type), the algorithmic-work comparison with jaxpb, classification (a-d) and recommendation per stage,
what could not be measured. Stop after this.
