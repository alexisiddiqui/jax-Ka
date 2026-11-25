Performance review request (READ-ONLY: do not edit or create files, do not run compute, no git writes). Answer in your final message.

Folder: this directory (jaxpb-opt), a git working copy at commit bdffc91 (= jax-Ka jaxpb 2d8aa19). jaxpb is a batched JAX
Poisson-Boltzmann + Monte Carlo replacement for PyPKA (see DESIGN.md). Phases 1-3 are done (BUILD-OPT*.md, OPT*-REPORT.md,
results/opt1b.json, opt2.json, opt3b.json; per-op profile in results/profile2.json; PyPKA reference timings in
results/bench.json). On the A40 the pipeline is ~10 s/state; MC on CPU is now fine (33 s/state at 12 CPUs).

The remaining problem is CPU PB (XLA CPU, 12 Slurm CPUs = 6 physical cores of an AMD EPYC 9224, 32 MB L3 per CCD,
2 GB/CPU): ~464 s/state = maps ~248, SOR ~164, geometry ~44, plus setup. PyPKA does the whole job (PB+assembly+MC) in
~86 s/state on the same 12 CPUs (538 s single core; its PB is DelPhi in Fortran/C, 179 s single core).

Files: src/jaxpb/geometry.py (f64 SES probes), maps.py (eps/ion/boundary grids; CSR/dense cell tables, CPU tiles), sor.py
(XLA red/black SOR; on CPU the full-grid masked stencil plus a conditional convergence reduction; CPU lane widths 4/8/16
in records.py), pb.py (orchestration, focusing levels).

Hard constraint: results must stay bitwise identical to the current CPU outputs (energies, site potentials, SOR sweep
counts/maxchange), or, where that is impossible, the change must be argued and measured as numerically equivalent within
DelPhi-unconverged-solver tolerance (note phase 2 rejected CPU packed-colour SOR because the energy changed by 3.9e-4).
Compile counts must stay flat (fixed buckets). Memory <= ~20 GB host.

Task:
1. Maps (248 s): with file:line, what dominates on CPU (gathers, probe queries, reach masks, tiles vs L3, f64, thread
   utilisation of XLA CPU). Rank by estimated cost (label estimates).
2. SOR (164 s): what limits the XLA CPU stencil (memory bandwidth, masked full-grid work, lane layout, fusion, threads).
   Is a bitwise-preserving CPU speed-up possible (exact per-point update expression and order of evaluation)? Options:
   layout changes, different lane chunking, a custom CPU kernel via jax.pure_callback / FFI (C or numba if present),
   Pallas-on-CPU, fewer checks; give expected gain and parity risk.
3. Geometry (44 s): what is slow and what can change without changing outputs.
4. Whether the 12-CPU allocation is used well at all (XLA CPU intra-op threading, parallelism across states/sites
   vs within, process-level parallelism over states).
5. A realistic floor: could jaxpb CPU PB reach PyPKA's ~86 s/state (or what is the best plausible number), and what
   would it take.
Finish with a ranked plan (top 5) for phase 4+: change, file:line, expected gain (estimate), memory, parity risk, effort.
Be specific; do not pad.
