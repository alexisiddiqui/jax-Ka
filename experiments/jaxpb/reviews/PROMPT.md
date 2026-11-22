Code review request (READ-ONLY: do not edit, create or run anything that changes files; do not run heavy compute).

Repository folder: this directory (jaxpb-pipeline). `src/jaxpb/` is a batched JAX Poisson–Boltzmann + Monte Carlo
titration pipeline that replaces PyPKA (DelPhi PB + Cython MC) for pKa label generation. Design: `DESIGN.md`.
Modules: records, geometry (SES probe generation, f64), maps (eps/ion/boundary grids per site, B=16 site batches), pb
(setup/post, DelPhi-parity numerics incl. its unconverged SOR stop rule), sor (Pallas Triton SOR kernel + XLA reference),
assemble (interaction matrix W), titrate (Pallas MC kernel + XLA reference), api (Config, sharded resumable driver),
compile_budget, prep_pypka (CPU structure prep via PyPKA).

Evidence (read these first):
- results/bench.json — native PyPKA (1/12 CPUs) vs jaxpb on CPU (12 CPUs) vs jaxpb on one A40, 20 states.
  Per state warm: PyPKA 538 s (1 CPU) / 86 s (12 CPUs); jaxpb A40 12.2 s; jaxpb CPU PB ~1,700 s with 21–25 GB host RSS
  regardless of state size (OOM at 24.6 GB); jaxpb CPU MC 3–9x slower than PyPKA single-core MC.
- results/profile2.json — profile of the current code: A40 Pallas vs A40 XLA backend vs CPU, per stage
  (geometry, maps, SOR, post, assembly, MC), SOR lane waste, GPU idle, CPU memory vs site-batch size.
- results/profile.json (pre-speed-up profile), results/speed.json (the shipped speed-ups), results/M5.json, M6.json.

Hard constraints any proposal must respect: outputs must stay bitwise-identical (or within the stated parity tolerances
vs DelPhi/PyPKA: energies <= 1e-6 rel, sweep counts equal) and deterministic run-to-run; flat compile counts via
bucketing/padding (no per-shape recompiles); Pallas kernels stay optional with XLA references.

Task: the non-Pallas path ("main JAX path": XLA backend on GPU and the whole pipeline on CPU) is far too slow and
memory-hungry. Review the code and say concretely what needs addressing, ranked by expected impact:
1. SOR on XLA (GPU and CPU): why is it ~10x slower than the Pallas kernel; what restructuring would fix it
   (e.g. red-black formulation, while_loop/scan structure, convergence-check frequency, lane pooling, fusion, dtype, layout).
2. Maps and geometry on CPU: what makes them ~100x+ slower than on the A40; algorithmic waste vs the A40 path.
3. Memory: what drives the 21–25 GB host RSS on CPU independent of state size (buffers sized for B=16 / max buckets?
   XLA temporaries? gathered probe tables?), and how to bound it.
4. MC on CPU vs PyPKA's Cython MC: what the XLA MC does that a CPU-friendly formulation wouldn't.
5. Anything else: host<->device syncs, recompiles, padding waste, dead code, correctness risks you notice.

For each finding give: file:line, what is wrong, why it costs time/memory (tie to the numbers in the evidence where
possible), the proposed change, expected gain (labelled as an estimate), and whether it risks bitwise parity.
Finish with a prioritised top-5 plan. Be specific; do not pad.
