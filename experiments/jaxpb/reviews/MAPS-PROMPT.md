Code review request (READ-ONLY: do not edit or create files, do not run compute, no git writes). Answer in your final message.

Folder: this directory (jaxpb-opt), a git working copy. `git diff ef64a2a -- src/jaxpb/maps.py` is phase 1 of an
optimisation of `src/jaxpb/maps.py` (batched JAX eps/ion/boundary grid maps for a Poisson–Boltzmann pipeline; see
DESIGN.md, BUILD-OPT1.md, OPT1-REPORT.md, results/opt1.json "comparisons"/"changes"/"tile_budgets_bytes",
results/profile2.json for the pre-change per-op maps split, reviews/COMPARISON.md for why the change was made).

Phase 1 replaced the dense per-site [ncell^3, K] cell table with CSR (order/start/stop) ranges, streams occupied
32-slot slabs with a fori_loop when K > 32 (preserving the original offset-major/slot-minor tie rank), byte-budgeted
query/reach tiles (CPU 64/32 MiB, GPU 512/256 MiB), lax.cond/lax.map wrappers so padded sites do no work, and on CPU
processes B=16 sites as 4 groups of 4 (lax.map over groups). All outputs stay bitwise identical.

Result: memory fell (A40 peak 16.2 -> 4.35 GB together with setup changes; CPU host 21-25 -> 14 GB), but maps got
SLOWER: A40 Pallas 3.40 -> 5.73 s/state, A40 XLA 3.12 -> 5.31, CPU (12 threads) 370 -> 376 (no gain). Change A
(unrelated) left maps unchanged at 3.40, so the +2.34 s is entirely change B.

Task:
1. Identify, with file:line, what in the new maps code causes the A40 slowdown (e.g. slab fori_loop serialisation
   and per-slab argmin/take_along_axis, smaller query chunks than before, lax.cond around a vmapped body breaking
   fusion, the `jnp.all(active)` cond choosing between vmap and lax.map, reach tile size, dynamic trip counts,
   gather patterns of order[start[cid]+slot]). Rank causes by estimated cost, labelled as estimates.
2. Why CPU maps did not speed up despite the memory drop; what actually dominates CPU maps time.
3. Concrete fixes that restore or beat the baseline speed on the A40 while keeping most of the memory saving, and
   that make CPU maps faster. Options to weigh: backend-specific paths (dense table on GPU when it fits a budget,
   CSR on CPU), larger slabs or one-shot reconstruction for typical occupancy, hoisting padding checks out of the
   hot loop, sorting queries by cell, compacting candidates, different chunk sizes. For each: expected gain
   (estimate), memory cost, and bitwise-parity risk (exact candidate order, first-candidate tie rule, distance
   expression and overflow detection must be preserved; flat compile count, no per-shape recompiles).
4. Any correctness or parity risk you see in the new code.
Finish with a ranked fix plan (top 3-5). Be specific; do not pad.
