# jaxpb optimisation, phase 1b: fix the maps slowdown (2026-11-22, user approved "implement fixes")

Context: phase 1 (BUILD-OPT1.md, OPT1-REPORT.md, results/opt1.json) cut memory and kept every bitwise gate, but maps got
slower: A40 Pallas 3.40 -> 5.73 s/state, A40 XLA 3.12 -> 5.31, CPU 370 -> 376. Two reviews explain why:
reviews/maps-codex-sol.md and reviews/maps-gemini-neon.md, merged in reviews/MAPS-COMPARISON.md (read all three). Work on top
of the current uncommitted phase-1 tree. Do NOT commit; the user reviews `git diff`. Scope: src/jaxpb/maps.py, plus the
maps call sites in pb.py if needed. Do not touch SOR, MC or geometry (that is phase 2+).

## Changes (in order; measure each separately; keep or revert each on its own)
1. GPU tile sizes: separate static budgets per query kind (atom reach-1, cull reach-2, probe, brute, reach_mask), with GPU
   bytes/candidate that reflect fusion, so the A40 row counts are at least the baseline ones (6,472 atom, 1,398 cull, reach
   ch as in ef64a2a, brute 65,536). Hoist grid_coords out of the reach loop. CPU budgets are tuned separately (step 4).
2. Budget-gated dense tables: when B*ncell^3*K*4 <= a static backend budget, build the baseline dense [ncell^3,K] table
   and use the baseline gather/_pick path. Expected: all atom tables dense (GPU and CPU); focused and model probe tables
   dense on the GPU; coarse probes stay CSR (the 11 GB win). Static trace-time choice, no extra compiles.
3. Padding and tails: drop the GPU B=1 partial_group/lax.map path. Run padded sites inside the full-width batch with zero
   counts, or as fixed groups of 8 with a whole-group skip; pick by measurement. Remove the redundant all(active) cond around
   stage0. On CPU, hoist the activity checks out of the per-tile loops in _dyn_rows and reach_mask_b, so no full group ever
   falls back to B=1 per tile (that path already flipped esolvS bits once). Pass site validity explicitly from the site
   layout instead of inferring it from any(srad>0). Make padded-lane outputs clean (mask res/ctr by active). Pad a CPU B
   that is not a multiple of the group size internally instead of raising.
4. Probe queries and CPU: replace the 32-slot slab loop for the remaining CSR queries with either compacted CSR streaming
   (per-row prefix sum over the 27 cells, fixed tile T, strict < merge in offset-major/slot-minor order) or a larger fixed
   slab (128-256) with the ranked fallback. Choose by measurement on A40 and CPU. Tune CPU tile budgets to stay
   L3-resident (try 8/16/32 MiB). Cast order/rank/carry to int32.
5. Small: one searchsorted for start/stop with an explicit stable argsort; build focused probe cells from preg instead of
   all probes ONLY if maps outputs stay bitwise (check ov_p semantics; report any diagnostic change).

## Hard gates (same as phase 1, against the frozen baseline outputs from phase 1's opt1-baseline runs)
1. A40 Pallas bitwise (energies, site potentials, sweep counts) on the 104 L3 states and the 20 bench states; XLA backend
   bitwise on >= 12 states. 2. CPU bitwise on the 5 profile2 CPU states. 3. Determinism (2 passes + fresh process).
4. Flat warm compile counts (report vs phase-1 35/36/34). 5. CPU unit tests + tests/test_opt1.py (extend it for K=48, B not
   divisible by 4, a zero-radius real site, padded-lane outputs). Also compare maps outputs (ex/ey/ez, ion, bidx, surf,
   counters) bytewise vs baseline for at least one state per class.
Targets (report, not gates): A40 maps <= 3.40 s/state Pallas and <= 3.12 XLA; GPU peak <= ~5.5 GB; CPU maps faster
than 370 s/state; CPU peak host RSS not above phase 1's 14.1 GB.

## Compute rules (strict, unchanged from BUILD-OPT1.md)
Coulson only via `ssh Coulson-Chemistry`; base ~/_runtime/jax-Ka/pkabench/. A40: partition oc, node comp1400,
--gres=gpu:a40:1, <= 12 CPUs, --mem-per-cpu=2G. CPU: partition amd96, <= 12 CPUs, --mem-per-cpu=2G. Env
envs/ogqt-main-7c3e3e9 plus jobs/a40site shim (reuse jobs/opt1*.sh patterns). Frozen code copy per run in NEW folders
tierB/jaxpb-pipeline/opt1b-<tag>/; outputs tierB/jaxpb-pipeline/opt1b/. Reuse the phase-1 baseline outputs; do not rerun
the baseline unless needed. Never modify existing folders; never touch jobs you did not submit (pkai-scale-*, abag-fv-*,
cov-*, masif-*, pool5-*), never touch ~/jax-Ka on Coulson; no login-node compute; no installs, downloads, envs, commits or
pushes. Keep file and folder counts low: prefer one frozen folder per measured variant, not per retry.

## Output
- src/ and tests/ edits (uncommitted); job scripts under jobs/opt1b*.
- results/opt1b.json: per change and cumulative: gates, s/state per stage (geometry, maps, setup, SOR, PB) on A40 Pallas,
  A40 XLA, CPU (5 states), GPU peak, host peak RSS, compile counts, job ids, frozen paths, what was reverted and why.
- OPT1B-REPORT.md (<= 350 words): changes kept (functions), gate table, timing/memory table vs baseline and phase 1, reverts,
  open risks. Stop after this; do not start phase 2.
