# jaxpb optimisation, phase 2: XLA SOR + lane waste (2026-11-22, user approved "Start phase 2")

Context: phases 1/1b are committed here as 10a3ddc (= jax-Ka jaxpb a39e03b). Read reviews/COMPARISON.md items 2-3,
reviews/codex-astra.md and reviews/gemini-neon.md (SOR sections), results/profile2.json (SOR lane waste, per-sweep cost:
Pallas 0.125 ns/point, XLA 0.46 ns/point plus 2.6-3.1x lockstep waste, CPU 8.3 ns/point), OPT1B-REPORT.md and
results/opt1b.json (current numbers: A40 XLA SOR 40.4 s/state, CPU SOR 368 s/state, XLA GPU peak 8.68 GB set by SOR).
Do NOT commit; the user reviews `git diff 10a3ddc`. Scope: src/jaxpb/sor.py, records.py lane settings, and the SOR call
sites/operator construction in pb.py. Do not touch maps, geometry, assembly or MC (later phases).

## Changes (in order; measure each; keep or revert each on its own)
1. XLA SOR colour-compressed sweeps (sor.py sor_chunk/sor_block, ~:52-80): today each half-sweep computes the full-grid
   stencil for both colours and masks (2x work) and gathers 4 f32 coefficient fields per block via exE[op]. Restructure
   to update only the active colour's points (packed red/black layout or strided views), using compact per-point
   operator codes + a small LUT (as the Pallas kernel does: op_code/op_lut, ~:168-180) instead of 4 gathered f32 fields.
   Keep the exact per-point update expression and evaluation order, the colour order, omega handling, and the stop rule:
   max-change check every RC.CHECK sweeps, threshold, NLIT cap, done-lane freezing at the same sweep. Avoid any eager
   per-block host sync (keep the loop on device).
2. Convergence-check cost: only the checked sweep needs the max-change reduction; the other CHECK-1 sweeps should not
   compute it, provided the recorded maxchange/sweeps stay bitwise identical.
3. Lane waste: finer LANE_WIDTHS (e.g. 8/16/32/64/128/256 or similar fixed set) and freeing finished lanes at block
   boundaries with compaction, for both backends. Compile count must stay flat and bounded (report the number of SOR
   executables per width); if a finer set raises compiles beyond a few, pick the best fixed set by measurement.
4. CPU-specific: tune the XLA CPU path (layout, chunking over lanes, fusion) for the 12-thread EPYC node; target the
   memory too (SOR sets the 13.6 GB CPU peak and the 8.68 GB XLA GPU peak - find and shrink the dominant buffers).

## Hard gates (all must pass; if one cannot pass bitwise, localise it, revert that change and report why)
1. A40 Pallas bitwise (energies, site potentials, sweep counts, maxchange) vs 10a3ddc on the 104 L3 states and 20 bench
   states - Pallas must be untouched by XLA changes, and lane-width changes must keep it bitwise.
2. A40 XLA backend bitwise vs the 10a3ddc XLA backend on >= 12 states; CPU bitwise vs 10a3ddc CPU on the 5 profile2 CPU
   states. If FMA/fusion makes exact XLA parity impossible for a change, report the max abs/rel energy difference and
   sweep-count agreement and REVERT it unless sweeps are 100% equal and energies within 1e-7 rel - and flag it clearly.
3. Determinism (2 passes + fresh process, A40). 4. Flat warm compile counts (report vs 35/36/34).
5. CPU unit tests incl. tests/test_opt1.py, plus a new SOR unit test that checks the restructured XLA sweep against the
   10a3ddc sor_chunk on random small grids/operators (bitwise) and the stop rule on lanes finishing at different blocks.
Reuse phase-1b outputs (tierB/jaxpb-pipeline/opt1b/, final frozen opt1b-s5-20261122) as the reference where they match
10a3ddc; rerun a reference only if needed.
Targets (report, not gates): A40 XLA SOR well below 40.4 s/state (reviews estimate 2-9x), CPU SOR well below 368 s/state,
XLA GPU peak and CPU host peak below phase 1b; A40 Pallas SOR not slower than 4.15 s/state.

## Compute rules (strict, unchanged)
Coulson only via `ssh Coulson-Chemistry`; base ~/_runtime/jax-Ka/pkabench/. A40: partition oc, node comp1400,
--gres=gpu:a40:1, <= 12 CPUs, --mem-per-cpu=2G. CPU: partition amd96, <= 12 CPUs, --mem-per-cpu=2G. Env
envs/ogqt-main-7c3e3e9 plus jobs/a40site shim (reuse jobs/opt1b* patterns). Frozen code copies in NEW folders
tierB/jaxpb-pipeline/opt2-<tag>/; outputs tierB/jaxpb-pipeline/opt2/. Never modify existing folders; never touch jobs you
did not submit (pkai-scale-*, abag-fv-*, cov-*, masif-*, pool5-*), never touch ~/jax-Ka on Coulson; no login-node compute;
no installs, downloads, envs, commits or pushes. Keep file and folder counts low (one frozen folder per measured variant).

## Output
- src/ and tests/ edits (uncommitted); job scripts under jobs/opt2*.
- results/opt2.json: per change and cumulative: gates, s/state per stage (geometry, maps, setup, SOR, PB; SOR sweeps/s,
  ns/point, lane occupancy) on A40 Pallas, A40 XLA, CPU (5 states), GPU peak, host peak RSS, compile counts, job ids,
  frozen paths, reverts and why.
- OPT2-REPORT.md (<= 350 words): changes kept (functions), gate table, timing/memory vs baseline / 1b, reverts, open
  risks, and what phase 3 (CPU MC) should know. Stop after phase 2.
