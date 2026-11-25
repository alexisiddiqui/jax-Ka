# jaxpb optimisation, phase 4: CPU PB (2026-11-24, user approved "phase 4")

Context: phases 1-3b are committed here as bdffc91 (= jax-Ka jaxpb 2d8aa19). Do NOT commit; the user reviews `git diff bdffc91`.
Read reviews/CPUPB-COMPARISON.md first, then reviews/cpupb-codex-astra.md and reviews/cpupb-gemini-argon.md for detail,
plus results/profile2.json, results/bench.json, OPT2-REPORT.md (CPU PB 464 s/state on the 5 OPT2 states: maps 248,
SOR 164, geometry 44, setup ~8). Matched PyPKA on those 5 states: 8.8 s PB, ~25-29 s total, so the gap is ~17x. Goal:
shrink CPU PB as far as bitwise-safe changes allow; the native SOR prototype decides how close we can get.
Scope: src/jaxpb (maps, geometry, SOR/setup as needed), api.py if needed. Do not touch MC (titrate.py) or the Pallas MC kernel.
A40/GPU PB paths must stay bitwise unchanged.

## Step 0: measure first (report in results/opt4.json)
Per-op CPU time split of the maps stage (atom queries, reach masks, probe projection, per-site cell builds), geometry
(geo_probes candidate search, burial, geometry_sites) and SOR (per sweep) on the 5 OPT2 states at 1 and 12 CPUs. The existing
profile is from the A40 before phase 1b; the reviews' numbers are estimates. Use this to confirm or reorder steps 1-4.

## Steps (measure each; keep or revert each on its own)
1. Maps atom queries: visit only occupied cell slots (not all 27x24=648 per row), in the ORIGINAL (offset, slot) order with a
   running minimum, so the first-min tie rule and f32 norms are bit-identical.
2. Maps reach masks: invert to query the `need` midpoints against nearby probes (early exit) instead of scattering 1,064
   offsets per probe. Result must be identical (same boolean masks).
3. Geometry: precomputed neighbour spans / direct cell lookup instead of 54 searchsorted per point, compacted candidates;
   keep f64 burial and the same candidate order. Also check the 8.52 GB geometry_sites temporary.
4. Smaller items if the Step-0 split supports them: boundary-only Laplacian in rf_energy/post instead of 6 full-grid rolls,
   setup2 dummy-grid padding, 2 pinned 3-core workers for maps groups (report gain; keep only if simple and >10%).
5. Native SOR PROTOTYPE (arm only, NOT default): in-place red/black C kernel (CPU FFI or pure_callback; check which compilers
   and headers exist in the env: gcc/clang, jax.ffi headers; no installs, no downloads). Same f32 expression and per-colour
   order as the XLA path; the hard question is XLA's FMA contraction, so test with -ffp-contract=off and =fast variants and
   report which (if any) matches the XLA sweep bit for bit after 1, 10 and 100 sweeps on real lanes. Report s/sweep per lane at
   1 and 12 threads vs XLA (Astra expects 2-4x, Argon 15-25x; settle it). Do not wire it into the default path; if bitwise, expose
   it behind an explicit backend flag so the user can decide. Note: any change to the split-check schedule changes g by ~5e-6
   (phase-2 finding), so keep the check schedule and compare the full converged phi/g.

## Hard gates (steps 1-4)
1. CPU PB outputs bitwise identical to bdffc91 on the 5 OPT2 states (maps, geometry, phi/g, energies, label shards) and on >= 20
   bench states (energies bitwise). If a step cannot be bitwise, report exactly where it differs, the size, and revert it
   unless the difference is below float noise and you flag it clearly.
2. A40: PB outputs bitwise unchanged on the 20 bench states (a check job; reuse patterns from jobs/opt2*).
3. Determinism (two runs + fresh process), flat compile counts, bounded memory (report peak RSS per arm).
4. Unit tests for each new maps/geometry routine against the old routine on toy and real-shaped inputs, including empty
   cells, ties, and probes at grid edges.
Targets (report): CPU PB s/state per stage at 1 and 12 CPUs vs 464 s and vs PyPKA 8.8 s.

## Compute rules (strict, unchanged)
Coulson only via `ssh Coulson-Chemistry`; CPU partition amd96 and A40 (oc, comp1400, gpu:a40:1), <= 12 CPUs,
--mem-per-cpu=2G, env envs/ogqt-main-7c3e3e9 + jobs/a40site shim. New frozen folders tierB/jaxpb-pipeline/opt4-<tag>/, outputs
tierB/jaxpb-pipeline/opt4/. Never modify existing folders or touch others' jobs (pkai-scale-*, abag-fv-*, cov-*, masif-*,
pool5-*) or ~/jax-Ka; no login-node compute, installs, downloads, commits or pushes. Keep file counts low (inode quota).

## Output
src/tests edits (uncommitted), jobs/opt4*, results/opt4.json, OPT4-REPORT.md (<= 400 words): step-0 split, per-step gain and
keep/revert, gate table, native SOR prototype findings (bitwise or not, speedup), updated CPU PB floor estimate, and what
phase 5 should be. Stop after this.
