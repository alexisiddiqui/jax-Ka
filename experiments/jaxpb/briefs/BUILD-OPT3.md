# jaxpb optimisation, phase 3: CPU Monte Carlo titration (2026-11-22, user approved "phase 3")

Context: phases 1, 1b and 2 are committed here as a5d5c1d (= jax-Ka jaxpb c0c12d7). Read reviews/COMPARISON.md item 4,
the MC sections of reviews/codex-astra.md and reviews/gemini-neon.md, results/profile2.json ("mc_per_bucket_S32",
"mc_20_state_mean_s_per_state"), results/bench.json (jaxpb CPU MC is 3-9x slower than PyPKA's single-core Cython MC; A40
Pallas MC 1.7 s/state at 200k sweeps). Do NOT commit; the user reviews `git diff a5d5c1d`. Scope: src/jaxpb/titrate.py
(and its backend selection in api.py if needed). Do not touch PB/maps/SOR/geometry/assembly or the Pallas MC kernel.

Problem: mc_ref (titrate.py:111-206) is the "ref" XLA backend: per-chain fori_loops over sweeps/site moves/pair moves,
vmapped over chains and states. Under vmap the lax.cond accept branches become selects, so rejected moves still pay the
full update, and every move pays O(Mp)/O(Np) masked reductions over the padded bucket. On CPU this runs far slower than a
scalar chain should.

## Changes (measure each; keep or revert each on its own)
1. A CPU-specific MC backend ("cpu"): a scalar-indexed chain that touches only the moved site's (and pair's) W row and the
   real (unpadded) sites, with real branch skipping on rejection. Options, choose by measurement: (a) JAX on CPU without
   vmap over chains (lax.map or a loop over chains/states so lax.cond really branches; parallelise across chains/states
   with fixed-size batches or host threads); (b) plain NumPy/Python is too slow - only if vectorised across chains; (c)
   numba ONLY if it is already importable in the env (check, do not install). Whatever is chosen must use the SAME RNG
   stream (hash32/urand counter scheme, chain_keys), the same move order (site moves then pair moves per sweep, BLOCK
   structure, mcblocks), the same accept rule (invexp/accept) and the same f32 arithmetic and accumulation order for E,
   X and curv, so results are bitwise identical to the "ref" backend.
2. Default selection: backend='auto' picks pallas on GPU (unchanged) and the new cpu backend on CPU only if gate 1 holds.
3. Keep compile counts flat (fixed buckets; no per-state shapes) and memory bounded.

## Hard gates
1. CPU MC outputs (counts/curves, pKa, all titration outputs written to label shards) bitwise identical to the current
   "ref" backend on CPU at 200k sweeps on the 5 profile2 CPU states and at 50k on >= 20 states (use the bench states' saved
   energies, so PB need not be rerun: reuse the label shards / energies from tierB/jaxpb-pipeline/opt2 or bench outputs).
   If exact bitwise is impossible for a formulation, report where it differs and revert it.
2. A40: Pallas MC and the GPU "ref" backend outputs unchanged (bitwise vs a5d5c1d on the 20 bench states).
3. Determinism (two runs + fresh process on CPU). 4. Flat compile counts. 5. Unit tests: existing titrate tests plus a new
   test comparing the cpu backend to ref bitwise on toy states (titrate.toy / brute) incl. pair moves, odd bucket sizes and
   states with 1 site.
Targets (report, not gates): CPU MC s/state at 200k sweeps vs PyPKA single-core MC (222 s/state in bench.json: 538 total
- 179 PB - 137 assembly) and vs the current jaxpb CPU MC; scaling at 1 and 12 CPUs.

## Compute rules (strict, unchanged)
Coulson only via `ssh Coulson-Chemistry`; base ~/_runtime/jax-Ka/pkabench/. CPU: partition amd96, <= 12 CPUs,
--mem-per-cpu=2G. A40 (gate 2 only): partition oc, node comp1400, --gres=gpu:a40:1, <= 12 CPUs, --mem-per-cpu=2G. Env
envs/ogqt-main-7c3e3e9 plus jobs/a40site shim (reuse jobs/opt2* patterns). Frozen code copies in NEW folders
tierB/jaxpb-pipeline/opt3-<tag>/; outputs tierB/jaxpb-pipeline/opt3/. Never modify existing folders; never touch jobs you
did not submit (pkai-scale-*, abag-fv-*, cov-*, masif-*, pool5-*), never touch ~/jax-Ka on Coulson; no login-node compute;
no installs, downloads, envs, commits or pushes. Keep file and folder counts low.

## Output
- src/ and tests/ edits (uncommitted); job scripts under jobs/opt3*.
- results/opt3.json: gates, MC s/state per arm (CPU 1 and 12 CPUs; A40 unchanged check), per bucket, compile counts,
  memory, job ids, frozen paths, reverts and why.
- OPT3-REPORT.md (<= 300 words): backend design, gate table, timing vs PyPKA MC and old jaxpb CPU MC, reverts, open risks,
  and what phase 4 (CPU geometry and remaining CPU maps cost) should know. Stop after phase 3.
