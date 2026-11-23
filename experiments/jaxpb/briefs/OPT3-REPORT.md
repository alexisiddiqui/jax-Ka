# OPT3 — phase 3

Added `cpu`: scalar XLA chains on ≤12 host threads sharing W, indexed current/site access, real n/P loops and rejection skipping. A fixed sorting network prunes the reference pair-sum tree. E/X/count order, RNG, BLOCK structure and acceptance are preserved. Fast path: validated Linux AVX2/JAX 0.11.1, Mp64–1024, ≤8 microstates/site. Other inputs retain ref. `auto` uses this gated CPU dispatcher; GPU selection is unchanged.

| Gate | Result |
|---|---|
| CPU vs a5d5c1d ref | Bitwise: 20/50k + 5/200k; raw counts, curves, pKa, titration shard fields |
| A40 Pallas / ref | Both bitwise: 20/50k |
| Determinism | Two runs, fresh 1-CPU process, final public auto/API |
| Compiles | Five bucket executables; zero later compiles |
| Tests | Existing titration, new pair/odd/single-site/brute/fallback, API shard/resume: pass |

Measured 200k MC seconds/state, all 20 states: **238.50 (1 CPU), 41.66 (12 CPUs)**; scaling **5.72×**. PyPKA single-core: 222 s/state (speedup ratios 0.93× / 5.33×). Old PROFILE2 CPU12 projection: 588.2 s/state (14.12×). Matched five-state CPU12: 303.96→22.09. Allocations were non-exclusive. One task CPU has affinity 1; Slurm reserves two CPUs.

Reverted ordinary scalar reductions: raw counts differed in two states at 200k; one curve differed by one count at pH 6.75/site 105 of 70c9c3eaa126c736_AB. Naive four-term sums also change rounding. Unsupported domains fall back; generic sorting was slower. Numba was absent; nothing installed.

Peak CPU MC RSS: 0.75 GB; Mp1024 compiled temporary: 22,084 bytes/chain. Runtime/compiler changes require new gates.

Phase 4: unchanged CPU geometry 44.35 s/state and maps 248.18 dominate; preserve numerical ordering. Per-bucket measurements, jobs, frozen paths and reverts: `results/opt3.json` and its evidence archive. No commits; stopped after phase 3.
