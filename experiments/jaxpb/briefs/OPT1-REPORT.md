# OPT1 — phase 1

`api.MemTrack` initializes lazily; `_run_shard` keeps bounded timing aggregates. `pb._solve_state` bounds promotion, uses host `spec` and releases buffers. `maps.build_cells`, `_q_one`, `maps_batch`, `reach_mask_b` use CSR ranges, ranked candidate slabs, byte budgets, fixed CPU groups and padding skips. `coulomb_bc`/`setup1`/`setup2` evaluate 38,402 faces only and select backend operators. Timing readers stream shard records.

| Gate | Result |
|---|---|
| Pallas energies, potentials, sweeps | Exact: 104 L3 + 20 bench, A/B/cumulative |
| XLA / CPU | Exact: 12 / 5 final states; A-only CPU: 4 exact, fifth OOM |
| Determinism | Exact: two 104-state passes + fresh process |
| Compiles / CPU tests | Zero warm compiles; four existing modes + seven edge tests pass |
| Final CPU source | Identical StableHLO at all 10 normal/XL variants |

Seconds/state: geometry/maps/setup/SOR/PB. GPU: instrumented warm passes; CPU: compiler-subtracted stage totals. Baselines: PROFILE2 fine timings; XLA memory: matched control. GB are decimal.

| Arm/change | g / m / setup / SOR / PB | GPU / host GB |
|---|---|---|
| Pallas20 baseline | 2.08/3.40/0.20/4.17/10.76 | 16.15/— |
| Pallas20 A | 2.05/3.40/0.20/4.15/10.54 | 16.52/4.86 |
| Pallas20 A+B | 2.07/5.74/0.20/4.15/12.92 | 8.62/5.47 |
| Pallas20 A+B+C | 2.00/5.73/0.10/4.16/12.74 | 4.35/5.46 |
| XLA12 baseline | 1.60/3.12/0.19/40.53/46.31 | 15.91/3.81 |
| XLA12 A | 1.59/3.11/0.19/40.42/46.03 | 15.91/3.84 |
| XLA12 A+B | 1.59/5.32/0.19/40.42/48.26 | 11.08/4.31 |
| XLA12 A+B+C | 1.59/5.31/0.10/40.40/48.13 | 8.68/4.35 |
| CPU5/12cores baseline | 45.54/370.08/12.83/373.27/807.91 | —/21.2–25.1 |
| CPU4+OOM A | 28.75/213.26/9.71/231.95/488.56 | —/25.88 |
| CPU5/12cores A+B | 48.12/378.57/13.29/388.53/833.97 | —/21.61 |
| CPU5/12cores A+B+C | 47.67/375.89/1.55/378.05/808.61 | —/14.08 |

Compiles: Pallas 35/104 (33/bench); XLA 36/12. CPU 34/5. BUILD’s 40/33 covers different sets; matched controls: 35/36/34.

GPU PB change: +18.4%. Reverted CPU single-site tail maps after model-energy bit differences; query/reach skips retain group-of-four projection arithmetic. A-only CPU OOM was resolved by B/C. CPU peak 14.08 GB misses the 12 GB target; SOR retains the peak. Phase 2: colour-compressed XLA SOR, preserving arithmetic and ten-sweep checks.

Evidence, job IDs, frozen paths and per-change deltas: `results/opt1.json` and `results/opt1-evidence.tar.gz`. Stopped after phase 1; no commits.
