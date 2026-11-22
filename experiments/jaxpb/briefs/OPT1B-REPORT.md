# OPT1B

GPU tiles: `_q_chunk`, `_brute_chunk`, `reach_mask_b`. `build_cells` uses budgeted dense tables and one bounds search; coarse probes remain CSR. `maps_batch`/`_maps_group`/`build_maps_b`/`_dyn_rows` preserve group width and clean padding; `pb._site_layout` supplies validity. `_q_one` uses 128-slot slabs and int32 ranks.

| Gate | Result |
|---|---|
| Pallas | Exact: 104 L3 + 20 bench |
| XLA / CPU | Exact: 12 / 5 |
| Determinism | Two passes + fresh process exact |
| Direct maps | Bytewise: five capacities, GPU/CPU |
| Tests / GPU warm compiles | CPU modes + 11 edge tests pass / zero |

A40 Pallas/XLA; CPU: 12 cores. s/state: geometry/maps/setup/SOR/PB; baseline PROFILE2; CPU compiler time excluded.

| Arm | Baseline | Phase 1 | OPT1B |
|---|---|---|---|
| pallas | 2.08/3.40/0.20/4.17/10.76 | 2.00/5.73/0.10/4.16/12.74 | 2.07/3.19/0.10/4.15/10.27 |
| xla | 1.60/3.12/0.19/40.53/46.31 | 1.59/5.31/0.10/40.40/48.13 | 1.59/2.93/0.10/40.40/45.80 |
| cpu | 45.54/370.08/12.83/373.27/807.91 | 47.67/375.89/1.55/378.05/808.61 | 41.27/246.36/1.67/368.18/663.22 |

Peak GPU/host GB (matched baseline controls).

| Arm | Baseline | Phase 1 | OPT1B |
|---|---|---|---|
| pallas | 16.52/4.82 | 4.35/5.46 | 4.35/4.48 |
| xla | 15.91/3.81 | 8.68/4.35 | 8.68/3.58 |
| cpu | —/25.97 | —/14.08 | —/13.64 |

Compile counts (phase 1: 35/36/34): 35/36/34.

Kept GPU groups of 8, 128-slot slabs and 32 MiB CPU tiles. Rejected focused-probe compaction: arrays matched, but `cell_probes` changed (337→270). Class 4 uses the largest real state padded upward. XLA’s 8.68 GB peak misses the GPU target; unchanged SOR sets that peak.

Per-change evidence: `results/opt1b.json`. No commits; stopped after OPT1B.
