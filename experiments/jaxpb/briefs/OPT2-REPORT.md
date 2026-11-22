# OPT2 — phase 2

Kept `sor_chunk_packed`/`_packed_block`: GPU red/black packing and byte-code/LUT operators; `sor_chunk`: CPU conditional convergence reduction. `device_solve` compacts at stop blocks without host synchronization. `records` fixes GPU widths at 16/64/128/256 and CPU tiles at 4/8/16. `setup1`/`setup2` select compact GPU operators. Pallas retains its original device solver.

| Gate | Result |
|---|---|
| Pallas energies, potentials, sweeps, maxchange | Bitwise: 104 L3 + 20 bench |
| XLA / CPU | Bitwise: 12 / 5 |
| A40 determinism | Two passes + fresh process, both backends |
| CPU / GPU SOR tests | Pass; rejected CPU packing excluded |
| A40 warm compiles / audits | Zero / pass |

Seconds/state: geometry/maps/setup/SOR/PB. CPU compilation excluded; GPU warm passes.

| Arm | Baseline | OPT1B | OPT2 |
|---|---|---|---|
| pallas | 2.08/3.40/0.20/4.17/10.76 | 2.07/3.19/0.10/4.15/10.27 | 2.07/3.19/0.10/4.13/10.28 |
| xla | 1.60/3.12/0.19/40.53/46.31 | 1.59/2.93/0.10/40.40/45.80 | 1.59/2.92/0.09/9.19/14.50 |
| cpu | 45.54/370.08/12.83/373.27/807.91 | 41.27/246.36/1.67/368.18/663.22 | 44.35/248.18/1.71/163.81/464.02 |

Peak GPU/host GB; Pallas includes all 104 states.

| Arm | Baseline | OPT1B | OPT2 |
|---|---|---|---|
| pallas | 16.52/4.82 | 4.35/4.48 | 4.35/4.50 |
| xla | 15.91/3.81 | 8.68/3.58 | 6.11/3.64 |
| cpu | —/25.97 | —/13.64 | —/10.57 |

Compiled SOR temporary GB: GPU 4.90→3.13 (W256); CPU 8.16→0.58 (W256→16).

Cold compiles (104/12/5): 35/36/32 versus 35/36/34. SOR executables: one per width (Pallas 256; XLA 16/64/128/256; CPU 4/8/16).

Reverted Pallas compaction (4.69 versus 4.15 s/state), CPU packed colours (localised energy error 3.89e-4; changed sweeps) and a split checked sweep (5.25e-6; sweeps equal), both CPU trials beyond tolerance. CPU keeps the original stencil; geometry-site buffers now limit host memory. CPU tile 16 trades 1.4% pool time for smaller buffers than 32.

Per-change measurements, throughput, occupancy, buffers, job IDs and frozen paths: `results/opt2.json`. Phase 3: CPU MC remains untouched; preserve move ordering and measure it separately. No commits; stopped after phase 2.
