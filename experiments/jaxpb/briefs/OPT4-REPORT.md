# OPT4 report

Five-state mean PB stage sums: **1377.3s at 1 CPU; 351.2s at 12 CPUs** (compilation excluded; 12-CPU fresh warm pass, one L3 group), versus Step-0 1831.8/463.0s. The 12-CPU result is 1.32× faster than 464s and 39.9× PyPKA’s 12-CPU 8.8s PB (its total was 25–29s); the 1-CPU ratio is 156.5×. The separate two-L3 allocation measured 443.7s.

| Stage, seconds/state | Before 1/12 | After 1/12 |
|---|---:|---:|
| Maps | 1255.1 / 249.0 | 787.7 / 167.8 |
| Global geometry | 78.6 / 41.6 | 54.0 / 15.8 |
| Model geometry | 9.0 / 3.5 | 7.1 / 2.6 |
| SOR | 473.7 / 164.9 | 508.4 / 161.2 |

Step-0 isolated maps atom/reach/projection/cells: 894/219/100/15.4s (1 CPU), 179/48.4/18.3/4.23s (12). These separately compiled diagnostics are nonadditive; cells overlap stage0. From-scratch geometry search/burial: 25.6/67.0s and 18.5/24.3s; production reuses AB geometry. SOR: .01514/.00527s per useful lane-sweep. Diagnostic splitting alone changed float32 surf (audit max 1.53e-5); masks/counters matched. Original maps fed reference solves.

Keep occupied-slot queries plus wider tiles (1.41× at five cores; 1.46× at 12 CPUs; slots alone 1.03× at five cores), direct geometry spans plus compact candidates (2.47×; spans alone 1.57×), and 65536-point model tiles (1.54×). Geometry/site arm studies used five physical cores. Model temporary: 8.52GB → 152MB. Revert inverse reach (0.62–0.76×), 128-slot slabs, smaller geometry chunks, and two pinned three-core workers (0.91×). Defer setup/padding/post: combined setup/post <0.7% of Step-0 PB.

| Gate | Result |
|---|---|
| Five CPU states: maps/geometry/φ/g/energies, full label shards | Bitwise |
| CPU20; A40 Pallas20 and XLA20 | Bitwise |
| CPU12/A40 repeats + fresh processes; warm compiles | Pass; zero growth |
| Five unit groups; peak selected CPU RSS | Pass; 7.14GiB |

Parent allocation sampled aggregate peaked at 15.32GiB including its held reference process (<24GiB).

GCC8.5/FFI headers existed; clang absent. Native off/fast costs .001117/.001030s per lane-sweep at 1 thread (15.2/16.5× XLA), .000338/.000349s at 12 (13.4/13.0×); XLA costs .017031/.004539s. Neither matches after 1/10/100 sweeps. Full-state φ errors are 5.34e-5/7.78e-3; g errors 7.15e-6/3.89e-4. Fast shifts one stop by ten sweeps despite unchanged checks. Prototype only; no backend flag.

Unchanged SOR sets an approximately 161s/state floor. Observed non-SOR work is 190s/state. Matching native rounding would suggest ~202s PB, optimistically excluding callback overhead. Phase 5 should match per-operation FMA choices before revisiting remaining queries. MC unchanged; no commits.
