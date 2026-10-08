# Capped tighter GQT capacities

Test whether smaller padded tensor capacities improve full-graph GQT throughput
without changing sampled batch membership, order, optimizer-update count,
features, labels or graph context. All inputs use the verified mmap backend.

Cap the experiment at 12 compiled shapes across five independently sampled
epoch plans. Every rounded dimension is also capped by its previous bucket
capacity, so a tighter policy can never enlarge a batch. Schemes producing
32--204 shapes are rejected before GPU profiling.

| Policy | Capacity rule | Shapes across five plans | Actual/padded edges |
|---|---|---:|---:|
| Current | Three fixed capacities | 3 | 0.262 |
| Residue only | N rounded to 128; old K/Q | 8 | 0.267 |
| Selected | N rounded to 128; K to 64; old Q | 12 | 0.290 |
| Rejected fine policy | N to 64; K to 32; old Q | 32 | 0.325 |

The timed epoch encountered six residue-only shapes and nine selected-policy
shapes. Warm throughput was stable across forward and reverse-order repeats.

| Model | Current epoch | Residue-only | Selected N128/K64 | Selected gain |
|---|---:|---:|---:|---:|
| 49,709 parameters | 70.04 s | 68.85 s | 62.12 s | 11.3% |
| 209,645 parameters | 91.10 s | 89.35 s | 80.88 s | 11.2% |

The residue-only policy is not worth increasing the shape count for a 1.7--1.9%
gain. The capped N128/K64 policy is the production candidate. Checked padded
prefixes were exact; prediction and one-step update differences remained below
the accepted 5e-6 relative float32 tolerance. Persistent compilation caching
can amortize restart compilation, but the 12-shape hard cap remains part of the
policy.
