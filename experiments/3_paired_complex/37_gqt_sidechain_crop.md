# Side-chain GQT crop-radius screen

Authorized 2026-10-08. This screen compares complete graphs and 5, 15, and
20 Å query-centred crops using the existing 50,001-parameter side-chain GQT.
The only model-input change relative to the matched backbone screen is the
addition of 32 named native side-chain heavy-atom coordinate/presence slots.

All arms use the cleaned 5k cohort, explicit historical-pKPDB signed-shift MSE,
batch size eight, full float32, indexed Triton encoder attention, AdamW with
weight decay `1e-4`, no dropout, and three seeds. The learning rate remains
`1e-3` through epoch 10 then follows a per-update cosine decay to `1e-5` by
epoch 20. Crops occur on 25% of training structure draws, select one eligible
query uniformly, retain nodes by Cα distance, and supervise only the centre
query. Validation always uses complete, unmodified side-chain graphs.

An exact read-only mmap bundle is built once from the existing 5k side-chain
graphs. The conversion verifies every graph field bit-for-bit and is shared by
all runs. Reports include validation MAE, crop retention, end-to-end wall time,
and JAX allocator peak VRAM. No test data are read.

| Stage | Slurm job |
|---|---:|
| Registration | 745217 |
| Exact mmap conversion | 745218 |
| Four-arm Triton smoke | 745219 |
| 12-run GPU array | 745220 |
| Aggregate report | 745221 |

Runtime location: `pretraining/gqt-sidechain-crop-radius-v1-triton`.
