# GQT query-centred crop-radius screen

Authorized 2026-10-08. This experiment compares full-graph training with 10,
15 and 20 Å query-centred spatial crops on the fixed 5,000-structure cohort.
All arms use the 49,709-parameter backbone GQT, explicit historical-pKPDB shift
target, uniform structure loss, AdamW with weight decay 1e-4, batch size eight,
full float32, and indexed Triton encoder attention.

Crops apply to 25% of deterministic structure/epoch draws. A cropped draw
selects one eligible query uniformly, retains residue nodes whose C-alpha lies
within the registered radius, removes all graph edges crossing the crop boundary,
and supervises only the centre query. Other draws retain the complete graph and
all eligible queries. Crop decisions and selected queries are identical across
radius arms for a given seed. The full arm never crops. Validation is always the
complete, unmodified graph.

The first implementation retains padded tensor shapes and represents the crop
through node and edge masks. It therefore isolates the scientific augmentation
effect and makes no throughput claim. Graph compaction is deferred until a crop
demonstrates an accuracy benefit.

The learning rate remains 1e-3 through epoch 10, then follows a per-update cosine
decay to 1e-5 by epoch 20. Validation group-macro MAE selects checkpoints with
`min_delta=0.001` and patience eight. Seeds are 17, 29 and 43. No test data are
read. Runtime artifacts: `pretraining/gqt-50k-crop-radius-v1-triton`.

Submitted 2026-10-08 after the geometry/provenance gate passed:

| Stage | Slurm job |
|---|---:|
| Geometry and crop-boundary test | 744278 |
| Manifest registration | 744280 |
| Four-arm Triton smoke | 744281 |
| 12-run GPU array (maximum four concurrent) | 744284 |
| Dependent aggregate report | 744285 |

**Follow-up (2026-10-11):** `experiments/4_backbone/08_gqt_production_crops.md` retested crops on the production
oGQT recipe (dropout 0.1 plus burial auxiliary). Those crops supervise every site within an inner radius rather than
only the centre. They hurt at the 10% pool, and at 50% they improved interface paired MAE at a cost to pKPDB
validation MSE, so they were not adopted.
