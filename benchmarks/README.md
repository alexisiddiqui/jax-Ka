# Benchmarking

Performance and Regression testing will be performed using the foldbench set
Download:
https://drive.google.com/file/d/17KdWDXKATaeHF6inPxhPHIRuIzeqiJxS/view


## Order of testing:
Proteins-only
Protein-Protein
Protein-Ligand
Other

## Synthetic graph query transformer training profile

Run the production graph loader and fused forward/backward/clipped-Adam update
on generated float32 graphs, including the verified read-only mmap backend:

```sh
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python \
  benchmarks/profile_gqt_synthetic.py \
  --output reports/gqt-synthetic-cpu \
  --data-root /private/tmp/jax-ka-gqt-synthetic-v1 --trace
```

The environment needs the project's `train` dependencies. Defaults profile the
49,709- and 209,645-parameter models at batch size 8, with 16 training structures
and three validation structures per bucket. Padded `(nodes, neighbors, queries)`
capacities are `(384, 64, 128)`, `(768, 96, 256)` and `(1024, 128, 384)`.
These synthetic capacities do not reproduce the historical cohort's sizes.

The profile checks matched NPZ/mmap batches and updates, finite gradients,
parameter changes, Adam update counts and mmap immutability after 5% context
masking. Each backend carries parameters and optimizer state across two epochs;
NPZ/mmap/mmap/NPZ rounds restart from matched initialization. Compilation and
synchronized stage timings are separate from normal prefetched throughput.
Validation uses the production NPZ path and 2,000 bootstrap replicates each epoch.

Outputs include `profile.json`, `summary.txt`, host cProfile statistics, and
optional JAX traces. `--models`, `--batch-size`, `--structures-per-bucket`,
`--epochs`, `--repeats`, `--backend-cycles` and `--context-probability` adjust the workload. Reusing a
data root requires the same synthetic data specification. Mmap conversion and
verification warm the filesystem cache; the profile does not flush it.
Warm epoch comparisons exclude epoch 1 when multiple epochs are requested.
For longer comparisons, add `--epochs 4 --backend-cycles 2 --repeats 10`.
