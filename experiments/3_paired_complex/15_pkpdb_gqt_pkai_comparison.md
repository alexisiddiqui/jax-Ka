# 5k historical pKPDB raw/clean pretraining comparison

Authorized 2026-10-06: train the backbone GQT and pKAI from scratch on the
same temporary 5,000-structure cohort, comparing raw and cleaned labels.
Use `pkpdb-5k-v2/pilot.json`, its fixed sequence exclusions and masks, and the
existing clean 142-complex validation set. No test evaluation. Historical
training labels and current validation labels differ in teacher provenance.

| Setting | GQT | pKAI |
|---|---|---|
| Initialization | Random, seed 17 | Random, seed 17; no pretrained weights |
| Parameters | 49,709 | 3,608,001 |
| Inputs | Strict backbone, 20 Å Cα graph | Native atom features, 15 Å functional-atom cutoff |
| Architecture | Width 44, FF 88, existing two graph layers + query layer | 4008 → 800 → 400 → 200 → 1; ReLU |
| Dropout | Existing architecture, none | 0.5 / 0.125 / 0.03125 |
| Optimizer | Existing Adam, 0.001, gradient clip 1 | Adam, 0.000001, weight decay 0.0001 |
| Sampling | Existing group/complex-uniform, accumulation 8 | Shuffled sites, batch 256 |
| Loss | Absolute scalar pKa MSE | Native model-compound shift MSE |
| Duration/selection | Fixed 20 epochs | Validation site-MSE early stopping, delta .001, patience 5 epochs; cap 200 |
| Arithmetic | Full float32 | Full float32 |

pKAI settings follow the [authors' training methods](https://assets-eu.researchsquare.com/files/rs-949180/v2_covered.pdf).
Its released TorchScript file supplies architecture/dropout evidence. The
published run used 16-bit arithmetic; this pilot keeps full float32 per the
user's preference. The epoch cap and epoch interpretation of patience are
explicit pilot choices. No pKAI+ shrinkage penalty is used.

Use the installed pKAI encoder semantics, including its inverse-square distance
features (the paper describes inverse distance), atom classification, sorted
250-neighbor limit, and six supported side-chain site types. No termini are
invented. Observed canonical protein atoms are exported with reversible numbering;
the same conformer resolution and component stripping are applied in both arms.
Missing side-chain atoms are not rebuilt. The cleaned arm changes label eligibility,
not coordinates. Structures or sites outside native pKAI support are counted.
Each structure checks one accelerated feature row against the untouched encoder.

GQT and pKAI differ in input information, capacity, sampling and checkpoint
selection. This compares native model recipes, not isolated architectural effects.
Report full support and the identical validation-site intersection. Bootstrap by
the existing validation sequence groups, 2,000 replicates. Validation-selected
pKAI results remain development results; never label them final test performance.

CPU preparation: 32 cores / 64 GB, exclude comp1400. GPU training: one A40 and
8 CPUs / 16 GB per arm, up to four arms together, within the 400-core user cap.
Outputs: `pretraining/pkpdb-5k-comparison-v1`. Preparation and feature equivalence
must pass before the corresponding model starts. Record manifests, environment
provenance, checkpoints, per-site predictions, and a shared comparison table.

## Correction: true GQT structure batching

The first GQT implementation mistakenly retained serial one-structure gradients
with accumulation over eight structures. It was stopped after epoch 1 checkpoints
were committed in both arms. Those original artifacts are preserved.

The replacement processes eight same-capacity structures simultaneously with
`vmap` and one compiled loss/gradient/Adam update. Each structure still contributes
its mean site loss equally. Tail padding contributes zero loss and no weight.
Group-uniform sampled membership is preserved, then bucket batches are shuffled;
batch membership and subsequent RNG consumption consequently differ from the
serial implementation. CPU loading is threaded with one host batch prefetched.

The first default-precision probe measured 3.66× speedup in the ≤384-residue
bucket, but the next bucket's gradient relative difference (2.70e-4) exceeded
the 2e-4 gate. Preserve that incomplete probe; do not claim equivalence from it.
The replacement probe and batched training use `JAX_DEFAULT_MATMUL_PRECISION=highest`
to require strict float32 matrix arithmetic. This precision change is recorded
alongside the batching change; the original epoch used the previous default.

Gate on numerical agreement with eight separate gradients and measure complete
optimizer steps for eight real structures in each of the three size buckets.
Only after that gate passes, resume weights, Adam state and RNG from the committed
epoch 1 checkpoints into `gqt-batched-raw` and `gqt-batched-clean`. Record parent
manifest/checkpoint hashes; do not overwrite the original runs. The target remains
20 total epochs, with the inherited serial epoch explicitly identified. pKAI
runs remain unchanged. Report measured throughput rather than assuming a speedup.

The strict-float32 gate passed on all three buckets (job 738283), with relative
gradient differences 1.06–1.55e-7. Median complete optimizer-step timings over
three warm repetitions, on one A40:

| Padded residues | Eight serial gradients + update | True batch of eight + update | Speedup |
|---:|---:|---:|---:|
| 384 | 0.3103 s | 0.0734 s | 4.23× |
| 768 | 0.3102 s | 0.1316 s | 2.36× |
| 1056 | 0.4022 s | 0.2112 s | 1.90× |

These are optimizer-step timings, not end-to-end epoch estimates. Measured JAX
peak live allocation remained below 1.1 GB in the probe. Eleven tests passed.
The first restart submissions (738277 / 738278) held an older Slurm script without
the strict-matmul export; the configuration guard stopped them before training.
They were resubmitted with the corrected script. Report job 738115 follows the
replacement jobs.

Replacement jobs 738285 (raw) and 738286 (clean) passed their checks and entered
epoch 2 with `batch_size=8`; both restored the committed epoch 1 checkpoint.
The first observed raw-arm progress was 3,032 structures / 62.3 seconds; the
clean arm was still amortizing compilation. Await complete epochs for an ETA.
