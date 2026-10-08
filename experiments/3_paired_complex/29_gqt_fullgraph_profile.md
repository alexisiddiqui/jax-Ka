# Full-graph GQT runtime profile

Profile the existing full-graph training problem before changing storage,
bucketing, graph representation or model context. Compare the 49,709- and
209,645-parameter backbone GQT models on all three existing capacity buckets.

Measure synchronized stage attribution separately from normal overlapped
throughput. Preserve the fused forward/backward/update kernel as the primary
compute benchmark, four parallel structure workers, one-batch prefetch, exact
batch membership and full validation every epoch. Profile validation separately,
including repeated graph hashing, loading, inference and 2,000-replicate
bootstrap aggregation.

Persistent compilation caching is an independent, explicit option. Namespace it
by software, device, precision, architecture and code; bound each namespace at
5 GiB and require locked concurrent access. Report it only as a startup/restart
optimization.

Add an exact-array mmap backend only if steady-state loader waits exceed 5% of
epoch time or its projected end-to-end gain reaches 10%. Cropping is a separate
accuracy experiment and is not implemented here.

## Baseline result

The A40 profile completed for both models with the existing full validation
pass. The 49,709-parameter model took 86.92 s per warm training epoch and spent
19.63% of that epoch waiting for prefetched input. The 209,645-parameter model
took 102.77 s and spent 11.59% waiting. Validation took 5.44 s and 5.38 s,
respectively, including 2,000 group-bootstrap replicates. The loader therefore
passes the pre-registered 5% gate for an mmap prototype at both model sizes.

The opt-in persistent compilation cache passed a concurrent two-writer and
one-reader integrity check. A cached reader reached the first fused call in
2.69 s, compared with 20.69--20.91 s for the two cold writers. This is a
restart improvement; it does not alter warm epoch throughput.

The mmap follow-up stores the existing arrays exactly in read-only `.npy`
files, indexed by the frozen manifest. Conversion checks every field of every
structure against the source NPZ and records full-file SHA-256 checksums.
Training augmentation copies only the four fields it mutates before the normal
padding copy. The matched benchmark uses the same initialization, sampled
batches, update count and validation timing, in NPZ--mmap--mmap--NPZ order.
Adoption requires at least 10% improvement in training plus validation for both
models. Batches, forwards and scalar losses must agree exactly in every
capacity bucket. CUDA gradient reductions are not bitwise repeatable even for
the same input, so one-step gradient-derived updates must remain below 1e-5
absolute and 5e-6 relative L2, the accepted float32 gradient tolerance.
Full-epoch parameter checksums are diagnostic only: reduction differences
compound through Adam, and repeats of the same storage backend are not bitwise
reproducible.

## mmap result

The conversion produced a 28.2 GiB read-only bundle for 5,142 structures in
18.5 minutes. Every raw field was checked bit-for-bit against its source NPZ.
Representative padded batches, model forwards and scalar losses were also
bit-identical in all three capacity buckets. One-step update differences stayed
within the accepted float32 gradient tolerance.

| Model | NPZ epoch | mmap epoch | Training gain | Training + validation gain |
|---|---:|---:|---:|---:|
| 49,709 parameters | 89.44 s | 70.11 s | 21.6% | 20.4% |
| 209,645 parameters | 103.37 s | 91.43 s | 11.6% | 11.0% |

Both models pass the 10% end-to-end adoption gate. The backend remains explicit
through `PKATRAIN_GRAPH_BACKEND=mmap`; runs record the backend, bundle path and
verification checksum. Setting `PKATRAIN_GRAPH_MMAP_VERIFY=1` additionally
rehashes all bundle files at open time and is intended for audits rather than
normal training. Full validation remains enabled every epoch and continues to
use the unchanged NPZ path, whose measured contribution is about 5.4 seconds.
