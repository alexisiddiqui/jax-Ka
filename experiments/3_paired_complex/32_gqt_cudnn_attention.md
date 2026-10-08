# GQT cuDNN gathered-neighbour attention probe

## Question

Can `jax.nn.dot_product_attention(..., implementation="cudnn")` replace GQT's
native gathered-neighbour attention without changing the model and improve A40
training throughput?

## Required adaptation

The obvious node-as-batch mapping cannot be trained with JAX 0.6.2/cuDNN:

- the native GQT query length is one, while cuDNN training with additive bias
  requires even query and key lengths;
- learned edge-bias gradients require bias batch size one and a bias head axis
  equal to the query head axis.

The probe therefore maps `node x model_head` to the cuDNN head axis, keeps batch
size one, duplicates the query to length two, and pads head dimensions 11 and
23 to 16 and 24 respectively. It passes the original `1/sqrt(d)` scale, so
padding does not alter the scale. The distance switch is added as `log(switch)`
to the learned edge bias. Empty rows receive a zero-valued sentinel and are
zeroed after attention.

## Registered gates

1. The adapter must retain gradients for the learned edge bias.
2. Empty neighbourhoods must produce exactly zero.
3. Forward outputs and gradients must meet the established `5e-6` relative
   float32 comparison tolerance on ordinary, non-floor-triggering inputs.
4. Forward and forward-plus-backward attention must be faster than native
   float32 attention on the comp1400 A40.
5. The `1e-8` normalization-floor case must either match or have a cheaper,
   explicit fallback.

Failure of either numerical or speed gates stops integration into `pkanet`.

## Execution

- Job: 743914
- Node/GPU: comp1400, one A40
- JAX: 0.6.2, x64 disabled
- CPU allocation: 8 CPUs, 2 GiB per CPU
- Benchmark: one-off capability probe, removed after the optimization was rejected
- Raw result: `_runtime/jax-Ka/pkabench/audits/gqt-cudnn-attention-v1/capability.json`

## Result

The packed layout trains and handles empty rows, but fails the numerical and
speed gates. cuDNN is slower in all four cases. Relative output error is about
`2.8e-3` to `2.9e-3`; relative gradient error is about `4.4e-3`. The original
floor semantics cannot be expressed by normalized FlashAttention: in the
forced floor case, the native output is scaled down by `1e-4`, whereas cuDNN
renormalizes it to unit mass.

The direct mapping also fails exactly as predicted: query length one is
unsupported for training with bias; after padding the query, learned bias
backpropagation rejects bias batch size greater than one.

## Decision

Do not add this backend to production GQT. A native fallback for the floor
would require detecting the denominator from the native logits and softmax,
duplicating the work the fused kernel was intended to replace. The adapter
also retains gathered key/value tensors, so it does not address the principal
memory cost. The probe and submission wrapper were removed after recording the
raw result and source hash, so no rejected implementation remains in the model
or benchmark source tree.

If attention itself becomes dominant after tighter buckets, the next viable
kernel experiment is a fused gather plus sparse attention kernel with an
optimized backward pass. The current profile does not justify that work yet.
