"""Import before pkatrain / pkanet when JAX_PLATFORMS=cpu: the snapshot's attention is a Triton (CUDA) kernel, so on CPU
pkanet.triton_attention is replaced by the plain-JAX forward of jax-Ka pkanet/attention_backend.indexed_attention_reference
(CLI parity with Triton: max 0.002 pKa, audits/default-evals-v1). Forward only (scoring); a no-op on GPU."""
import os
import sys
import types

if os.environ.get("JAX_PLATFORMS") == "cpu":
    import jax
    import jax.numpy as jnp

    def indexed_attention_one(q, k, v, neighbors, bias, mask, switch):
        scale = 1.0 / jnp.sqrt(jnp.asarray(q.shape[-1], q.dtype))
        logits = jnp.einsum("qhd,qkhd->qkh", q, k[neighbors]) * scale + bias
        logits = jnp.where(mask[..., None], logits, -1.0e9)
        weighted = jax.nn.softmax(logits, axis=1) * switch[..., None] * mask[..., None].astype(q.dtype)
        weights = weighted / jnp.maximum(weighted.sum(axis=1, keepdims=True), 1e-8)
        return jnp.einsum("qkh,qkhd->qhd", weights, v[neighbors])
    sys.modules["pkanet.triton_attention"] = types.SimpleNamespace(indexed_attention_one=indexed_attention_one)
