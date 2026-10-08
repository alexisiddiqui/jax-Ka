import numpy as np
import pytest

jax_triton = pytest.importorskip("jax_triton")

import jax
import jax.numpy as jnp

from pkanet.model import initialize, predict_shift, predict_shift_indexed
from pkatrain.graph_data import pad
from test_pkanet import graph


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="Triton attention requires CUDA")


def relative(left, right):
    numerator = denominator = 0.0
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        numerator += float(np.sum((a-b)**2)); denominator += float(np.sum(a**2))
    return np.sqrt(numerator/max(denominator,1e-30))


def test_indexed_model_matches_native_under_batch_and_gradient():
    # The repository test suite enables x64 globally for the physical solver;
    # GQT training and this optional kernel are deliberately full float32.
    with jax.experimental.enable_x64(False):
        params = initialize(jax.random.PRNGKey(17), width=92, ff=184)
        one, _ = graph(); one, _, _ = pad(one, np.zeros(4,np.float32),(32,32,16))
        np.testing.assert_allclose(
            jax.jit(predict_shift_indexed)(params,one),
            jax.jit(predict_shift)(params,one),rtol=5e-6,atol=5e-6)
        graphs = jax.tree.map(lambda x:jnp.stack((x,x)),one)
        native = jax.jit(jax.vmap(predict_shift,in_axes=(None,0)))
        indexed = jax.jit(jax.vmap(predict_shift_indexed,in_axes=(None,0)))
        left = native(params,graphs); right = indexed(params,graphs)
        np.testing.assert_allclose(right,left,rtol=5e-6,atol=5e-6)
        native_grad = jax.jit(jax.grad(lambda p:jnp.sum(native(p,graphs)**2)))(params)
        indexed_grad = jax.jit(jax.grad(lambda p:jnp.sum(indexed(p,graphs)**2)))(params)
        assert relative(native_grad,indexed_grad)<2e-4
