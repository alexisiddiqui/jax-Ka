"""Numerical bridge validation; synthetic fixtures are not accuracy benchmarks."""
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from jaxpropka import ModelConfig, TitrationModel
from jaxpropka.torch import TorchReadout, torch_readout


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vector_vjp_noncontiguous_and_repeated_backward(jit, dtype):
    fn = lambda x: jnp.sin(x) + x ** 2
    wrapped = torch_readout(fn, jit=jit)
    base = torch.linspace(-0.6, 0.8, 12, dtype=dtype).reshape(3, 4).requires_grad_()
    x = base.T
    weights = torch.linspace(0.2, 1.2, 12, dtype=dtype).reshape(3, 4).T
    y = wrapped(x)
    np.testing.assert_allclose(y.detach(), fn(jnp.asarray(x.detach().numpy())), rtol=1e-6, atol=1e-7)
    (y * weights).sum().backward(retain_graph=True)
    expected = ((torch.cos(x) + 2*x) * weights).T
    torch.testing.assert_close(base.grad, expected)
    (y * weights).sum().backward()
    torch.testing.assert_close(base.grad, 2*expected)


def test_scalar_gradcheck():
    wrapped = TorchReadout(lambda x: jnp.sum(jnp.sin(x) * x))
    x = torch.tensor([0.2, -0.5, 0.9], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(wrapped, (x,), eps=1e-6, atol=1e-5)


def test_torch_model_parameter_gradients_match_jax(cache):
    read = TitrationModel(cache, ModelConfig(steps=16)).charge([5.0, 7.0])
    wrapped = torch_readout(read)
    layer = torch.nn.Linear(4, 20, bias=True, dtype=torch.float64)
    torch.manual_seed(19)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight)*0.1)
        layer.bias.copy_(torch.randn_like(layer.bias)*0.1)
    features = torch.randn(3, 4, dtype=torch.float64)
    probabilities = layer(features).softmax(-1)
    actual = wrapped(probabilities)
    loss = actual.square().mean()
    loss.backward()

    def reference(w, b):
        p = jax.nn.softmax(jnp.asarray(features.numpy()) @ w.T + b, -1)
        return jnp.mean(read(p)**2)

    expected_loss, (gw, gb) = jax.value_and_grad(reference, argnums=(0, 1))(
        jnp.asarray(layer.weight.detach().numpy()), jnp.asarray(layer.bias.detach().numpy()))
    np.testing.assert_allclose(loss.detach(), expected_loss, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(layer.weight.grad, gw, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(layer.bias.grad, gb, rtol=1e-8, atol=1e-10)
    assert layer.weight.grad.norm() > 0


def test_batched_readout(cache):
    read = TitrationModel(cache, ModelConfig(steps=8)).charge(7.)
    fn = jax.vmap(read)
    p = torch.full((2, 3, 20), .05, dtype=torch.float64, requires_grad=True)
    y = torch_readout(fn)(p)
    y.sum().backward()
    expected = jax.grad(lambda x: fn(x).sum())(jnp.asarray(p.detach().numpy()))
    np.testing.assert_allclose(p.grad, expected, rtol=1e-8, atol=1e-10)


def test_no_grad_and_output_ownership():
    wrapped = torch_readout(lambda x: x)
    x = torch.ones(4, requires_grad=True)
    with torch.no_grad():
        y = wrapped(x)
        y.add_(3)
    assert not y.requires_grad
    torch.testing.assert_close(x, torch.ones(4))


def test_input_mutation_is_detected():
    x = torch.ones(4, requires_grad=True)
    y = torch_readout(lambda x: x*x)(x)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()


def test_double_backward_rejected():
    x = torch.tensor([.4], requires_grad=True)
    y = torch_readout(lambda x: x**3)(x)
    g, = torch.autograd.grad(y.sum(), x, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(g.sum(), x)


def test_contract_errors():
    wrapped = torch_readout(lambda x: x)
    with pytest.raises(TypeError, match="torch.Tensor"):
        wrapped(np.ones(3))
    with pytest.raises(TypeError, match="dtype"):
        wrapped(torch.ones(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="CPU and CUDA"):
        wrapped(torch.empty(3, device="meta"))
    previous_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            wrapped(torch.ones(3, dtype=torch.float64))
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    with pytest.raises(TypeError, match="one floating JAX array"):
        torch_readout(lambda x: {"value": x})(torch.ones(3))
    with pytest.raises(TypeError, match="float32 or float64"):
        torch_readout(lambda x: x > 0)(torch.ones(3))


def test_core_import_does_not_import_torch():
    subprocess.run([sys.executable, "-c", "import sys; import jaxpropka; assert 'torch' not in sys.modules"], check=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PyTorch CUDA")
def test_cuda_value_and_gradient():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("requires JAX CUDA")
    if not devices:
        pytest.skip("requires JAX CUDA")
    x = torch.tensor([.2, -.4], device="cuda", requires_grad=True)
    y = torch_readout(lambda x: jnp.sin(x))(x)
    assert y.device == x.device
    y.sum().backward()
    torch.testing.assert_close(y, x.sin())
    torch.testing.assert_close(x.grad, x.cos())


@pytest.mark.parametrize("kind", ["curve", "pka", "grid"])
def test_selected_structured_fields_match_jax(cache, kind):
    model = TitrationModel(cache, ModelConfig(steps=64))
    if kind == "curve":
        read = model.curves([6., 7.])
        fn = lambda p: read(p).residue_charge
    elif kind == "pka":
        read = model.pka_sites([(cache.keys[0], "ASP")])
        fn = lambda p: read(p).value
    else:
        read = model.pka_from_grid(np.linspace(-2, 16, 37), groups=["ASP"])
        fn = lambda p: read(p).value
    p = torch.full((3, 20), .05, dtype=torch.float64, requires_grad=True)
    jp = jnp.asarray(p.detach().numpy())
    if kind != "curve":
        assert np.asarray(read(jp).valid).all()
    y = torch_readout(fn)(p)
    y.square().sum().backward()
    expected, grad = jax.value_and_grad(lambda x: jnp.sum(fn(x)**2))(jp)
    np.testing.assert_allclose(y.detach().square().sum(), expected, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(p.grad, grad, atol=1e-8, rtol=1e-7)
