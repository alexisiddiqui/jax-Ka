"""Pinned endpoint parity, coordinate derivatives and partner isolation."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pkanet.differentiable import StructureFeatures
from pkanet.relational import RelationalStructureFeatures
from pkatrain.production_defaults import load_frozen_default

ROOT = Path(__file__).resolve().parents[1]


def case(state='bound'):
    base, inputs, _ = StructureFeatures.from_file(ROOT/'tests/data/two_chains.pdb')
    return base, RelationalStructureFeatures(base, state), tuple(map(jnp.asarray, inputs))


@pytest.mark.parametrize('state', ['bound', 'free'])
def test_matches_production(state):
    if jax.default_backend() != 'gpu':
        pytest.skip('Production indexed forward requires GPU')
    jax.config.update('jax_enable_x64', False)
    _, features, inputs = case(state)
    _, reference, _ = load_frozen_default()
    _, portable, _ = load_frozen_default(input_gradients=True)
    graph = features(*inputs)
    np.testing.assert_allclose(jax.jit(portable)(graph), jax.jit(reference)(graph), atol=3e-5, rtol=3e-5)


def test_gradients_and_chain_isolation():
    jax.config.update('jax_enable_x64', True)
    base, free, inputs = case('free')
    inputs = tuple(x.astype(jnp.float64) for x in inputs)
    _, predict, _ = load_frozen_default(input_gradients=True)
    bound = RelationalStructureFeatures(base, 'bound')
    fn = jax.jit(lambda seq, bb, sc: predict(bound(seq, bb, sc)))
    weights = jnp.linspace(.1, 1., len(base.graph['query_site']))
    objective = lambda *x: jnp.sum(fn(*x)*weights)
    gradients = jax.grad(objective, argnums=(0,1,2))(*inputs)
    for k in (0,1):
        g = gradients[k]
        assert np.isfinite(g).all() and np.linalg.norm(g)>1e-6
        direction = g/jnp.linalg.norm(g)
        plus=list(inputs); minus=list(inputs); eps=.00001
        plus[k]=plus[k]+eps*direction; minus[k]=minus[k]-eps*direction
        fd=(objective(*plus)-objective(*minus))/(2*eps)
        np.testing.assert_allclose(fd, jnp.sum(g*direction), rtol=.025, atol=.003)
    np.testing.assert_array_equal(gradients[2], 0.)
    # Query outputs for chain A cannot depend on chain B in the free branch.
    queries = base.chain[base.graph['query_residue']] == base.chain[0]
    chain_b = base.chain != base.chain[0]
    free_fn = jax.jit(lambda bb: predict(free(inputs[0],bb,inputs[2])))
    moved=inputs[1]+jnp.asarray(chain_b)[:,None,None]*jnp.array([2.,1.,-.5])
    np.testing.assert_allclose(free_fn(moved)[queries],free_fn(inputs[1])[queries],atol=1e-6)
    free_grad=jax.grad(lambda bb:jnp.sum(free_fn(bb)*queries))(inputs[1])
    np.testing.assert_array_equal(free_grad[chain_b],0.)
    assert np.max(np.abs(np.asarray(fn(inputs[0],moved,inputs[2])-fn(*inputs))[queries]))>1e-5


def test_torch_default():
    torch=pytest.importorskip('torch')
    from pkanet.torch import TorchStructureReadout
    base,features,inputs=case()
    module=TorchStructureReadout.from_default(base,compilation_cache=False,output='shift')
    assert not list(module.parameters())
    x=[torch.tensor(np.asarray(v),requires_grad=True) for v in inputs]
    y=module(*x); y.sum().backward()
    _,predict,info=load_frozen_default(input_gradients=True)
    np.testing.assert_allclose(y.detach().numpy(),predict(features(*inputs)),atol=3e-5,rtol=3e-5)
    assert module.model_metadata['params_sha256']==info['params_sha256']
    assert all(torch.isfinite(v.grad).all() for v in x)
    assert x[1].grad.abs().max()>1e-6
    with pytest.raises(ValueError):
        TorchStructureReadout.from_default(base,state='invalid')
