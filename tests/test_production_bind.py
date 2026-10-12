import numpy as np
import pytest

from pkatrain import production_bind as bind


def setup(width=8, seed=0):
    import jax
    import jax.numpy as jnp
    params = bind.initialize({"head": {"w": jnp.zeros((width, 1))}}, seed)["aux_bind"]
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    return params, jax.random.normal(k1, (2, 5, width)), jax.random.normal(k2, (2, 5, width))


def test_head_starts_neutral_and_static_weights_give_the_plain_forms():
    from pkanet.model import norm
    params, bound, free = setup()
    for operand in bind.OPERANDS:
        assert np.allclose(bind.logits(params, bound, free, operand), 0.0)
    with pytest.raises(ValueError):
        bind.logits(params, bound, free, "sum")
    unit = {**params, "w2": {**params["w2"], "b": np.ones(8, np.float32)}}
    a, b = np.asarray(norm(params["norm"], bound)), np.asarray(norm(params["norm"], free))
    assert np.allclose(bind.logits(unit, bound, free, "product"), (a * b).sum(-1), atol=1e-4)
    assert np.allclose(bind.logits(unit, bound, bound, "difference"), 0.0, atol=1e-5)


def test_weight_block_and_both_branches_train():
    import jax
    params, bound, free = setup()
    loss = lambda p, x, y: jax.numpy.sum((bind.logits(p, x, y, "product") - 1.0) ** 2)
    grads = jax.grad(loss)(params, bound, free)
    assert float(np.abs(grads["w2"]["w"]).sum()) > 0 and float(np.abs(grads["w2"]["b"]).sum()) > 0
    moved = {**params, "w2": {"w": 0.1 * jax.random.normal(jax.random.PRNGKey(5), (8, 8)), "b": params["w2"]["b"]}}
    token_grads = jax.grad(loss, argnums=(1, 2))(moved, bound, free)
    assert all(float(np.abs(g).sum()) > 0 for g in token_grads)
