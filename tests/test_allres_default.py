"""Default all-residue GQT: attention backend parity, sparse/dense graph agreement, free-state isolation, gradients,
rigid invariance, Torch endpoint and the ogqt predict CLI."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pkanet.allres import AllResidueStructureFeatures, graph_from_features, state_graph
from pkanet.attention_backend import indexed_attention_reference
from pkanet.differentiable import StructureFeatures
from pkanet.features import featurise
from pkatrain.production_defaults import DECOY_EXPORT, DECOY_RUN, DEFAULT_RUN, load_default

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / 'tests/data/two_chains.pdb'


@pytest.fixture(scope='module')
def model():
    return load_default(input_gradients=True)


def case(state='bound'):
    base, inputs, _ = StructureFeatures.from_file(PDB)
    return base, AllResidueStructureFeatures(base, state), tuple(map(jnp.asarray, inputs))


def test_reference_attention_matches_gathered_softmax():
    rng = np.random.default_rng(0); q, n, k, h, d = 7, 9, 5, 4, 16
    args = (rng.normal(size=(q, h, d)), rng.normal(size=(n, h, d)), rng.normal(size=(n, h, d)), rng.integers(0, n, (q, k)),
            rng.normal(size=(q, k, h)), rng.random((q, k)) > .3, rng.random((q, k)))
    qq, kk, vv, nb, bias, mask, switch = (jnp.asarray(a, jnp.float32) if a.dtype.kind == 'f' else jnp.asarray(a) for a in args)
    logits = jnp.einsum('qhd,qkhd->qkh', qq, kk[nb]) / 4.0 + bias
    soft = jax.nn.softmax(jnp.where(mask[..., None], logits, -1e9), axis=1) * mask[..., None] * switch[..., None]
    want = jnp.einsum('qkh,qkhd->qhd', soft / jnp.maximum(soft.sum(1, keepdims=True), 1e-8), vv[nb])
    np.testing.assert_allclose(indexed_attention_reference(qq, kk, vv, nb, bias, mask, switch), want, atol=1e-6)


def test_pinned_export(model):
    assert model.info['run'] == DEFAULT_RUN and model.info['parameter_count'] == 180000
    assert 'allres' in model.info['protocol']['production_variant']


@pytest.mark.parametrize('state', ['bound', 'free'])
def test_sparse_cli_graph_matches_dense_template(model, state):
    """The cached-file graph (pkanet.features, sparse 20 A lists) and the differentiable dense template agree."""
    _, features, inputs = case(state)
    dense = jax.jit(model.outputs)(features(*inputs))
    graph, _ = featurise(PDB)
    sparse = jax.jit(model.outputs)(state_graph({k: jnp.asarray(v) for k, v in graph_from_features(graph).items()}, state))
    for key in ('query_pka', 'shift', 'gate_logit', 'burial_logit'):
        np.testing.assert_allclose(sparse[key], dense[key], atol=2e-4, rtol=2e-4)


def test_free_isolation_gradients_and_invariance(model):
    jax.config.update('jax_enable_x64', True)
    try:
        base, free, inputs = case('free'); bound = AllResidueStructureFeatures(base, 'bound')
        inputs = tuple(x.astype(jnp.float64) for x in inputs)
        chain_b = base.chain != base.chain[0]; queries = base.chain[base.graph['query_residue']] == base.chain[0]
        free_fn = jax.jit(lambda bb: model.outputs(free(inputs[0], bb, inputs[2]))['query_pka'])
        bound_fn = jax.jit(lambda bb: model.outputs(bound(inputs[0], bb, inputs[2]))['query_pka'])
        moved = inputs[1] + jnp.asarray(chain_b)[:, None, None] * jnp.array([2., 1., -.5])
        np.testing.assert_allclose(free_fn(moved)[queries], free_fn(inputs[1])[queries], atol=1e-6)
        assert np.max(np.abs(np.asarray(bound_fn(moved) - bound_fn(inputs[1]))[queries])) > 1e-5
        grad = jax.grad(lambda bb: jnp.sum(free_fn(bb) * queries))(inputs[1])
        assert np.isfinite(grad).all() and np.linalg.norm(grad) > 1e-6
        np.testing.assert_array_equal(grad[chain_b], 0.)
        theta = .7; rot = jnp.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        rigid = inputs[1] @ rot.T + jnp.array([5., -3., 2.])
        np.testing.assert_allclose(bound_fn(rigid), bound_fn(inputs[1]), atol=1e-5)
    finally:
        jax.config.update('jax_enable_x64', False)


def test_edge_pair_logits_symmetric(model):
    _, features, inputs = case('free')
    z = model.outputs(features(*inputs))['edge_embedding']; a = jnp.array([0, 1, 2]); b = jnp.array([5, 6, 7])
    np.testing.assert_allclose(model.pair_logits(z, a, b), model.pair_logits(z, b, a), atol=1e-6)


def test_torch_default(model):
    torch = pytest.importorskip('torch')
    from pkanet.torch import TorchStructureReadout
    base, features, inputs = case()
    module = TorchStructureReadout.from_default(base, compilation_cache=False, output='shift')
    assert not list(module.parameters()) and module.model_metadata['run'] == DEFAULT_RUN
    x = [torch.tensor(np.asarray(v), requires_grad=True) for v in inputs]
    y = module(*x); y.sum().backward()
    np.testing.assert_allclose(y.detach().numpy(), model.outputs(features(*inputs))['query_shift'], atol=3e-5, rtol=3e-5)
    assert x[1].grad.abs().max() > 1e-6
    burial = TorchStructureReadout.from_default(base, compilation_cache=False, output='burial')(*x)
    assert burial.shape == (len(base.chain),) and bool(((burial >= 0) & (burial <= 1)).all())
    with pytest.raises(ValueError):
        TorchStructureReadout.from_default(base, output='invalid')


def test_cli_default(tmp_path):
    from pkanet.cli import main
    out = tmp_path / 'out.jsonl'
    main(['predict', str(PDB), '--cache', str(tmp_path / 'cache'), '--output', str(out), '--no-compilation-cache'])
    row = json.loads(out.read_text().splitlines()[0])
    assert row['model'] == DEFAULT_RUN and row['states'] == ['bound', 'free']
    site = row['sites'][0]
    assert {'pka_bound', 'pka_free', 'binding_shift', 'gate_bound'} <= set(site)
    assert abs(site['binding_shift'] - (site['pka_bound'] - site['pka_free'])) < 1e-9
    residue = row['residues'][0]; assert 'burial_bound' in residue and 'interface_logit' in residue
    main(['predict', str(PDB), '--cache', str(tmp_path / 'cache'), '--output', str(out), '--state', 'bound', '--no-residues',
          '--no-compilation-cache'])
    row = json.loads(out.read_text().splitlines()[0]); assert 'residues' not in row and 'pka' in row['sites'][0]


def test_decoy_export_selectable(model, tmp_path):
    """The decoy-ranking export loads by name (same tree, extra protocol keys), differs from the default, and the CLI
    takes it with --model; an unpinned export is refused."""
    import shutil
    from pkanet.cli import main
    decoy = load_default(DECOY_EXPORT)
    assert decoy.info['run'] == DECOY_RUN and decoy.info['parameter_count'] == 180000
    assert {'decoy_retype', 'decoy_batch'} <= set(decoy.info['protocol'])
    assert jax.tree.structure(decoy.params) == jax.tree.structure(model.params)
    graph, _ = featurise(PDB); g = {k: jnp.asarray(v) for k, v in graph_from_features(graph).items()}
    a, b = model.outputs(g)['query_pka'], decoy.outputs(g)['query_pka']
    assert np.isfinite(b).all() and np.max(np.abs(np.asarray(a - b))) > 1e-3
    out = tmp_path / 'out.jsonl'
    main(['predict', str(PDB), '--model', DECOY_EXPORT, '--cache', str(tmp_path / 'cache'), '--output', str(out),
          '--state', 'bound', '--no-residues', '--no-compilation-cache'])
    row = json.loads(out.read_text().splitlines()[0])
    assert row['model'] == DECOY_RUN
    # the CLI runs on CPU; on a GPU host the direct forward uses GPU matmul precision (~3e-4 apart)
    np.testing.assert_allclose([s['pka'] for s in row['sites']], b, atol=2e-3)
    copy = tmp_path / 'unpinned'; shutil.copytree(decoy.path, copy)
    info = json.loads((copy / 'model.json').read_text()); info['run'] = 'other'; (copy / 'model.json').write_text(json.dumps(info))
    with pytest.raises(ValueError):
        load_default(copy)
