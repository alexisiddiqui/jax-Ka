"""Feature parity, finite-difference input gradients, masks and adapter contracts."""
import itertools
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from pkanet.differentiable import StructureFeatures
from pkanet.torch import TorchStructureReadout
from pkanet.graph import sidechain_features, SIDECHAIN_ATOMS
from pkanet.ogqt import initialize, predict_with_trace
from pkanet.inference import load_model

ROOT = Path(__file__).resolve().parents[1]


def setup_case(sidechains=True):
    features, inputs, _ = StructureFeatures.from_file(ROOT/'tests/data/two_chains.pdb',sidechains=sidechains)
    params = initialize(jax.random.PRNGKey(4),width=8,ff=16,node_dim=152 if sidechains else 24)
    params = jax.tree.map(lambda x:jnp.asarray(x,dtype=jnp.float32),params)
    module = TorchStructureReadout.from_ogqt(params,features,output='shift',compilation_cache=False)
    return features, inputs, params, module


def test_features_match_existing_and_backbone_checkpoint():
    features,inputs,params,module=setup_case()
    g=features(*inputs);old=features.graph
    for name in ('edge','switch','edge_mask'):
        actual=np.asarray(g[name])[np.arange(len(inputs[0]))[:,None],old['neighbors']]
        np.testing.assert_allclose(actual[old['edge_mask']],old[name][old['edge_mask']],atol=2e-6)
    for name in ('site_edge','site_switch','site_edge_mask'):
        actual=np.asarray(g[name])[np.arange(len(old['site_residue']))[:,None],old['site_neighbors']]
        np.testing.assert_allclose(actual[old['site_edge_mask']],old[name][old['site_edge_mask']],atol=2e-6)
    atoms=[{name:inputs[2][i,j] for j,name in enumerate(SIDECHAIN_ATOMS) if features.atom_present[i,j]}
           for i in range(len(inputs[0]))]
    side=sidechain_features(inputs[1],atoms)
    np.testing.assert_allclose(g['nodes'][:,24:],side,atol=2e-6)
    ref=dict(old,nodes=np.concatenate((old['nodes'],side),-1))
    actual=module(*(torch.tensor(x) for x in inputs)).detach().numpy()
    np.testing.assert_allclose(actual,predict_with_trace(params,ref)['predicted_shift'],atol=2e-5,rtol=2e-5)
    features,inputs,_=StructureFeatures.from_file(ROOT/'tests/data/two_chains.pdb')
    trained,_=load_model(ROOT/'models/ogqt-production-50k')
    layer=TorchStructureReadout.from_ogqt(trained,features,output='shift',compilation_cache=False)
    np.testing.assert_allclose(layer(*(torch.tensor(x) for x in inputs)).detach().numpy(),
        predict_with_trace(trained,features.graph)['predicted_shift'],atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_forward_backward_finite_difference_and_masks(device):
    if device=='cuda' and not torch.cuda.is_available():
        if os.environ.get('OGQT_REQUIRE_GPU'):pytest.fail('CUDA required')
        pytest.skip('CUDA unavailable')
    features,inputs,params,module=setup_case()
    x=[torch.tensor(v,device=device,requires_grad=True) for v in inputs]
    weights=torch.linspace(.1,1.,len(features.graph['query_site']),device=device)
    value=module(*x);(value*weights).sum().backward()
    gradients=[t.grad.detach().clone() for t in x]
    assert all(torch.isfinite(g).all() and g.abs().max()>1e-7 for g in gradients)
    # Central differences through the entire feature+model computation.
    for k in range(3):
        direction=gradients[k]/gradients[k].norm()
        plus=[t.detach().clone() for t in x];minus=[t.detach().clone() for t in x]
        eps=.002;plus[k]+=eps*direction;minus[k]-=eps*direction
        fd=((module(*plus)-module(*minus))*weights).sum()/(2*eps)
        analytic=(gradients[k]*direction).sum()
        torch.testing.assert_close(fd,analytic,rtol=.02,atol=.003)
    # All eight group selections preserve the forward, detach only requested groups.
    groups=('sequence','backbone','sidechains')
    for flags in itertools.product((False,True),repeat=3):
        xx=[t.detach().clone().requires_grad_() for t in x]
        y=module(*xx,gradients=tuple(g for g,on in zip(groups,flags) if on))
        torch.testing.assert_close(y,value)
        if any(flags):
            (y*weights).sum().backward()
            for t,expected,on in zip(xx,gradients,flags):
                if on:torch.testing.assert_close(t.grad,expected)
                else:assert t.grad is None
        else:assert not y.requires_grad
    # Atom-level and residue-level masks; absent atoms always have zero derivative.
    xx=[t.detach().clone().requires_grad_() for t in x]
    sm=torch.ones(len(x[0]),device=device,dtype=torch.bool);sm[0]=False
    bm=torch.ones(x[1].shape[:2],device=device,dtype=torch.bool);bm[:,0]=False
    cm=torch.tensor(features.atom_present,device=device);cm[0]=False
    y=module(*xx,sequence_mask=sm,backbone_mask=bm,sidechain_mask=cm)
    (y*weights).sum().backward()
    for t,g,m in zip(xx,gradients,(sm[:,None],bm[...,None],cm[...,None])):
        torch.testing.assert_close(t.grad,g*m)
    assert not gradients[2][~torch.tensor(features.atom_present,device=device)].any()
    # Combined heavy-atom input gives the same coordinate derivatives.
    atoms=torch.cat([x[1].detach(),x[2].detach()],dim=1).requires_grad_()
    (module.forward_atoms(x[0].detach(),atoms)*weights).sum().backward()
    torch.testing.assert_close(atoms.grad,torch.cat(gradients[1:],dim=1))


def test_custom_angle_features_soft_sequence_and_degenerate_frames():
    # A different side-chain input shape and feature mapping, without bridge changes.
    def features(sequence,backbone,angles):
        return dict(seq=sequence,geometry=jnp.sum(backbone**2),torsions=jnp.sin(angles))
    layer=TorchStructureReadout(features,lambda g:jnp.sum(g['seq']**2)+g['geometry']+jnp.sum(g['torsions']),compilation_cache=False)
    logits=torch.randn(3,20,requires_grad=True);bb=torch.randn(3,4,3,requires_grad=True)
    angles=torch.randn(3,4,requires_grad=True)
    layer(logits.softmax(-1),bb,angles).backward()
    torch.testing.assert_close(angles.grad,angles.detach().cos())
    torch.testing.assert_close(bb.grad,2*bb.detach())
    assert logits.grad.abs().sum()>0
    f,inputs,_,module=setup_case()
    inputs=list(inputs);inputs[1]=inputs[1].copy();inputs[1][0]=inputs[1][0,1]
    x=[torch.tensor(v,requires_grad=True) for v in inputs]
    module(*x).sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in x)


def test_switch_weight_coordinate_derivative():
    features, inputs, _, _ = setup_case()
    inputs=list(inputs); inputs[1]=inputs[1].copy()
    inputs[1][1] += inputs[1][0,1]+np.array([19.,0.,0.])-inputs[1][1,1]
    layer=TorchStructureReadout(features,lambda g:g['switch'][0,1],compilation_cache=False)
    x=[torch.tensor(v,requires_grad=True) for v in inputs]
    value=layer(*x);value.backward()
    np.testing.assert_allclose(value.detach(),.5,atol=2e-5)
    np.testing.assert_allclose(x[1].grad[1,1,0],-np.pi/4,atol=2e-5)
