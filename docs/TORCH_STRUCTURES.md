# Gradients through structural features and oGQT

`pkanet.torch.TorchStructureReadout` places the PyTorch autograd boundary outside
both continuous featurisation and prediction. Its three inputs are sequence,
backbone and side chains; all are Torch floating tensors on the same CPU/CUDA
device. First-order reverse-mode gradients flow back to those inputs and their
upstream Torch operations. Model parameters captured by the readout are fixed;
this is not a Torch parameter-training wrapper.

## Standard backbone / named-heavy-atom inputs

```python
import jax
import jax.numpy as jnp
import torch
from pkanet.differentiable import StructureFeatures, ATOM_NAMES
from pkanet.ogqt import initialize
from pkanet.torch import TorchStructureReadout

features, arrays, metadata = StructureFeatures.from_file(
    'protein.pdb', sidechains=True)
# A dummy 152-input model for smoke tests; supply trained parameters when available.
params = initialize(jax.random.PRNGKey(0), node_dim=152)
params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)
model = TorchStructureReadout.from_ogqt(params, features)
sequence, backbone, sidechains = [
    torch.tensor(x, dtype=torch.float32, requires_grad=True) for x in arrays]
pka = model(sequence, backbone, sidechains,
            gradients=('sequence', 'backbone', 'sidechains'))
pka.sum().backward()
# sequence.grad [N,20], backbone.grad [N,4,3], sidechains.grad [N,32,3]
```

Sequence channels use `jaxpropka.parameters.ALPHABET` order. The model accepts
continuous residue features/probabilities; to get gradients on logits, supply
`logits.softmax(-1)`. It does not argmax the differentiable sequence input.
Sequence gradients are conditional on fixed query/site types, atom presence,
termini, chain membership and residue topology. They are not derivatives through
creating/removing residues, atoms or titratable sites. Reprepare the template
after discrete changes. Soft residue mixtures are an extension of a model
trained on categorical inputs, not a calibrated mutation-effect estimate.

Backbone atom order is N, CA, C, O. Side-chain order is `SIDECHAIN_ATOMS` in
`pkanet.graph`; combined order is `ATOM_NAMES`. Coordinates are in Angstroms.
`model.forward_atoms(sequence, atoms)` accepts [N,36,3] named heavy-atom slots,
splits backbone/side chains and returns their derivatives to the original tensor.
It does not model hydrogens. The current 24/152-feature models do not use backbone
O; its derivative is correctly zero. Missing side-chain slots also have zero
derivative. `sidechains=False` uses the 24-feature backbone model and gives zero
side-chain derivatives.

## Gradient selection

`gradients` can contain any subset of `sequence`, `backbone`, `sidechains`.
A disabled group is detached; existing values still participate in prediction.
`sequence_mask`, `backbone_mask`, and `sidechain_mask` are optional boolean Torch
tensors with prefix shapes: [N] per residue, [N,atoms] per atom, or the full input
shape. True passes gradients; False zeros gradients, without changing forward
values. An entirely detached input normally has `.grad is None`. With no enabled
groups the result does not require grad. This selection is independent from
feature/presence masks used to describe the actual structure.

## Replaceable features, including torsion angles

The generic constructor accepts any pure JAX callables:

```python
model = TorchStructureReadout(featurise_fn, predict_graph_fn)
result = model(sequence_probabilities, backbone_xyz, sidechain_angles,
               gradients=('backbone', 'sidechains'))
```

`featurise_fn(sequence, backbone, sidechains)` returns any JAX-compatible feature
pytree; `predict_graph_fn(features)` returns one floating JAX array. The third
input can be angles [N,chi], sin/cos pairs, coordinates, or another continuous
representation. Its gradient has exactly that representation's shape. Implement
angle-to-coordinate/feature calculations inside the JAX callable to retain their
chain rule; no changes to the Torch bridge are needed. An angular adapter smoke
test exercises this contract. This does not supply a rotamer/kinematics model or
assume a particular future angle feature definition. `forward_atoms` is only the
convenience entry point for the named-coordinate representation.

## Geometry and execution limits

The supplied structural featuriser recomputes local frames, radial bases,
orientations, switch weights and side-chain features inside every differentiable
forward. Dense candidate pairs cover newly entering neighbors as coordinates
move, so it never silently reuses an obsolete radius graph. Its memory cost is
quadratic in residues/sites; large-structure sparse training needs a separate
neighbor-capacity/refresh policy. Parsing and discrete topology preparation stay
outside autograd. Frame validity and radius membership are piecewise/discrete;
finite derivatives at degenerate frames are numerical conventions, not a smooth
extension of topology changes.

The standard adapter uses portable JAX on CPU or GPU. The existing indexed Triton
attention VJP omits switch-weight derivatives, so using it would give incomplete
coordinate gradients. Other production variants (extra orientation/dihedral/H-bond
inputs or altered forwards) require matching feature/model callables; they are
not automatically inferred from checkpoints. `from_ogqt` targets the standard
24/152-feature architecture. The 152-feature test uses initialized parameters;
the released 24-feature checkpoint also has forward-parity coverage.

Construct once for JIT reuse. Persistent compilation-cache configuration is
shared with inference; pass `compilation_cache=PATH` if needed. Both forward and
backward executables use it. Initial configuration also works after JAX parameter
initialization; repeated configuration of the same directory is a no-op. Captured parameters
and topology must not change between forward and backward. Backward recomputes a
JAX VJP; double backward, torch.func transforms and torch.compile fullgraph are
not supported. Tensor gradient masks do not require new model JITs.

The [6SC6 ABC-context validation](validation/6sc6-abc-gradients/RESULTS.md)
checks A–B interface gradients on CPU/GPU and records GPU memory and disk-cache reuse.
