# PyTorch autograd integration

Install the optional dependency with `python -m pip install -e '.[torch,test]'`.
Importing the core `jaxpropka` package does not require or import PyTorch.

`jaxpropka.torch.torch_readout` wraps a prepared JAX callable as a
`torch.nn.Module`. It accepts one floating tensor and returns one floating
tensor, retaining the gradient path to an upstream PyTorch model:

```python
import torch
from jaxpropka import TitrationModel, StructureCache
from jaxpropka.torch import torch_readout

model = TitrationModel(StructureCache.load("structure-cache.npz"))
charge = torch_readout(model.charge(ph=7.0))  # construct once
network = torch.nn.Linear(8, 20)
features = torch.randn(len(model.cache.keys), 8)
probabilities = network(features).softmax(dim=-1)
predicted_charge = charge(probabilities)
loss = predicted_charge.square().mean()
loss.backward()  # gradients populate network.weight.grad and network.bias.grad
```

The example shows differentiation mechanics, not a validated scientific
objective. Inputs to the model's readouts are probabilities in the documented
20-amino-acid alphabet. Integer indices and argmax break the gradient path.
If you need the model's allowed-identity or frozen-position handling, wrap a
closure that calls `model.probabilities_from_logits` before its readout.

The wrapper uses the same approach as the BioFeaturisers Torch backend: DLPack
conversion, a custom `torch.autograd.Function`, and a JAX VJP recomputed during
backward. It preserves arbitrary output cotangents, so losses and reductions
may be computed in Torch. `jit=True` compiles forward and backward once per
shape/dtype/device specialization. Use `jit=False` for debugging.

## Contract and limitations

- The callable must be a pure, deterministic `JAX array -> JAX array` function.
  Its captured state must stay fixed between forward and backward. Geometry,
  caches, model constants and other captured parameters are not Torch parameters.
- CPU and CUDA float32/float64 tensors are supported. Float64 requires
  `jax.config.update("jax_enable_x64", True)` before use. There is no silent
  device or precision fallback; a CPU-only DLPack compatibility fallback may
  copy through NumPy. CUDA requires compatible JAX and PyTorch CUDA installs
  with the same visible-device ordering. MPS is not supported by this bridge.
- The input selects the execution device. The wrapper has no parameters or
  buffers to move with `.to()`. Captured JAX arrays should be prepared for the
  intended device; output-device mismatches raise an error.
- Non-contiguous inputs are accepted. Outputs are copied into Torch-owned
  storage so an in-place Torch operation cannot corrupt JAX buffers.
- Only first-order reverse mode is supported. Double backward, `torch.func`
  transforms and `torch.compile(fullgraph=True)` are not supported. Eager Torch
  autograd can surround the internally JIT-compiled JAX readout.
- Wrap `jax.vmap(readout)` to accept a leading batch dimension for one structure.
  Wrapping does not itself change the readout's shape contract.
- Structured results are deliberately rejected. Select a numerical field in a
  closure, e.g. `torch_readout(lambda p: curve_readout(p).residue_charge)`.
  For pKa results, check `valid`, convergence and reliability diagnostics
  separately; a zero sentinel is not a valid pKa measurement. Wrapping `.value`
  does not turn an invalid solve into a usable loss.

## Validation

`tests/test_torch_interop.py` checks vector VJPs, finite-difference gradcheck,
non-contiguous tensors, repeated backward, input mutation detection, output
ownership, batching, and end-to-end PyTorch linear-model parameter gradients
against native JAX through a synthetic titration fixture. A CUDA test skips
unless both frameworks have CUDA devices. Synthetic numerical validation is
not a molecular-accuracy benchmark.
