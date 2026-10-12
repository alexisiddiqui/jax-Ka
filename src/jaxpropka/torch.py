"""Optional PyTorch autograd bridge for prepared, single-array JAX readouts.

The bridge follows BioFeaturisers' DLPack / recomputed-VJP approach. Importing
jaxpropka itself does not import PyTorch. Only first-order reverse mode is
supported; geometry and any values closed over by the readout remain fixed.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
import torch
from torch.autograd.function import once_differentiable


def _jax_device(tensor: torch.Tensor):
    if tensor.device.type == "cpu":
        return jax.devices("cpu")[0]
    if tensor.device.type == "cuda":
        try:
            devices = jax.devices("gpu")
        except RuntimeError as exc:
            raise ValueError("CUDA tensors require a CUDA-enabled JAX installation") from exc
        index = tensor.device.index
        for device in devices:
            if device.local_hardware_id == index:
                return device
        raise ValueError(f"No local JAX GPU matches {tensor.device}")
    raise ValueError("torch_readout supports CPU and CUDA tensors only")


def _to_jax(tensor: torch.Tensor, device):
    tensor = tensor.detach().resolve_conj().resolve_neg().contiguous()
    try:
        array = jax.dlpack.from_dlpack(tensor, device=device)
    except (BufferError, RuntimeError, TypeError):
        if tensor.device.type != "cpu":
            raise
        # Compatibility fallback for CPU DLPack implementations only.
        array = jax.device_put(np.array(tensor.numpy(), copy=True), device)
    expected = np.dtype("float64" if tensor.dtype == torch.float64 else "float32")
    if np.dtype(array.dtype) != expected:
        raise TypeError("JAX changed the input dtype during conversion")
    return array


def _to_torch(array, device):
    if not isinstance(array, jax.Array):
        raise TypeError("The readout must return one floating JAX array; select a result field first")
    if np.dtype(array.dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("The readout must return a float32 or float64 JAX array")
    try:
        tensor = torch.utils.dlpack.from_dlpack(array)
    except (BufferError, RuntimeError, TypeError):
        if device.type != "cpu" or array.device.platform != "cpu":
            raise
        tensor = torch.from_numpy(np.array(array, copy=True))
    if tensor.device != device:
        raise ValueError(f"Readout returned {tensor.device}, expected {device}")
    # Torch consumers may modify their output; do not expose JAX-owned buffers.
    return tensor.clone()


class _ReadoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, readout, value):
        device = _jax_device(value)
        ctx.readout = readout
        ctx.jax_device = device
        ctx.save_for_backward(value)
        with jax.default_device(device):
            result = readout._forward(_to_jax(value, device))
        return _to_torch(result, value.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        (value,) = ctx.saved_tensors
        # Recompute the VJP instead of retaining a JAX pullback per forward.
        # save_for_backward also detects in-place changes to the input.
        with jax.default_device(ctx.jax_device):
            gradient = ctx.readout._backward(
                _to_jax(value, ctx.jax_device),
                _to_jax(cotangent, ctx.jax_device),
            )
        return None, _to_torch(gradient, value.device)


class TorchReadout(torch.nn.Module):
    """Use a pure ``JAX array -> JAX array`` callable inside a Torch model.

    Construct once and call with a float32/float64 CPU or CUDA tensor. Device
    and precision follow the input; float64 requires JAX x64 to be enabled.
    The callable must be deterministic and its captured state must stay fixed
    between forward and backward. There are no trainable Torch parameters in
    this module: gradients propagate through its input to upstream parameters.

    Scalar and array outputs are supported. For structured readouts select a
    numeric field with a closure; validate the accompanying diagnostics
    separately. Batch on the JAX side with ``jax.vmap`` before wrapping.
    Double backward, torch.func transforms, torch.compile fullgraph, MPS and
    gradients with respect to closed-over values are not supported.
    """

    def __init__(self, function: Callable, *, jit: bool = True):
        super().__init__()
        if not callable(function):
            raise TypeError("function must be a callable JAX readout")

        def vjp(value, cotangent):
            _, pullback = jax.vjp(function, value)
            return pullback(cotangent)[0]

        self._forward = jax.jit(function) if jit else function
        self._backward = jax.jit(vjp) if jit else vjp

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if value.layout != torch.strided:
            raise TypeError("Input must be a dense strided tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError("Input must have dtype torch.float32 or torch.float64")
        if value.dtype == torch.float64 and not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before using float64 Torch inputs")
        return _ReadoutFunction.apply(self, value)


def torch_readout(function: Callable, *, jit: bool = True) -> TorchReadout:
    """Wrap a prepared JAX readout as a PyTorch module with input gradients."""
    return TorchReadout(function, jit=jit)


__all__ = ["TorchReadout", "torch_readout"]
