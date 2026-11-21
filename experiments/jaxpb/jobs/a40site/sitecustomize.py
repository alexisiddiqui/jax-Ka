"""A40 shim (Coulson comp1400 only; on PYTHONPATH via jobs/a40.sbatch, never on Isambard).
JAX 0.11.1's Pallas Triton lowering looks the device kind up in jax._src.pallas.triton.gpu_info and has no entry for
"NVIDIA A40" (it lists A10 = sm_86, the same architecture), so every Triton pallas_call fails on the A40 with
"No supported GPU devices found". gpu_info.registry is the module's extension point for unlisted device kinds: register
the A40 as arch 8.6 / compute capability 86. Changes nothing on devices JAX already knows."""
try:
    from jax._src.pallas.triton import gpu_info as _gi
    _gi.registry.setdefault("NVIDIA A40", lambda: _gi.GpuInfo(gpu_version=None, arch_name="8.6", compute_capability=86))
except Exception as _e:  # pragma: no cover
    import sys
    print(f"[a40site] gpu_info registration failed: {_e!r}", file=sys.stderr)
