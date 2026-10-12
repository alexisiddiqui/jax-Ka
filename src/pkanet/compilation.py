"""Process-wide persistent JAX executable cache configuration."""
from __future__ import annotations

import os
from pathlib import Path

import jax
from jax.experimental.compilation_cache import compilation_cache

_configured: Path | None = None


def configure_compilation_cache(directory=None):
    """Configure once; repeating the same request is a no-op.

    JAX owns executable keys (computation, shapes, compiler and device). Model
    weights stay runtime arguments, so compatible checkpoints reuse executables.
    The cache is process-wide: use a new process to switch cache directories.
    """
    global _configured
    current = jax.config.jax_compilation_cache_dir
    if directory is None:
        directory = current or os.environ.get('JAX_COMPILATION_CACHE_DIR')
        if not directory:
            root = os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache'))
            directory = Path(root)/'jax-Ka'/'ogqt'
    path = Path(directory).expanduser().resolve()
    if _configured is not None:
        if path != _configured:
            raise ValueError('JAX compilation cache is process-wide; use a new process to change its directory')
        return path
    if current and Path(current).expanduser().resolve() != path:
        raise ValueError('Requested cache conflicts with the configured JAX compilation cache directory')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    # JAX may have initialized its cache before a directory was configured
    # (e.g. during parameter initialization). Reset its handle, not disk files.
    compilation_cache.reset_cache()
    jax.config.update('jax_compilation_cache_dir', str(path))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    _configured = path
    return path
