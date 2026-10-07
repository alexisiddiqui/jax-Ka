"""Explicit, bounded persistent JAX cache configuration for GQT workloads."""
import hashlib
import json
import os
from pathlib import Path


MAX_BYTES=5*1024**3


def configure(config, code_hashes, *, base=None):
    """Enable the cache only when explicitly requested, before the first JIT."""
    import jax
    import jaxlib
    requested=base or os.environ.get('PKATRAIN_COMPILATION_CACHE_DIR')
    if not requested:return dict(enabled=False)
    try:
        import filelock
    except ImportError as error:
        raise RuntimeError('Opt-in shared compilation caching requires filelock') from error
    device=jax.devices()[0]
    identity=dict(jax=jax.__version__,jaxlib=jaxlib.__version__,platform=device.platform,
        device_kind=device.device_kind,precision=str(jax.config.jax_default_matmul_precision),
        architecture=config.get('architecture',{}),dtype=config.get('dtype','float32'),code_hashes=code_hashes)
    token=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    namespace=f"jax-{jax.__version__}_{device.platform}_{token}"
    path=Path(requested)/namespace;path.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_compilation_cache_dir',str(path))
    jax.config.update('jax_compilation_cache_max_size',MAX_BYTES)
    return dict(enabled=True,path=str(path),namespace=namespace,max_bytes=MAX_BYTES,
                filelock_version=getattr(filelock,'__version__','unknown'),identity=identity)
