"""Verify persistent executable reuse across fresh Python interpreters."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_persistent_cache_restart_and_runtime_weights(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = '''
import json,sys
import jax
import jax.numpy as jnp
# Simulate model initialization before the wrapper configures its cache.
jax.jit(lambda x: x + 7)(jnp.ones(3)).block_until_ready()
events=[]
jax.monitoring.register_event_listener(lambda name, **kwargs: events.append(name))
from pkanet.compilation import configure_compilation_cache
path=configure_compilation_cache(sys.argv[1])
assert configure_compilation_cache(path/'.') == path
try:
    configure_compilation_cache(path/'other')
except ValueError:
    pass
else:
    raise AssertionError('cache directory change was accepted')
@jax.jit
def predict(weights, x):
    return weights*x + 1
result=predict(jnp.asarray(float(sys.argv[2])), jnp.arange(5,dtype=jnp.float32))
print(json.dumps(dict(values=result.tolist(),hits=events.count('/jax/compilation_cache/cache_hits'))))
'''
    env = dict(os.environ, PYTHONPATH=str(root/'src'), JAX_PLATFORMS='cpu',
               JAX_DEBUG_LOG_MODULES='jax._src.compiler', JAX_ENABLE_COMPILATION_CACHE='true')
    env.pop('JAX_COMPILATION_CACHE_DIR', None)
    cache = tmp_path/'executables'
    def run(weight):
        return subprocess.run([sys.executable,'-c',script,str(cache),str(weight)],
                              env=env,text=True,capture_output=True,check=True)
    first=run(2)
    files={p.name:p.read_bytes() for p in cache.glob('*-cache')}
    assert files
    second=run(3)
    assert json.loads(first.stdout)['values']==[1,3,5,7,9]
    assert json.loads(second.stdout)['values']==[1,4,7,10,13]
    assert json.loads(second.stdout)['hits'] > 0
    assert files=={p.name:p.read_bytes() for p in cache.glob('*-cache')}
