"""Compute-only CLI for the versioned shared-training experiment."""
import argparse
from pathlib import Path
import traceback
from pkabench.runtime import require_compute,atomic_json


def main():
    import os
    require_compute(threads=int(os.environ.get('PKATRAIN_THREADS','1')),gpu_benchmark=os.environ.get('PKATRAIN_GPU')=='1')
    import jax
    jax.config.update('jax_enable_x64',True)
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['init','register-fast','register-float32','buckets','speed-gates','prepare','gate','union-gradient','gates','smoke','train','profile','profiles','baseline','report'])
    p.add_argument('out',type=Path); p.add_argument('argument',nargs='?'); a=p.parse_args(); out=a.out.resolve()
    from . import records,validation,experiment
    if (out/'manifest.json').exists():jax.config.update('jax_enable_x64',records.read(out/'manifest.json')['config'].get('dtype')!='float32')
    if a.command=='init': records.initialize(out)
    elif a.command=='register-fast': records.register_fast(out,Path(a.argument).resolve())
    elif a.command=='register-float32': records.register_fast(out,Path(a.argument).resolve(),full_float32=True)
    elif a.command=='buckets':
        from .buckets import build
        build(out)
    elif a.command=='speed-gates': validation.collect_speed(out)
    elif a.command=='prepare': records.prepare(out,a.argument)
    elif a.command=='gate': validation.real_gate(out,a.argument)
    elif a.command=='union-gradient': validation.union_gradient(out,a.argument)
    elif a.command=='gates': validation.collect_gates(out)
    elif a.command=='smoke': experiment.train(out,17,smoke=True)
    elif a.command=='train': experiment.train(out,int(a.argument))
    elif a.command=='profile': validation.profile(out,a.argument)
    elif a.command=='profiles': validation.collect_profiles(out)
    elif a.command=='baseline':
        import jax.numpy as jnp
        experiment.evaluate(out,jnp.zeros(3),out/'baseline',records.read(out/'manifest.json')['val'],diagnostics=True)
    else:
        from .report import collect
        collect(out)

if __name__=='__main__': main()
