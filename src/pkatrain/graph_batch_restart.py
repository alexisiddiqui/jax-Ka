"""Versioned checkpoint handover after the true-GPU-batch gate passes."""
import os
from pathlib import Path
import shutil
from pkabench.runtime import atomic_json,digest,require_compute
from .records import read


def register(root,arm):
    base=root/'pretraining/pkpdb-5k-comparison-v1';gate=read(base/'batch-probe.json')
    assert gate['passed'] and all(r['gradient_relative_difference']<2e-4 for r in gate['rows'])
    source=base/f'gqt-{arm}';dest=base/f'gqt-batched-{arm}'
    m=read(source/'manifest.json');seed=m['config']['seed']
    checkpoint=source/f'seed-{seed}'/'checkpoints'/read(source/f'seed-{seed}/checkpoints/latest.json')['checkpoint']
    meta=read(checkpoint/'metadata.json');assert meta['manifest_sha256']==digest(source/'manifest.json')
    m['parent']=dict(path=str(source),manifest_sha256=digest(source/'manifest.json'))
    m['resume_checkpoint']=dict(path=str(checkpoint),metadata_sha256=digest(checkpoint/'metadata.json'),epoch=meta['epoch'])
    m['batch_probe_sha256']=digest(base/'batch-probe.json')
    m['config'].update(batch_size=8,accumulation=1,matmul_precision='highest',
        batching='vmap over eight structures in one size bucket; shuffle bucket batches; mask tail slots',
        optimizer_weighting='mean site MSE per structure, then mean across real structures',
        ordering_change='Same group-uniform draws; bucket grouping changes optimizer batch membership and subsequent RNG stream')
    dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists():
        assert read(dest/'manifest.json')==m;return
    (dest/'data').symlink_to(source/'data',target_is_directory=True)
    shutil.copy2(source/'preparation.json',dest/'preparation.json')
    shutil.copy2(source/'train_type_means.json',dest/'train_type_means.json')
    run=dest/f'seed-{seed}';run.mkdir(exist_ok=True)
    if (source/f'seed-{seed}/initial.json').exists():shutil.copy2(source/f'seed-{seed}/initial.json',run/'initial.json')
    history=read(source/f'seed-{seed}/history.json')[:meta['epoch']]
    atomic_json(run/'history.json',history)
    atomic_json(dest/'manifest.json',m)
    atomic_json(dest/'handover.json',dict(inherited_epochs=meta['epoch'],checkpoint=str(checkpoint),
        parent_checkpoint_verified=True,optimizer_and_rng_restored=True,batch_size=8,gradient_accumulation=False))


if __name__=='__main__':
    import sys
    require_compute(threads=8,gpu_benchmark=True)
    register(Path(os.environ['PKABENCH_RUNTIME']),sys.argv[1])
