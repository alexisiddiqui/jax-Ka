"""Complete-assembly feature lookup and optional GPU upload; no training."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
from pkabench.runtime import atomic_json,require_compute,digest
from pkatrain.masif_store import MaSIFStore


def run(source,output,gpu=False,max_assemblies=0):
    start=time.perf_counter();store=MaSIFStore(source);initialization=time.perf_counter()-start
    parents=[r for r in store.parents.values() if r['status']=='covered'];parents.sort(key=lambda r:(r['dataset'],r['cid']))
    population=len(parents);rng=random.Random(20261121);rng.shuffle(parents)
    if max_assemblies:parents=parents[:max_assemblies]
    passes=[];device=None
    if gpu:
        import jax
        devices=jax.devices()
        if len(devices)!=1 or devices[0].platform!='gpu':raise RuntimeError('exactly one allocated GPU must be visible')
        device=devices[0].device_kind
        jax.block_until_ready(jax.device_put(np.zeros((1024,80),np.float32)))
    def load(row):
        t=time.perf_counter();value=store.load_assembly(row['dataset'],row['cid'],split=row['split'])
        descriptors=[u['arrays']['descriptors'] for u in value['units']]
        n=sum(len(x) for x in descriptors);latency=time.perf_counter()-t;transfer=0.
        if gpu:
            # Actual dense descriptor arrays, no network or synthetic training
            # computation. Bound the staging batch to 1024 surface points.
            batch=np.concatenate([x[:1024] for x in descriptors],axis=0)[:1024]
            t=time.perf_counter();jax.block_until_ready(jax.device_put(batch));transfer=time.perf_counter()-t
        return dict(lookup_s=latency,points=n,surfaces=len(descriptors),device_put_s=transfer)
    try:
        for workers in [1,2,4]:
            for repeat in range(2):
                t=time.perf_counter()
                if workers==1:rows=[load(row) for row in parents]
                else:
                    with ThreadPoolExecutor(workers) as pool:rows=list(pool.map(load,parents))
                wall=time.perf_counter()-t
                passes.append(dict(workers=workers,repeat=repeat,assemblies=len(rows),surfaces=sum(x['surfaces'] for x in rows),
                    points=sum(x['points'] for x in rows),wall_s=wall,assemblies_per_s=len(rows)/wall,
                    lookup_median_ms=float(np.median([x['lookup_s'] for x in rows])*1000),
                    lookup_p95_ms=float(np.percentile([x['lookup_s'] for x in rows],95)*1000),
                    device_put_sum_s=sum(x['device_put_s'] for x in rows)))
                atomic_json(output,dict(status='running',passes=passes,device=device))
        result=dict(status='complete',passes=passes,device=device,reader_initialization_s=initialization,
            covered_assemblies=population,sampled_assemblies=len(parents),surfaces=len(store.store.rows),source_manifest_sha256=digest(Path(source)/'manifest.json'),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            coverage=dict(Counter(r['dataset']+'/'+r['split'] for r in parents)),
            limits=['Small size/profile-spread pilot; does not measure full-release random-I/O or model training throughput',
                'OS page cache is not evicted; first-read and repeated passes are reported separately',
                'GPU timing measures bounded uploads of actual fingerprints, not a model forward/backward pass'],training_enabled=False)
        atomic_json(output,result);print(json.dumps(result),flush=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--max-assemblies',type=int,default=0);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True,gpu_benchmark=a.gpu)
    run(a.source,a.output,a.gpu,a.max_assemblies)
