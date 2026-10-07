"""True structure batches, preserving per-structure loss and sampled membership."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from threading import Lock
import time
import os
import numpy as np
import jax
from pkabench.runtime import digest
from .graph_data import bucket,pad,mask_features


def epoch_batches(order,byid,rng,batch_size):
    buckets=defaultdict(list)
    for cid in order:buckets[bucket(byid[cid])].append(cid)
    batches=[members[i:i+batch_size] for members in buckets.values() for i in range(0,len(members),batch_size)]
    rng.shuffle(batches)
    return batches


class LoaderTelemetry:
    """Opt-in loader timings. Disabled loaders retain the original hot path."""
    def __init__(self):
        self._lock=Lock();self.structures=[];self.batches=[];self.prefetch_waits=[]

    def structure(self,record):
        with self._lock:self.structures.append(record)

    def batch(self,record):
        with self._lock:self.batches.append(record)

    def prefetch_wait(self,record):
        with self._lock:self.prefetch_waits.append(record)

    def snapshot(self):
        with self._lock:
            return dict(structures=list(self.structures),batches=list(self.batches),
                        prefetch_waits=list(self.prefetch_waits))


class BatchLoader:
    def __init__(self,out,manifest,batch_size,telemetry=None,backend=None):
        self.out=Path(out);self.manifest=manifest;self.batch_size=batch_size
        self.telemetry=telemetry
        self.byid={r['complex_id']:r for r in manifest['records']}
        self.verified=set();self.pool=ThreadPoolExecutor(max_workers=4)
        self.backend=backend or os.environ.get('PKATRAIN_GRAPH_BACKEND','npz')
        if self.backend not in ('npz','mmap'):raise ValueError(f'Unknown graph backend: {self.backend}')
        self.store=None
        if self.backend=='mmap':
            from .graph_mmap import GraphMMap,default_bundle
            location=os.environ.get('PKATRAIN_GRAPH_MMAP_DIR')
            self.store=GraphMMap(location or default_bundle(self.out),manifest['records'],
                verify_files=os.environ.get('PKATRAIN_GRAPH_MMAP_VERIFY')=='1')
        self.context_plan=None;self.context_mask=None
        if manifest['config'].get('context_mask_probability',0):
            import json
            path=Path(manifest['context_path'])/'plan.json'
            assert digest(path)==manifest['context_plan_sha256']
            self.context_plan=json.loads(path.read_text())

    def set_epoch(self,epoch):
        if self.context_plan is None:return None
        from .context_augmentation import epoch_mask,mask_digest
        cfg=self.manifest['config']
        self.context_mask=epoch_mask(self.context_plan,cfg['seed'],epoch,cfg['context_mask_probability'])
        return mask_digest(self.context_mask)

    def one(self,cid,capacities=None):
        began=time.perf_counter() if self.telemetry is not None else None
        row=self.byid[cid];path=self.out/'data'/cid/'graph.npz'
        verified=False;verify_seconds=0.
        if self.backend=='npz':
            if cid not in self.verified:
                tv=time.perf_counter() if self.telemetry is not None else None
                assert digest(path)==row['sha256'];self.verified.add(cid);verified=True
                if tv is not None:verify_seconds=time.perf_counter()-tv
            t=time.perf_counter() if self.telemetry is not None else None
            with np.load(path,allow_pickle=False) as f:
                graph={k:f[k] for k in f.files if k!='labels'};labels=f['labels']
        else:
            t=time.perf_counter() if self.telemetry is not None else None
            graph,labels=self.store.raw(cid)
        read_seconds=0. if t is None else time.perf_counter()-t
        actual_edges=int(graph['edge_mask'].sum());actual_nodes=int(row['n']);actual_queries=len(labels)
        t=time.perf_counter() if self.telemetry is not None else None
        if self.context_plan is not None and row['split']=='train':
            from .context_augmentation import mask_graph
            assert self.context_mask is not None
            r=self.context_plan['structures'][cid]
            if self.backend=='mmap':
                # The canonical mmap is immutable. Only augmentation-mutated
                # fields need copies; pad() copies every field afterwards.
                graph=dict(graph)
                for name in ('nodes','edge','edge_mask','switch'):graph[name]=graph[name].copy()
            graph=mask_graph(graph,self.context_mask[r['start']:r['stop']])
        augment_seconds=0. if t is None else time.perf_counter()-t
        t=time.perf_counter() if self.telemetry is not None else None
        graph,y,mask=pad(graph,labels,capacities or self.manifest['capacities'][bucket(row)])
        result=mask_features(graph,self.manifest['config']),y,mask
        if self.telemetry is not None:
            self.telemetry.structure(dict(complex_id=cid,bucket=bucket(row),verified=verified,
                verify_seconds=verify_seconds,read_seconds=read_seconds,augment_seconds=augment_seconds,
                pad_and_mask_seconds=time.perf_counter()-t,total_seconds=time.perf_counter()-began,
                actual_nodes=actual_nodes,actual_edges=actual_edges,actual_queries=actual_queries))
        return result

    def load(self,cids,capacities=None):
        began=time.perf_counter() if self.telemetry is not None else None
        assert 0<len(cids)<=self.batch_size
        assert len({bucket(self.byid[cid]) for cid in cids})==1
        items=list(self.pool.map(self.one,cids,repeat(capacities)))
        valid=np.arange(self.batch_size)<len(items)
        while len(items)<self.batch_size:
            g,y,m=items[-1];items.append((g,np.zeros_like(y),np.zeros_like(m)))
        t=time.perf_counter() if self.telemetry is not None else None
        graph=jax.tree.map(lambda *x:np.stack(x),*[item[0] for item in items])
        result=graph,np.stack([x[1] for x in items]),np.stack([x[2] for x in items]),valid
        if self.telemetry is not None:
            stack_seconds=time.perf_counter()-t
            n,k=graph['neighbors'].shape[1:];q=graph['query_residue'].shape[1]
            self.telemetry.batch(dict(complex_ids=list(cids),bucket=bucket(self.byid[cids[0]]),
                real_structures=len(cids),load_seconds=time.perf_counter()-began,stack_seconds=stack_seconds,
                actual_nodes=sum(int(item[0]['node_mask'].sum()) for item in items[:len(cids)]),
                actual_edges=sum(int(item[0]['edge_mask'].sum()) for item in items[:len(cids)]),
                actual_queries=sum(int(item[2].sum()) for item in items[:len(cids)]),
                padded_nodes=self.batch_size*n,padded_edges=self.batch_size*n*k,padded_queries=self.batch_size*q))
        return result

    def iterate(self,batches):
        # Prefetch the next host batch while the current GPU update runs.
        def unpack(spec):
            if isinstance(spec,dict):return spec['cids'],spec['capacities']
            return spec,None
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            first=unpack(batches[0]) if batches else None
            future=prefetch.submit(self.load,*first) if first else None
            for i,spec in enumerate(batches):
                cids,_=unpack(spec)
                t=time.perf_counter() if self.telemetry is not None else None
                inputs=future.result()
                if self.telemetry is not None:
                    self.telemetry.prefetch_wait(dict(batch=i,startup=i==0,seconds=time.perf_counter()-t,
                                                      complex_ids=list(cids)))
                following=unpack(batches[i+1]) if i+1<len(batches) else None
                future=prefetch.submit(self.load,*following) if following else None
                yield cids,inputs

    def provenance(self):
        if self.backend=='npz':return dict(backend='npz',per_graph_sha256_on_first_access=True)
        return dict(backend='mmap',path=str(self.store.path),read_only=True,
            verification_sha256=self.store.verification_sha256,
            full_file_verification_at_open=os.environ.get('PKATRAIN_GRAPH_MMAP_VERIFY')=='1')

    def close(self):
        self.pool.shutdown()
        if self.store is not None:self.store.close()
