"""Reusable, fixed-shape oGQT inference with host preparation and device batching."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from .features import load_features
from .model import PKPDB_PK_MOD
from .ogqt import predict_shift, predict_with_trace


def load_model(path):
    """Load the checked parameter export without importing training/optimizer code."""
    path = Path(path)
    info = json.loads((path / 'model.json').read_text())
    if hashlib.sha256((path / 'params.npz').read_bytes()).hexdigest() != info['params_sha256']:
        raise ValueError('Checkpoint SHA256 does not match model.json')
    tree = {}; count = 0
    with np.load(path / 'params.npz', allow_pickle=False) as data:
        for name in data.files:
            array = data[name]
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f'Invalid checkpoint parameter {name}')
            count += array.size
            node = tree; parts = name.split('/')
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = array
    if count != info['parameter_count']:
        raise ValueError('Checkpoint parameter count mismatch')
    def lists(node):
        if not isinstance(node, dict): return node
        if node and all(k.isdigit() for k in node):
            return [lists(node[str(i)]) for i in range(len(node))]
        return {k: lists(v) for k, v in node.items()}
    tree = lists(tree)
    if tree['embed']['w'].shape != (24, info['architecture']['width']):
        raise ValueError('Checkpoint is not the supported 24-feature backbone oGQT')
    return tree, info


def capacities(dimensions):
    """Power-of-two capacities yield reusable shapes across successive requests."""
    return tuple(1 << (max(int(n), minimum)-1).bit_length()
                 for n, minimum in zip(dimensions, (32, 8, 16, 16, 8)))


def shapes(cap):
    n, k, q, s, sk = cap
    return dict(nodes=(n,24), node_mask=(n,), neighbors=(n,k), edge=(n,k,20),
        edge_mask=(n,k), switch=(n,k), query_residue=(q,), query_group=(q,),
        site_residue=(s,), site_type=(s,), site_mask=(s,), site_neighbors=(s,sk),
        site_edge=(s,sk,34), site_edge_mask=(s,sk), site_switch=(s,sk), query_site=(q,))


def pad_graph(graph, cap):
    expected = shapes(cap)
    if set(graph) != set(expected): raise ValueError('Unexpected graph fields')
    result = {}
    for name, value in graph.items():
        shape = expected[name]
        if value.ndim != len(shape) or any(a>b for a,b in zip(value.shape,shape)):
            raise ValueError(f'Bucket overflow for {name}: {value.shape} > {shape}; no truncation')
        padded = np.zeros(shape, value.dtype)
        padded[tuple(slice(0,n) for n in value.shape)] = value
        result[name] = padded
    return result


def plans(records, batch_size):
    if batch_size < 1: raise ValueError('batch_size must be positive')
    groups = defaultdict(list)
    for i, (_, metadata) in enumerate(records):
        groups[capacities(metadata['dimensions'])].append(i)
    for cap, indices in sorted(groups.items()):
        for start in range(0,len(indices),batch_size):
            yield cap, indices[start:start+batch_size]


def assemble(records, indices, cap, batch_size):
    items = [pad_graph(load_features(records[i][0])[0], cap) for i in indices]
    # Repeated dummy structures are independent vmap lanes; their outputs are discarded.
    items += [items[-1]] * (batch_size-len(items))
    return jax.tree.map(lambda *x: np.stack(x), *items)


class Predictor:
    """Load once; reuse compilation across many structures and predict() calls.

    JAX is portable CPU/GPU inference. Triton is the existing indexed-attention
    implementation and requires the repository's optional CUDA dependencies.
    """
    def __init__(self, model, *, device='cpu', backend='jax', batch_size=4,
                 max_input_bytes=512*1024**2, compilation_cache=None):
        if batch_size < 1 or max_input_bytes < 1: raise ValueError('Invalid batch/memory limit')
        if backend not in ('jax','triton'): raise ValueError('backend must be jax or triton')
        from .compilation import configure_compilation_cache
        self.compilation_cache = (None if compilation_cache is False else
                                  configure_compilation_cache(compilation_cache))
        self.device = jax.devices(device)[0]
        if backend == 'triton' and self.device.platform not in ('gpu','cuda'):
            raise ValueError('Triton requires a GPU')
        params, self.model_info = load_model(model)
        self.params = jax.device_put(params, self.device)
        self.batch_size = batch_size; self.max_input_bytes = max_input_bytes
        self.backend = backend; self.executables = {}; self.last_stats = {}
        def one(p, g):
            # Use the established portable reference; unused attention diagnostics
            # are eliminated by JIT. The accelerated path uses indexed Triton.
            shift = predict_shift(p,g) if backend == 'triton' else predict_with_trace(p,g)['predicted_shift']
            return shift, shift + PKPDB_PK_MOD[g['query_group']]
        self._batch = jax.jit(jax.vmap(one, in_axes=(None,0)))

    def predict(self, records):
        """Yield (input index, per-site shift, per-site pKa), in bucket order.

        At most the active and one prefetched host batch are resident. Feature
        records contain only cache paths and metadata, never the whole corpus.
        """
        specs = iter(plans(records, self.batch_size))
        stats = dict(batches=0, structures=0, compiled_shapes=0,
                     compile_seconds=0., execution_seconds=0., backend=self.backend,
                     device=str(self.device),
                     compilation_cache=str(self.compilation_cache) if self.compilation_cache else None)
        self.last_stats = stats
        def load(spec):
            cap, indices = spec
            # Exact input allocation bound (not an estimate of peak GPU working memory).
            sizes = shapes(cap)
            boolean = {'node_mask','edge_mask','site_mask','site_edge_mask'}
            size = self.batch_size * sum(int(np.prod(v))*(1 if k in boolean else 4) for k,v in sizes.items())
            if size > self.max_input_bytes:
                raise ValueError(f'Bucket {cap} needs {size} input bytes; reduce batch size or raise max_input_bytes')
            return cap, indices, assemble(records, indices, cap, self.batch_size)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = next(specs, None)
            if first is None: return
            future = pool.submit(load, first)
            while future is not None:
                cap, indices, host = future.result()
                upcoming = next(specs, None)
                future = pool.submit(load, upcoming) if upcoming is not None else None
                batch = jax.device_put(host, self.device)
                if cap not in self.executables:
                    start = time.perf_counter()
                    self.executables[cap] = self._batch.lower(self.params,batch).compile()
                    stats['compile_seconds'] += time.perf_counter()-start
                    stats['compiled_shapes'] += 1
                start = time.perf_counter()
                shift, pka = jax.device_get(self.executables[cap](self.params,batch))
                stats['execution_seconds'] += time.perf_counter()-start
                stats['batches'] += 1; stats['structures'] += len(indices)
                for lane, index in enumerate(indices):
                    q = records[index][1]['dimensions'][2]
                    if not np.isfinite(pka[lane,:q]).all():
                        raise FloatingPointError(f'Nonfinite predictions for input {index}')
                    yield index, shift[lane,:q].copy(), pka[lane,:q].copy()
