"""Torch BatchSource for row-aligned site arrays (packed pKAI features, targets, weights); see pkatrain.loading.

PackedSiteSource serves a batch of row indices from a dict of equally long arrays (np.load(..., mmap_mode="r") is fine):
- resident: when the arrays fit `budget_fraction` of free GPU memory they are copied to the device once, in chunks
  (bounded host memory), and a batch is an on-device index_select;
- streaming: otherwise worker threads gather rows from the host arrays (indices sorted for read locality, then put
  back in the requested order) and Prefetcher copies them through pinned memory on a side CUDA stream.
Either way a batch is the requested rows in the requested order, so batch membership, order and loss values are the
same as indexing the arrays directly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .loading import LoaderConfig

CHUNK_BYTES = 256 << 20


class PackedSiteSource:
    def __init__(self, arrays, *, device="cuda", mode="auto", budget_fraction=0.6, config: LoaderConfig | None = None):
        import torch
        self.torch = torch; self.device = torch.device(device); self.config = config or LoaderConfig()
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1: raise ValueError(f"arrays differ in length: { {k: len(v) for k, v in arrays.items()} }")
        self.length = lengths.pop(); self.nbytes = int(sum(value.nbytes for value in arrays.values()))
        if mode == "auto":
            free = torch.cuda.mem_get_info(self.device)[0] if self.device.type == "cuda" else 0
            mode = "resident" if self.device.type == "cuda" and self.nbytes <= budget_fraction * free else "stream"
        if mode not in ("resident", "stream"): raise ValueError(mode)
        self.mode = mode; self.arrays = arrays; self.tensors = None; self.pool = None; self.stream = None
        if mode == "resident":
            self.tensors = {name: self._upload(value) for name, value in arrays.items()}
        else:
            self.pool = ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="site-gather")
            if self.device.type == "cuda": self.stream = torch.cuda.Stream(device=self.device)

    def _upload(self, value):
        torch = self.torch; out = torch.empty(value.shape, dtype=getattr(torch, np.dtype(value.dtype).name), device=self.device)
        rows = max(1, CHUNK_BYTES // max(1, value[:1].nbytes))
        for start in range(0, len(value), rows):
            stop = min(start + rows, len(value)); out[start:stop].copy_(torch.from_numpy(np.array(value[start:stop], copy=True)))
        return out

    def load(self, index):
        index = np.asarray(index, np.int64)
        if self.mode == "resident":
            idx = self.torch.as_tensor(index, device=self.device)
            return {name: tensor.index_select(0, idx) for name, tensor in self.tensors.items()}
        order = np.argsort(index, kind="stable"); ordered = index[order]
        def gather(item):
            name, value = item; rows = np.asarray(value[ordered]); out = np.empty_like(rows); out[order] = rows
            return name, out
        return dict(self.pool.map(gather, self.arrays.items()))

    def to_device(self, batch):
        """Prefetcher hook (streaming): pinned, non_blocking copy on a side stream; returns (tensors, event)."""
        if self.mode == "resident" or self.device.type != "cuda":
            return batch if self.mode == "resident" else {k: self.torch.from_numpy(v) for k, v in batch.items()}, None
        torch = self.torch
        with torch.cuda.stream(self.stream):
            tensors = {name: torch.from_numpy(value).pin_memory().to(self.device, non_blocking=True) for name, value in batch.items()}
            event = torch.cuda.Event(); event.record(self.stream)
        return tensors, event

    def on_consume(self, value):
        """Prefetcher hook: make the consumer's current stream wait for the copy before use."""
        if isinstance(value, tuple):
            tensors, event = value
            if event is not None:
                self.torch.cuda.current_stream(self.device).wait_event(event)
                for tensor in tensors.values(): tensor.record_stream(self.torch.cuda.current_stream(self.device))
            return tensors
        return value

    def hooks(self):
        """(to_device, on_consume) for Prefetcher."""
        return (self.to_device, self.on_consume) if self.mode == "stream" else (None, None)

    def close(self):
        if self.pool is not None: self.pool.shutdown(wait=True)
        self.tensors = None

    def provenance(self):
        return {"source": "packed-sites", "mode": self.mode, "rows": self.length, "bytes": self.nbytes, "workers": self.config.workers}


class CombinedSource:
    """Several sources loaded together; spec and batch are dicts keyed like `sources` (missing keys skipped)."""

    def __init__(self, sources):
        self.sources = sources

    def load(self, spec):
        return {name: self.sources[name].load(value) for name, value in spec.items()}

    def to_device(self, batch):
        out = {}
        for name, value in batch.items():
            hook = self.sources[name].hooks()[0]; out[name] = hook(value) if hook else value
        return out

    def on_consume(self, batch):
        out = {}
        for name, value in batch.items():
            hook = self.sources[name].hooks()[1]; out[name] = hook(value) if hook else value
        return out

    def close(self):
        for source in self.sources.values(): source.close()

    def provenance(self):
        return {name: source.provenance() for name, source in self.sources.items()}
