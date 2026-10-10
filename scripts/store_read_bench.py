"""Random-structure read throughput of a packed GQT store (pkatrain.production_graphs.ProductionStore), copying every
field of each structure as a training loader would. Each run uses its own random structures so the page cache does not
serve repeats.  python scripts/store_read_bench.py STORE_DIR LABEL [threads ...]"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pkatrain.production_graphs import ProductionStore

path, label = sys.argv[1], sys.argv[2]; threads = [int(x) for x in sys.argv[3:]] or [8, 32]
store = ProductionStore(path, verify=False); rng = np.random.default_rng(abs(hash(label)) % 2**32)
order = rng.permutation(len(store.ids)); used = 0
def one(cid): return sum(np.array(v, copy=True).nbytes for v in store.raw(cid).values())
for t in threads:
    ids = [store.ids[i] for i in order[used:used + 600]]; used += 600
    began = time.time()
    with ThreadPoolExecutor(t) as pool: total = sum(pool.map(one, ids))
    dt = time.time() - began
    print(f"{label} threads={t} structures/s={len(ids) / dt:.0f} MB/s={total / dt / 1e6:.0f}", flush=True)
