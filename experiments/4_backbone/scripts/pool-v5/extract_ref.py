"""abdata-v5 parity reference (Isambard, read-only I/O, no decoding): raw store-v2 / store-hotspot blobs and side-table
rows for the ids in IDS_TSV -> one npz.   python extract_ref.py IDS_TSV OUT_NPZ"""
import csv, json, os, sys
from pathlib import Path
import numpy as np
P = Path('/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench/training/gqt-production-v2/pinder')
ids = [r['id'] for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')]; out = {}
def blobs(store, tag):
    ix = np.load(P / store / 'index.npz'); sid = list(ix['ids'].astype(str)); off = ix['offsets']; fd = os.open(P / store / 'records.bin', os.O_RDONLY)
    for c in ids:
        if c in sid:
            i = sid.index(c); out[f'{tag}|{c}'] = np.frombuffer(os.pread(fd, int(off[i + 1] - off[i]), int(off[i])), np.uint8); out[f'{tag}-dims|{c}'] = ix['dims'][i]
    os.close(fd)
blobs('store-v2', 'store'); blobs('store-hotspot-v1', 'hot1'); blobs('store-hotspot-v2', 'hot2')
for table, arrays in (('backbone-v1', ['features']), ('sidechains-v1', ['features']), ('contacts-v1', ['scores']), ('residue-rsa-v1', ['rsa', 'chain'])):
    index = {c: i for i, c in enumerate(json.loads((P / table / 'ids.json').read_text()))}; offs = np.load(P / table / 'offsets.npy')
    for a in arrays:
        m = np.load(P / table / f'{a}.npy', mmap_mode='r')
        for c in ids: k = index[c]; out[f'{table}:{a}|{c}'] = np.asarray(m[offs[k]:offs[k + 1]])
    if table == 'residue-rsa-v1':
        po = np.load(P / table / 'pair_offsets.npy')
        for a in ('i', 'j', 'd'):
            m = np.load(P / table / f'pair_{a}.npy', mmap_mode='r')
            for c in ids: k = index[c]; out[f'{table}:pair_{a}|{c}'] = np.asarray(m[po[k]:po[k + 1]])
meta = {t: json.loads((P / t / 'metadata.json').read_text()) for t in ('store-v2', 'store-hotspot-v1')}
out['_meta'] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
np.savez(sys.argv[2], **out); print(len(out), 'arrays')
