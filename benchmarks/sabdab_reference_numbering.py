"""Enrich legacy CDR-only evidence with full IMGT numbering, without edits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path


def number_chunk(items):
    from anarci import anarci
    nums, alignments, _ = anarci(items, scheme='imgt', allowed_species=None,
                                allow={'H', 'K', 'L'}, assign_germline=False, ncpu=1)
    out = {}
    for (key, _), domains, info in zip(items, nums, alignments):
        out[key] = [dict(chain_type=a['chain_type'], start=int(start), end=int(end),
                         numbering=[[int(p), str(i).strip(), aa] for (p, i), aa in numbering])
                    for (numbering, start, end), a in zip(domains or [], info or [])]
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_MEM_PER_CPU') != '2048':
        raise RuntimeError('Use a real Slurm allocation with --mem-per-cpu=2G')
    items = sorted(json.loads(args.source.read_text()).items())
    chunks = [items[i:i+32] for i in range(0, len(items), 32)]
    out = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(number_chunk, chunks):
            out.update(result)
    args.output.write_text(json.dumps(dict(numbered=out, version=importlib.metadata.version('anarci')))+'\n')
    print('Enriched numbering for', len(out), 'reference sequences', flush=True)
