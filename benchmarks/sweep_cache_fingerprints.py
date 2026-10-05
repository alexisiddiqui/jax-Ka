"""Full-campaign check: vectorized native caches vs production-1024-v2 receipt fingerprints.

``run`` handles one shard (complexes with index % shards == shard, in
residue-count order so shards are balanced); ``summarize`` merges shards.
"""
import argparse
import json
import os
import resource
import time
from pathlib import Path


def run(campaign, shard, shards, out):
    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    from pkabench.prep import read_cif
    from pkabench.schema import read_table
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache, native_identities
    structures = sorted(read_table(campaign/'structures.parquet'), key=lambda s: (s['n_residues'], s['complex_id']))
    mine = structures[shard::shards]; records = []
    for s in mine:
        cid = s['complex_id']; receipt = json.loads((campaign/'jobs/jaxka'/f'{cid}.json').read_text())
        for state in ('AB', 'A', 'B'):
            top = load_topology(read_cif(campaign/'structures'/cid/f'{state}.cif'), gap_policy='cap', freeze_disulfides=True)
            lib = build_candidates(top, missing_sidechain='error')
            t0 = time.perf_counter(); cache = build_cache(top, lib, identities=native_identities(top))
            records.append(dict(complex_id=cid, state=state, n_residues=int(top.n_residues),
                                seconds=time.perf_counter()-t0,
                                reference_seconds_logged=receipt['extra'][state]['cache_seconds'],
                                match=cache.fingerprint() == receipt['extra'][state]['cache_fingerprint']))
    atomic_json(out/f'shard-{shard:03d}.json', dict(shard=shard, shards=shards, node=os.uname().nodename,
        job=os.environ.get('SLURM_JOB_ID'), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        records=records))
    print(json.dumps(dict(shard=shard, states=len(records), matches=sum(r['match'] for r in records))), flush=True)


def summarize(campaign, shards, out):
    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    parts = [json.loads((out/f'shard-{i:03d}.json').read_text()) for i in range(shards) if (out/f'shard-{i:03d}.json').exists()]
    records = [r for p in parts for r in p['records']]
    from pkabench.schema import read_table
    expected = 3*len(read_table(campaign/'structures.parquet'))
    vec = sum(r['seconds'] for r in records); ref = sum(r['reference_seconds_logged'] for r in records)
    summary = dict(shards_present=len(parts), shards_expected=shards, states=len(records), states_expected=expected,
                   matches=sum(r['match'] for r in records),
                   mismatches=[{k: r[k] for k in ('complex_id', 'state')} for r in records if not r['match']],
                   vectorized_hours=vec/3600, reference_hours_logged=ref/3600, speedup_total=ref/vec if vec else None,
                   max_state_seconds=max(r['seconds'] for r in records),
                   max_shard_peak_rss_mib=max(p['peak_rss_mib'] for p in parts),
                   passed=len(parts) == shards and len(records) == expected and all(r['match'] for r in records))
    atomic_json(out/'summary.json', summary); print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['run', 'summarize'])
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--shards', type=int, default=30)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    (run(a.campaign, a.shard, a.shards, a.output) if a.stage == 'run' else summarize(a.campaign, a.shards, a.output))
