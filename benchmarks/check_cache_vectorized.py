"""Vectorized build_cache on production complexes: fingerprints vs production-1024-v2 receipts.

v2 receipts record the cache fingerprint computed by the loop implementation
(now build_cache_reference). A bitwise-identical vectorized cache reproduces
every fingerprint. Selection is by residue-count quantiles only (no labels or
prediction outcomes). The reference is re-timed on the same node for a subset.
"""
import argparse
import json
import os
import resource
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--reference-timing', type=int, default=3,
                        help='re-time the reference on this many of the selected complexes (smallest first)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from pkabench.runtime import require_compute, atomic_json
    require_compute()
    from pkabench.prep import read_cif
    from pkabench.schema import read_table
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache, build_cache_reference, native_identities
    structures = sorted(read_table(args.campaign/'structures.parquet'), key=lambda s: s['n_residues'])
    picks = sorted({round(q*(len(structures)-1)/(args.count-1)) for q in range(args.count)})
    chosen = [structures[i] for i in picks]
    records = []
    for rank, s in enumerate(chosen):
        cid = s['complex_id']
        receipt = json.loads((args.campaign/'jobs/jaxka'/f'{cid}.json').read_text())
        for state in ('AB', 'A', 'B'):
            top = load_topology(read_cif(args.campaign/'structures'/cid/f'{state}.cif'), gap_policy='cap', freeze_disulfides=True)
            lib = build_candidates(top, missing_sidechain='error')
            t0 = time.perf_counter()
            cache = build_cache(top, lib, identities=native_identities(top))
            vec = time.perf_counter()-t0
            rec = dict(complex_id=cid, state=state, n_residues=int(top.n_residues), vectorized_seconds=vec,
                       v2_reference_seconds_logged=receipt['extra'][state].get('cache_seconds'),
                       fingerprint_match=cache.fingerprint() == receipt['extra'][state]['cache_fingerprint'])
            if rank < args.reference_timing:
                t0 = time.perf_counter()
                ref = build_cache_reference(top, lib, identities=native_identities(top))
                rec['reference_seconds_same_node'] = time.perf_counter()-t0
                rec['reference_fingerprint_match'] = ref.fingerprint() == cache.fingerprint()
            records.append(rec)
            print(json.dumps(rec), flush=True)
            atomic_json(args.output, dict(status='running', records=records))
    summary = dict(status='complete', node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                   states=len(records), fingerprint_matches=sum(r['fingerprint_match'] for r in records),
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, records=records)
    atomic_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
