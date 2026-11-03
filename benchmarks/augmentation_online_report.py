"""Summarise the synchronised online-feature pilot without touching payloads."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);args=p.parse_args()
report=json.loads((args.root/'report.json').read_text())
projection=json.loads((args.root/'storage-projection.json').read_text())
rows=[]
for comparison in report['comparisons']:
    key=(comparison['dataset'],comparison['bucket'])
    resident={r['kind']:r for r in report['resident'] if (r['dataset'],r['bucket'])==key}
    old=resident['cached']['milliseconds'];new=resident['online']['milliseconds']
    rows.append(dict(dataset=key[0],bucket=int(key[1]),batch=resident['online']['micro'],
        geometry_ms=resident['online']['geometry_only_ms'],cached_resident_ms=old,online_resident_ms=new,
        resident_overhead_percent=100*(new/old-1),cached_structures_s=comparison['cached'],
        online_structures_s=comparison['online'],end_to_end_ratio=comparison['throughput_ratio'],
        online_wait_percent=100*comparison['online_worst_wait_fraction']))
with (args.root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['GPU feature construction pilot','==============================','',
    f"Job {report['job']}; {report['gpu']}; {report['wall_s']:.1f} seconds including conversion/compile. "
    f"All numerical checks passed: {report['passed']}.",'',
    'The cached and online arms use the same sampled views, exact neighbour lists, live native graphs, '
    'global production capacities, batch 16, loader workers 16 and prefetch 2. Each size bucket has two '
    'alternating-order repeats of 20 synchronised forward/backward batches. The model is the frozen '
    'base-A width64/ff128 orientation+dihedrals/dropout0.1 protocol with Triton indexed attention. '
    'The objective is a synthetic paired/burial proxy; no optimiser or training occurs.','',
    'CPU tests cover the NumPy reconstruction and real packed single-/paired-branch loaders. GPU '
    'checks compare every non-derived input, active geometry, objective and parameter gradients. '
    'Only the tiny geometry dot products request full float32 precision; model precision is unchanged.','',
    'Geometry-only time includes materialising expanded GPU outputs. Resident step time fuses '
    'construction into the forward/backward program; end-to-end timing also includes host reading, '
    'assembly and host-to-device copies. These are separate measurements, not additive costs.','',
    f"Device-prefetched batches: {report.get('device_prefetch',False)} (production whole-batch loading path).",'',
    'Dataset   Bucket  Geometry ms  Resident cached/online ms  Compute change  End-to-end speedup',
    '=======   ======  ===========  =========================  ==============  ==================']
for r in rows:
    lines.append(f"{r['dataset']:<9} {r['bucket']:>6}  {r['geometry_ms']:>11.3f}  "
        f"{r['cached_resident_ms']:>10.3f}/{r['online_resident_ms']:<10.3f}     "
        f"{r['resident_overhead_percent']:>10.1f}%     {r['end_to_end_ratio']:>13.2f}x")
lines+=['=======   ======  ===========  =========================  ==============  ==================','',
    f"Active feature max absolute error: {max(max(c['active_feature_max_abs'].values()) for c in report['checks']):.3g}. "
    f"Worst relative loss error: {max(c['loss_relative'] for c in report['checks']):.3g}; "
    f"worst relative gradient L2 error: {max(c['gradient_relative_l2'] for c in report['checks']):.3g}.",'',
    f"Pilot compressed features: {report['storage']['expanded_bytes']/1e9:.3f} GB expanded versus "
    f"{report['storage']['compact_bytes']/1e6:.2f} MB compact ({1/report['storage']['ratio']:.1f}x smaller). "
    f"Full compact reader projection: {projection['projected_reader_bytes']/1e9:.2f} GB, including "
    'the existing metadata/coverage allowance. This uses exact eligible-view counts by '
    'dataset/split/partner/unit-length stratum and mean compact probability-sample frame bytes; '
    'stress-only examples are excluded. All population strata are covered.','',
    'Limits: warm pilot working set; full-release cold random I/O is unmeasured. Neighbour search '
    'remains precomputed. PINDER still reads and decodes expanded live native graphs, then drops '
    'derived fields before transfer; both native/free branches are reconstructed on the GPU. '
    'Only sampled size buckets were timed. The full compact release has not been built or activated.','']
(args.root/'report.rst').write_text('\n'.join(lines))
print(json.dumps(dict(passed=report['passed'],cases=len(rows),geometry_ms_range=[min(r['geometry_ms'] for r in rows),max(r['geometry_ms'] for r in rows)],
    resident_overhead_percent_range=[min(r['resident_overhead_percent'] for r in rows),max(r['resident_overhead_percent'] for r in rows)],
    end_to_end_speedup_range=[min(r['end_to_end_ratio'] for r in rows),max(r['end_to_end_ratio'] for r in rows)],
    projected_compact_reader_GB=projection['projected_reader_bytes']/1e9),indent=2))
