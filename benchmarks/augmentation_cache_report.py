"""Summarise the completed cache pilot without training or expanding shared files."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

from pkatrain.augmentation_cache_pilot import stratum

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--coordinates',type=Path,required=True)
p.add_argument('--quota',type=Path,required=True);p.add_argument('--results',type=Path)
args=p.parse_args();root=args.root;results=args.results or root
build=json.loads((root/'build-summary.json').read_text());quota=json.loads(args.quota.read_text())
population=Counter()
for row in json.loads((args.coordinates/'index.json').read_text())['records']:
    if row['summary']['selected']>0:population[stratum(row)]+=1
sample=defaultdict(list)
for row in build['parents_detail']:
    if not row['stress_only'] and row['status']=='built':sample[row['stratum']].append(row)
if set(population)!=set(sample):raise ValueError('availability calibration has an unsampled stratum')
weights={row['parent']:population[key]/len(rows) for key,rows in sample.items() for row in rows}
projected_bytes=sum(weights.get(row['parent'],0)*row['feature_bytes'] for row in build['parents_detail'])
worker_h=sum(weights.get(row['parent'],0)*row['wall_s'] for row in build['parents_detail'])/3600
pack=json.loads((root/'packing-probe.json').read_text()) if (root/'packing-probe.json').exists() else None
pack_h=None
verify_h=None
if pack is not None:
    pack_h=sum(weights.get(row['parent'].rsplit('/',1)[0],0)*row['pack_write_s'] for row in pack['measurements'])/3600
    verify_h=pack['finish_verify_s']/pack['compressed_bytes']*projected_bytes/3600
shards=math.ceil(projected_bytes/(2*1024**3));coordinate_bytes=json.loads((args.coordinates/'manifest.json').read_text())['compressed_bytes']
consolidated_files=shards+7;distributed_inodes=shards+223+224*8+227
additional_bytes=math.ceil(projected_bytes*1.3)+coordinate_bytes+2*2*1024**3+1024**3
additional_inodes=distributed_inodes+452+32
budget=dict(passed=quota['used_bytes']+additional_bytes<=.9*quota['limit_bytes'] and
            quota['used_files']+additional_inodes<=.9*quota['limit_files'] and additional_inodes<=5000,
    additional_bytes=additional_bytes,additional_inodes=additional_inodes,
    projected_used_bytes=quota['used_bytes']+additional_bytes,projected_used_inodes=quota['used_files']+additional_inodes,
    limits=quota,policy='30% payload allowance, full coordinate release, two transfer partial shards, 1 GiB metadata reserve, conservative 224-worker files/directories; shared publication avoids a second full cache copy')
gpu={}
for dataset in ('pkpdb','pinder'):
    path=results/f'gpu-{dataset}.json'
    if path.exists():
        d=json.loads(path.read_text());gpu[dataset]=dict(passed=d['passed'],viable_settings=d['viable_settings'],
            comparisons=d['comparisons'],finite_probes=len(d['finite_probes']),peak_rss_GiB=d['peak_rss_GiB'],
            device_memory_stats=d['device_memory_stats'],job=d['job'],source_report=d.get('source_report'),
            population_weighted_proxy_estimate=d.get('population_weighted_proxy_estimate'))
compat=json.loads((root/'live-compatibility.json').read_text()) if (root/'live-compatibility.json').exists() else None
result=dict(build_passed=build['passed'],sample_parents=build['parents'],representative=build['representative'],
    built_parents=build['counts']['built'],unavailable=build['counts'].get('coordinate_unavailable',0),records=build['records'],
    projected_feature_bytes=projected_bytes,projected_full_payload_shards=shards,projected_consolidated_files=consolidated_files,
    projected_build_worker_hours=worker_h,projected_serial_pack_write_hours=pack_h,
    projected_single_writer_finish_verify_hours=verify_h,
    conditional_build_planning={str(w):dict(ideal_overlap_h=max(worker_h/w,pack_h)+verify_h,
        planning_h_30pct_margin=(max(worker_h/w,pack_h)+verify_h)*1.3) for w in (4,16)} if pack_h is not None else None,
    conditional_build_assumptions='Coulson timings, bounded worker queue feeding one writer, one final verification; excludes queueing/transfers and unmeasured full-scale cold I/O. This is not an implemented full-run submission.',
    population_augmented_parents=sum(population.values()),calibrated_weights=weights,
    projection='Post-stratify the random sample within dataset/split/partner/length using the exact known count of parents with retained views; excluded source failures have exactly zero cache allocation. Stress controls have zero projection weight.',
    budget=budget,live_compatibility=compat,gpu=gpu,training_enabled=False,
    ready_for_full_cache=bool(build['passed'] and compat and compat['passed'] and budget['passed'] and len(gpu)==2 and all(g['passed'] for g in gpu.values())),
    limitations=build['limitations']+['Pilot verification and compilation warm a ~5 GB working set; full-dataset cold random I/O remains unmeasured',
        'CPU packing and graph build costs were measured on Coulson, not GH200 CPUs; no full-build wall-time guarantee',
        'The GPU objective is a matched compute proxy, without native teacher-label reuse or optimiser updates',
        'PINDER timing changes one sampled partner; simultaneous A/B perturbation and mixed-dataset training are unmeasured',
        'A full build still needs bounded/resumable worker outputs, exported compact native partitions, global capacities and memory-sized finalisation'])
(root/'pilot-summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['Packed ANM feature-cache pilot','==============================','',
    f"Sample: {build['parents']} parents ({build['representative']} probability sample plus stress controls); {build['counts']['built']} built; {build['records']} cached views.",
    f"Projected full free-partner payload: {projected_bytes/1e12:.3f} TB; {shards} payload shards, approximately {consolidated_files} consolidated files.",
    f"CPU graph build and independent checking: {worker_h:.2f} weighted worker-hours. Serial pack/write estimate: {pack_h:.2f} hours; final verification estimate: {verify_h:.2f} hours." if pack_h is not None else 'Packing probe pending.',
    f"Conservative storage/inode budget passed: {budget['passed']}.",
    f"Native Isambard compatibility: {compat['passed'] if compat else 'pending'}.",
    f"GPU results: { {d:dict(passed=g['passed'],settings=g['viable_settings']) for d,g in gpu.items()} }.",
    f"Ready for a full feature-cache build: {result['ready_for_full_cache']}.",'',
    result['projection'],'']
for dataset,g in gpu.items():
    rows=[x for x in g['comparisons'] if x['workers']==16 and x['prefetch']==2]
    if rows:
        lines.append(f"{dataset}: 16 workers/prefetch 2; {sum(x['passed'] for x in rows)}/{len(rows)} buckets pass; throughput retention {min(x['throughput_ratio'] for x in rows):.3f}..{max(x['throughput_ratio'] for x in rows):.3f}; worst wait {max(x['worst_wait_fraction'] for x in rows):.3%}.")
        for x in rows:
            if not x['passed']:lines.append(f"Unmet gate: {dataset} bucket {x['bucket']} retains {x['throughput_ratio']:.1%} of native throughput (required 90%).")
    if g.get('population_weighted_proxy_estimate'):
        z=g['population_weighted_proxy_estimate']['2'];lines.append(f"{dataset}: population-weighted warm proxy throughput retention {z['relative_throughput']:.1%}; not an actual training epoch measurement.")
lines+=['',result['conditional_build_assumptions'],str(result['conditional_build_planning']),'']+result['limitations']
(root/'report.rst').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('calibrated_weights','gpu')},indent=2))
