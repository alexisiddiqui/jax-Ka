"""Measure level-1 packing and verification separately from parallel graph building."""
import argparse
import json
from pathlib import Path
import time

from pkabench.runtime import atomic_json,require_compute,digest
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter

p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
require_compute(threads=1,allow_comp1400=True)
root=args.root;rows=[];start=time.perf_counter()
with AugmentationStore(root/'features') as source,ReleaseWriter(root/'packing-probe',
    dict(kind='packing-cost-probe-v1',source_manifest_sha256=digest(root/'features/manifest.json')),zstd_level=1) as writer:
    for row in source.rows:
        t=time.perf_counter();record=source.raw(row['id']);read_s=time.perf_counter()-t
        t=time.perf_counter();written=writer.add(row['id'],record['metadata'],record['arrays'],summary=row['summary'])
        rows.append(dict(parent=row['id'],weight=row['summary']['weight'],read_s=read_s,pack_write_s=time.perf_counter()-t,
                         compressed_bytes=written['bytes']))
    t=time.perf_counter();manifest=writer.finish();finish_s=time.perf_counter()-t
projected_bytes=sum(r['weight']*r['compressed_bytes'] for r in rows)
result=dict(passed=True,records=len(rows),wall_s=time.perf_counter()-start,finish_verify_s=finish_s,
    projected_serial_pack_write_hours=sum(r['weight']*r['pack_write_s'] for r in rows)/3600,
    projected_finish_verify_hours=finish_s/manifest['compressed_bytes']*projected_bytes/3600,
    compressed_bytes=manifest['compressed_bytes'],measurements=rows,
    cache='uncontrolled, recently built pilot on shared Coulson storage; not an Isambard filesystem bandwidth measurement')
atomic_json(root/'packing-probe.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='measurements'}),flush=True)
