"""Bounded indexing of the next coordinate acquisition batch."""
import json
from pathlib import Path
from .runtime import require_compute,atomic_json,digest


def index(out,shard,shards):
    require_compute()
    from .pool import index as build
    out=Path(out).resolve(); work=out/'index-shards'/str(shard); work.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((out/'download-manifest.json').read_text())
    atomic_json(work/'download-manifest.json',{**manifest,'assemblies':manifest['assemblies'][shard::shards]})
    for name in ('sources','receipts'): (work/name).symlink_to(out/name,target_is_directory=True)
    build(work)


def collect(out,shards):
    require_compute(); out=Path(out); parts=[json.loads((out/'index-shards'/str(s)/'index.json').read_text()) for s in range(shards)]
    rows=[r for p in parts for r in p['candidates']]; records=[r for p in parts for r in p['assemblies']]
    expected=json.loads((out/'download-manifest.json').read_text())['assemblies']
    assert len(records)==len(expected) and len({r['pdb_id'] for r in records})==len(expected)
    assert len({r['complex_id'] for r in rows})==len(rows)
    atomic_json(out/'index.json',{'candidates':rows,'assemblies':records,'download_manifest_sha256':digest(out/'download-manifest.json'),'selection':parts[0]['selection'],'production_allowed':False})
    print(json.dumps({'assemblies':len(records),'candidates':len(rows),'pipeline_errors':sum(r.get('status')=='pipeline_error' for r in records)}))
