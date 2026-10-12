import os,json,statistics,hashlib
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
O=Path(os.environ['PKABENCH_RUNTIME'])/'pretraining/atlas-prepared-v1/graph-v1'
results=json.loads((O/'pilot-build.json').read_text());assert not any(r['status']=='error' for r in results)
sizes=[(O/'graphs'/r['id']/'record.zst').stat().st_size for r in results if r['status']=='accepted']
assert sizes
estimated=round(statistics.mean(sizes)*155677);free=os.statvfs(O).f_bavail*os.statvfs(O).f_frsize
# Cache plus final packed binary, with conservative headroom.
assert free>2.5*estimated,dict(free=free,estimated=estimated)
receipt=dict(pilot=len(results),accepted=sum(r['status']=='accepted' for r in results),excluded=sum(r['status']=='excluded' for r in results),estimated_packed_bytes=estimated,available_bytes=free)
(O/'preflight.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
