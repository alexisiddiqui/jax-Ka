import os,json,urllib.request
from collections import Counter
from pathlib import Path
from pkabench.runtime import require_compute,atomic_json,digest
require_compute(); out=Path(os.environ['PKABENCH_RUNTIME'])/'experimental/pilot-v1'; src=out/'sources'
url='https://compbio.clemson.edu/pkad-r/file?kind=dt&name=PKAD-R-250211.json'
path=src/'PKAD-R-250211.json'
if not path.exists(): path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
rows=json.loads(path.read_text()); assert isinstance(rows,list)
atomic_json(src/'pkadr-data-receipt.json',{'url':url,'sha256':digest(path),'rows':len(rows)})
selected=[r for r in rows if any(x in str(r.get('Protein Name',r)).lower() for x in ('barnase','barstar','protein g','neonatal','immunoglobulin'))]
atomic_json(out/'pkadr-lead-rows.json',selected)
print(json.dumps({'rows':len(rows),'columns':list(rows[0]),'example':rows[0],'lead_rows':selected},indent=2),flush=True)
