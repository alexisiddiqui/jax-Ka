"""Archive public primary-source XML for curation, without admitting labels."""
import os,json,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
from pkabench.runtime import require_compute,atomic_json,digest
require_compute()
out=Path(os.environ['PKABENCH_RUNTIME'])/'audits/set2-primary-v2'; out.mkdir(exist_ok=False)
receipts=[]
for name,url in [('protein-g-fc','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2673305/fullTextXML'),('barnase-barstar','https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:8494892%20AND%20SRC:MED&format=json&resultType=core')]:
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'pkabench academic curation'})
        data=urllib.request.urlopen(req,timeout=90).read(); path=out/(name+'.raw'); path.write_bytes(data)
        if name=='protein-g-fc':
            root=ET.fromstring(data); paragraphs=[' '.join(el.itertext()) for el in root.iter() if el.tag in ('p','table-wrap','fig')]
            (out/(name+'.txt')).write_text('\n\n'.join(paragraphs))
        receipts.append({'source':url,'path':str(path),'sha256':digest(path),'status':'downloaded_not_admitted'})
    except Exception as e: receipts.append({'source':url,'status':'unavailable','error':str(e)})
atomic_json(out/'receipts.json',receipts)
print(json.dumps(receipts,indent=2))
