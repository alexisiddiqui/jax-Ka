"""Archive primary literature and deposited structure metadata on compute only."""
import os,json,urllib.request,xml.etree.ElementTree as ET,concurrent.futures,datetime
from pathlib import Path
from pkabench.runtime import require_compute,atomic_json,digest
require_compute(); out=Path(os.environ['PKABENCH_RUNTIME'])/'experimental/pilot-v1/sources'; out.mkdir(parents=True,exist_ok=True)
sources=[]
for pmc in ('PMC2673305','PMC11164218'):
 for label,url in [('epmc',f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML'),('ncbi',f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/?report=xml'),('bioc',f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmc}/unicode')]: sources.append((pmc+'-'+label,url))
for pmid in ('8494892','7578107'):
 sources.append((pmid,f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json&resultType=core'))
for pdb in ('1BRS','1FCC','1FRT','4N0U'):
 sources.append((pdb+'-entry',f'https://data.rcsb.org/rest/v1/core/entry/{pdb}'))
 sources.append((pdb+'-cif',f'https://files.rcsb.org/download/{pdb}.cif'))
def fetch(item):
 name,url=item; path=out/(name+'.raw')
 try:
  if not path.exists():
   req=urllib.request.Request(url,headers={'User-Agent':'pkabench research curation/1.0'})
   with urllib.request.urlopen(req,timeout=45) as response: data=response.read()
   path.write_bytes(data)
  data=path.read_bytes(); status='archived'
  try:
   root=ET.fromstring(data); paragraphs=[' '.join(el.itertext()) for el in root.iter() if el.tag in ('p','table-wrap','fig','passage')]
   if paragraphs: (out/(name+'.txt')).write_text('\n\n'.join(paragraphs))
   status='xml' if root.tag in ('article','pmc-articleset','collection') else 'other_xml'
  except ET.ParseError: pass
  return dict(id=name,url=url,path=str(path),sha256=digest(path),bytes=len(data),status=status)
 except Exception as e: return dict(id=name,url=url,status='unavailable',error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: receipts=list(pool.map(fetch,sources))
atomic_json(out/'receipts.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':receipts})
print(json.dumps(receipts,indent=2),flush=True)
