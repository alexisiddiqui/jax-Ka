import os,json,urllib.request,re,html,xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from pkabench.runtime import require_compute,atomic_json,digest
require_compute(); out=Path(os.environ['PKABENCH_RUNTIME'])/'experimental/pilot-v1'; src=out/'sources'
class Parser(HTMLParser):
 def __init__(self): super().__init__(); self.parts=[]; self.links=[]; self.skip=0
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag in ('script','style'): self.skip+=1
  if tag in ('a','script','link'): self.links.append(dict(tag=tag,**attrs))
  if tag in ('p','tr','h1','h2','h3','table','section'): self.parts.append('\n')
 def handle_endtag(self,tag):
  if tag in ('script','style'): self.skip=max(0,self.skip-1)
  if tag in ('p','tr','h1','h2','h3','table','section'): self.parts.append('\n')
 def handle_data(self,data):
  if not self.skip: self.parts.append(data)
for name in ('PMC2673305-ncbi','PMC11164218-ncbi'):
 parser=Parser(); parser.feed((src/(name+'.raw')).read_text()); (src/(name+'.txt')).write_text(''.join(parser.parts)); atomic_json(src/(name+'-links.json'),parser.links)
url='https://compbio.clemson.edu/pkad-r/'; data=urllib.request.urlopen(url,timeout=45).read(); path=src/'pkadr.html'; path.write_bytes(data)
p=Parser(); p.feed(data.decode()); (src/'pkadr.txt').write_text(''.join(p.parts)); atomic_json(src/'pkadr-links.json',p.links)
root=ET.parse(src/'PMC11164218-epmc.raw').getroot(); tables=[]
for table in root.iter('table-wrap'):
 tables.append({'id':table.get('id'),'label':' '.join(table.findtext('label','').split()),'caption':' '.join(' '.join(table.find('caption').itertext()).split()) if table.find('caption') is not None else '',
 'rows':[[' '.join(' '.join(cell.itertext()).split()) for cell in row if cell.tag in ('th','td')] for row in table.iter('tr')]})
atomic_json(out/'fcrn_tables.json',tables)
# Deposit sequence metadata, components and missing-coordinate declarations.
from biotite.structure.io.pdbx import CIFFile
summary=[]
for pdb in ('1BRS','1FCC','1FRT','4N0U'):
 block=CIFFile.read(src/(pdb+'-cif.raw')).block
 def rows(category):
  if category not in block: return []
  c=block[category]; return [{k:str(c[k].as_array()[i]) for k in c} for i in range(c.row_count)]
 summary.append({'pdb_id':pdb,'entity':rows('entity'),'entity_poly':rows('entity_poly'),'struct_ref_seq_dif':rows('struct_ref_seq_dif'),
  'resolution':rows('refine'),'nonpoly':rows('pdbx_entity_nonpoly'),'unobserved':rows('pdbx_unobs_or_zero_occ_residues')})
atomic_json(out/'structure_metadata.json',summary)
print(json.dumps({'tables':[(x['id'],x['label'],len(x['rows'])) for x in tables],'pkadr_links':p.links,'structures':[x['pdb_id'] for x in summary]},indent=2),flush=True)
