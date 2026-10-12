import csv,json,hashlib,collections,pathlib
p=pathlib.Path(__file__).resolve().parent
src=p/'information_for_complexes.csv'
h=hashlib.file_digest(src.open('rb'),'sha256').hexdigest()
assert h=='08740c654b0be90ed0284ece0f50e6c506f6ba2725de8c882d394cd9c43b36d3',h
counts=collections.Counter(); hc=collections.Counter();rows=[];seqs={};bad=[];seen=set();missing=collections.Counter()
with src.open() as f:
 for r in csv.DictReader(f):
  counts[(r['type'],r['class'])]+=1
  if not r['LIS'] or not r['LIA']:
   missing[(r['type'],r['class'])]+=1;continue
  if float(r['LIS'])<.203 or float(r['LIA'])<3432:continue
  chains=r.pop('sequence').split(':')
  if len(chains)!=2 or any(not s for s in chains):bad.append(r['id']);continue
  assert r['id'] not in seen,r['id'];seen.add(r['id'])
  ids=[]
  for s in chains:
   s=''.join(s.split()).upper();key='s'+hashlib.sha256(s.encode()).hexdigest()[:24];seqs[key]=s;ids.append(key)
  r['chains']=ids;r['lengths']=[len(s) for s in chains];rows.append(r);hc[(r['type'],r['class'])]+=1
(p/'atlas-high-confidence.json').write_text(json.dumps(rows))
(p/'atlas.fasta').write_text(''.join('>'+k+'\n'+v+'\n' for k,v in sorted(seqs.items())))
summary=dict(source_sha256=h,source_bytes=src.stat().st_size,rows=sum(counts.values()),high_confidence=len(rows),unique_chains=len(seqs),bad_chain_records=bad,source_counts={str(k):v for k,v in counts.items()},high_confidence_counts={str(k):v for k,v in hc.items()},criteria='LIS >= 0.203 and LIA >= 3432',missing_confidence={str(k):v for k,v in missing.items()},metadata_revision='15b1cce925520e267e3c71c96a9eff582e480e66')
(p/'atlas-source-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
