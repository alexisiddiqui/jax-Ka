"""Download official archives, verify SHA256, retain only the frozen high-confidence IDs.
No source archive is executed or extracted to filesystem paths. Outputs are tar shards of PDB.gz data.
"""
import argparse,concurrent.futures,hashlib,io,json,os,re,struct,sys,tarfile,time,urllib.request,urllib.parse,zipfile
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--limit',type=int);P.add_argument('--workers',type=int,default=4);a=P.parse_args()
BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]/'pretraining/atlas-high-confidence-v1'
for d in ['structures','receipts','download_work','metadata']: (ROOT/d).mkdir(parents=True,exist_ok=True)
HC={r['id'] for r in json.loads((BASE/'atlas-high-confidence.json').read_text())}
(ROOT/'metadata/atlas-high-confidence.json').write_bytes((BASE/'atlas-high-confidence.json').read_bytes())
(ROOT/'metadata/atlas-source-summary.json').write_bytes((BASE/'atlas-source-summary.json').read_bytes())
(ROOT/'metadata/chains.fasta').write_bytes((BASE/'atlas.fasta').read_bytes())
plan=[]
for m in sorted((BASE/'manifests').glob('*.files.json')):
 repo=m.name.removesuffix('.files.json')
 for f in json.loads(m.read_text())['Data']['Files']:
  if f['Name'].endswith('.zip'):plan.append((repo,f))
if a.limit:plan=plan[:a.limit]
def url(repo,f):return 'https://www.modelscope.cn/api/v1/datasets/anneCoder/'+repo+'/repo?'+urllib.parse.urlencode(dict(Source='SDK',Revision=f['Revision'],FilePath=f['Path']))
def request(u,headers=None):return urllib.request.urlopen(urllib.request.Request(u,headers=headers or {}),timeout=180)
def central(u,size):
 start=max(0,size-1048576)
 with request(u,{'Range':f'bytes={start}-{size-1}'}) as r:
  if r.status!=206:raise ValueError('Range not honoured')
  b=r.read()
 off=b.rfind(b'PK\x05\x06');e=struct.unpack_from('<4s4H2LH',b,off)
 length,pos=e[5:7]
 if pos<start:
  with request(u,{'Range':f'bytes={pos}-{pos+length-1}'}) as r:b=r.read()
 else:b=b[pos-start:pos-start+length]
 names=[];i=0
 while i<len(b):
  v=struct.unpack_from('<4s6H3L5H2L',b,i);assert v[0]==b'PK\x01\x02'
  n,x,c=v[10:13];names.append(b[i+46:i+46+n].decode());i+=46+n+x+c
 return names
def run(item):
 repo,f=item; stem=repo+'__'+f['Name'][:-4];receipt=ROOT/'receipts'/(stem+'.json');target=ROOT/'structures'/(stem+'.tar')
 if receipt.exists():
  j=json.loads(receipt.read_text())
  if j['source_sha256']==f['Sha256'] and (j.get('parser_version')==2 or j['ids']) and (not j['ids'] or target.exists()):return j
 u=url(repo,f);t=time.time();last=None
 for attempt in range(4):
  try:
   names=central(u,f['Size']);selected=[]
   for name in names:
    match=re.match(r'((?:EID|ID)_\d+)(?:_|\.|$)',Path(name).name)
    if match and match[1] in HC and name.endswith(('.pdb.gz','.pdb')):selected.append((name,match[1]))
   j=dict(parser_version=2,repository=repo,source_file=f['Path'],source_revision=f['Revision'],source_sha256=f['Sha256'],source_bytes=f['Size'],ids=[],structures=[],archive_entries=len(names))
   if selected:
    tmp=ROOT/'download_work'/(stem+'.zip.part');h=hashlib.sha256()
    with request(u) as r,tmp.open('wb') as w:
     while b:=r.read(8*1024*1024):h.update(b);w.write(b)
    assert tmp.stat().st_size==f['Size'], 'size mismatch'
    assert h.hexdigest()==f['Sha256'],'source checksum mismatch'
    pending=target.with_suffix('.tar.part')
    with zipfile.ZipFile(tmp) as z,tarfile.open(pending,'w') as tar:
     for name,cid in selected:
      data=z.read(name);arc=cid+'/'+Path(name).name
      info=tarfile.TarInfo(arc);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
      j['ids'].append(cid);j['structures'].append(dict(id=cid,member=arc,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    os.replace(pending,target);j['retained_bytes']=target.stat().st_size
    j['shard_sha256']=hashlib.file_digest(target.open('rb'),'sha256').hexdigest();tmp.unlink()
   j['seconds']=round(time.time()-t,2);pending=receipt.with_suffix('.json.part');pending.write_text(json.dumps(j));os.replace(pending,receipt)
   print(json.dumps(dict(archive=stem,kept=len(j['ids']),seconds=j['seconds'])),flush=True);return j
  except Exception as e:
   last=str(e);print(json.dumps(dict(archive=stem,attempt=attempt+1,error=last)),flush=True);time.sleep(2*(attempt+1))
 return dict(repository=repo,source_file=f['Path'],error=last,ids=[])
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:results=list(ex.map(run,plan))
seen={i for r in results for i in r['ids']};summary=dict(expected=len(HC),downloaded_unique=len(seen),missing_ids=sorted(HC-seen),archives=len(plan),failed=[r for r in results if 'error' in r],complete=seen==HC and not any('error' in r for r in results),retained_bytes=sum(r.get('retained_bytes',0) for r in results),source_bytes=sum(r.get('source_bytes',0) for r in results if r['ids']))
(ROOT/('download-pilot.json' if a.limit else 'download-report.json')).write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='missing_ids'}),flush=True)
if not a.limit and not summary['complete']:sys.exit(1)
