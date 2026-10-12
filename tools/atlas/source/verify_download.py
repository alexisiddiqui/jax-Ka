"""Read-only verification of the staged release; run within the one-GPU allocation."""
import argparse,collections,gzip,hashlib,json,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args()
B=Path(__file__).resolve().parent;R=B.parents[1]/'pretraining/atlas-high-confidence-v1'
atlas={r['id']:r for r in json.loads((B/'atlas-high-confidence.json').read_text())}
receipts=[json.loads(f.read_text()) for f in (R/'receipts').glob('*.json')]
counts=collections.Counter(cid for r in receipts for cid in r['ids'])
expected_archives=sum(f['Name'].endswith('.zip') for m in (B/'manifests').glob('*.files.json') for f in json.loads(m.read_text())['Data']['Files'])
summary=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 expected_complexes=len(atlas),downloaded_unique=len(counts),verified_archive_receipts=sum(r.get('source_kind')!='companion_individual_structures' for r in receipts),expected_archives=expected_archives,
 companion_structures=sum(len(r['ids']) for r in receipts if r.get('source_kind')=='companion_individual_structures'),
 retained_bytes=sum(r.get('retained_bytes',0) for r in receipts),source_bytes=sum(r.get('source_bytes',0) for r in receipts if r['ids']),
 duplicate_ids={c:n for c,n in counts.items() if n>1},missing_count=len(set(atlas)-counts.keys()),extra_count=len(counts.keys()-set(atlas)))
summary['complete']=summary['verified_archive_receipts']==expected_archives and not summary['missing_count'] and not summary['extra_count'] and not summary['duplicate_ids']
summary['by_repository']={repo:dict(receipts=sum(r['repository']==repo for r in receipts),archive_entries=sum(r['archive_entries'] for r in receipts if r['repository']==repo),retained=sum(len(r['ids']) for r in receipts if r['repository']==repo)) for repo in sorted({r['repository'] for r in receipts})}
summary['source_qc_exclusions']=[dict(id=s['id'],reason='Companion mean CA pLDDT differs from metadata by >=0.15',observed_plddt=s['observed_plddt'],metadata_plddt=float(atlas[s['id']]['pLDDT'])) for r in receipts for s in r['structures'] if s.get('plddt_verified') is False]
if a.full:
 errors=[];samples=[];sampled=set()
 aa=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))
 for r in receipts:
  if not r['ids']:continue
  shard=R/'structures'/(r['repository']+'__'+Path(r['source_file']).stem+'.tar')
  with shard.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  if sha!=r['shard_sha256']:errors.append(str(shard)+': checksum mismatch')
  if r['repository'] in sampled:continue
  sampled.add(r['repository']);s=r['structures'][0]
  with tarfile.open(shard) as t:data=t.extractfile(s['member']).read()
  assert hashlib.sha256(data).hexdigest()==s['sha256']
  if s['member'].endswith('.gz'):data=gzip.decompress(data)
  chains=collections.defaultdict(dict)
  for line in data.decode().splitlines():
   if line.startswith('ENDMDL'):break
   if line.startswith('ATOM  ') and line[12:16].strip()=='CA':
    chains[line[21]].setdefault(line[22:27],aa.get(line[17:20],'X'))
  observed=[''.join(res.values()) for res in chains.values()]
  keys=sorted('s'+hashlib.sha256(s.encode()).hexdigest()[:24] for s in observed)
  match=keys==sorted(atlas[s['id']]['chains'])
  samples.append(dict(repository=r['repository'],id=s['id'],chains=len(chains),sequence_match=match,lengths=list(map(len,observed))))
  if not match:errors.append(s['id']+': sampled structure sequence differs from metadata')
 metadata=R/'metadata/information_for_complexes.csv.gz'
 if metadata.exists():
  with gzip.open(metadata,'rb') as f:summary['uncompressed_metadata_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
  if summary['uncompressed_metadata_sha256']!=json.loads((B/'atlas-source-summary.json').read_text())['source_sha256']:errors.append('Metadata SHA256 mismatch')
 else:errors.append('Full original metadata absent')
 summary.update(shard_errors=errors,structure_samples=samples,verified_all_shards=True)
 summary['complete']=summary['complete'] and not errors
 (R/'verification.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
 if not summary['complete']:raise SystemExit(1)
else:
 (B/'download-progress.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
