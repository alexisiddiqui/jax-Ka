"""Create an indexed, provenance-preserving raw source dataset. No training changes."""
import csv,hashlib,json,shutil,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[1]/'pretraining/atlas-high-confidence-v1'
verification=json.loads((R/'verification.json').read_text());assert verification['complete']
excluded={r['id'] for r in verification.get('source_qc_exclusions',[])}
atlas={r['id']:r for r in json.loads((B/'atlas-high-confidence.json').read_text())}
with (R/'structure-index.tsv').open('w') as out:
 fields=['id','shard','member','offset','compressed_bytes','sha256','chain_A','chain_B','source_qc_pass']
 w=csv.DictWriter(out,fields,delimiter='\t');w.writeheader();n=0
 for path in sorted((R/'receipts').glob('*.json')):
  r=json.loads(path.read_text())
  if not r['ids']:continue
  shard='structures/'+r['repository']+'__'+Path(r['source_file']).stem+'.tar'
  with tarfile.open(R/shard) as t:members={m.name:m for m in t.getmembers()}
  for s in r['structures']:
   m=members[s['member']];assert m.size==s['bytes'];cid=s['id']
   w.writerow(dict(zip(fields,[cid,shard,s['member'],m.offset_data,m.size,s['sha256'],*atlas[cid]['chains'],int(cid not in excluded)])));n+=1
assert n==len(atlas)
for f in ['atlas-screening.tsv','overlap-summary.tsv','overlap-summary.json','analysis-provenance.json','screening-summary.json','atlas-cohort-stats.json','reference-scope-check.json','missing-structure-investigation.json']:
 shutil.copyfile(B/f,R/'metadata'/f)
shutil.copytree(B/'manifests',R/'metadata/source-manifests',dirs_exist_ok=True)
manifest=dict(name='ATLAS high-confidence complexes',version='atlas-high-confidence-v1',
 source_collection='https://www.modelscope.cn/collections/protein_complex_atlas-2ae5e7d4f4a343',
 companion_source='https://www.biopredictnavigator.cn',companion_recovered_structures=verification.get('companion_structures',0),
 companion_provenance='Exact canonical sequences verified; mean CA pLDDT checked with explicit source-QC exclusions. LIS/LIA from official CSV; locally calculated hashes. Per-file receipts document the expired TLS certificate exception.',
 paper='https://www.nature.com/articles/s41467-026-70884-4',license='CC-BY-NC-4.0',
 confidence_filter=dict(LIS_min=.203,LIA_min=3432),complexes=n,
 unique_sequences=171265,structure_index='structure-index.tsv',
 index_sha256=hashlib.file_digest((R/'structure-index.tsv').open('rb'),'sha256').hexdigest(),
 structures_format='TAR shards containing source PDB.gz members; offset and compressed_bytes support pread',
 metadata='metadata/atlas-high-confidence.json',sequences='metadata/chains.fasta',
 overlap_flags='metadata/atlas-screening.tsv',verification='verification.json',audit_directory=str(B),
 status='raw_source_download_verified',prepared_for_training=False,source_qc_excluded_ids=sorted(excluded),
 next_steps=['Apply project-specific fragment and antibody/CDR leakage rules in addition to preliminary whole-chain flags.',
 'Select representatives or cluster-aware weights for redundant sequence pairs.',
 'Run existing structure/site preparation and pKAI AB/A/B labeling, preserving predicted-source confidence.',
 'Build graph records and a separate ATLAS training manifest; retain existing validation cohorts.'])
(R/'dataset.json').write_text(json.dumps(manifest,indent=2))
if (R/'download-report.json').exists() and not (R/'modelscope-download-report.json').exists():
 shutil.copyfile(R/'download-report.json',R/'modelscope-download-report.json')
(R/'download-report.json').write_text(json.dumps(dict(verification,modelscope_report='modelscope-download-report.json',dataset_manifest='dataset.json'),indent=2))
print(json.dumps(manifest,indent=2),flush=True)
