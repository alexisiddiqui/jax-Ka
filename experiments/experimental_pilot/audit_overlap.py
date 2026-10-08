"""New experimental sequences against the actual frozen synthetic assignments."""
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pkabench.runtime import require_compute, atomic_json, digest, config_hash

require_compute()
import pyarrow.parquet as pq
runtime=Path(os.environ['PKABENCH_RUNTIME'])
out=runtime/'experimental/pilot-v2'
structures=json.loads((out/'structure_preflight.json').read_text())
queries={r['pdb_id']:r['sequences'][0]['sequence'] for r in structures if r['status']=='prepared'}
fasta=out/'queries.fasta'
fasta.write_text(''.join(f'>{k}\n{v}\n' for k,v in queries.items()))
target=runtime/'universe/combined-split-v1/sequence/sequences.fasta'
binary=runtime/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
hits=out/'sequence_hits.tsv'
command=[str(binary),'easy-search',str(fasta),str(target),str(hits),str(Path(os.environ['TMPDIR'])/'experimental-search'),
    '--threads','1','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--alignment-mode','3','--max-seqs','10000','-s','7.5',
    '--split-memory-limit','2G','--format-output','query,target,fident,qcov,tcov']
with (out/'sequence_search.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
bytarget=defaultdict(list)
for line in hits.read_text().splitlines():
    query,target_id,identity,qcov,tcov=line.split('\t')
    if float(identity)>=.3 and min(float(qcov),float(tcov))>=.8:
        bytarget[target_id].append(dict(query=query, identity=float(identity), qcov=float(qcov), tcov=float(tcov)))
freeze=runtime/'universe/structural-freeze-v1'
assignments={r['complex_id']:r for r in pq.read_table(freeze/'assignments.parquet').to_pylist()}
matches=[]
for candidate in json.loads((runtime/'universe/combined-split-v1/index.json').read_text())['candidates']:
    assignment=assignments.get(candidate['complex_id'])
    if assignment is None: continue
    for chain in candidate['chains']:
        key='s'+config_hash(chain['sequence'])[:20]
        for hit in bytarget.get(key,[]):
            matches.append(dict(complex_id=candidate['complex_id'], pdb_id=candidate['pdb_id'], chain=chain['chain'],
                split=assignment['split'], **hit))
summary={q:{s:len({m['complex_id'] for m in matches if m['query']==q and m['split']==s}) for s in ('train','val','test')} for q in queries}
atomic_json(out/'synthetic_overlap.json',dict(counts=summary, matches=matches, command=command,
    query_sha256=digest(fasta), target_sha256=digest(target), assignment_sha256=digest(freeze/'assignments.parquet'),
    hits_sha256=digest(hits), split_changed=False,
    scope='30% identity and 80% bidirectional coverage; isolated-chain query; does not certify remote homologue or domain-containment independence'))

# Archive the model authors' description of their synthetic/experimental split.
urls=[('pkai-paper.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9369009/fullTextXML'),
      ('pkai-paper.html','https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9369009/?report=xml')]
receipts=[]
for name,url in urls:
    path=out/'sources'/name
    try:
        if not path.exists(): path.write_bytes(urllib.request.urlopen(url,timeout=45).read())
        receipts.append(dict(url=url, path=str(path), sha256=digest(path)))
        if name.endswith('.xml'):
            tree=ET.parse(path)
            paragraphs=[' '.join(' '.join(p.itertext()).split()) for p in tree.iter('p')]
            (out/'sources/pkai-paper-paragraphs.txt').write_text('\n\n'.join(paragraphs))
            atomic_json(out/'sources/pkai-split-paragraphs.json',[p for p in paragraphs if any(x in p.lower() for x in ('pkad','similarity','training set','experimental test'))])
    except Exception as exc: receipts.append(dict(url=url,error=repr(exc)))
atomic_json(out/'pkai_source_receipts.json',receipts)
print(json.dumps(summary,indent=2),flush=True)
