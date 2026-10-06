"""Collect the full PKAD-R structure audit and measure family/split overlap."""
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, config_hash, digest

require_compute()
import pyarrow.parquet as pq

runtime = Path(os.environ['PKABENCH_RUNTIME'])
root = runtime / 'experimental/pkadr-full-v1'
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['code_sha256'] == digest(Path(os.environ['PKABENCH_SOURCE'])/'experiments/experimental_pilot/full_inventory.py')
records = {r['record_id']: r for r in json.loads((root/'records.json').read_text())}
tasks = json.loads((root/'tasks.json').read_text())
results = []
for task in tasks:
    path = root/'rows'/f"{task['task_id']}.json"
    assert path.exists(), task['task_id']
    result = json.loads(path.read_text())
    assert result['task_id'] == task['task_id'] and sorted(result['record_ids']) == sorted(task['record_ids'])
    results.append(result)

joined = []
for result in results:
    structural = {r['record_id']:r for r in result.get('records',[])}
    for record_id in result['record_ids']:
        record = records[record_id]
        row = dict(record)
        row.update(pdb=result['pdb'], author_chain=result['chain'], task_status=result['status'],
                   task_reason=result.get('reason'), task_detail=result.get('detail'),
                   sequence=result.get('sequence'), prepared_sha256=result.get('prepared_sha256'),
                   omitted_protein_chains=result.get('omitted_protein_chains'),
                   component_count=result.get('component_count'), missing_residue_count=result.get('missing_residue_count'),
                   alternate_residue_count=result.get('alternate_residue_count'))
        row.update(structural.get(record_id, dict(mapped=False,structural_train_mask=False,
                                                  structural_eval_mask=False,reason='task_not_prepared')))
        row['training_eligible'] = False
        row['independent_evaluation_eligible'] = False
        joined.append(row)
assert len(joined)==1024 and len({r['record_id'] for r in joined})==1024

# Search prepared sequences against each other to form prospective family blocks.
prepared = [r for r in results if r['status']=='prepared']
query = root/'prepared-sequences.fasta'
query.write_text(''.join(f">{r['task_id']}\n{r['sequence']}\n" for r in prepared))
mmseqs = runtime/'audits/foldbench-full-v1/tools/mmseqs/bin/mmseqs'
self_hits = root/'prepared-self-hits.tsv'
cmd = [str(mmseqs),'easy-search',str(query),str(query),str(self_hits),str(Path(os.environ['TMPDIR'])/'pkadr-self'),
       '--threads','2','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--alignment-mode','3',
       '--max-seqs','10000','-s','7.5','--split-memory-limit','4G',
       '--format-output','query,target,fident,qcov,tcov']
with (root/'prepared-self-search.log').open('w') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
parent = {r['task_id']:r['task_id'] for r in prepared}
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    a,b=find(a),find(b)
    if a!=b: parent[max(a,b)]=min(a,b)
for line in self_hits.read_text().splitlines():
    a,b,identity,qcov,tcov=line.split('\t')
    if float(identity)>=.3 and min(float(qcov),float(tcov))>=.8: union(a,b)
groups={}
for task_id in parent:
    root_id=find(task_id); groups.setdefault(root_id,[]).append(task_id)
family_id={t:'family-'+config_hash(sorted(members))[:12] for members in groups.values() for t in members}

# Search prepared sequences against the actual frozen candidate universe.
target = runtime/'universe/combined-split-v1/sequence/sequences.fasta'
frozen_hits = root/'frozen-sequence-hits.tsv'
cmd2 = [str(mmseqs),'easy-search',str(query),str(target),str(frozen_hits),str(Path(os.environ['TMPDIR'])/'pkadr-frozen'),
        '--threads','2','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--alignment-mode','3',
        '--max-seqs','10000','-s','7.5','--split-memory-limit','4G',
        '--format-output','query,target,fident,qcov,tcov']
with (root/'frozen-sequence-search.log').open('w') as log:
    subprocess.run(cmd2,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
by_target=defaultdict(list)
for line in frozen_hits.read_text().splitlines():
    q,t,i,qc,tc=line.split('\t')
    if float(i)>=.3 and min(float(qc),float(tc))>=.8:
        by_target[t].append(dict(task_id=q,identity=float(i),qcov=float(qc),tcov=float(tc)))
freeze=runtime/'universe/structural-freeze-v1'
assignments={r['complex_id']:r for r in pq.read_table(freeze/'assignments.parquet').to_pylist()}
index=json.loads((runtime/'universe/combined-split-v1/index.json').read_text())['candidates']
matches=[]
for candidate in index:
    assignment=assignments.get(candidate['complex_id'])
    if assignment is None: continue
    for chain in candidate['chains']:
        target_id='s'+config_hash(chain['sequence'])[:20]
        for hit in by_target.get(target_id,[]):
            matches.append(dict(complex_id=candidate['complex_id'],pdb_id=candidate['pdb_id'],
                                chain=chain['chain'],split=assignment['split'],**hit))
task_overlap={r['task_id']:Counter() for r in prepared}
for m in matches: task_overlap[m['task_id']][m['split']]+=1

for row in joined:
    row['family_id']=family_id.get(row['task_id'])
    overlap=task_overlap.get(row['task_id'],{})
    row['frozen_train_matches']=overlap.get('train',0)
    row['frozen_val_matches']=overlap.get('val',0)
    row['frozen_test_matches']=overlap.get('test',0)
    row['local_independent_candidate']=bool(row['structural_eval_mask'] and row['label_kind']=='point' and
                                            not row['frozen_train_matches'] and not row['frozen_val_matches'])

reason_counts=Counter(r['task_reason'] or 'prepared' for r in joined)
task_reasons=Counter(r.get('reason') or 'prepared' for r in results)
summary={
    'records':len(joined),'tasks':len(results),'prepared_tasks':len(prepared),'held_tasks':len(results)-len(prepared),
    'prepared_records':sum(r['task_status']=='prepared' for r in joined),
    'mapped_records':sum(r['mapped'] for r in joined),
    'structural_train_records':sum(r['structural_train_mask'] for r in joined),
    'structural_eval_records':sum(r['structural_eval_mask'] for r in joined),
    'structural_eval_point_records':sum(r['structural_eval_mask'] and r['label_kind']=='point' for r in joined),
    'local_independent_point_candidates':sum(r['local_independent_candidate'] for r in joined),
    'families_prepared':len(set(family_id.values())),
    'families_with_eval_points':len({r['family_id'] for r in joined if r['structural_eval_mask'] and r['label_kind']=='point'}),
    'families_local_independent':len({r['family_id'] for r in joined if r['local_independent_candidate']}),
    'record_outcomes':dict(reason_counts),'task_outcomes':dict(task_reasons),
    'label_kinds':dict(Counter(r['label_kind'] for r in joined)),
    'residue_types_eval_points':dict(Counter(r['raw']['ResName'] for r in joined if r['structural_eval_mask'] and r['label_kind']=='point')),
    'frozen_overlap_tasks':{s:len({m['task_id'] for m in matches if m['split']==s}) for s in ('train','val','test')},
    'admission_status':'structural candidates only; primary-source, mutation/construct, condition, upstream-model leakage and family split gates remain closed'}
atomic_json(root/'joined-records.json',joined)
atomic_json(root/'sequence-families.json',[dict(family_id='family-'+config_hash(sorted(v))[:12],task_ids=sorted(v)) for v in groups.values()])
atomic_json(root/'frozen-overlap.json',dict(matches=matches,counts=summary['frozen_overlap_tasks'],
    assignment_sha256=digest(freeze/'assignments.parquet'),query_sha256=digest(query),target_sha256=digest(target),hits_sha256=digest(frozen_hits)))
atomic_json(root/'summary.json',summary)

# Held tasks with database-suggested alternative PDBs are a recovery queue, not
# silently substituted structures.
recovery=[]
for row in joined:
    if row['task_status']=='prepared': continue
    alternatives=[x.strip().upper() for x in str(row['raw'].get('Alternative_PDBs','')).split(',') if x.strip()]
    if alternatives:
        recovery.append(dict(record_id=row['record_id'],primary_pdb=row['pdb'],chain=row['author_chain'],
                             hold_reason=row['task_reason'],alternatives=alternatives))
atomic_json(root/'alternative-recovery-queue.json',recovery)

lines=['# Full PKAD-R structural audit — v1','',
       'This is a structural and local-overlap preflight. It does not yet admit labels to training or certify independent evaluation.','',
       '| Quantity | Count |','|---|---:|',
       f"| PKAD-R records | {summary['records']} |",f"| Unique PDB/author-chain tasks | {summary['tasks']} |",
       f"| Prepared tasks | {summary['prepared_tasks']} |",f"| Held tasks | {summary['held_tasks']} |",
       f"| Records on prepared tasks | {summary['prepared_records']} |",f"| Mapped target records | {summary['mapped_records']} |",
       f"| Structurally retained evaluation records | {summary['structural_eval_records']} |",
       f"| Structurally retained numeric point labels | {summary['structural_eval_point_records']} |",
       f"| Local train/validation-disjoint point candidates | {summary['local_independent_point_candidates']} |",
       f"| Prepared sequence families | {summary['families_prepared']} |",
       f"| Families with retained point labels | {summary['families_with_eval_points']} |",
       f"| Locally disjoint families | {summary['families_local_independent']} |",'',
       '## Task outcomes','', '| Outcome | Tasks | Records |','|---|---:|---:|']
for reason,n in task_reasons.most_common():
    lines.append(f"| {reason} | {n} | {reason_counts[reason]} |")
lines += ['', 'Buried/coordinated metals and declared connected components remain whole-task holds under the agreed policy. Database-provided alternative structures are recorded in alternative-recovery-queue.json for explicit recovery and sequence/construct verification.','',
          '## Retained point-label composition','', '| Residue/group | Records |','|---|---:|']
for name,n in Counter(summary['residue_types_eval_points']).most_common(): lines.append(f'| {name} | {n} |')
lines += ['', '## Frozen synthetic overlap','', '| Split | Prepared tasks with a sequence match |','|---|---:|']
for split,n in summary['frozen_overlap_tasks'].items(): lines.append(f'| {split} | {n} |')
lines += ['', 'Matches use 30% identity and 80% bidirectional coverage against every frozen candidate chain. Records matching train or validation are not local independent-evaluation candidates. Test matches are retained as evaluation-family evidence. This local check does not establish independence from upstream pKAI, PROPKA, PypKa parameterization or literature databases.','',
          '## Remaining admission gates','',
          'Resolve alternative structures for held records; verify mutant and experimental constructs; normalize measurement conditions without merging distinct measurements; deduplicate repeated measurements and alternative structures; reserve whole sequence families; then run frozen baselines on a family-balanced, shift-stratified evaluation set. Primary-source verification can be prioritized by selected family rather than blocking this structural inventory.','',
          'Machine-readable outputs include joined-records.json, sequence-families.json, frozen-overlap.json, alternative-recovery-queue.json and summary.json. The original 1,024 source rows are preserved verbatim.']
(root/'report.md').write_text('\n'.join(lines)+'\n')
atomic_json(root/'release_manifest.json',dict(code_sha256=digest(Path(__file__)),source_manifest_sha256=digest(root/'manifest.json'),
    outputs_sha256={p.name:digest(p) for p in root.iterdir() if p.is_file() and p.name!='release_manifest.json'},job=os.environ['SLURM_JOB_ID']))
print(json.dumps(summary,indent=2),flush=True)
