"""Reproducible metadata-only subsets of an existing frozen split."""
import collections,csv,hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'outputs/atlas-full-portions-v1';out.mkdir(exist_ok=True)
train_path=root/'outputs/atlas-train.tsv';val_path=root/'outputs/atlas-validation.tsv'
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
train=read(train_path);val=read(val_path)
assert len(train)==153677 and len(val)==2000
train_ids={r['id'] for r in train};val_ids={r['id'] for r in val}
assert len(train_ids)==len(train) and len(val_ids)==len(val) and train_ids.isdisjoint(val_ids)
assert {r['sequence_component'] for r in train}.isdisjoint({r['sequence_component'] for r in val})
seed='atlas-full-portions-v1|20261012'
def rank(r):return hashlib.sha256((seed+'|'+r['id']).encode()).hexdigest()
strata=collections.defaultdict(list)
for row in train:strata[row['stratum']].append(row)
ranked=[]
for group in strata.values():
 for i,row in enumerate(sorted(group,key=rank)):
  ranked.append(((i+.5)/len(group),rank(row),row))
ordered=[row for _,__,row in sorted(ranked)]
report={'scope':'Original full ATLAS frozen split; membership metadata only','seed':seed,'method':'Within-stratum SHA256 order, interleaved by normalized within-stratum rank; exact floor percentage prefixes','sampling_unit':'complex; sequence components may span training portions, never training and validation','parent_train':{'count':len(train),'sha256':sha(train_path)},'parent_validation':{'count':len(val),'sha256':sha(val_path)},'portions':{},'nested':True,'shared_validation':True,'train_validation_id_overlap':0,'train_validation_component_overlap':0,'graph_preparation_run':False,'training_run':False}
previous=set()
for percent in [10,50,75]:
 rows=ordered[:len(train)*percent//100];path=out/f'train-{percent:03d}.tsv'
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['id','sequence_component','stratum'],delimiter='\t');w.writeheader();w.writerows(rows)
 loaded=read(path);ids={r['id'] for r in loaded}
 assert loaded==rows and len(ids)==len(rows) and previous<=ids<=train_ids and ids.isdisjoint(val_ids)
 previous=ids
 report['portions'][str(percent)]={'complexes':len(rows),'sha256':sha(path),'strata':dict(sorted(collections.Counter(r['stratum'] for r in rows).items()))}
shutil.copyfile(val_path,out/'validation.tsv');shutil.copyfile(train_path,out/'train-100.tsv')
assert sha(out/'validation.tsv')==sha(val_path) and sha(out/'train-100.tsv')==sha(train_path)
report['portions']['100']={'complexes':len(train),'sha256':sha(out/'train-100.tsv')}
report['script_sha256']=sha(Path(__file__))
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'directory':str(out),'training_portions':{p:r['complexes'] for p,r in report['portions'].items()},'shared_validation':len(val),'nested':True,'id_overlap':0,'component_overlap':0},indent=2))
