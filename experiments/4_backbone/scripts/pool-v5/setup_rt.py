"""abdata-v5: mini runtime for the original record/side-table builders (file ops only, login node OK).

  python setup_rt.py RT IDS_TSV
RT/pretraining/pinder-pkai-v1 -> symlink to the real (read-only) PINDER prep; RT/training/pool-v4/{pinder.tsv,
pinder-val.tsv, pool-v4.json} list ONLY the structures to build (IDS_TSV columns id, split [, any pool-v4 columns]), so
production_graphs ids / build / pack and every side-table builder run unchanged on just these records. min_fraction is
a placeholder (1.0) unless IDS_TSV carries one: the training pool's min_fraction comes from pool-v5 at manifest merge.
"""
import csv, json, os, sys
from pathlib import Path
REAL = Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench')
rt = Path(sys.argv[1]); rows = list(csv.DictReader(open(sys.argv[2]), delimiter='\t'))
if rt.exists(): raise FileExistsError(rt)
(rt / 'pretraining').mkdir(parents=True); os.symlink(REAL / 'pretraining/pinder-pkai-v1', rt / 'pretraining/pinder-pkai-v1')
pool = rt / 'training/pool-v4'; pool.mkdir(parents=True)
cols = [c for c in rows[0] if c != 'split'] + (['min_fraction'] if 'min_fraction' not in rows[0] else [])
for name, split in (('pinder.tsv', 'train'), ('pinder-val.tsv', 'val')):
    with open(pool / name, 'w', newline='') as f:
        w = csv.DictWriter(f, cols, delimiter='\t', extrasaction='ignore'); w.writeheader()
        for r in rows:
            if r['split'] == split: w.writerow({'min_fraction': '1.0', **r})
(pool / 'pool-v4.json').write_text(json.dumps({'note': 'abdata-v5 mini runtime: structure list for the builders only, NOT a training pool', 'source': sys.argv[2]}) + '\n')
print(json.dumps({'rt': str(rt), 'train': sum(r['split'] == 'train' for r in rows), 'val': sum(r['split'] == 'val' for r in rows)}))
