import os,json
from pathlib import Path
from pkabench.runtime import require_compute
from pkatrain.pkai_scratch import native
require_compute(threads=2);native()
from protein import Protein
root=Path(os.environ['PKABENCH_RUNTIME']);plan=json.loads((root/'pretraining/augmentation-v1/contexts/plan.json').read_text())
for cid,layout in plan['structures'].items():
    source=root/'pretraining/pkpdb-5k-comparison-v1/pkai-data/train'/cid
    req=json.loads((source/'request.json').read_text());residues=list(Protein(source/'input.pdb').iter_residues())
    n=layout['stop']-layout['start']
    if len(residues)!=n or n!=len(req['mapping']):
        print(json.dumps(dict(cid=cid,n=n,native=len(residues),mapped=len(req['mapping']),
            residues=[(r.resnumb,r.resname,req['mapping'][str(r.resnumb)]) for r in residues])),flush=True)
        break
