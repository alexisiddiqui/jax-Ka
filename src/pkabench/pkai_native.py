"""Released fixed-coordinate CPU pKAI protocol shared by native source adapters.

Prediction loop copied from the pinned whole-Fv labels-v3 implementation
(fv_labels.py SHA256 1a0c67f3276faea432eed8de8c6e00dfa74d697d9d0b482387d4965da93423b2).
Source-specific scope, termini and masks are established before calling it.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
from .runtime import atomic_json, digest, require_compute
MODELS = {'pkai': 'pKAI', 'pkai_plus': 'pKAI+'}
def read(path):
    return json.loads(Path(path).read_text())

def predict(request, destination):
    """Run in the SAME isolated pKAI environment as the v1 label build."""
    require_compute()
    import contextlib
    from importlib.util import find_spec
    import importlib.metadata
    sys.path.insert(0,str(Path(find_spec('pkai').origin).parent))
    from pKAI import pKAI
    from protein import Protein
    data=read(request)
    for path,expected in data['teacher_input_pins'].items():
        if digest(path)!=expected:raise ValueError('Teacher input changed: '+str(path))
    package=Path(find_spec('pkai').origin).parent
    expected_names={'pKAI.py','protein.py','residue.py','atom.py','models/pKAI_model.pt','models/pKAI+_model.pt'}
    actual={str(package/name) for name in expected_names}
    if actual!=set(data['teacher_input_pins']):raise ValueError('Teacher package differs from pinned protocol')
    labels={m:{} for m in MODELS};errors=[];timings={};states={}
    for state in ['AB','A','B']:
        mapping=data['mappings'][state]
        forward={(c,int(n)):(oc,int(on),oi) for c,n,oc,on,oi in mapping['residues']}
        assigned={(c,int(n)):(kind,oc,int(on)) for c,n,kind,oc,on in mapping['termini']}
        pdb=str(Path(request).parent/(state+'.pdb'))
        supported=bool(list(Protein(pdb).iter_residues(titrable_only=True)))
        states[state]='predicted' if supported else 'empty_no_supported_sites'
        for method,model in MODELS.items():
            started=time.monotonic();rows=[]
            if not supported:
                labels[method][state]=[];timings[method+'_'+state]=time.monotonic()-started
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result=pKAI(pdb,model_name=model,device='cpu',threads=1)
                for c,n,r,v in result:
                    if (c,int(n)) in assigned:
                        kind,oc,on=assigned[c,int(n)];key=forward[oc,on];group='NTERM' if kind=='NTR' else 'CTERM'
                    else:key=forward[c,int(n)];group=str(r)
                    value=float(v)
                    if not __import__('math').isfinite(value):raise ValueError('nonfinite pKAI output')
                    rows.append([*key,group,round(value,4)])
            except (RuntimeError,ZeroDivisionError,ValueError) as exc:
                errors.append([method,state,str(exc)[:400]])
                states[state]='prediction_failed_see_errors'
            labels[method][state]=rows;timings[method+'_'+state]=time.monotonic()-started
    atomic_json(destination,dict(**labels,errors=errors,termini=True,version=importlib.metadata.version('pKAI'),
        device='cpu',timings_s=timings,state_status=states))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('request',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();predict(args.request,args.destination)
