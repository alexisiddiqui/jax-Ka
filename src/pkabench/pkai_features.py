"""Extract the installed pKAI representation without rounding training outputs."""
import json
import sys
from pathlib import Path
from importlib.util import find_spec


def main():
    from .runtime import require_compute,atomic_json,digest
    require_compute()
    import torch
    sys.path.insert(0,str(Path(find_spec('pkai').origin).parent))
    from protein import Protein,PK_MODS
    from pKAI import load_model
    torch.set_num_threads(1)
    req=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]); out.mkdir(exist_ok=False)
    mapping={(r['chain'],r['resnum']):r['original'] for r in json.loads(Path(req['mapping']).read_text())}
    protein=Protein(req['pdb']); protein.apply_cutoff(); residues=list(protein.iter_residues(titrable_only=True))
    assert residues,'No supported pKAI sites'
    x=torch.stack([r.input_layer for r in residues]); assert torch.isfinite(x).all()
    model=load_model('pKAI','cpu'); model.eval()
    with torch.no_grad(): prediction=torch.cat([model(batch).reshape(-1) for batch in x.split(64)]).tolist()
    keys=[]; values=[]
    for r,v in zip(residues,prediction):
        chain,num,icode=mapping[r.chain,r.resnumb]
        keys.append([chain,num,icode,r.resname]); values.append(float(v)+PK_MODS[r.resname])
    atomic_json(out/'features.json',{'x':x.tolist(),'absolute_pka':values})
    atomic_json(out/'keys.json',keys)
    atomic_json(out/'receipt.json',{'input_sha256':digest(Path(req['pdb'])),'json_sha256':digest(out/'features.json'),
        'sites':len(keys),'width':int(x.shape[1]),'source':'Installed pKAI Protein.apply_cutoff, native 15 A cutoff; frozen model output before 2-decimal reporting round.'})

if __name__=='__main__': main()
