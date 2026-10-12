"""Native pKAI features and randomly initialized pKAI training on the 5k cohort."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from pkabench.runtime import atomic_json, digest, require_compute


def native():
    root=Path(os.environ['PKABENCH_RUNTIME'])
    package=root/'envs/pkai/lib/python3.11/site-packages'
    sys.path.append(str(package));sys.path.insert(0,str(package/'pkai'))
    import torch
    torch.set_num_threads(int(os.environ.get('PKAI_THREADS','1')))
    return torch,package/'pkai'


# Neighbour-slot encodings: "atom16" is native pKAI (16 functional-atom classes per slot, 4,008 inputs); "aa20" puts a
# 20-amino-acid one-hot of the neighbour atom's residue in each slot instead (5,008 inputs); "atom16aa20" keeps both,
# the 16 atom classes then the 20 residue types (36 per slot, 9,008 inputs), in the native slot order (2026-10-10).
# "atom16aa20sc" appends backbone/side-chain channels (38 per slot); active flag channels also use 1/d^2.
# All keep the 250 nearest environment atoms by distance, the 1/d^2 value and the 8-class site one-hot.
AA20 = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
AA20_ALIASES = {"MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
                "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "CYX": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
SLOT_WIDTH = {"atom16": 16, "aa20": 20, "atom16aa20": 36, "atom16aa20sc": 38, "atom16aa20ori": 39}
# Environment geometry (2026-10-10): the cutoff (A) and the number of nearest-atom slots; native pKAI is 15 A / 250.
# Other values (PKAI_CUTOFF, PKAI_SLOTS) change the input width and get their own stores, runs and validation package.
CUTOFF = float(os.environ.get("PKAI_CUTOFF", "15"))
SLOTS = int(os.environ.get("PKAI_SLOTS", "250"))
GEOMETRY = "" if (CUTOFF, SLOTS) == (15.0, 250) else f"-r{CUTOFF:g}s{SLOTS}"


def feature_width(encoding="atom16"):
    return SLOTS * SLOT_WIDTH[encoding] + 8


def aa20_index(resname):
    name = AA20_ALIASES.get(str(resname).strip(), str(resname).strip())
    if name not in AA20: raise ValueError(f"non-canonical residue in pKAI environment: {resname}")
    return AA20.index(name)


# Compact slot form (2026-10-10). Every encoding is a function of, per site, the 250 slot values (1/d^2; 0 in empty
# slots), each slot's atom class (atom16, atom16aa20) and/or residue type (aa20, atom16aa20), and the site class:
# 1.25-1.5 kB per site against 16-36 kB dense. compact() inverts a dense matrix and checks that expand() rebuilds it
# exactly; expand_torch() rebuilds batches on the GPU.
def compact_fields(encoding="atom16"):
    return ("value",) + (("atom",) if encoding != "aa20" else ()) + (("aa",) if encoding != "atom16" else ()) + (("sc",) if encoding == "atom16aa20sc" else ()) + (("orientation",) if encoding == "atom16aa20ori" else ()) + ("site",)


def _aa_offset(encoding): return 16 if encoding.startswith("atom16aa20") else 0


def expand(slots, encoding="atom16"):
    slot = SLOT_WIDTH[encoding]; n = len(slots["site"]); x = np.zeros((n, feature_width(encoding)), np.float32)
    rows = np.arange(n)[:, None]; base = np.arange(SLOTS) * slot
    if "atom" in slots: x[rows, base + slots["atom"]] = slots["value"]
    if "aa" in slots: x[rows, base + _aa_offset(encoding) + slots["aa"]] = slots["value"]
    if "sc" in slots: x[rows, base + 36 + slots["sc"]] = slots["value"]
    if "orientation" in slots: x[:, :SLOTS * slot].reshape(n, SLOTS, slot)[..., 36:39] = slots["orientation"]
    x[np.arange(n), SLOTS * slot + slots["site"].astype(np.int64)] = 1.0
    return x


def compact(x, encoding="atom16"):
    slot = SLOT_WIDTH[encoding]; blocks = x[:, :SLOTS * slot].reshape(len(x), SLOTS, slot)
    parts = {} if encoding == "aa20" else {"atom": blocks[..., :16]}
    if encoding != "atom16": parts["aa"] = blocks[..., _aa_offset(encoding):_aa_offset(encoding) + 20]
    if encoding == "atom16aa20sc": parts["sc"] = blocks[..., 36:38]
    out = {"value": next(iter(parts.values())).max(-1)}
    for name, part in parts.items(): out[name] = part.argmax(-1).astype(np.uint8)
    if encoding == "atom16aa20ori": out["orientation"] = blocks[..., 36:39].copy()
    out["site"] = x[:, SLOTS * slot:].argmax(-1).astype(np.uint8)
    out = {name: out[name] for name in compact_fields(encoding)}
    if not np.array_equal(expand(out, encoding), x): raise ValueError("dense pKAI features are not in compact slot form")
    return out


def expand_torch(torch, slots, encoding="atom16"):
    value = slots["value"]; slot = SLOT_WIDTH[encoding]; device = value.device
    x = torch.zeros((value.shape[0], feature_width(encoding)), dtype=torch.float32, device=device)
    base = torch.arange(SLOTS, device=device) * slot
    if "atom" in slots: x.scatter_(1, base + slots["atom"].long(), value)
    if "aa" in slots: x.scatter_(1, base + _aa_offset(encoding) + slots["aa"].long(), value)
    if "sc" in slots: x.scatter_(1, base + 36 + slots["sc"].long(), value)
    if "orientation" in slots: x[:, :SLOTS * slot].reshape(value.shape[0], SLOTS, slot)[..., 36:39] = slots["orientation"]
    x.scatter_(1, SLOTS * slot + slots["site"].long()[:, None], 1.0)
    return x


def feature_matrix(protein, encoding="atom16"):
    """Vectorized distances/OHE; use native atom classification and exact ordering."""
    if encoding == "atom16aa20ori": raise ValueError("orientation encoding is backbone-only")
    from residue import AA_ATOMS, ATOM_OHE, RES_OHE
    atoms=list(protein.iter_atoms());coords=np.array([a.coords for a in atoms],dtype=np.float64)
    residues=list(protein.iter_residues(titrable_only=True))
    slot=SLOT_WIDTH[encoding];matrix=np.zeros((len(residues),feature_width(encoding)),np.float32)
    for i,r in enumerate(residues):
        centers=np.array([a.coords for a in r.iter_atoms() if a.aname in AA_ATOMS[r.resname]])
        if len(centers):
            distances=np.sqrt(((coords[:,None]-centers[None,:])**2).sum(-1).min(-1))
            keep=np.array([a.residue is not r for a in atoms]) & (distances<CUTOFF)
            ids=np.flatnonzero(keep)
            if np.any(distances[ids]==0): raise ValueError('Coincident pKAI environment/reference atoms')
            r.env_anames=[atoms[j].aname for j in ids]
            r.env_resnames=[atoms[j].residue.resname for j in ids]
            if encoding=="atom16":
                r.encode_atoms()
                order=sorted(zip(distances[ids],r.env_oheclasses))[:SLOTS]
                for j,(distance,cls) in enumerate(order):matrix[i,j*16+ATOM_OHE.index(cls)]=1/(float(distance)**2)
            elif encoding in ("atom16aa20", "atom16aa20sc"):
                r.encode_atoms()
                flags = [int(atoms[k].aname not in ("N", "CA", "C", "O", "OXT")) for k in ids]
                order=sorted(zip(distances[ids],r.env_oheclasses,[aa20_index(name) for name in r.env_resnames], flags))[:SLOTS]
                for j,(distance,cls,aa,sc) in enumerate(order):
                    matrix[i,j*slot+ATOM_OHE.index(cls)]=matrix[i,j*slot+16+aa]=1/(float(distance)**2)
                    if encoding == "atom16aa20sc": matrix[i,j*slot+36+sc]=1/(float(distance)**2)
            else:
                order=sorted(zip(distances[ids],[aa20_index(name) for name in r.env_resnames]))[:SLOTS]
                for j,(distance,cls) in enumerate(order):matrix[i,j*slot+cls]=1/(float(distance)**2)
        matrix[i,SLOTS*slot+RES_OHE.index(r.resname)]=1.
    assert np.isfinite(matrix).all()
    return residues,matrix


def features(dest):
    torch,package=native()
    from protein import Protein,PK_MODS
    req=json.loads((dest/'request.json').read_text())
    residues,x=feature_matrix(Protein(dest/'input.pdb'))
    # Each structure checks one real site against the untouched installed encoder.
    check=list(Protein(dest/'input.pdb').iter_residues(titrable_only=True))
    if check:
        check[0].calc_cutoff_atoms(15);check[0].encode_input()
        np.testing.assert_allclose(x[0],check[0].input_layer.numpy(),rtol=2e-6,atol=1e-8)
    lookup={}
    for i,r in enumerate(residues):
        key=(*req['mapping'][str(r.resnumb)],r.resname)
        assert key not in lookup, ('ambiguous pKAI mapping',key)
        lookup[key]=i
    selected=[];rows=[];unsupported=0
    for site in req['sites']:
        key=(site['chain'],site['resnum'],site['icode'],site['group'])
        if key not in lookup:unsupported+=1;continue
        selected.append(lookup[key]);rows.append(dict(site,component_id=req['component_id'],split=req['split'],
                                                    model_pka=PK_MODS[site['group']]))
    np.savez_compressed(dest/'features.npz',x=x[selected])
    atomic_json(dest/'rows.json',rows)
    atomic_json(dest/'receipt.json',dict(complex_id=req['complex_id'],split=req['split'],
        path=str(dest),sites=len(rows),clean_sites=sum(r['train_mask'] for r in rows),unsupported_sites=unsupported,
        native_parity_checked=bool(check),sha256=digest(dest/'features.npz'),rows_sha256=digest(dest/'rows.json'),
        pdb_sha256=digest(dest/'input.pdb'),native_hashes={n:digest(package/n) for n in ('protein.py','residue.py','atom.py')}))


def model_class(torch, hidden=(800, 400, 200), inputs=4008):
    hidden=tuple(int(value) for value in hidden)
    if len(hidden)!=3 or any(value<=0 for value in hidden):
        raise ValueError(f'Expected three positive hidden widths, got {hidden}')
    class PKAI(torch.nn.Module):
        def __init__(self):
            super().__init__()
            widths=(int(inputs),)+hidden+(1,)
            self.layers=torch.nn.ModuleList([torch.nn.Linear(a,b) for a,b in zip(widths[:-1],widths[1:])])
            self.dropouts=torch.nn.ModuleList([torch.nn.Dropout(p) for p in (.5,.125,.03125)])
        def forward(self,x):
            for layer,dropout in zip(self.layers[:-1],self.dropouts):x=dropout(torch.relu(layer(x)))
            return self.layers[-1](x).reshape(-1)
    return PKAI


def architecture_gate():
    torch,package=native();torch.manual_seed(17)
    reference=torch.jit.load(str(package/'models/pKAI_model.pt'),map_location='cpu').eval()
    model=model_class(torch)();assert sum(p.numel() for p in model.parameters())==3608001
    assert set(model.state_dict())==set(dict(reference.named_parameters()))
    # Copy only for implementation validation, never as training initialization.
    model.load_state_dict(dict(reference.named_parameters()));model.eval()
    x=torch.randn(8,4008)
    with torch.no_grad():torch.testing.assert_close(model(x),reference(x).reshape(-1),rtol=1e-6,atol=1e-6)
    model.train();assert not torch.equal(model(x),model(x)), 'Dropout must be active during fitting'
    torch.manual_seed(17);fresh=model_class(torch)()
    assert all(not torch.equal(p,dict(reference.named_parameters())[n]) for n,p in fresh.named_parameters())
    loss=fresh(x).square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fresh.parameters())
    return dict(passed=True,parameters=3608001,dropout=[.5,.125,.03125],native_model_sha256=digest(package/'models/pKAI_model.pt'))


def pack(out):
    info=json.loads((out/'pkai-features.json').read_text());assert info['passed']
    folder=out/'pkai-packed';folder.mkdir(exist_ok=True)
    rows=[];total=sum(r['sites'] for r in info['records'])
    x=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(total,4008))
    offset=0
    for r in info['records']:
        path=Path(r['path']);assert digest(path/'features.npz')==r['sha256'] and digest(path/'rows.json')==r['rows_sha256']
        with np.load(path/'features.npz') as f: x[offset:offset+r['sites']]=f['x']
        rows.extend(json.loads((path/'rows.json').read_text()));offset+=r['sites']
    x.flush();del x
    atomic_json(folder/'rows.json',rows)
    atomic_json(folder/'verification.json',dict(passed=True,sites=total,features_sha256=digest(folder/'features.npy'),
        rows_sha256=digest(folder/'rows.json'),source_sha256=digest(out/'pkai-features.json')))


def evaluate(rows,pred):
    from collections import defaultdict
    from pkabench.frozen_score import measures,aggregate
    grouped=defaultdict(list)
    for row,value in zip(rows,pred):grouped[row['complex_id']].append((row,value))
    scores=[]
    for cid,items in grouped.items():
        y=np.array([r['pka']-r['model_pka'] for r,v in items]);p=np.array([v-r['model_pka'] for r,v in items])
        scores.append(dict(complex_id=cid,component_id=items[0][0]['component_id'],n=len(items),**measures(y,p)))
    return aggregate(scores,replicates=2000)[0]


def train(out,arm,*,destination=None,context_path=None,seed=17,batch_size=256):
    torch,package=native();require_compute(threads=8,gpu_benchmark=True)
    assert torch.cuda.is_available();torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    gate=architecture_gate()
    torch.manual_seed(seed)  # Gate must not change the registered initialization stream.
    model=model_class(torch)().cuda()
    config=dict(seed=seed,arm=arm,optimizer='Adam',learning_rate=1e-6,weight_decay=1e-4,batch_size=batch_size,
        patience=5,min_delta=.001,max_epochs=200,precision='float32',selection='minimum clean validation site-MSE',
        objective='MSE on pKa minus native model-compound pKa',dropout=[.5,.125,.03125],
        initialization='PyTorch Linear defaults; no pretrained weights',
        precision_deviation='Published recipe used 16-bit; this pilot uses full float32 as requested',
        paper='https://assets-eu.researchsquare.com/files/rs-949180/v2_covered.pdf')
    dest=Path(destination) if destination is not None else out/f'pkai-{arm}';dest.mkdir(parents=True,exist_ok=True)
    context=None
    if context_path is not None:
        from .context_augmentation import PKAIContext
        context=PKAIContext(context_path)
    config['context_mask_probability']=.05 if context is not None else 0.
    packed=out/'pkai-packed';v=json.loads((packed/'verification.json').read_text());assert v['passed']
    assert digest(packed/'features.npy')==v['features_sha256'] and digest(packed/'rows.json')==v['rows_sha256']
    rows=json.loads((packed/'rows.json').read_text());x=np.load(packed/'features.npy',mmap_mode='r')
    trainids=np.array([i for i,r in enumerate(rows) if r['split']=='train' and (arm=='raw' or r['train_mask'])])
    valid=np.array([i for i,r in enumerate(rows) if r['split']=='val']);valrows=[rows[i] for i in valid]
    assert len(trainids)>0 and len(valid)>0
    if context is not None:
        checkids=trainids[np.linspace(0,len(trainids)-1,min(128,len(trainids)),dtype=int)]
        context.verify_reference(checkids,np.array(x[checkids]))
    y=torch.tensor([r['pka']-r['model_pka'] for r in rows],dtype=torch.float32,device='cuda')
    vx=torch.tensor(np.array(x[valid]),device='cuda');vy=y[torch.tensor(valid,device='cuda')]
    opt=torch.optim.Adam(model.parameters(),lr=1e-6,weight_decay=1e-4)
    provenance=dict(config=config,gate=gate,packed_sha256=digest(packed/'verification.json'),
        code_sha256=digest(Path(__file__)),train_sites=len(trainids),validation_sites=len(valid),
        augmentation_code_sha256=digest(Path(__file__).with_name('context_augmentation.py')),
        context_verification_sha256=digest(Path(context_path)/'verification.json') if context_path is not None else None,
        train_structures=len({rows[i]['complex_id'] for i in trainids}),test_data_included=False)
    atomic_json(dest/'manifest.json',provenance)
    history=[];best=float('inf');anchor=float('inf');stall=0;start=0;bestepoch=0
    if (dest/'latest.pt').exists():
        state=torch.load(dest/'latest.pt');assert state['provenance']==provenance
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=state['epoch'];best=state['best'];anchor=state['anchor'];stall=state['stall'];bestepoch=state['bestepoch']
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);history=state['history']
    began=time.monotonic()
    for epoch in range(start,config['max_epochs']):
        if stall>=config['patience']:break
        model.train();t0=time.monotonic();losses=[];counts=[]
        context_sha256=context.set_epoch(seed,epoch+1) if context is not None else None
        for ids in np.array_split(np.random.permutation(trainids),range(batch_size,len(trainids),batch_size)):
            features=np.array(x[ids])
            if context is not None:features=context.augment(ids,features)
            xb=torch.tensor(features,device='cuda');yb=y[torch.tensor(ids,device='cuda')]
            opt.zero_grad(set_to_none=True);loss=(model(xb)-yb).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pKAI loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite pKAI gradient')
            opt.step();losses.append(float(loss.detach()));counts.append(len(ids))
        model.eval()
        with torch.no_grad():p=torch.cat([model(batch) for batch in vx.split(batch_size)]);mse=float((p-vy).square().mean())
        if mse<best:
            best=mse;bestepoch=epoch+1;torch.save(model.state_dict(),dest/'best.pending.pt');os.replace(dest/'best.pending.pt',dest/'best.pt')
        if mse<anchor-config['min_delta']:anchor=mse;stall=0
        else:stall+=1
        row=dict(epoch=epoch+1,train_mse=float(np.average(losses,weights=counts)),validation_mse=mse,
                 seconds=time.monotonic()-t0,stall=stall,best_epoch=bestepoch,context_mask_sha256=context_sha256)
        history.append(row);atomic_json(dest/'history.json',history);atomic_json(dest/'progress.json',row)
        torch.save(dict(provenance=provenance,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,
            best=best,anchor=anchor,stall=stall,bestepoch=bestepoch,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),history=history),dest/'latest.pending.pt')
        os.replace(dest/'latest.pending.pt',dest/'latest.pt');print(json.dumps(row),flush=True)
        if time.monotonic()-began>36000 and stall<config['patience']:
            atomic_json(dest/'resume_required.json',dict(epoch=epoch+1));return
    model.load_state_dict(torch.load(dest/'best.pt'));model.eval()
    with torch.no_grad():pred=torch.cat([model(batch) for batch in vx.split(batch_size)]).cpu().numpy()
    pred+=np.array([r['model_pka'] for r in valrows]);assert np.isfinite(pred).all()
    from pkabench.frozen_score import write_csv
    write_csv(dest/'validation_predictions.csv',[dict(r,teacher_pka=r['pka'],predicted_pka=float(p)) for r,p in zip(valrows,pred)])
    atomic_json(dest/'final.json',dict(metrics=evaluate(valrows,pred),best_epoch=bestepoch,epochs=len(history),
        validation_selected=True,cap_reached=len(history)==config['max_epochs'] and stall<config['patience']))
    atomic_json(dest/'verification.json',dict(passed=True,finite_gradients=True,complete=True,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        manifest_sha256=digest(dest/'manifest.json')))


if __name__=='__main__':
    action=sys.argv[1];dest=Path(sys.argv[2]);gpu=action in ('train','train-aug','train-batch')
    require_compute(threads=8 if gpu else 1,gpu_benchmark=gpu)
    if action=='features': features(dest)
    elif action=='pack': pack(dest)
    elif action=='gate': atomic_json(dest,architecture_gate())
    elif action=='train': train(dest,sys.argv[3])
    elif action=='train-aug':
        root=Path(os.environ['PKABENCH_RUNTIME'])
        train(root/'pretraining/pkpdb-5k-comparison-v1','clean',destination=dest,
            context_path=root/'pretraining/augmentation-v1/contexts' if sys.argv[3]=='mask' else None)
    elif action=='train-batch':
        root=Path(os.environ['PKABENCH_RUNTIME'])
        train(root/'pretraining/pkpdb-5k-comparison-v1','clean',destination=dest,batch_size=int(sys.argv[3]))
    else:raise ValueError(action)
