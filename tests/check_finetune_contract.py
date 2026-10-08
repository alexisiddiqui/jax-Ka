"""Compute-node checks for aggregation and paired gradient semantics."""
import os,sys
from pathlib import Path
import numpy as np
from pkabench.finetune_experiment import macro,weights
from pkabench.runtime import require_compute
require_compute()
rows=[]; pred=[]
for cid,g,n,error in [('a','g1',100,1.),('b','g1',1,3.),('c','g2',1,6.)]:
    for _ in range(n): rows.append({'complex_id':cid,'component_id':g,'target_delta_pka':0.}); pred.append(error)
s,groups=macro(rows,pred,list(range(len(rows))))
assert s['mae']==4.,s
w=weights(rows,list(range(len(rows))))
assert np.isclose(w[:101].sum(),w[101:].sum())
sys.path.append(str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/pkai/lib/python3.11/site-packages'))
import torch
m=torch.jit.load(str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/pkai/lib/python3.11/site-packages/pkai/models/pKAI_model.pt')); m.eval()
torch.manual_seed(17)
a=torch.rand(4,4008); b=torch.rand(4,4008)
assert torch.equal(m(a)-m(a),torch.zeros(4,1))
for name,p in m.named_parameters(): p.requires_grad_(name.startswith('layers.3.'))
loss=((m(a)-m(b)-1)**2).mean(); loss.backward()
assert m.layers.__getattr__('3').weight.grad.abs().sum()>0
assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('layers.3.'))
print('PASS: unequal-size group aggregation, group-balanced weights, paired cancellation and last-layer-only gradients')
