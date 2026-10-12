"""6SC6 ABC-context masked gradient, persistent-cache and GPU-memory diagnostic.

Set CHECK_INPUT to prepared input directory and CHECK_OUT to results directory.
See docs/TORCH_STRUCTURES.md for scope and docs/validation/6sc6-abc-gradients/README.md
for reproduction commands. Run each workload in a fresh process.
"""
import os,time,json,argparse,threading,subprocess
from pathlib import Path
if hasattr(os,'sched_setaffinity'): os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
import numpy as np
import torch
import jax
import jax.numpy as jnp
from pkanet.differentiable import StructureFeatures
from pkanet.torch import TorchStructureReadout
from pkanet.ogqt import initialize
from pkanet.inference import load_model
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['backbone','sidechain'],required=True);p.add_argument('--mode',choices=['forward','backward','check'],required=True);p.add_argument('--run',default='1');p.add_argument('--device',choices=['cpu','gpu'],default='gpu');a=p.parse_args()
torch_device='cuda' if a.device=='gpu' else 'cpu'
out=Path(os.environ['CHECK_OUT']);src=Path(os.environ['CHECK_INPUT']);out.mkdir(parents=True,exist_ok=True)
events=[];jax.monitoring.register_event_listener(lambda name,**kwargs:events.append(name))
def hits():return events.count('/jax/compilation_cache/cache_hits')
phase='setup';samples=[];stop=threading.Event()
def sample():
 while not stop.is_set():
  try:
   r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
   for row in r.stdout.splitlines():
    parts=row.split(',')
    if len(parts)==2 and parts[0].strip()==str(os.getpid()):samples.append(dict(phase=phase,mib=float(parts[1].strip())))
  except (OSError,ValueError,subprocess.TimeoutExpired):pass
  stop.wait(.1)
thread=threading.Thread(target=sample,daemon=True);thread.start() if a.device=='gpu' else None
f,arrays,meta=StructureFeatures.from_file(src/'6SC6_ABC.pdb',sidechains=a.variant=='sidechain')
interface=np.load(src/'interface.npy');selection=json.loads((src/'interface.json').read_text())
assert len(interface)==len(arrays[0])
if a.variant=='backbone':params,_=load_model(os.environ.get('CHECK_MODEL',str(Path(__file__).resolve().parents[1]/'models/ogqt-production-50k')))
else:params=initialize(jax.random.PRNGKey(4),width=44,ff=88,node_dim=152)
params=jax.tree.map(lambda x:jnp.asarray(x,dtype=jnp.float32),params)
layer=TorchStructureReadout.from_ogqt(params,f,output='shift',compilation_cache=out/'jit-cache')
x=[torch.tensor(v,device=torch_device,requires_grad=a.mode!='forward') for v in arrays]
mask=torch.tensor(interface,device=torch_device);cm=mask[:,None]&torch.tensor(f.atom_present,device=torch_device)
qmask=interface[f.graph['query_residue']];weights=torch.tensor(qmask.astype(np.float32)/qmask.sum(),device=torch_device)
kwargs=dict(sequence_mask=mask,backbone_mask=mask,sidechain_mask=cm)
def evaluate(values):return (layer(*values,**kwargs)*weights).sum()
def sync():
 if a.device=='gpu': torch.cuda.synchronize()
def memory():
 sync(); stats=jax.devices(a.device)[0].memory_stats() or {}
 return dict(jax={k:int(v) for k,v in stats.items() if isinstance(v,(int,np.integer))},torch_peak_allocated=int(torch.cuda.max_memory_allocated()) if a.device=='gpu' else None,torch_peak_reserved=int(torch.cuda.max_memory_reserved()) if a.device=='gpu' else None)
report=dict(variant=a.variant,mode=a.mode,run=a.run,torch=torch.__version__,jax=jax.__version__,device=torch.cuda.get_device_name() if a.device=='gpu' else 'cpu',residues=len(interface),interface_residues=int(interface.sum()),interface_query_sites=int(qmask.sum()),input_parameter_dtype='float32',context_chains=['A','B','C'],model='released 24-feature checkpoint' if a.variant=='backbone' else 'initialized 152-feature width44 ff88 seed4',objective='mean predicted shift over supported query sites at the A-B interface')
phase='forward';h=hits();t=time.perf_counter();value=evaluate(x);sync()
report['forward']=dict(seconds=time.perf_counter()-t,cache_hits=hits()-h,value=float(value.detach()),output_dtype=str(value.dtype),memory=memory())
if a.mode!='forward':
 phase='backward';h=hits();t=time.perf_counter();value.backward();sync()
 report['backward']=dict(seconds=time.perf_counter()-t,cache_hits=hits()-h,memory=memory())
 gradients=[v.grad.detach().clone() for v in x]
 report['finite_gradients']=all(bool(torch.isfinite(g).all()) for g in gradients)
 report['outside_interface_max']=max(float(g[~mask].abs().max()) for g in gradients)
 np.savez(out/f'{a.variant}-{a.mode}-{a.run}-gradients.npz',**{k:g.cpu().numpy() for k,g in zip(['sequence','backbone','sidechains'],gradients)})
 if a.mode=='check':
  phase='finite_difference';checks=[];chain=np.array([row['chain'] for row in meta['sites']]) # metadata for outputs only
  residue_chain=np.empty(len(interface),dtype='U1')
  # The source topology keeps all chain A residues before B; use the graph chain mapping.
  residue_chain[:]='C';residue_chain[np.asarray(f.chain)==0]='A';residue_chain[np.asarray(f.chain)==1]='B'
  for k,group in enumerate(['sequence','backbone','sidechains']):
   for ch in ['A','B']:
    keep=torch.tensor(interface&(residue_chain==ch),device=torch_device).reshape(-1,*([1]*(x[k].ndim-1)))
    direction=gradients[k]*keep
    if k==0:direction=direction-direction.mean(-1,keepdim=True) # zero-sum residue-feature direction
    norm=direction.norm()
    if float(norm)<1e-10:
     checks.append(dict(group=group,chain=ch,zero_gradient=True,expected=a.variant=='backbone' and k==2));continue
    direction=direction/norm;analytic=float((gradients[k]*direction).sum());estimates=[]
    for eps in [.01,.003]:
     plus=[v.detach() for v in x];minus=[v.detach() for v in x]
     plus[k]=plus[k]+eps*direction;minus[k]=minus[k]-eps*direction
     fd=float((evaluate(plus)-evaluate(minus))/(2*eps));error=abs(fd-analytic)
     estimates.append(dict(epsilon=eps,finite_difference=fd,absolute_error=error,relative_error=error/max(abs(analytic),1e-8),passed=error<=5e-5+.02*abs(analytic)))
    checks.append(dict(group=group,chain=ch,autograd=analytic,estimates=estimates,passed=all(v['passed'] for v in estimates)))
  report['checks']=checks
  report['passed']=report['finite_gradients'] and report['outside_interface_max']==0 and all(v.get('passed',v.get('expected',False)) for v in checks)
else:
 # Repeat forward to ensure sampled process memory sees the resident executable/buffers.
 for _ in range(3):evaluate(x);sync()
phase='done';sync();stop.set();thread.join(timeout=6) if a.device=='gpu' else None
report['sampled_process_peak_mib']=max((v['mib'] for v in samples),default=None)
report['sampled_process_peak_by_phase_mib']={ph:max(v['mib'] for v in samples if v['phase']==ph) for ph in {v['phase'] for v in samples}}
report['memory_samples']=len(samples)
(out/f'{a.variant}-{a.mode}-{a.run}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if a.mode=='check' and not report['passed']:raise SystemExit('Gradient check failed')
