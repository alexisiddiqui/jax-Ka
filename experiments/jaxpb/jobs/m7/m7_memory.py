"""4096-state real-driver replay: bounded summaries, real assembly, synthetic PB/MC."""
import os,json,dataclasses,time,gc
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np
import bench_jaxpb as BJ
from jaxpb import api as A,records as R
import sys
sys.path.insert(0,os.getcwd()+'/tests')
import test_api as F
O=Path(BJ.TP)/'m7';D=O/('memory-'+os.environ['SLURM_JOB_ID']);D.mkdir()
r=F.synth('memory_A',1);inp=str(D/'input.npz');R.save_states(inp,[r])
A.solve_pb=F.fake_pb
# The driver filters nonnumeric timing fields; this string is deliberately not
# retained. The measured replay exercises actual numeric rows and batch history.
fake=F.fake_pb
payload='x'*32768
def pb(*args,**kw):
 out,tim=fake(*args,**kw);tim['replay_payload']=payload;return out,tim
A.solve_pb=pb

def titrate(E,cfg,chain_ids=None,on_batch=None,**kw):
 if on_batch:on_batch(dict(Mp=64,states=len(E),seconds=0.,compiles=0))
 return [dict(curves=np.zeros((73,len(e.nst))),pka=np.full(len(e.nst),np.nan),Mp=64,method='replay',promoted=False,flag=None) for e in E]
A.titrate=titrate
original=A._run_shard;samples=[]
def shard(*args,**kw):
 original(*args,**kw);gc.collect();s=args[-1];samples.append(dict(states=s['state_timing']['count'],rss_kb=BJ.status('VmRSS'),hwm_kb=BJ.status('VmHWM'),history_rows={k:len(s[k]) for k in ('titration_batches','compiles_after_shard','shard_bytes')}))
 BJ.jdump(dict(samples=samples),str(O/'memory-progress.json'))
A._run_shard=shard
items=[dict(input=inp,state_id=r.state_id,alias=f'{r.state_id}@{i}') for i in range(4096)]
s=A.run(items,str(D/'run'),A.Config(backend='xla',shard_size=128,s_flush=1,mc_sweeps=1000),echo=False)
warm=[v for v in samples if v['states']>=1024];delta=warm[-1]['rss_kb']-warm[0]['rss_kb'];span=max(v['rss_kb'] for v in warm)-min(v['rss_kb'] for v in warm)
result=dict(env=BJ.env_info(),states=len(items),warm_interval_states=[1024,4096],samples=samples,rss_delta_kb=delta,rss_span_kb=span,summary_bytes=len(json.dumps(s)),history_counts=s['history_totals'],history_limit=s['history_limit'],passed=span<64*1024,scope='actual API.run, fake PB/MC, real assembly/shards/resume bookkeeping; input plan preallocated; numeric timing rows retained, nonnumeric 32KiB field filtered')
assert len(s['titration_batches'])==A.SUMMARY_HISTORY_LIMIT and s['history_totals']['titration_batches']['count']==4096
BJ.jdump(result,str(O/'memory.json'));print(result,flush=True)
