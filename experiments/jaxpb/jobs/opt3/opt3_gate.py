"""Phase-3 saved-energy gates/timing. Scientific work is Slurm-only."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID'), 'Coulson compute allocations only'
import numpy as np
import jax
from jaxpb import titrate as T, api as API
from jaxpb.compile_budget import CompileCounter
import bench_jaxpb as BJ

O = Path(BJ.TP) / 'opt3'
CPU5 = ['4066edfcdb3ea8e3_AB','4066edfcdb3ea8e3_A','4066edfcdb3ea8e3_B',
        '4de4ae106157408f_A','70c9c3eaa126c736_AB']
BASE = Path(BJ.TP) / 'opt2-finalr-20261122/src/jaxpb/titrate.py'
BASE_SHA = '84070647a7060777856c35d879d14f72ebff72dbc9fd2d623874e0b3b99a1598'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
spec = importlib.util.spec_from_file_location('jaxpb._opt3_baseline', BASE)
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)

def digest(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['cpu','baseline','gpu','fresh','units','timing'],required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--reference')
    a=p.parse_args()
    r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),env=BJ.env_info(),mode=a.mode,
           baseline_source=str(BASE),baseline_sha256=BASE_SHA,runs=[])
    target=O/(a.tag+'.json')
    if a.mode=='units':
        sys.path.insert(0,str(Path.cwd()/'tests'))
        import test_opt3, test_titrate
        r['new']=test_opt3.test_cpu_ref();r['existing']=test_titrate.local()
        r['passed']=r['new']['passed'] and r['existing']['passed']
        BJ.jdump(r,str(target));return 0 if r['passed'] else 1
    # Use one complete saved bench energy pack on every device; PB is never rerun.
    paths=[BJ.TP+'/bench/gpu/energies_gpu.npz']
    E=BJ.load_energies(paths)
    assert set(BJ.STATES) <= set(E), set(BJ.STATES)-set(E)
    r['cpu_fast_runtime']=T.cpu_available() if hasattr(T,'cpu_available') else None
    r['energy_paths']=paths;r['energy_hashes']={s:{k:digest(getattr(E[s],k)) for k in ('g','occ','W','start','nst','pairs')} for s in BJ.STATES}
    ph=jax.device_put(T.PH_TERM)
    refs=json.load(open(a.reference)) if a.reference else None
    def save(): BJ.jdump(r,str(target))
    def run(mod,backend,ids,sw,label,S):
        groups={}
        for sid in ids:
            s=E[sid];mp=T.fit_bucket(len(s.g),len(s.nst),len(s.pairs))[0]
            groups.setdefault(mp,[]).append(sid)
        row=dict(label=label,backend=backend,sweeps=sw,S=S,states=ids,buckets=[],outputs={},raw_counts={})
        fn=getattr(mod, 'mc_' + backend)
        shard=[]
        for mp, members in sorted(groups.items()):
            tstart=time.perf_counter(); n0=cc.n_compiled
            for offset in range(0,len(members),S):
                names=members[offset:offset+S];states=[E[s] for s in names]
                chain_ids=np.array([API.chain_id_of(s) for s in names],np.int64)
                keys=T.chain_keys(1,chain_ids[:,None]*T.NPH+np.arange(T.NPH)).reshape(len(names),T.NPH)
                B=T.pack_bucket(states,list(range(len(states))),mp,sw//T.BLOCK,keys,S=S)
                args=(ph,)+tuple(B[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))
                t=time.perf_counter(); cnt=np.asarray(jax.block_until_ready(fn(*args)));dt=time.perf_counter()-t
                for j,sid in enumerate(names):
                    e=E[sid];prot,tot=T.counts_to_prot(e,cnt[j]);curves=prot/sw
                    assert (tot==sw).all(),sid
                    tr=dict(method='mc',flag=None,Mp=mp,promoted=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[1],
                            prot=prot,tot=tot,curves=curves,pka=T.curves_to_pka(curves))
                    row['outputs'][sid]={k:digest(tr[k]) for k in ('prot','tot','curves','pka')}
                    row['raw_counts'][sid]=digest(cnt[j,:,:len(e.g)])
                    arr=API._state_arrays(len(e.nst),None,tr)
                    shard.append(dict(state_id=sid,alias=sid,input='saved bench energies',status='ok',arrays=arr,
                                      diag={k:tr[k] for k in ('method','flag','Mp','promoted')}))
                    row.setdefault('batch_seconds',[]).append(dict(Mp=mp,state=sid,seconds=dt/len(names)))
            row['buckets'].append(dict(Mp=mp,states=members,seconds=time.perf_counter()-tstart,compiles=cc.n_compiled-n0,
                                       host_hwm_kb=BJ.status('VmHWM')))
            print(a.tag,label,mp,'seconds',row['buckets'][-1]['seconds'],'compiles',row['buckets'][-1]['compiles'],flush=True)
            r['current']=row;save()
        path=O/(a.tag+'-'+label+'-labels.npz')
        API.write_shard(str(path),shard,dict(phase='OPT3',job=r['job'],sweeps=sw,backend=backend),{})
        # Compare the exact data written to label shards, including titration diagnostics.
        row['shard']=str(path)
        row['shard_fields']={d['state_id']:{**{k:digest(d[k]) for k in API.RAGGED},'diag':d['diag'],'status':d['status']} for d in API.load_shard(str(path),full_W=False)}
        row['wall_s']=sum(x['seconds'] for x in row['buckets'])
        row['s_per_state']=row['wall_s']/len(ids)
        if refs and 'warm' not in label:
            match=next(x for x in refs['runs'] if x['sweeps']==sw and set(x['states'])==set(ids) and 'warm' not in x['label'])
            row['reference_checks']={k:row[k]==match[k] for k in ('outputs','raw_counts','shard_fields')}
        r['runs'].append(row);r.pop('current',None);save()
        return row
    with CompileCounter() as cc:
        if a.mode == 'timing':
            run(T,'cpu',BJ.STATES,1000,'warm',32)
            run(T,'cpu',BJ.STATES,200000,'200k-all',32)
        elif a.mode in ('cpu','fresh'):
            run(T,'cpu',BJ.STATES,1000,'warm',32)
            first=run(T,'cpu',BJ.STATES,50000,'50k',32)
            five=run(T,'cpu',CPU5,200000,'200k',32)
            if a.mode=='cpu':
                again=run(T,'cpu',BJ.STATES,50000,'50k-repeat',32)
                r['repeat_exact']=all(first[k]==again[k] for k in ('outputs','raw_counts','shard_fields'))
                again=run(T,'cpu',CPU5,200000,'200k-repeat',32)
                r['repeat_200k_exact']=all(five[k]==again[k] for k in ('outputs','raw_counts','shard_fields'))
        elif a.mode=='baseline':
            run(OLD,'ref',BJ.STATES,1000,'warm',1)
            run(OLD,'ref',BJ.STATES,50000,'50k',1)
            run(OLD,'ref',CPU5,200000,'200k',1)
        else:
            for backend in ('pallas','ref'):
                run(OLD,backend,BJ.STATES,1000,backend+'-warm-old',32)
                old=run(OLD,backend,BJ.STATES,50000,backend+'-50k-old',32)
                new=run(T,backend,BJ.STATES,50000,backend+'-50k-new',32)
                r[backend+'_unchanged']=all(old[k]==new[k] for k in ('outputs','raw_counts','shard_fields'))
        r['compiles']=cc.summary()
    if a.mode in ('cpu','fresh','timing'):
        r['compiled_memory']={}
        for mp in sorted({T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[0] for e in E.values()}):
            sid=next(s for s in BJ.STATES if T.fit_bucket(len(E[s].g),len(E[s].nst),len(E[s].pairs))[0]==mp)
            b=T.pack_bucket([E[sid]],[0],mp,1,np.zeros((1,T.NPH),np.int32),S=1)
            args=(np.float32(T.PH_TERM[0]),)+tuple(jax.device_put(b[k][0]) for k in ('g','occ','W','start','nst','pa','pb','meta'))+(np.int32(0),)
            ma=T._mc_cpu_chain.lower(*args).compile().memory_analysis()
            r['compiled_memory'][str(mp)]={k:int(getattr(ma,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')}
        r['host_hwm_kb']=BJ.status('VmHWM')
    r['passed']=all(all(x.get('reference_checks',{'none':True}).values()) for x in r['runs']) and r.get('repeat_exact',True) and r.get('repeat_200k_exact',True) and r.get('pallas_unchanged',True) and r.get('ref_unchanged',True)
    save();print('RESULT',a.tag,r['passed'],flush=True)
    return 0 if r['passed'] else 1

if __name__=='__main__': sys.exit(main())
