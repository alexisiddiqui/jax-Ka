"""OPT3B saved-energy experiment. All numerical work requires a Slurm allocation."""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
import numpy as np
import jax
from jaxpb import titrate as T, api as API, records as RC
from jaxpb.compile_budget import CompileCounter
import bench_jaxpb as BJ

O = Path(BJ.TP) / 'opt3b'
CPU5 = ['4066edfcdb3ea8e3_AB','4066edfcdb3ea8e3_A','4066edfcdb3ea8e3_B',
        '4de4ae106157408f_A','70c9c3eaa126c736_AB']
BASE = Path(BJ.TP) / 'opt2-finalr-20261122/src/jaxpb/titrate.py'
BASE_SHA = '84070647a7060777856c35d879d14f72ebff72dbc9fd2d623874e0b3b99a1598'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
spec = importlib.util.spec_from_file_location('jaxpb._opt3b_baseline', BASE)
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)


def digest(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['units','cpu','ref','fresh','timing','gpu','gpu-final'], required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    target = O/(a.tag+'.json')
    r = dict(job=os.environ['SLURM_JOB_ID'], frozen=os.getcwd(), env=BJ.env_info(), mode=a.mode,
             baseline_source=str(BASE), baseline_sha256=BASE_SHA, runs=[])
    def save(): BJ.jdump(r, str(target))
    if a.mode == 'units':
        sys.path.insert(0, str(Path.cwd()/'tests'))
        import test_opt3, test_titrate
        r['new'] = test_opt3.test_cpu_ref(); save()
        r['existing'] = test_titrate.local(); save()
        subprocess.run([sys.executable, 'tests/test_api.py', '/tmp/opt3b-api-'+r['job']], check=True)
        r['api'] = True
        r['passed'] = r['new']['passed'] and r['existing']['passed']; save()
        return
    E = BJ.load_energies([BJ.TP+'/bench/gpu/energies_gpu.npz'])
    r['energy_paths'] = [BJ.TP+'/bench/gpu/energies_gpu.npz']
    if a.mode in ('gpu','gpu-final'):
        labels = {}
        for path in sorted(glob.glob(BJ.TP+'/a40/m6/m6/runA/labels_*.npz')):
            labels.update({x['alias']: x for x in API.load_shard(path)})
        assert len(labels) == 104
        for sid, lab in labels.items():
            E['L3:'+sid] = API.to_energies(lab)
        r['energy_paths'].append(BJ.TP+'/a40/m6/m6/runA/labels_*.npz')
        teacher = {}
        # Match M6's token mapping and finite-midpoint metric exactly.
        sys.path.insert(0, BJ.BASE+'/tierB/mf-surrogate-v1/code')
        import common as C
        for path in BJ.INPUTS:
            for rec in RC.load_states(path):
                if rec.state_id not in labels: continue
                toks = json.loads(rec.ref['mc_energies'])['all_sites']
                rows = {(x['chain'],x['resnum'],x['group']): x for x in json.loads(rec.ref['result'])['rows']}
                teacher['L3:'+rec.state_id] = np.array([np.nan if rows[C.token_key(t)]['pka'] is None
                                                     else rows[C.token_key(t)]['pka'] for t in toks])
        assert len(teacher) == 104
        np.savez_compressed(O/'teacher.npz', **teacher)
    r['energy_hashes'] = {s:{k:digest(getattr(e,k)) for k in ('g','occ','W','start','nst','pairs')} for s,e in E.items()}
    archive = {}
    if a.mode == 'gpu-final':
        previous=json.load(open(O/'gpu.json'))
        r['baseline_arms_source']=dict(job=previous['job'],frozen=previous['frozen'],file=str(O/'gpu.json'))
        r['runs']=[x for x in previous['runs'] if x['label'].startswith(('old-','noise-'))]
        z=np.load(O/'gpu-arrays.npz')
        archive={k:z[k] for k in z.files if k.startswith(('old-','noise-'))}
        assert all(any(x['label']==label for x in r['runs']) for label in ('old-l3','noise-l3','old-bench200k','noise-bench200k'))
    ph = jax.device_put(T.PH_TERM)
    with CompileCounter() as cc:
        def run(mod, backend, ids, sw, label, seed=1, S=32):
            groups = {}
            for sid in ids:
                e=E[sid]; mp=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[0]
                assert mp is not None
                groups.setdefault(mp,[]).append(sid)
            row = dict(label=label, backend=backend, seed=seed, sweeps=sw, states=ids,
                       buckets=[], outputs={}, raw_counts={}, shard_fields={})
            fn = getattr(mod, 'mc_'+backend)
            for mp,members in sorted(groups.items()):
                n0=cc.n_compiled; seconds=0
                for offset in range(0,len(members),S):
                    names=members[offset:offset+S]; states=[E[s] for s in names]
                    cids=np.array([API.chain_id_of(s.removeprefix('L3:')) for s in names],np.int64)
                    keys=T.chain_keys(seed,cids[:,None]*T.NPH+np.arange(T.NPH))
                    b=T.pack_bucket(states,list(range(len(states))),mp,sw//T.BLOCK,keys,S=S)
                    args=(ph,)+tuple(b[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))
                    t=time.perf_counter(); cnt=np.asarray(jax.block_until_ready(fn(*args))); seconds+=time.perf_counter()-t
                    for j,sid in enumerate(names):
                        e=E[sid]; prot,tot=T.counts_to_prot(e,cnt[j]); assert (tot==sw).all()
                        tr=dict(method='mc',flag=None,Mp=mp,promoted=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[1],
                                prot=prot,tot=tot,curves=prot/sw,pka=T.curves_to_pka(prot/sw))
                        row['outputs'][sid]={k:digest(tr[k]) for k in ('prot','tot','curves','pka')}
                        row['raw_counts'][sid]=digest(cnt[j,:,:len(e.g)])
                        fields=API._state_arrays(len(e.nst),None,tr)
                        row['shard_fields'][sid]={k:digest(v) for k,v in fields.items()}
                        if 'warm' not in label:
                            for k,v in dict(counts=cnt[j,:,:len(e.g)],prot=prot,pka=tr['pka']).items():
                                archive[f'{label}/{sid}/{k}']=v
                row['buckets'].append(dict(Mp=mp,states=len(members),seconds=seconds,compiles=cc.n_compiled-n0))
                r['current']=row; save()
                print(a.tag,label,mp,seconds,'compiles',cc.n_compiled-n0,flush=True)
            row['wall_s']=sum(x['seconds'] for x in row['buckets']); row['s_per_state']=row['wall_s']/len(ids)
            r['runs'].append(row); r.pop('current',None); save()
            np.savez_compressed(O/(a.tag+'-arrays.npz'),**archive)
            return row
        if a.mode in ('cpu','fresh','timing'):
            run(T,'cpu',BJ.STATES,1000,'warm')
            if a.mode != 'timing':
                first=run(T,'cpu',BJ.STATES,50000,'50k')
                five=run(T,'cpu',CPU5,200000,'200k')
                if a.mode=='cpu':
                    again=run(T,'cpu',BJ.STATES,50000,'50k-repeat')
                    repeat=run(T,'cpu',CPU5,200000,'200k-repeat')
                    r['repeat_exact']=all(first[k]==again[k] and five[k]==repeat[k]
                                          for k in ('outputs','raw_counts','shard_fields'))
            else:
                run(T,'cpu',BJ.STATES,200000,'200k-all')
        elif a.mode=='ref':
            for mod,name,seed in [(T,'new',1),(OLD,'old',1),(OLD,'noise',2)]:
                run(mod,'ref',BJ.STATES,1000,name+'-warm',seed,S=1)
                run(mod,'ref',BJ.STATES,50000,name+'-50k',seed,S=1)
                run(mod,'ref',CPU5,200000,name+'-200k',seed,S=1)
        else:
            l3=sorted(s for s in E if s.startswith('L3:'))
            for mod,name,seed in ([(T,'new',1)] if a.mode=='gpu-final' else [(OLD,'old',1),(T,'new',1),(OLD,'noise',2)]):
                run(mod,'pallas',BJ.STATES,1000,name+'-warm',seed)
                run(mod,'pallas',BJ.STATES,200000,name+'-bench200k',seed)
                run(mod,'pallas',l3,200000,name+'-l3',seed)
            # Matched 50k cross-backend, including CPU's five-state 200k gate.
            run(T,'pallas',BJ.STATES,50000,'pallas-50k')
            run(T,'ref',BJ.STATES,1000,'ref-warm',S=1)
            run(T,'ref',BJ.STATES,50000,'ref-50k',S=1)
            run(T,'pallas',CPU5,200000,'pallas-200k')
            run(T,'ref',CPU5,200000,'ref-200k',S=1)
        r['compiles']=cc.summary()
    if a.mode == 'fresh':
        r['public_api'] = {}
        assert API.mc_backend(API.Config()) == 'cpu'
        for label,ids,sw in [('50k',BJ.STATES,50000),('200k',CPU5,200000)]:
            out=API.titrate([E[s] for s in ids],API.Config(mc_sweeps=sw),chain_ids=[API.chain_id_of(s) for s in ids])
            expected=next(x for x in r['runs'] if x['label']==label)
            got={s:{k:digest(v[k]) for k in ('prot','tot','curves','pka')} for s,v in zip(ids,out)}
            rows=[dict(state_id=s,alias=s,input='saved energies',status='ok',arrays=API._state_arrays(len(E[s].nst),None,v),
                       diag={k:v[k] for k in ('method','flag','Mp','promoted')}) for s,v in zip(ids,out)]
            path=O/('fresh-'+label+'-labels.npz')
            API.write_shard(str(path),rows,dict(phase='OPT3B'),{})
            loaded=API.load_shard(str(path),full_W=False)
            fields={v['alias']:{k:digest(v[k].T if k=='curves' else v[k]) for k in API._state_arrays(len(E[v['alias']].nst),None,out[ids.index(v['alias'])])} for v in loaded}
            r['public_api'][label]=got==expected['outputs'] and fields==expected['shard_fields']
            save()
    r['host_hwm_kb']=BJ.status('VmHWM'); r['completed']=True; save()
    print('COMPLETE',a.tag,flush=True)

if __name__ == '__main__': main()
