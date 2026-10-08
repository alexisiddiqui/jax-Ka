"""Versioned labels with explicit semantics and immutable structural provenance."""
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pkabench.runtime import atomic_json, digest

KEY=('complex_id','chain','resnum','icode','group')
LABEL_KINDS=('occupancy_curve','scalar_pka','native_tautomer_energy','paired_observation')

def read(path): return json.loads(Path(path).read_text())


def register_fast(out,source,full_float32=False):
    """Register changed numerical settings without mutating previous provenance."""
    import shutil
    manifest=read(source/'manifest.json')
    if (out/'manifest.json').exists():raise FileExistsError(out/'manifest.json')
    out.mkdir(parents=True,exist_ok=True)
    manifest['parent']={'path':str(source),'manifest_sha256':digest(source/'manifest.json')}
    manifest['config'].update(seed_steps=None,lm_steps=32)
    if full_float32:
        manifest['config'].update(dtype='float32',seed_dtype=None,x64_enabled=False,
            batching='group-uniform sampling, then size-bucket ordering; eight complexes per optimizer update')
        manifest['reference_gates']={str(source/'real_gate.json'):digest(source/'real_gate.json'),
            str(source/'gpu-pilot/smoke/verification.json'):digest(source/'gpu-pilot/smoke/verification.json')}
        assert read(source/'real_gate.json')['passed'] and read(source/'gpu-pilot/smoke/verification.json')['passed']
    manifest['release_requirement']='Production 1024-step seed; fresh numerical gates; no old gate reuse'
    for cid,h in manifest['records_sha256'].items():
        path=source/'records'/f'{cid}.json';assert digest(path)==h
        (out/'records').mkdir(exist_ok=True);shutil.copy2(path,out/'records'/path.name)
    atomic_json(out/'manifest.json',manifest)
    for cid in manifest['train']+manifest['val']:
        previous=source/'prepared'/cid
        if not (previous/'receipt.json').exists():continue
        receipt=read(previous/'receipt.json')
        assert receipt['manifest_sha256']==manifest['parent']['manifest_sha256']
        dest=out/'prepared'/cid;dest.mkdir(parents=True,exist_ok=True)
        for path in previous.glob('*.npz'):
            # Immutable structural/label payloads; only their manifest receipt changes.
            os.link(path,dest/path.name)
        receipt['manifest_sha256']=digest(out/'manifest.json')
        atomic_json(dest/'receipt.json',receipt)
    print(json.dumps({'registered':str(out),'seed_steps':None,'effective_seed_steps':1024,'lm_steps':32,'released':False}),flush=True)

def initialize(out):
    root=Path(os.environ['PKABENCH_RUNTIME']); native=root/'tierB/native-v2'
    assert read(native/'readiness.json')['passed']
    index=read(native/'native_state_index.json')
    assert digest(native/'native_state_index.json')==read(native/'verification.json')['index_sha256']
    pilotpath=root/'campaigns/production-1024-v1/pilot.json'; pilot=set(read(pilotpath)['complex_ids'])
    source=Path(read(native/'manifest.json')['source'])
    assignments=pq.read_table(native/'assignments.parquet').to_pylist()
    structures={s['complex_id']:s for s in pq.read_table(native/'structures.parquet').to_pylist()}
    states=defaultdict(dict)
    for s in index: states[s['complex_id']][s['state']]=s
    requested=[a for a in assignments if a['split']=='val' or a['complex_id'] in pilot]
    assert len(pilot)==500 and all(a['split']=='train' for a in assignments if a['complex_id'] in pilot)
    assert not {a['component_id'] for a in requested if a['split']=='train'} & {a['component_id'] for a in requested if a['split']=='val'}
    records=[]; excluded=[]
    if (out/'manifest.json').exists(): raise FileExistsError('Dataset manifest already exists')
    out.mkdir(parents=True,exist_ok=True)
    for a in requested:
        cid=a['complex_id']
        if set(states[cid])!={'AB','A','B'}:
            excluded.append(dict(complex_id=cid,reason='missing native state',split=a['split'])); continue
        sites={}
        for st,s in states[cid].items():
            p=Path(s['export'])/'sites.json'; assert digest(p)==s['sites_sha256']
            sites[st]={tuple(r[k] for k in KEY):r for r in read(p) if r['supervision_eligible'] and r['interface']}
        keys=sites['AB'].keys() & (sites['A'].keys() | sites['B'].keys())
        if not keys:
            excluded.append(dict(complex_id=cid,reason='no paired eligible interface site',split=a['split'])); continue
        paths={st:str(source/'structures'/cid/f'{st}.cif') for st in ('AB','A','B')}
        record=dict(a,n_residues=structures[cid]['n_residues'],structures=paths,
                    structure_sha256={st:digest(p) for st,p in paths.items()},states=states[cid],
                    label_availability={k:k in ('occupancy_curve','scalar_pka','paired_observation') for k in LABEL_KINDS},
                    label_source='native-v2 current PypKa',conditions={'temperature_K':298.15,'ionic_strength_M':.1},
                    paired_interface_sites=len(keys))
        atomic_json(out/'records'/f'{cid}.json',record); records.append(record)
    train=[r for r in records if r['split']=='train']; smoke=[]
    for role in sorted({r['role'] for r in train}):
        candidates=sorted((r for r in train if r['role']==role),key=lambda r:(r['n_residues'],r['complex_id']))
        smoke += [candidates[round(i*(len(candidates)-1)/3)]['complex_id'] for i in range(min(4,len(candidates)))]
    assert len(smoke)==8 and len(set(smoke))==8
    config={'epochs':20,'seeds':[17,29,43],'learning_rate':.001,'gradient_clip':1.,'accumulate':8,
        'paired_ramp_epochs':5,'prior':.001,'profile_scales':[.25,.5,.75,1.,1.5,2.,4.],
        'initialization':'production','damped_steps':1024,'lm_steps':512,'lm_tolerance':1e-6,'residual_tolerance':2e-5,
        'dtype':'float64','per_complex_missing_limit':.05,'epoch_missing_limit':.01,'ph':[-2,16,.25]}
    atomic_json(out/'manifest.json',{'config':config,'native':str(native),'source':str(source),
        'native_readiness_sha256':digest(native/'readiness.json'),'pilot_sha256':digest(pilotpath),
        'records_sha256':{r['complex_id']:digest(out/'records'/f"{r['complex_id']}.json") for r in records},
        'train':[r['complex_id'] for r in train],'val':[r['complex_id'] for r in records if r['split']=='val'],
        'smoke':smoke,'excluded':excluded,'test_data_included':False})
    print(json.dumps({'train':len(train),'val':len(records)-len(train),'smoke':smoke,'excluded':len(excluded)}),flush=True)


def prepare(out,cid):
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache,native_identities
    from jaxpropka.parameters import GROUPS,GROUP_AA
    from pkabench.prep import read_cif
    from .branches import paired_inputs
    manifest=read(out/'manifest.json'); path=out/'records'/f'{cid}.json'
    assert digest(path)==manifest['records_sha256'][cid]; record=read(path)
    dest=out/'prepared'/cid
    if (dest/'receipt.json').exists(): return
    dest.mkdir(parents=True,exist_ok=True); caches={}; native_labels={}
    for st in ('AB','A','B'):
        cif=record['structures'][st]; assert digest(cif)==record['structure_sha256'][st]
        topology=load_topology(read_cif(cif),gap_policy='cap',freeze_disulfides=True)
        candidates=build_candidates(topology,missing_sidechain='error')
        cache=build_cache(topology,candidates,identities=native_identities(topology),dtype=np.float64)
        caches[st]=cache; cache.save(dest/f'{st}.npz')
        state=record['states'][st]; raw=Path(state['source'])
        for name,h in state['source_hashes'].items(): assert digest(raw/name)==h
        sitepath=Path(state['export'])/'sites.json'; assert digest(sitepath)==state['sites_sha256']
        masks={tuple(r[k] for k in KEY):r for r in read(sitepath)}
        mapping={(r['chain'],r['resnum']):r['original'] for r in read(raw/'mapping.json')}
        rows={}
        for r in read(raw/'result.json')['rows']:
            key=(cid,*mapping[r['chain'],r['resnum']],r['group']); mask=masks[key]
            if mask['supervision_eligible'] and mask['interface']:
                rows[key]={'curve':r['curve'],'pka':r['pka']}
        native_labels[st]=rows
    inputs,layout=paired_inputs(caches['AB'],caches['A'],caches['B'])
    n=layout['N']; labels=np.zeros((2,73,n,9)); eligible=np.zeros((n,9),bool)
    teacher_pka=np.full((2,n,9),np.nan)
    lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(caches['AB'].keys)}
    omitted=[]
    free=native_labels['A']|native_labels['B']
    for key in sorted(native_labels['AB'].keys() & free.keys()):
        _,chain,num,ins,group=key; i=lookup.get((chain,num,ins)); g=GROUPS.index(group)
        if i is None or not inputs['arrays']['group_mask'][:,i,g].all() or (g<7 and caches['AB'].native_index[i]!=GROUP_AA[g]):
            omitted.append(list(key)); continue
        labels[0,:,i,g]=native_labels['AB'][key]['curve']; labels[1,:,i,g]=free[key]['curve']; eligible[i,g]=True
        for branch,row in enumerate((native_labels['AB'][key],free[key])):
            if row['pka'] is not None: teacher_pka[branch,i,g]=row['pka']
    assert eligible.any(),'No representable paired interface labels'
    assert np.isfinite(labels).all() and np.all((labels>=0)&(labels<=1))
    arrays={f'arrays__{k}':v for k,v in inputs['arrays'].items()}
    arrays.update({k:v for k,v in inputs.items() if k!='arrays'}); arrays.update(reference=labels,eligible=eligible,teacher_pka=teacher_pka)
    np.savez_compressed(dest/'paired.npz',**arrays)
    atomic_json(dest/'receipt.json',{'complex_id':cid,'layout':layout,'sites':int(eligible.sum()),'omitted_model_sites':omitted,
        'output_sha256':digest(dest/'paired.npz'),'cache_sha256':{st:digest(dest/f'{st}.npz') for st in caches},
        'record_sha256':digest(path),'manifest_sha256':digest(out/'manifest.json')})
    print(json.dumps({'complex_id':cid,'layout':layout,'sites':int(eligible.sum())}),flush=True)


def load(out,cid,dtype=None):
    folder=out/'prepared'/cid; receipt=read(folder/'receipt.json')
    assert receipt['manifest_sha256']==digest(out/'manifest.json')
    assert digest(folder/'paired.npz')==receipt['output_sha256']
    with np.load(folder/'paired.npz',allow_pickle=False) as f:
        arrays={k.split('__',1)[1]:f[k] for k in f.files if k.startswith('arrays__')}
        inputs={'arrays':arrays,**{k:f[k] for k in ('probabilities','active','active_valid','teacher_pka')}}
        reference=f['reference']
        if dtype is not None:
            arrays={k:(v.astype(dtype) if np.issubdtype(v.dtype,np.floating) else v) for k,v in arrays.items()}
            inputs=dict(inputs,arrays=arrays,probabilities=inputs['probabilities'].astype(dtype))
            reference=reference.astype(dtype)
        return inputs,reference,f['eligible'],receipt
