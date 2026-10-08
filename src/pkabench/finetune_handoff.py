"""Explicit experiment-02 label contract and paired structural features."""
import json
import os
import subprocess
from pathlib import Path
from collections import Counter,defaultdict
from .runtime import require_compute,atomic_json,digest
from .schema import read_table,key,GROUPS


def initialise(out):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    runtime=Path(os.environ['PKABENCH_RUNTIME']); source=runtime/'campaigns/production-nojax-v1'
    assert json.loads((source/'verification.json').read_text())['passed']
    pilot=json.loads((runtime/'campaigns/production-1024-v1/pilot.json').read_text())['complex_ids']
    assignments={r['complex_id']:r for r in read_table(source/'assignments.parquet')}
    selected=set(pilot)|{cid for cid,a in assignments.items() if a['split']=='val'}
    assert all(assignments[c]['split']=='train' for c in pilot) and len(pilot)==500
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    masks={key(r):r for r in read_table(source/'site_masks.parquet') if r['complex_id'] in selected}
    sites=defaultdict(list)
    for r in read_table(source/'sites.parquet'):
        if r['complex_id'] in selected: sites[r['complex_id']].append(r)
    structures=[]; target=[]; excluded=Counter()
    for s in read_table(source/'structures.parquet'):
        cid=s['complex_id']
        if cid not in selected: continue
        folder=(source/'structures'/cid).resolve(); a=assignments[cid]
        structures.append(s|{'structure_root':str(folder),'role':a['role'],'state_sha256':json.dumps({st:digest(folder/f'{st}.cif') for st in ('AB','A','B')},sort_keys=True)})
        index={}
        for method in ('pypka','propka','pkai'):
            side=json.loads((source/'jobs'/method/f'{cid}.json').read_text())
            path=source/'jobs'/method/f'{cid}.parquet'; assert digest(path)==side['output_sha256']
            index[method]={(key(r),r['state']):{'status':r['status'],'pka':r['pka']} for r in read_table(path)}
        for s in sites[cid]:
            k=key(s); mask=masks[k]
            if not (mask['training_eligible'] if a['split']=='train' else mask['evaluation_eligible']): continue
            def pair(method):
                rr=[index[method].get((k,st)) for st in ('AB',s['partner'])]
                if not all(r and r['status']=='ok' and r['pka'] is not None for r in rr): return None
                return rr[0]['pka'],rr[1]['pka'],rr[0]['pka']-rr[1]['pka']
            teacher=pair('pypka'); physics=pair('propka'); baseline=pair('pkai')
            if teacher is None: excluded[a['split']+':teacher_unavailable']+=1; continue
            row={name:s[name] for name in ('complex_id','chain','resnum','icode','group','partner','residue_delta_sasa','functional_delta_sasa','min_partner_distance')}
            row.update(split=a['split'],component_id=a['component_id'],role=a['role'],interface=mask['interface'],shell_0_20=s['min_partner_distance']<=20,
                pypka_ab=teacher[0],pypka_free=teacher[1],target_delta_pka=teacher[2],
                propka_ab=physics[0] if physics else None,propka_free=physics[1] if physics else None,
                propka_delta=physics[2] if physics else None,catboost_target=teacher[2]-physics[2] if physics else None,
                frozen_pkai_ab=baseline[0] if baseline else None,frozen_pkai_free=baseline[1] if baseline else None,
                pkai_group_supported=s['group'] in ('ASP','GLU','HIS','CYS','TYR','LYS'))
            target.append(row)
    pq.write_table(pa.Table.from_pylist(structures),out/'structures.parquet')
    pq.write_table(pa.Table.from_pylist(target),out/'targets.parquet')
    pq.write_table(pa.Table.from_pylist([assignments[c] for c in sorted(selected)]),out/'assignments.parquet')
    pq.write_table(pa.Table.from_pylist(list(masks.values())),out/'site_masks.parquet')
    summary={split:{'complexes_selected':sum(s['split']==split for s in structures),
        'teacher_sites':sum(r['split']==split for r in target),'interface_sites':sum(r['split']==split and r['interface'] for r in target),
        'complexes_with_teacher_interface':len({r['complex_id'] for r in target if r['split']==split and r['interface']})} for split in ('train','val')}
    atomic_json(out/'manifest.json',{'version':'finetune-handoff-v1','source':str(source),'source_verification_sha256':digest(source/'verification.json'),
        'pilot_sha256':digest(runtime/'campaigns/production-1024-v1/pilot.json'),'summary':summary,'excluded':dict(excluded),
        'artifacts_sha256':{name:digest(out/name) for name in ('structures.parquet','targets.parquet','assignments.parquet','site_masks.parquet')},
        'contract':{'key':['complex_id','chain','resnum','icode','group'],'label':'PypKa AB minus free partner at identical prepared coordinates',
            'catboost':'Predict PypKa delta minus PROPKA delta; add PROPKA delta at inference. Never subtract teacher from itself.',
            'selection':'500 fixed training complexes plus all 151 frozen validation complexes; test labels/structures excluded.',
            'loss_selection':'Interface flag is primary diagnostic support; shell_0_20 retained for registered secondary analyses. Fit scaling and weights on training rows only.',
            'pKAI':'Shared weights f(AB)-f(free), native local features. Model-compound offsets cancel for paired same-group sites.',
            'geometry_proxies':'Coordination/density and fixed formal-charge changes are proxies; donor/acceptor proximity is not a directional hydrogen-bond assignment. No exact burial-depth feature claimed.'},
        'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in ('finetune_handoff.py','pkai_features.py')},'training_started':False})
    (out/'features').mkdir()
    print(json.dumps(summary,indent=2),flush=True)


def features(out,shard,shards):
    require_compute()
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree
    from .prep import read_cif,export_pdb
    from .annotate import SITE_ATOMS
    from jaxpropka.parameters import DONORS,ACCEPTORS
    out=Path(out).resolve(); manifest=json.loads((out/'manifest.json').read_text())
    for n,sha in manifest['code_sha256'].items(): assert digest(Path(__file__).with_name(n))==sha
    targets=defaultdict(list)
    for r in read_table(out/'targets.parquet'): targets[r['complex_id']].append(r)
    structures=sorted(read_table(out/'structures.parquet'),key=lambda s:s['complex_id'])
    for s in structures[shard::shards]:
        cid=s['complex_id']; dest=out/'features'/cid; dest.mkdir(exist_ok=False); folder=Path(s['structure_root']); errors={}
        state_hashes=json.loads(s['state_sha256'])
        for state,sha in state_hashes.items(): assert digest(folder/f'{state}.cif')==sha
        # Original annotations define centroid; no label enters geometry.
        atoms=read_cif(folder/'AB.cif'); residue_atoms=defaultdict(list)
        for i in range(len(atoms)): residue_atoms[str(atoms.chain_id[i]),int(atoms.res_id[i]),str(atoms.ins_code[i])].append(i)
        site_metadata=read_table(folder/'sites.parquet')
        iface_keys={(r['chain'],r['resnum'],r['icode']) for r in site_metadata if r['residue_delta_sasa']>10}
        iface_idx=[i for k in iface_keys for i in residue_atoms[k] if atoms.atom_name[i]=='CA']
        centroid=np.mean(atoms.coord[iface_idx],axis=0) if iface_idx else None
        formal={'ASP':-1.,'GLU':-1.,'LYS':1.,'ARG':1.}; partners={p:set(s[f'partner_{p}_chains']) for p in ('A','B')}
        geometric=[]
        for r in targets[cid]:
            inds=residue_atoms[r['chain'],r['resnum'],r['icode']]
            functional=[i for i in inds if atoms.atom_name[i] in SITE_ATOMS[r['group']]]
            if not functional: errors['site:'+str(key(r))]='missing functional atoms'; continue
            center=np.mean(atoms.coord[functional],axis=0); other='B' if r['partner']=='A' else 'A'
            pidx=np.flatnonzero(np.isin(atoms.chain_id,list(partners[other])))
            coords=atoms.coord[pidx]; dist=np.linalg.norm(coords-center,axis=1)
            f={name:r[name] for name in ('complex_id','chain','resnum','icode','group')}
            f.update(delta_sasa=r['residue_delta_sasa'],functional_delta_sasa=r['functional_delta_sasa'],
                min_partner_distance=r['min_partner_distance'],distance_interface_centroid=float(np.linalg.norm(center-centroid)) if centroid is not None else None,
                delta_heavy_count_6A=int((dist<=6).sum()),delta_heavy_count_10A=int((dist<=10).sum()))
            f['potential_partner_donor_atoms_4A']=sum(dist[j]<=4 and atoms.atom_name[i] in DONORS.get(str(atoms.res_name[i]),()) for j,i in enumerate(pidx))
            f['potential_partner_acceptor_atoms_4A']=sum(dist[j]<=4 and atoms.atom_name[i] in ACCEPTORS.get(str(atoms.res_name[i]),()) for j,i in enumerate(pidx))
            charge6=charge10=0.; opposite=[]
            for k,ii in residue_atoms.items():
                if k[0] not in partners[other]: continue
                aa=str(atoms.res_name[ii[0]]); charge=formal.get(aa,0.)
                if not charge: continue
                jj=[i for i in ii if atoms.atom_name[i] in SITE_ATOMS[aa]]
                if not jj: continue
                d=float(np.min(np.linalg.norm(atoms.coord[jj]-center,axis=1)))
                if d<=6: charge6+=charge
                if d<=10: charge10+=charge
                sitecharge={'ASP':-1,'GLU':-1,'CTERM':-1,'LYS':1,'ARG':1,'NTERM':1}.get(r['group'],0)
                if charge*sitecharge<0: opposite.append(d)
            f.update(delta_formal_charge_6A=charge6,delta_formal_charge_10A=charge10,salt_bridge_proxy_count_4A=sum(d<=4 for d in opposite),nearest_opposite_charge_distance=min(opposite) if opposite else None)
            for g in GROUPS: f['residue_'+g]=int(r['group']==g)
            geometric.append(f)
        if geometric: pq.write_table(pa.Table.from_pylist(geometric),dest/'geometry.parquet')
        for state in ('AB','A','B'):
            work=dest/state; work.mkdir(); mapping=export_pdb(read_cif(folder/f'{state}.cif'),work/'input.pdb')
            atomic_json(work/'mapping.json',[{'chain':c,'resnum':n,'original':v} for (c,n),v in mapping.items()])
            atomic_json(work/'request.json',{'pdb':str(work/'input.pdb'),'mapping':str(work/'mapping.json')})
            try:
                with (work/'worker.log').open('w') as log:
                    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/pkai/bin/python'),'-m','pkabench.pkai_features',str(work/'request.json'),str(work/'encoded')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                encoded=work/'encoded'; data=json.loads((encoded/'features.json').read_text())
                np.savez_compressed(encoded/'features.npz',x=np.asarray(data['x'],dtype=np.float32),absolute_pka=np.asarray(data['absolute_pka']))
                rec=json.loads((encoded/'receipt.json').read_text()); rec['feature_sha256']=digest(encoded/'features.npz')
                atomic_json(encoded/'receipt.json',rec)
            except Exception as exc: errors[state]=str(exc)
        atomic_json(dest/'receipt.json',{'complex_id':cid,'errors':errors,'geometry_rows':len(geometric),
            'state_sha256':state_hashes,'manifest_sha256':digest(out/'manifest.json')})
        print(json.dumps({'complex_id':cid,'errors':errors,'sites':len(targets[cid])}),flush=True)


def verify(out):
    require_compute()
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    out=Path(out); manifest=json.loads((out/'manifest.json').read_text())
    assert not (out/'INVALID.json').exists(),'Handoff explicitly invalidated'
    for name,sha in manifest['artifacts_sha256'].items(): assert digest(out/name)==sha
    targets=read_table(out/'targets.parquet'); indexed=defaultdict(list)
    for r in targets: indexed[r['complex_id']].append(r)
    paired=[]; geometry=[]; exclusions=Counter(); checked=0; feature_hashes={}
    for s in read_table(out/'structures.parquet'):
        cid=s['complex_id']; folder=out/'features'/cid; receipt=folder/'receipt.json'
        assert receipt.exists(),f'Missing feature receipt: {cid}'
        receipt=json.loads(receipt.read_text()); arrays={}; keys={}; predictions={}
        if (folder/'geometry.parquet').exists(): geometry.extend(read_table(folder/'geometry.parquet'))
        for state in ('AB','A','B'):
            base=folder/state/'encoded'
            if state in receipt['errors']: continue
            rec=json.loads((base/'receipt.json').read_text()); assert digest(base/'features.npz')==rec['feature_sha256']
            feature_hashes[str(base/'features.npz')]=rec['feature_sha256']
            kk=json.loads((base/'keys.json').read_text()); keys[state]={tuple(k):i for i,k in enumerate(kk)}
            with np.load(base/'features.npz') as data:
                arrays[state]=data['x']; predictions[state]=data['absolute_pka']
                assert arrays[state].shape==(len(kk),4008) and np.isfinite(arrays[state]).all()
        for r in indexed[cid]:
            k=(r['chain'],r['resnum'],r['icode'],r['group']); state=r['partner']
            if k not in keys.get('AB',{}) or k not in keys.get(state,{}): exclusions[r['split']+':pkai_unsupported_or_feature_failure']+=1; continue
            i=keys['AB'][k]; j=keys[state][k]
            for st,ix,name in (('AB',i,'frozen_pkai_ab'),(state,j,'frozen_pkai_free')):
                if r[name] is not None:
                    assert abs(float(predictions[st][ix])-r[name])<=.0052,(cid,k,st,float(predictions[st][ix]),r[name]); checked+=1
            paired.append(r|{'ab_feature_file':str(folder/'AB/encoded/features.npz'),'ab_feature_row':i,
                'free_feature_file':str(folder/state/'encoded/features.npz'),'free_feature_row':j})
    assert checked>0 and all(any(r['split']==split for r in paired) for split in ('train','val')),'Missing validated pKAI feature support'
    pq.write_table(pa.Table.from_pylist(paired),out/'pkai_pairs.parquet')
    pq.write_table(pa.Table.from_pylist(geometry),out/'geometry.parquet')
    geom={key(r):r for r in geometry}; cat=[r|{k:v for k,v in geom[key(r)].items() if k not in r} for r in targets if r['catboost_target'] is not None and key(r) in geom]
    pq.write_table(pa.Table.from_pylist(cat),out/'catboost_pairs.parquet')
    assert all(r['split'] in ('train','val') for r in paired+cat)
    report={'passed':True,'summary':manifest['summary'],'pkai_rows':dict(Counter(r['split'] for r in paired)),
        'pkai_interface_rows':dict(Counter(r['split'] for r in paired if r['interface'])),
        'catboost_rows':dict(Counter(r['split'] for r in cat)),'frozen_pkai_state_predictions_checked':checked,
        'exclusions':dict(exclusions),'feature_files_sha256':feature_hashes,
        'artifacts_sha256':{name:digest(out/name) for name in ('pkai_pairs.parquet','catboost_pairs.parquet','geometry.parquet')},
        'training_started':False,'test_data_included':False}
    atomic_json(out/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='feature_files_sha256'},indent=2),flush=True)
