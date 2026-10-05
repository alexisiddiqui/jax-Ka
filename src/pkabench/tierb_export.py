"""Export existing train/validation teacher data at native tautomer resolution."""
import argparse,json,os
from collections import Counter,defaultdict
from pathlib import Path
from .runtime import require_compute,atomic_json,digest
from .schema import read_table,key

def init(out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    runtime=Path(os.environ['PKABENCH_RUNTIME']); source=runtime/'campaigns/production-nojax-v1'; original=runtime/'campaigns/production-1024-v1'; recovery=runtime/'campaigns/pilot-state-recovery-v1'
    assert json.loads((source/'verification.json').read_text())['passed']
    rec=json.loads((recovery/'recovery_report.json').read_text()); assert digest(recovery/'teacher-pilot-v2.parquet')==rec['prediction_sha256']
    out.mkdir(parents=True,exist_ok=False); (out/'complexes').mkdir()
    assignments=[r for r in read_table(source/'assignments.parquet') if r['split'] in ('train','val')]; ids={r['complex_id'] for r in assignments}
    assert Counter(r['split'] for r in assignments)=={'train':778,'val':151}
    assert not {r['component_id'] for r in assignments if r['split']=='train'}&{r['component_id'] for r in assignments if r['split']=='val'}
    structures=[r for r in read_table(source/'structures.parquet') if r['complex_id'] in ids]
    masks=[r for r in read_table(source/'site_masks.parquet') if r['complex_id'] in ids]
    for name,rows in [('assignments',assignments),('structures',structures),('site_masks',masks)]: pq.write_table(pa.Table.from_pylist(rows),out/f'{name}.parquet')
    replacements={(r['complex_id'],r['state']) for r in rec['records'] if r['status']=='complete'}
    selected={r['complex_id']:r['split'] for r in assignments}; assert all(selected[c]=='train' for c,s in replacements)
    atomic_json(out/'manifest.json',{'source':str(source),'original':str(original),'recovery':str(recovery),'code_sha256':digest(Path(__file__)),
      'source_verification_sha256':digest(source/'verification.json'),'recovery_report_sha256':digest(recovery/'recovery_report.json'),
      'recovery_prediction_sha256':rec['prediction_sha256'],'replacements':[list(k) for k in sorted(replacements)],
      'input_sha256':{n:digest(out/n) for n in ('assignments.parquet','structures.parquet','site_masks.parquet')},
      'policy':'Train/validation only; original split/masks; native microstate energies in kBT. No scalar pair reduction; no new PB calculations; test excluded.'})
    print(json.dumps({'selected':dict(Counter(r['split'] for r in assignments)),'recovered_states':len(replacements)}),flush=True)

def export(out,shard,shards):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    m=json.loads((out/'manifest.json').read_text()); assert digest(Path(__file__))==m['code_sha256']
    for n,h in m['input_sha256'].items(): assert digest(out/n)==h
    source=Path(m['source']); original=Path(m['original']); recovery=Path(m['recovery'])
    assert digest(recovery/'recovery_report.json')==m['recovery_report_sha256'] and digest(recovery/'teacher-pilot-v2.parquet')==m['recovery_prediction_sha256']
    overlay=defaultdict(list)
    for r in read_table(recovery/'teacher-pilot-v2.parquet'): overlay[r['complex_id']].append(r)
    replacements={tuple(k) for k in m['replacements']}; masks={key(r):r for r in read_table(out/'site_masks.parquet')}
    assignments=sorted(read_table(out/'assignments.parquet'),key=lambda r:r['complex_id'])
    for a in assignments[shard::shards]:
        cid=a['complex_id']; dest=out/'complexes'/cid; dest.mkdir(exist_ok=False)
        sidepath=original/'jobs/pypka'/f'{cid}.json'; side=json.loads(sidepath.read_text()); predpath=original/'jobs/pypka'/f'{cid}.parquet'
        assert digest(predpath)==side['output_sha256']; old=read_table(predpath); rows=overlay.get(cid,old)
        oldmap={(key(r),r['state']):r for r in old}; newmap={(key(r),r['state']):r for r in rows}; assert len(newmap)==len(rows) and newmap.keys()==oldmap.keys()
        unchanged=0
        for k,r in oldmap.items():
            assert newmap[k]['config_sha256']==r['config_sha256'] and newmap[k]['method']=='pypka'
            if (cid,r['state']) not in replacements: assert newmap[k]==r; unchanged+=1
        site_rows=read_table(source/'structures'/cid/'sites.parquet'); targets=[]; before=after=0
        for site in site_rows:
            k=key(site); mask=masks[k]; eligible=mask['training_eligible'] if a['split']=='train' else mask['evaluation_eligible']
            def valid(mp): return all(mp[k,st]['status']=='ok' and mp[k,st]['pka'] is not None for st in ('AB',site['partner']))
            if eligible and valid(oldmap): before+=1
            if eligible and valid(newmap):
                after+=1; ab=newmap[k,'AB']; free=newmap[k,site['partner']]
                targets.append({n:site[n] for n in ('complex_id','chain','resnum','icode','group','partner')}|{'split':a['split'],'component_id':a['component_id'],'role':a['role'],
                    'interface':mask['interface'],'shell_0_20':site['min_partner_distance']<=20,'pka_ab':ab['pka'],'pka_free':free['pka'],'delta_pka':ab['pka']-free['pka'],
                    'curve_ab':ab['curve'],'curve_free':free['curve']})
        if targets: pq.write_table(pa.Table.from_pylist(targets),dest/'paired_sites.parquet')
        states=[]
        for state in ('AB','A','B'):
            replaced=(cid,state) in replacements
            if replaced:
                base=recovery/'states'/cid/state; receipt=json.loads((base/'receipt.json').read_text()); raw=base/'raw'/state
                assert receipt['status']=='complete' and digest(base/'predictions.parquet')==receipt['output_sha256']
                assert digest(source/'structures'/cid/f'{state}.cif')==receipt['input_sha256']
                rr=read_table(base/'predictions.parquet'); assert all(newmap[key(r),state]==r for r in rr)
            else:
                if state in side['errors']: states.append({'state':state,'status':'teacher_failed'}); continue
                raw=Path(side['workdir'])/state
                assert digest(source/'structures'/cid/f'{state}.cif')==side['input_state_sha256'][state]
            try:
                data=json.loads((raw/'mc-energies.json').read_text()); result=json.loads((raw/'result.json').read_text()); mapping={(r['chain'],r['resnum']):r['original'] for r in json.loads((raw/'mapping.json').read_text())}
                resultmap={(r['chain'],r['resnum'],r['group']):r for r in result['rows']}
                counts=np.asarray(data['npossible_states'],dtype=np.int32); total=int(counts.sum()); n=len(counts)
                assert n==len(data['all_sites']) and (counts>=2).all()
                interactions=np.asarray(data['interactions'],dtype=np.float64); assert interactions.shape==(total,total)
                owners=np.repeat(np.arange(n),counts); allowed=owners[:,None]!=owners[None,:]
                assert np.isfinite(interactions[allowed]).all() and not np.any(interactions[allowed]==-999999)
                symmetry=float(np.max(abs(interactions-interactions.T))); assert symmetry<1e-8
                flat_g=[]; flat_occ=[]; sites=[]; offset=0
                for i,token in enumerate(data['all_sites']):
                    chain,group,num=token.rsplit('_',2); number=int(num)-(5000 if group in ('NTR','CTR') else 0); group={'NTR':'NTERM','CTR':'CTERM'}.get(group,group)
                    orig=mapping[chain,number]; k=(cid,*orig,group); mask=masks[k]; nr=int(counts[i]); res=resultmap[chain,number,group]
                    g=np.asarray(data['possible_states_g'][i][:nr],dtype=float); occ=np.asarray(data['possible_states_occ'][i][:nr],dtype=int); lookup=data['interactions_look'][i][:nr]
                    assert np.isfinite(g).all() and set(occ)<={0,1} and g[-1]==0 and lookup==list(range(offset,offset+nr))
                    names=sorted(res['intrinsic_tautomers']); pki=np.asarray([res['intrinsic_tautomers'][name] for name in names]); assert len(names)==nr-1
                    expected=pki*np.log(10)*(1-2*occ[:-1]); assert np.allclose(g[:-1],expected,atol=1e-8,rtol=1e-10)
                    assert np.array_equal(np.asarray(res['curve'],dtype=np.float32),np.asarray(newmap[k,state]['curve'],dtype=np.float32)),'Native curve differs after the schema-mandated float32 conversion'
                    sites.append({'complex_id':cid,'chain':orig[0],'resnum':orig[1],'icode':orig[2],'group':group,'offset':offset,'count':nr,'tautomers':names+['reference'],
                      'intrinsic_pka_tautomers':pki.tolist(),'supervision_eligible':mask['training_eligible'] if a['split']=='train' else mask['evaluation_eligible'],'interface':mask['interface']})
                    flat_g.extend(g); flat_occ.extend(occ); offset+=nr
                # Undefined same-site entries are explicitly zeroed and marked, never learned.
                interactions[~allowed]=0.; stateout=dest/state; stateout.mkdir()
                np.savez_compressed(stateout/'native_states.npz',g_kbt=np.asarray(flat_g),occupancy=np.asarray(flat_occ,dtype=np.int8),counts=counts,owner=owners,interactions_kbt=interactions)
                atomic_json(stateout/'sites.json',sites)
                states.append({'state':state,'status':'exported','sites':n,'microstates':total,'supervision_sites':sum(s['supervision_eligible'] for s in sites),
                    'max_symmetry_error':symmetry,'source':str(raw),'source_hashes':{name:digest(raw/name) for name in ('mc-energies.json','result.json','mapping.json')},
                    'array_sha256':digest(stateout/'native_states.npz'),'sites_sha256':digest(stateout/'sites.json'),'recovered':replaced})
            except Exception as e:
                import traceback
                states.append({'state':state,'status':'rejected_intermediates','error':repr(e),'traceback':traceback.format_exc(),'source':str(raw)})
        atomic_json(dest/'receipt.json',{'complex_id':cid,'split':a['split'],'component_id':a['component_id'],'states':states,'paired_sites_before':before,'paired_sites_after':after,
            'interface_sites':sum(r['interface'] for r in targets),'unchanged_prediction_rows_checked':unchanged,'source_receipt_sha256':digest(sidepath),'manifest_sha256':digest(out/'manifest.json'),
            'paired_sha256':digest(dest/'paired_sites.parquet') if targets else None})
        print(json.dumps({'complex_id':cid,'pairs':after,'states':Counter(r['status'] for r in states)}),flush=True)

def collect(out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    m=json.loads((out/'manifest.json').read_text()); assert digest(Path(__file__))==m['code_sha256']
    counts=Counter(); statuses=Counter(); paired=[]; rejected=[]; index=[]; unchanged=0
    for a in read_table(out/'assignments.parquet'):
        cid=a['complex_id']; dest=out/'complexes'/cid; rec=json.loads((dest/'receipt.json').read_text()); assert rec['manifest_sha256']==digest(out/'manifest.json')
        for k in ('paired_sites_before','paired_sites_after','interface_sites'): counts[a['split']+':'+k]+=rec[k]
        counts[a['split']+':complexes_with_interface']+=bool(rec['interface_sites']); unchanged+=rec['unchanged_prediction_rows_checked']
        if rec['paired_sha256']:
            assert digest(dest/'paired_sites.parquet')==rec['paired_sha256']; paired.extend(read_table(dest/'paired_sites.parquet'))
        for s in rec['states']:
            statuses[a['split']+':'+s['status']]+=1
            if s['status']=='exported':
                base=dest/s['state']; assert digest(base/'native_states.npz')==s['array_sha256'] and digest(base/'sites.json')==s['sites_sha256']
                index.append({'complex_id':cid,'split':a['split'],'component_id':a['component_id'],**s,'export':str(base)})
            elif s['status']=='rejected_intermediates': rejected.append({'complex_id':cid,**s})
    pq.write_table(pa.Table.from_pylist(paired),out/'paired_sites.parquet'); atomic_json(out/'native_state_index.json',index)
    report={'complete':True,'counts':dict(counts),'state_statuses':dict(statuses),'unchanged_prediction_rows_checked':unchanged,'rejected':rejected,'test_data_included':False,
      'ready_for_native_state_model':not rejected,'ready_for_scalar_binary_pair_model':False,'reason':'Native tautomer states retained; scalar site-pair projection is not validated.',
      'paired_sha256':digest(out/'paired_sites.parquet'),'index_sha256':digest(out/'native_state_index.json')}
    atomic_json(out/'verification.json',report); print(json.dumps({k:v for k,v in report.items() if k!='rejected'},indent=2),flush=True)

if __name__=='__main__':
    require_compute(); p=argparse.ArgumentParser(); p.add_argument('stage',choices=['init','export','collect']); p.add_argument('--out',type=Path,required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=32); a=p.parse_args()
    if a.stage=='init': init(a.out)
    elif a.stage=='export': export(a.out,a.shard,a.shards)
    else: collect(a.out)
