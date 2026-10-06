"""Single-state graph pilot: inherited split/masks, all eligible AB scalar labels."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from pkabench.runtime import atomic_json,digest
from .records import read,KEY


def prepare(source,out):
    from jaxpropka.topology import load_topology
    from jaxpropka.parameters import GROUPS,GROUP_AA
    from pkabench.prep import read_cif
    from pkanet.graph import geometry
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    parent=read(source/'manifest.json'); rows=[]
    for split in ('train','val'):
        for cid in parent[split]:
            recordpath=source/'records'/f'{cid}.json'
            assert digest(recordpath)==parent['records_sha256'][cid]
            r=read(recordpath); state=r['states']['AB']; raw=Path(state['source'])
            folder=out/'data'/cid; folder.mkdir(parents=True,exist_ok=True)
            expected={'record_sha256':digest(recordpath),'source_manifest_sha256':digest(source/'manifest.json')}
            if (folder/'receipt.json').exists():
                receipt=read(folder/'receipt.json')
                assert all(receipt[k]==v for k,v in expected.items())
                assert digest(folder/'graph.npz')==receipt['sha256'];rows.append(receipt);continue
            cif=r['structures']['AB'];assert digest(cif)==r['structure_sha256']['AB']
            for name in ('mapping.json','result.json'):
                assert digest(raw/name)==state['source_hashes'][name]
            sitepath=Path(state['export'])/'sites.json';assert digest(sitepath)==state['sites_sha256']
            masks={tuple(s[k] for k in KEY):s for s in read(sitepath)}
            mapping={(m['chain'],m['resnum']):tuple(m['original']) for m in read(raw/'mapping.json')}
            topology=load_topology(read_cif(cif),gap_policy='cap',freeze_disulfides=True)
            lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(topology.keys)}
            graph,frames_valid=geometry(topology.backbone,topology.chain_index)
            graph['nodes']=np.concatenate((np.eye(20,dtype=np.float32)[topology.native_index],
                np.stack((topology.nterm,topology.cterm,topology.disulfide,frames_valid),axis=-1)),axis=-1).astype(np.float32)
            graph['node_mask']=np.ones(topology.n_residues,bool)
            queries=[]; labels=[]; keys=[]; skipped=Counter(); seen=set()
            for s in read(raw/'result.json')['rows']:
                original=mapping[s['chain'],s['resnum']];key=(cid,*original,s['group'])
                assert key not in seen;seen.add(key)
                if not masks[key]['supervision_eligible']:skipped['quality_mask']+=1;continue
                if s['pka'] is None or not np.isfinite(s['pka']):skipped['no_finite_midpoint']+=1;continue
                assert original in lookup,('unmapped site',key)
                i=lookup[original];g=GROUPS.index(s['group'])
                if g<7:assert topology.native_index[i]==GROUP_AA[g],key
                queries.append((i,g));labels.append(s['pka']);keys.append(list(key))
            assert labels,('no eligible scalar sites',cid)
            query=np.asarray(queries,np.int32)
            graph.update(query_residue=query[:,0],query_group=query[:,1])
            np.savez_compressed(folder/'graph.npz',**graph,labels=np.asarray(labels,np.float32))
            receipt=dict(expected,complex_id=cid,split=split,component_id=r['component_id'],role=r['role'],
                n=topology.n_residues,k=graph['neighbors'].shape[1],q=len(labels),keys=keys,skipped=dict(skipped),
                sha256=digest(folder/'graph.npz'))
            atomic_json(folder/'receipt.json',receipt);rows.append(receipt)
            if len(rows)%25==0:print(json.dumps({'prepared':len(rows),'total':len(parent['train'])+len(parent['val'])}),flush=True)
    assert not {r['component_id'] for r in rows if r['split']=='train'} & {r['component_id'] for r in rows if r['split']=='val'}
    # Three measured capacity buckets; no neighbor truncation, no label-based bucketing.
    capacities={}
    for limit in (384,768,100000):
        members=[r for r in rows if (384 if r['n']<=384 else 768 if r['n']<=768 else 100000)==limit]
        if members:capacities[str(limit)]=[int(np.ceil(max(r[x] for r in members)/32)*32) for x in ('n','k','q')]
    manifest=dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        train=parent['train'],val=parent['val'],records=rows,capacities=capacities,
        config=dict(seed=17,epochs=20,learning_rate=.001,accumulation=8,dtype='float32',objective='single-state scalar pKa MSE',
            state='AB',radius_A=20,paired_weight=0,selection='final epoch; validation is reporting only'),
        label_source='Current native-v2 PypKa, not historical pKPDB',
        scope='All eligible finite AB midpoints; not restricted to interface. Frozen component split and original quality masks. No test data.')
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'preparation.json',dict(passed=True,complexes=len(rows),
        sites={s:sum(r['q'] for r in rows if r['split']==s) for s in ('train','val')},
        exclusions=dict(sum((Counter(r['skipped']) for r in rows),Counter()))))


def pad(graph,labels,capacities):
    n,k,q=capacities; oldn,oldk=graph['neighbors'].shape;oldq=len(labels)
    assert oldn<=n and oldk<=k and oldq<=q
    shape=dict(nodes=(n,24),node_mask=(n,),neighbors=(n,k),edge=(n,k,20),edge_mask=(n,k),switch=(n,k),query_residue=(q,),query_group=(q,))
    result={}
    for name,value in graph.items():
        dest=np.zeros(shape[name],value.dtype)
        dest[tuple(slice(0,s) for s in value.shape)]=value;result[name]=dest
    target=np.zeros(q,np.float32);target[:oldq]=labels
    mask=np.arange(q)<oldq
    return result,target,mask


def load(out,row,capacities):
    path=out/'data'/row['complex_id']/'graph.npz'
    assert digest(path)==row['sha256']
    with np.load(path,allow_pickle=False) as f:
        graph={k:f[k] for k in f.files if k!='labels'};labels=f['labels']
    return pad(graph,labels,capacities)


def bucket(row):return str(384 if row['n']<=384 else 768 if row['n']<=768 else 100000)
