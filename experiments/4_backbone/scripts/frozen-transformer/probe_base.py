"""Frozen relational GQT: own-free site head and symmetric free-token pair head."""
import argparse, hashlib, io, json, os, time, zipfile
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('PKABENCH_RUNTIME','.'))
OUT=ROOT/'audits/frozen-interface-v1'

def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def packed_write(z,name,**arrays):
    b=io.BytesIO();np.savez(b,**arrays);z.writestr(name,b.getvalue())

def packed_read(z,name):
    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as f:return dict(f)

def records():
    from pkatrain.production_graphs import output,read
    from pkatrain.production_loading import select
    m=read(output(ROOT,'pinder')/'manifest-v1.json')
    return m, select(m,'train',.5)+select(m,'val')

def label_one(arg):
    cid,split,store_path=arg
    from scipy.spatial import cKDTree
    from jaxpropka.topology import load_topology
    from pkatrain.production_graphs import ProductionStore,source,PINDER
    from pkatrain.gqt_paired_pinder import _read_cif_gz
    store=ProductionStore(Path(store_path));raw=store.raw(cid);store.close()
    atoms=_read_cif_gz(source(ROOT,f'{PINDER}/entries/{cid}')/'AB.cif.gz')
    keys=load_topology(atoms,gap_policy='cap',freeze_disulfides=True).keys
    heavy=atoms[(atoms.element!='H')&(atoms.element!='D')&~atoms.hetero]
    atomkeys=list(zip(heavy.chain_id.astype(str),heavy.res_id.astype(int),np.char.strip(heavy.ins_code.astype(str))))
    qkeys=[(keys[int(i)].chain,keys[int(i)].number,keys[int(i)].insertion) for i in raw['query_residue']]
    # Free-state RSA gate is fixed in advance; bound coordinates are labels only.
    eligible=(raw['rsa_free']>=.1)&np.isfinite(raw['targets']).all(axis=0)&(raw['query_site']>=0)
    keep=np.flatnonzero(eligible);lookup={}
    for idx,q in enumerate(keep):lookup.setdefault(qkeys[q],[]).append(idx)
    chain=np.array([qkeys[q][0] for q in keep]);positives=set();tree=cKDTree(heavy.coord)
    own_atoms={}
    for a,k in enumerate(atomkeys):
        if k in lookup:own_atoms.setdefault(k,[]).append(a)
    for k,indices in lookup.items():
        aa=own_atoms.get(k,[])
        if not aa:raise ValueError((cid,'missing labelled atoms',k))
        for near in tree.query_ball_point(heavy.coord[aa],10.):
            for atom in near:
                partner=atomkeys[atom]
                if partner[0]==k[0]:continue
                for i in indices:
                    for j in lookup.get(partner,[]):positives.add((min(i,j),max(i,j)))
    pairs=np.asarray(sorted(positives),np.int32).reshape(-1,2)
    return cid,dict(keep=keep.astype(np.int32),chain=chain,positive=pairs,split=np.array(split))

def labels(limit=None):
    from concurrent.futures import ProcessPoolExecutor
    m,rows=records();rows=rows[:limit] if limit else rows
    OUT.mkdir(parents=True,exist_ok=True);dest=OUT/('labels-smoke.zip' if limit else 'labels.zip');t=time.time();counts={'structures':0,'sites':0,'positive_pairs':0}
    with ProcessPoolExecutor(max_workers=16) as pool,zipfile.ZipFile(dest.with_suffix('.partial'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for cid,a in pool.map(label_one,[(r['id'],r['split'],m['store']) for r in rows],chunksize=8):
            packed_write(z,cid,**a);counts['structures']+=1;counts['sites']+=len(a['keep']);counts['positive_pairs']+=len(a['positive'])
            if counts['structures']%200==0:print(json.dumps({**counts,'seconds':time.time()-t}),flush=True)
    dest.with_suffix('.partial').replace(dest)
    counts.update(seconds=time.time()-t,surface_rsa_min=.1,contact_distance_A=10.,fraction=.5,test_data_included=False)
    save_json(dest.with_suffix('.json'),counts);print(json.dumps(counts),flush=True)

def candidates(a):
    n=len(a['chain']);i,j=np.where(np.triu(a['chain'][:,None]!=a['chain'][None,:],1))
    y=np.zeros((n,n),bool)
    if len(a['positive']):y[tuple(a['positive'].T)]=True;y|=y.T
    return i.astype(np.int32),j.astype(np.int32),y[i,j].astype(np.float32),y.any(axis=1).astype(np.float32)

def extract(seed,smoke=False):
    import jax
    import jax.numpy as jnp
    from pkatrain.production_train import selected_model,_with_policy,run_dir
    from pkatrain.production_loading import PinderSource
    from pkatrain.production_inputs import BackboneTable
    from pkatrain.production_graphs import read
    from pkabench.runtime import digest
    m,rows=records();run=f'control-v2-50-relations-s{seed}';folder=run_dir(ROOT,run);selection=read(folder/'selection.json')
    checkpoint=folder/'checkpoints'/f"epoch-{selection['selected_epoch']:03d}"/'state.npz'
    before=digest(checkpoint);params,fn=selected_model(ROOT,run)
    token_fn=jax.jit(lambda g:fn[0](params,g))
    source=PinderSource(_with_policy(m,1),fraction=.5,sidechains=BackboneTable(ROOT,'pinder'))
    suffix='-smoke' if smoke else '';labelpath=OUT/f'labels{suffix}.zip';dest=OUT/f'features-s{seed}{suffix}.zip';start=time.time();count=0
    with zipfile.ZipFile(labelpath) as labels,zipfile.ZipFile(dest.with_suffix('.partial'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        ids=set(labels.namelist());rows=sorted([r for r in rows if r['id'] in ids],key=lambda r:r['n'])
        for r in rows:
            a=packed_read(labels,r['id']);graphs,*_=source.load([r['id']]);free=jax.tree.map(lambda x:jnp.asarray(x[0,1]),graphs)
            tok,q=token_fn(free);h=np.asarray(tok[q])[a['keep']]
            assert np.isfinite(h).all()
            if count==0:
                # Physical isolation: change every partner node and its coordinates, preserving own branch.
                chain=np.asarray(free['nodes'][:,-2]);valid=np.asarray(free['node_mask'],bool);first=chain[0];partner=valid&(chain!=first)
                changed={**free,'nodes':free['nodes'].at[partner,:24].add(3).at[partner,-14:-2].add(100)}
                t2,q2=token_fn(changed);own=(chain[np.asarray(free['query_residue'])]==first)&(np.asarray(free['query_site'])>=0)
                np.testing.assert_allclose(np.asarray(tok[q])[own],np.asarray(t2[q2])[own],atol=2e-5,rtol=2e-5)
                print('PASS frozen free-token partner isolation',flush=True)
            packed_write(z,r['id'],h=h.astype(np.float32));count+=1
            if count%200==0:print(json.dumps({'features':count,'seconds':time.time()-start,'seed':seed}),flush=True)
    source.close();dest.with_suffix('.partial').replace(dest);assert digest(checkpoint)==before
    save_json(dest.with_suffix('.json'),{'seed':seed,'run':run,'selected_epoch':selection['selected_epoch'],'checkpoint_sha256':before,'checkpoint_unchanged':True,'free_partner_isolation':True,'structures':count,'seconds':time.time()-start,'test_data_included':False})

def initialize(seed,kind):
    import jax,jax.numpy as jnp
    k1,k2=jax.random.split(jax.random.PRNGKey(seed));d=64 if kind=='site' else 128
    return {'w1':jax.random.normal(k1,(d,64))*jnp.sqrt(2/d),'b1':jnp.zeros(64),'w2':jax.random.normal(k2,(64,1))/8,'b2':jnp.zeros(1)}

def logits(p,a,b,kind):
    import jax,jax.numpy as jnp
    def norm(x):return (x-x.mean(-1,keepdims=True))/jnp.sqrt(x.var(-1,keepdims=True)+1e-5)
    a=norm(a)
    if kind=='pair':
        b=norm(b);a=jnp.concatenate((a*b,jnp.abs(a-b)),axis=-1)
    return (jax.nn.gelu(a@p['w1']+p['b1'])@p['w2']+p['b2'])[...,0]

def sample_indices(y,rng,n=256):
    pos=np.flatnonzero(y>0);neg=np.flatnonzero(y==0)
    if len(pos) and len(neg):return np.concatenate((rng.choice(pos,n//2),rng.choice(neg,n-n//2)))
    return rng.choice(len(y),n)

def binary_metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);assert len(y)>0 and np.isfinite(p).all()
    order=np.argsort(-p,kind='stable');yy=y[order];pp=p[order]
    ends=np.r_[np.flatnonzero(np.diff(pp)!=0),len(pp)-1]
    tp=np.cumsum(yy)[ends];fp=ends+1-tp;positives=float(y.sum());negatives=len(y)-positives
    ap=float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/positives) if positives else None
    if positives and negatives:
        tpr=np.r_[0,tp/positives];fpr=np.r_[0,fp/negatives]
        auc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    else:auc=None
    return {'n':len(y),'positives':int(positives),'average_precision':ap,'auroc':auc,'brier':float(np.mean((y-p)**2))}

def evaluate(p,kind,rows,predict):
    all_y=[];all_p=[];aps=[];edge_y=[];edge_p=[];edge_aps=[];per_complex=[]
    for cid,a,h in rows:
        i,j,y,site=candidates(a)
        if len(i)==0:continue
        if kind=='site':
            padded=np.zeros((2048,64),np.float32)
            assert len(h)<=2048
            padded[:len(h)]=h;score=np.asarray(predict(p,padded,padded))[:len(h)]
        else:
            scores=[]
            for st in range(0,len(i),4096):
                ix=i[st:st+4096];jx=j[st:st+4096];size=len(ix)
                aa=np.zeros((4096,64),np.float32);bb=aa.copy();aa[:size]=h[ix];bb[:size]=h[jx]
                scores.append(np.asarray(predict(p,aa,bb))[:size])
            ep=np.concatenate(scores);score=np.zeros(len(h),np.float32);np.maximum.at(score,i,ep);np.maximum.at(score,j,ep)
            em=binary_metrics(y,ep);edge_y.append(y);edge_p.append(ep)
            if em['average_precision'] is not None:edge_aps.append(em['average_precision'])
        sm=binary_metrics(site,score);per_complex.append({'id':cid,**sm})
        if sm['average_precision'] is not None:aps.append(sm['average_precision'])
        all_y.append(site);all_p.append(score)
    result={'site':binary_metrics(np.concatenate(all_y),np.concatenate(all_p)),'site_macro_ap':float(np.mean(aps)),
            'site_macro_ap_complexes':len(aps),'complexes':len(per_complex)}
    if kind=='pair':result.update(pair=binary_metrics(np.concatenate(edge_y),np.concatenate(edge_p)),pair_macro_ap=float(np.mean(edge_aps)))
    return result,per_complex

def train(seed,smoke=False):
    import jax,jax.numpy as jnp,optax
    suffix='-smoke' if smoke else '';rows=[]
    with zipfile.ZipFile(OUT/f'labels{suffix}.zip') as lab,zipfile.ZipFile(OUT/f'features-s{seed}{suffix}.zip') as feat:
        for cid in feat.namelist():
            a=packed_read(lab,cid);h=packed_read(feat,cid)['h']
            if len(h)>1 and len(np.unique(a['chain']))>1:rows.append((cid,a,h))
    training=[r for r in rows if str(r[1]['split'])=='train'];validation=[r for r in rows if str(r[1]['split'])=='val']
    if smoke:validation=training[:2]
    assert training and validation
    for kind in ('site','pair'):
        dest=OUT/f's{seed}{suffix}'/kind;dest.mkdir(parents=True,exist_ok=True)
        p=initialize(seed,kind);optimizer=optax.adamw(1e-3,weight_decay=1e-4);state=optimizer.init(p)
        def objective(p,a,b,y):return optax.sigmoid_binary_cross_entropy(logits(p,a,b,kind),y).mean()
        @jax.jit
        def step(p,state,a,b,y):
            loss,grad=jax.value_and_grad(objective)(p,a,b,y);updates,state=optimizer.update(grad,state,p);return optax.apply_updates(p,updates),state,loss
        predict=jax.jit(lambda p,a,b:jax.nn.sigmoid(logits(p,a,b,kind)))
        test=np.random.default_rng(seed).normal(size=(8,64)).astype(np.float32)
        if kind=='pair':np.testing.assert_allclose(predict(p,test,test[::-1]),predict(p,test[::-1],test),atol=1e-6)
        rng=np.random.default_rng(seed);best=-1.;history=[];start=time.time()
        for epoch in range(1,(2 if smoke else 20)+1):
            losses=[]
            for ix in rng.permutation(len(training)):
                _,a,h=training[ix];i,j,y,site=candidates(a)
                if kind=='site':idx=sample_indices(site,rng);aa=h[idx];bb=aa;target=site[idx]
                else:idx=sample_indices(y,rng);aa=h[i[idx]];bb=h[j[idx]];target=y[idx]
                p,state,loss=step(p,state,aa,bb,target);losses.append(float(loss))
                if not np.isfinite(loss):raise ValueError('nonfinite probe loss')
            metrics,detail=evaluate(p,kind,validation,predict)
            record={'epoch':epoch,'train_loss':float(np.mean(losses)),**metrics,'seconds':time.time()-start};history.append(record)
            print(json.dumps({'seed':seed,'head':kind,**record}),flush=True)
            if metrics['site_macro_ap']>best:
                best=metrics['site_macro_ap'];np.savez(dest/'params.npz',**{k:np.asarray(v) for k,v in p.items()})
                save_json(dest/'selection.json',record);save_json(dest/'per-complex.json',detail)
            save_json(dest/'history.json',history)
        save_json(dest/'protocol.json',{'seed':seed,'frozen_backbone':f'control-v2-50-relations-s{seed}','fraction':.5,'head':kind,'epochs':len(history),'training_complexes':len(training),'validation_complexes':len(validation),'target':'any 10A heavy-atom contact to an eligible pKa-labelled partner site','surface':'free RSA >= 0.1','sampler':'256 examples per complex per epoch, equally split positive/negative when both exist','selection':'validation macro site AP among positive-containing complexes','aggregation':'max pair probability over eligible partner sites' if kind=='pair' else 'direct site probability','free_definition':'partner-removed bound conformation, not independently determined unbound structure','test_data_included':False,'backbone_updates':False,'smoke':smoke})

def main():
    assert binary_metrics([0,1],[.5,.5])['average_precision']==.5
    assert binary_metrics([0,1],[.5,.5])['auroc']==.5
    assert binary_metrics([0,1],[0,1])['auroc']==1.
    assert binary_metrics([0,1],[1,0])['auroc']==0.
    p=argparse.ArgumentParser();p.add_argument('action',choices=['labels','run']);p.add_argument('--seed',type=int,default=17);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.action=='labels':labels(8 if a.smoke else None)
    else:extract(a.seed,a.smoke);train(a.seed,a.smoke)
if __name__=='__main__':main()
