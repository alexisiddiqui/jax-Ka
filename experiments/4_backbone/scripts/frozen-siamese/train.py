import argparse,json,time,zipfile
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,optax
import heads
from heads import init,forward,pack,evaluate
from probe_base import packed_read,packed_write,sample_indices,save_json
ROOT=heads.ROOT;CACHE=heads.CACHE;OUT=ROOT/'audits/frozen-siamese-v1'

def extract(seed,smoke=False):
    from pkatrain.production_train import selected_model,_with_policy,run_dir
    from pkatrain.production_loading import PinderSource
    from pkatrain.production_inputs import BackboneTable
    from probe_base import records
    from pkabench.runtime import digest
    m,rows=records();run=f'control-v2-50-relations-s{seed}';folder=run_dir(ROOT,run)
    selection=json.loads((folder/'selection.json').read_text());checkpoint=folder/'checkpoints'/f"epoch-{selection['selected_epoch']:03d}"/'state.npz'
    before=digest(checkpoint);original=json.loads((CACHE/f'features-s{seed}.json').read_text());assert before==original['checkpoint_sha256']
    params,fn=selected_model(ROOT,run);token_fn=jax.jit(lambda g:fn[0](params,g));source=PinderSource(_with_policy(m,1),fraction=.5,sidechains=BackboneTable(ROOT,'pinder'))
    suffix='-smoke' if smoke else '';dest=OUT/f'bound-s{seed}{suffix}.zip';OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    with zipfile.ZipFile(CACHE/'labels.zip') as lab,zipfile.ZipFile(CACHE/f'features-s{seed}.zip') as free,zipfile.ZipFile(dest.with_suffix('.partial'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        ids=set(free.namelist());rows=sorted([r for r in rows if r['id'] in ids],key=lambda r:r['n'])
        if smoke:rows=[r for r in rows if r['split']=='train'][:8]+[r for r in rows if r['split']=='val'][:8]
        for count,r in enumerate(rows,1):
            a=packed_read(lab,r['id']);graphs,*_=source.load([r['id']]);bound=jax.tree.map(lambda x:jnp.asarray(x[0,0]),graphs)
            tok,q=token_fn(bound);h=np.asarray(tok[q])[a['keep']];assert np.isfinite(h).all()
            if count==1:
                fg=jax.tree.map(lambda x:jnp.asarray(x[0,1]),graphs);ft,fq=token_fn(fg)
                np.testing.assert_allclose(np.asarray(ft[fq])[a['keep']],packed_read(free,r['id'])['h'],atol=2e-5,rtol=2e-5)
            packed_write(z,r['id'],h=h.astype(np.float32))
            if count%500==0:print(json.dumps({'bound_features':count,'seconds':time.time()-start}),flush=True)
    source.close();dest.with_suffix('.partial').replace(dest);assert digest(checkpoint)==before
    save_json(dest.with_suffix('.json'),{'checkpoint_sha256':before,'checkpoint_unchanged':True,'free_cache_alignment_checked':True,'records':len(rows),'seconds':time.time()-start})
    jax.clear_caches()

def loss_terms(zf,zb,y):
    free=optax.sigmoid_binary_cross_entropy(zf,y).mean();bound=optax.sigmoid_binary_cross_entropy(zb,y).mean()
    consistency=jnp.mean((jax.nn.sigmoid(zf)-jax.lax.stop_gradient(jax.nn.sigmoid(zb)))**2)
    return free+bound+consistency,jnp.stack((free,bound,consistency))

def checks():
    heads.OUT=OUT;heads.check()
    zf=jnp.array([-.5,1.]);zb=jnp.array([.2,-.4]);y=jnp.array([0.,1.])
    # Bound branch receives supervised gradients, never consistency-target gradients.
    g=jax.grad(lambda b:loss_terms(zf,b,y)[1][2])(zb);np.testing.assert_array_equal(g,jnp.zeros_like(g))
    np.testing.assert_allclose(loss_terms(zf,zf,y)[1][2],0.,atol=1e-7)
    x=jnp.ones((8,64));c=jnp.array([0]*4+[1]*4);v=jnp.ones(8,bool);idx=jnp.arange(8)
    for kind in ('small','mlp','transformer'):
        p=init(17,kind)
        def obj(p):return loss_terms(forward(p,x,c,v,kind),forward(p,x*2,c,v,kind),idx%2)[0]
        loss,g=jax.value_and_grad(obj)(p);assert np.isfinite(loss) and all(np.isfinite(a).all() for a in jax.tree.leaves(g))
    save_json(OUT/'siamese-checks.json',{'bound_consistency_target_detached':True,'identical_tokens_zero_consistency':True,'all_heads_finite_gradients':True})

def run(seed,smoke=False):
    suffix='-smoke' if smoke else '';rows=[]
    with zipfile.ZipFile(CACHE/'labels.zip') as lab,zipfile.ZipFile(CACHE/f'features-s{seed}.zip') as free,zipfile.ZipFile(OUT/f'bound-s{seed}{suffix}.zip') as bound:
        available=set(bound.namelist())
        for cid in free.namelist():
            if cid not in available:continue
            a=packed_read(lab,cid);hf=packed_read(free,cid)['h'];hb=packed_read(bound,cid)['h']
            assert hf.shape==hb.shape
            if len(hf)>1 and len(np.unique(a['chain']))>1:rows.append((str(a['split']),cid,*pack(a,hf),pack(a,hb)[0]))
    training=[r[1:] for r in rows if r[0]=='train'];validation=[r[1:] for r in rows if r[0]=='val'];del rows
    assert training and validation
    freeval=[r[:-1] for r in validation];boundval=[(cid,xb,c,v,y) for cid,xf,c,v,y,xb in validation]
    provenance=json.loads((CACHE/f'features-s{seed}.json').read_text());boundprov=json.loads((OUT/f'bound-s{seed}{suffix}.json').read_text())
    assert provenance['checkpoint_sha256']==boundprov['checkpoint_sha256']
    for kind in ('small','mlp','transformer'):
        dest=OUT/f's{seed}{suffix}'/kind;dest.mkdir(parents=True,exist_ok=True)
        p=init(seed,kind);optimizer=optax.adamw(1e-3,weight_decay=1e-4);state=optimizer.init(p)
        def objective(p,xf,xb,c,v,y,idx,key):
            # Same dropout mask on the two branches keeps consistency about input, not dropout noise.
            zf=forward(p,xf,c,v,kind,key)[idx];zb=forward(p,xb,c,v,kind,key)[idx]
            return loss_terms(zf,zb,y[idx])
        @jax.jit
        def step(p,state,xf,xb,c,v,y,idx,key):
            (loss,terms),g=jax.value_and_grad(objective,has_aux=True)(p,xf,xb,c,v,y,idx,key)
            updates,state=optimizer.update(g,state,p);return optax.apply_updates(p,updates),state,loss,terms
        predict=jax.jit(lambda p,x,c,v:jax.nn.sigmoid(forward(p,x,c,v,kind)))
        rng=np.random.default_rng(seed);key=jax.random.PRNGKey(seed);best=-1.;history=[];start=time.time()
        protocol={'seed':seed,'head':kind,'parameters':sum(a.size for a in jax.tree.leaves(p)),'epochs':2 if smoke else 20,'loss':'BCE(free,y) + BCE(bound,y) + mean((sigmoid(free)-stopgrad(sigmoid(bound)))^2)','weights':[1,1,1],'shared_head':True,'shared_branch_dropout_mask':True,'dropout':0 if kind=='small' else .1,'optimizer':'AdamW lr .001 weight_decay .0001','sampling':'256 class-balanced output sites per complex, same as free-only probes','selection':'free-only validation macro site AP','training_complexes':len(training),'validation_complexes':len(validation),'cached_backbone':provenance,'bound_cache':boundprov,'test_data_included':False,'smoke':smoke}
        save_json(dest/'protocol.json',protocol)
        for epoch in range(1,protocol['epochs']+1):
            sums=np.zeros(3)
            for ix in rng.permutation(len(training)):
                _,xf,c,v,y,xb=training[ix];idx=sample_indices(y[:int(v.sum())],rng);key,sub=jax.random.split(key)
                p,state,loss,terms=step(p,state,xf,xb,c,v,y,idx,sub);sums+=np.asarray(terms)
                if not np.isfinite(loss):raise ValueError('nonfinite loss')
            metrics,detail=evaluate(p,freeval,predict);bm,_=evaluate(p,boundval,predict)
            record={'epoch':epoch,'mean_loss_terms':(sums/len(training)).tolist(),**metrics,'bound_validation':bm,'seconds':time.time()-start};history.append(record)
            print(json.dumps({'seed':seed,'head':kind,**record}),flush=True)
            if metrics['site_macro_ap']>best:
                best=metrics['site_macro_ap'];np.savez(dest/'params.npz',**{k:np.asarray(a) for k,a in p.items()});save_json(dest/'selection.json',record);save_json(dest/'per-complex.json',detail)
            save_json(dest/'history.json',history)
        jax.clear_caches()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:checks()
    extract(a.seed,a.smoke);run(a.seed,a.smoke)
