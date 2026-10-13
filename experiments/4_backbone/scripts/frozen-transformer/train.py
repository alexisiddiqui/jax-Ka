import argparse,json,os,time,zipfile,hashlib
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,optax
from probe_base import packed_read,candidates,sample_indices,binary_metrics,save_json
ROOT=Path(os.environ['PKABENCH_RUNTIME']);CACHE=ROOT/'audits/frozen-interface-v1';OUT=ROOT/'audits/frozen-transformer-v1'

def init(seed,kind):
    keys=iter(jax.random.split(jax.random.PRNGKey(seed),12));p={}
    def dense(name,a,b):p[name+'w']=jax.random.normal(next(keys),(a,b))/np.sqrt(a);p[name+'b']=jnp.zeros(b)
    if kind=='mlp':dense('up',64,510);dense('out',510,1)
    else:
        for name in ('q','k','v','o'):dense(name,64,64)
        dense('up',64,128);dense('down',128,64);dense('out',64,1)
        for name in ('ln1','ln2','ln3'):p[name+'g']=jnp.ones(64);p[name+'b']=jnp.zeros(64)
    return p

def forward(p,x,chain,valid,kind,key=None):
    def norm(x):return (x-x.mean(-1,keepdims=True))*jax.lax.rsqrt(x.var(-1,keepdims=True)+1e-5)
    def linear(x,n):return x@p[n+'w']+p[n+'b']
    def ln(x,n):return norm(x)*p[n+'g']+p[n+'b']
    keys=iter(jax.random.split(key,3)) if key is not None else None
    def drop(x):return x if keys is None else x*jax.random.bernoulli(next(keys),.9,x.shape)/.9
    x=norm(x)*valid[:,None]
    if kind=='mlp':return linear(drop(jax.nn.gelu(linear(x,'up'))),'out')[:,0]*valid
    c=ln(x,'ln1');n=len(x)
    q,k,v=[linear(c,name).reshape(n,4,16) for name in ('q','k','v')]
    scores=jnp.einsum('ihd,jhd->hij',q,k,precision=jax.lax.Precision.HIGHEST)/4
    mask=valid[:,None]&valid[None,:]&(chain[:,None]==chain[None,:])
    a=jax.nn.softmax(jnp.where(mask[None],scores,-1e9),axis=-1)*mask[None]
    msg=jnp.einsum('hij,jhd->ihd',a,v,precision=jax.lax.Precision.HIGHEST).reshape(n,64)
    x=(x+drop(linear(msg,'o')))*valid[:,None]
    x=(x+drop(linear(drop(jax.nn.gelu(linear(ln(x,'ln2'),'up'))),'down')))*valid[:,None]
    return linear(ln(x,'ln3'),'out')[:,0]*valid

def pack(a,h):
    n=len(h);bucket=next(b for b in (32,64,128,256,512,1024,2048) if b>=n)
    x=np.zeros((bucket,64),np.float32);x[:n]=h
    chain=np.full(bucket,-1,np.int32);chain[:n]=np.unique(a['chain'],return_inverse=True)[1]
    valid=np.arange(bucket)<n;y=np.zeros(bucket,np.float32);y[:n]=candidates(a)[3]
    return x,chain,valid,y

def check():
    rng=np.random.default_rng(7);x=jnp.asarray(rng.normal(size=(8,64)),jnp.float32);c=jnp.array([0,0,0,0,1,1,1,1]);v=jnp.ones(8,bool)
    p=init(17,'transformer');fn=jax.jit(lambda x,c,v:forward(p,x,c,v,'transformer'))
    base=fn(x,c,v);changed=fn(x.at[4:].set(x[4:]*3+10),c,v)
    np.testing.assert_allclose(base[:4],changed[:4],atol=1e-6)
    grad=jax.grad(lambda x:forward(p,x,c,v,'transformer')[0])(x);np.testing.assert_array_equal(grad[4:],jnp.zeros_like(grad[4:]))
    perm=np.array([7,2,0,5,4,3,6,1]);np.testing.assert_allclose(fn(x[perm],c[perm],v),base[perm],atol=2e-5)
    np.testing.assert_allclose(fn(x,c+37,v),base,atol=1e-6)
    xp=jnp.pad(x,((0,8),(0,0)));cp=jnp.pad(c,(0,8),constant_values=-1);vp=jnp.arange(16)<8
    np.testing.assert_allclose(fn(xp,cp,vp)[:8],base,atol=2e-5)
    for kind in ('transformer','mlp'):
        pp=init(17,kind)
        loss,g=jax.value_and_grad(lambda p:jnp.mean(optax.sigmoid_binary_cross_entropy(forward(p,x,c,v,kind,jax.random.PRNGKey(1)),jnp.arange(8)%2)))(pp)
        assert np.isfinite(loss) and all(np.isfinite(z).all() for z in jax.tree.leaves(g))
    result={'free_chain_isolation':True,'cross_chain_gradient_zero':True,'padding_invariance':True,'permutation_equivariance':True,'chain_renaming_invariance':True,'finite_dropout_gradients':True,'parameters':{k:sum(a.size for a in jax.tree.leaves(init(17,k))) for k in ('transformer','mlp')}}
    save_json(OUT/'checks.json',result);print(json.dumps(result),flush=True)

def evaluate(p,rows,predict):
    ys=[];ps=[];aps=[];detail=[]
    for cid,x,c,v,y in rows:
        n=int(v.sum());pred=np.asarray(predict(p,x,c,v))[:n];label=y[:n];metric=binary_metrics(label,pred)
        detail.append({'id':cid,**metric});ys.append(label);ps.append(pred)
        if metric['average_precision'] is not None:aps.append(metric['average_precision'])
    return {'site':binary_metrics(np.concatenate(ys),np.concatenate(ps)),'site_macro_ap':float(np.mean(aps)),'site_macro_ap_complexes':len(aps),'complexes':len(rows)},detail

def run(seed,smoke=False):
    rows=[];start=time.time()
    with zipfile.ZipFile(CACHE/'labels.zip') as lab,zipfile.ZipFile(CACHE/f'features-s{seed}.zip') as feat:
        for cid in feat.namelist():
            a=packed_read(lab,cid);h=packed_read(feat,cid)['h']
            if len(h)>1 and len(np.unique(a['chain']))>1:rows.append((str(a['split']),cid,*pack(a,h)))
    training=[r[1:] for r in rows if r[0]=='train'];validation=[r[1:] for r in rows if r[0]=='val'];del rows
    if smoke:training=training[:8];validation=sorted(validation,key=lambda r:-r[-1].sum())[:3]
    provenance=json.loads((CACHE/f'features-s{seed}.json').read_text())
    assert provenance['checkpoint_unchanged'] and provenance['free_partner_isolation']
    print(json.dumps({'seed':seed,'loaded_train':len(training),'loaded_val':len(validation),'load_s':time.time()-start}),flush=True)
    for kind in ('transformer','mlp'):
        dest=OUT/f's{seed}{"-smoke" if smoke else ""}'/kind;dest.mkdir(parents=True,exist_ok=True)
        p=init(seed,kind);optimizer=optax.adamw(1e-3,weight_decay=1e-4);state=optimizer.init(p)
        def objective(p,x,c,v,y,idx,key):return optax.sigmoid_binary_cross_entropy(forward(p,x,c,v,kind,key)[idx],y[idx]).mean()
        @jax.jit
        def step(p,state,x,c,v,y,idx,key):
            loss,g=jax.value_and_grad(objective)(p,x,c,v,y,idx,key);updates,state=optimizer.update(g,state,p);return optax.apply_updates(p,updates),state,loss
        predict=jax.jit(lambda p,x,c,v:jax.nn.sigmoid(forward(p,x,c,v,kind)))
        rng=np.random.default_rng(seed);best=-1.;history=[];key=jax.random.PRNGKey(seed);start=time.time()
        protocol={'seed':seed,'kind':kind,'parameters':sum(z.size for z in jax.tree.leaves(p)),'width':64,'heads':4 if kind=='transformer' else 0,'ff_width':128 if kind=='transformer' else 510,'dropout':.1,'context':'all cached eligible surface site tokens on own chain only','sampling':'same 256 class-balanced sites per complex as prior probe; all context retained','optimizer':'AdamW lr .001 weight_decay .0001','epochs':2 if smoke else 20,'selection':'validation macro site AP on positive-containing complexes','training_complexes':len(training),'validation_complexes':len(validation),'cached_backbone':provenance,'test_data_included':False,'smoke':smoke}
        save_json(dest/'protocol.json',protocol)
        for epoch in range(1,protocol['epochs']+1):
            total=0.
            for ix in rng.permutation(len(training)):
                _,x,c,v,y=training[ix];idx=sample_indices(y[:int(v.sum())],rng);key,sub=jax.random.split(key)
                p,state,loss=step(p,state,x,c,v,y,idx,sub);total+=float(loss)
                if not np.isfinite(loss):raise ValueError('nonfinite loss')
            metrics,detail=evaluate(p,validation,predict);record={'epoch':epoch,'train_loss':total/len(training),**metrics,'seconds':time.time()-start};history.append(record)
            print(json.dumps({'seed':seed,'head':kind,**record}),flush=True)
            if metrics['site_macro_ap']>best:
                best=metrics['site_macro_ap'];np.savez(dest/'params.npz',**{k:np.asarray(z) for k,z in p.items()});save_json(dest/'selection.json',record);save_json(dest/'per-complex.json',detail)
            save_json(dest/'history.json',history)
        jax.clear_caches()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:check()
    run(a.seed,a.smoke)
