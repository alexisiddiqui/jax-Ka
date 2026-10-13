import argparse,json,time,zipfile,hashlib
import numpy as np
import jax,jax.numpy as jnp,optax
from heads import ROOT,CACHE,init,forward,pack,evaluate
from probe_base import packed_read,sample_indices,save_json
BOUND=ROOT/'audits/frozen-siamese-v1';OUT=ROOT/'audits/frozen-distill-v1'

def fingerprint(p):
    h=hashlib.sha256()
    for k in sorted(p):h.update(k.encode());h.update(np.asarray(p[k]).tobytes())
    return h.hexdigest()

def objective(z,y,teacher,weight):
    ce=optax.sigmoid_binary_cross_entropy(z,y).mean()
    kd=jnp.mean((jax.nn.sigmoid(z)-jax.lax.stop_gradient(teacher))**2)
    return ce+weight*kd,jnp.stack((ce,kd))

def check():
    z=jnp.array([-.5,.8]);y=jnp.array([0.,1.]);t=jnp.array([.3,.7])
    np.testing.assert_array_equal(jax.grad(lambda t:objective(z,y,t,1.)[0])(t),jnp.zeros_like(t))
    np.testing.assert_allclose(jax.grad(lambda z:objective(z,y,t,0.)[0])(z),jax.grad(lambda z:optax.sigmoid_binary_cross_entropy(z,y).mean())(z),atol=1e-7)
    np.testing.assert_allclose(objective(z,y,jax.nn.sigmoid(z),1.)[1][1],0.,atol=1e-7)
    save_json(OUT/'checks.json',{'teacher_gradient_zero':True,'weight_zero_matches_labels_only_gradient':True,'matching_probabilities_zero_distillation':True})

predict=jax.jit(lambda p,x,c,v:jax.nn.sigmoid(forward(p,x,c,v,'small')))

def fit(seed,name,training,validation,weight=0.,targets=None,smoke=False):
    dest=OUT/f's{seed}{"-smoke" if smoke else ""}'/name;dest.mkdir(parents=True,exist_ok=True)
    p=init(seed,'small');optimizer=optax.adamw(1e-3,weight_decay=1e-4);state=optimizer.init(p)
    def loss(p,x,c,v,y,idx,teacher):return objective(forward(p,x,c,v,'small')[idx],y[idx],teacher[idx],weight)
    @jax.jit
    def step(p,state,x,c,v,y,idx,teacher):
        (value,terms),g=jax.value_and_grad(loss,has_aux=True)(p,x,c,v,y,idx,teacher)
        updates,state=optimizer.update(g,state,p);return optax.apply_updates(p,updates),state,value,terms
    rng=np.random.default_rng(seed);best=-1.;history=[];start=time.time();epochs=2 if smoke else 20
    save_json(dest/'protocol.json',{'seed':seed,'head':'small MLP','parameters':4225,'role':name,'lambda':weight,'loss':'BCE(student,labels) + lambda*MSE(sigmoid(student),detached fixed teacher probabilities)','epochs':epochs,'lr':.001,'weight_decay':.0001,'dropout':0,'training_complexes':len(training),'validation_complexes':len(validation),'selection':'bound macro AP for teacher; free macro AP for students','backbone_frozen':True,'test_data_included':False,'smoke':smoke})
    for epoch in range(1,epochs+1):
        sums=np.zeros(2)
        for ix in rng.permutation(len(training)):
            cid,x,c,v,y=training[ix];idx=sample_indices(y[:int(v.sum())],rng);teacher=np.zeros_like(y) if targets is None else targets[cid]
            p,state,value,terms=step(p,state,x,c,v,y,idx,teacher);sums+=np.asarray(terms)
            if not np.isfinite(value):raise ValueError('nonfinite loss')
        metrics,detail=evaluate(p,validation,predict);record={'epoch':epoch,'mean_loss_terms':(sums/len(training)).tolist(),**metrics,'seconds':time.time()-start};history.append(record)
        print(json.dumps({'seed':seed,'role':name,**record}),flush=True)
        if metrics['site_macro_ap']>best:
            best=metrics['site_macro_ap'];np.savez(dest/'params.npz',**{k:np.asarray(v) for k,v in p.items()});save_json(dest/'selection.json',record);save_json(dest/'per-complex.json',detail)
        save_json(dest/'history.json',history)
    with np.load(dest/'params.npz') as f:chosen={k:jnp.asarray(f[k]) for k in f.files}
    return chosen

def run(seed,smoke=False):
    rows=[]
    with zipfile.ZipFile(CACHE/'labels.zip') as lab,zipfile.ZipFile(CACHE/f'features-s{seed}.zip') as free,zipfile.ZipFile(BOUND/f'bound-s{seed}.zip') as bound:
        assert set(free.namelist())==set(bound.namelist())
        for cid in free.namelist():
            a=packed_read(lab,cid);hf=packed_read(free,cid)['h'];hb=packed_read(bound,cid)['h'];assert hf.shape==hb.shape
            if len(hf)>1 and len(np.unique(a['chain']))>1:rows.append((str(a['split']),cid,*pack(a,hf),pack(a,hb)[0]))
    training=[r[1:] for r in rows if r[0]=='train'];validation=[r[1:] for r in rows if r[0]=='val'];del rows
    if smoke:training=training[:8];validation=sorted(validation,key=lambda r:-r[-2].sum())[:3]
    ft=[r[:-1] for r in training];fv=[r[:-1] for r in validation]
    bt=[(cid,xb,c,v,y) for cid,xf,c,v,y,xb in training];bv=[(cid,xb,c,v,y) for cid,xf,c,v,y,xb in validation]
    freeprov=json.loads((CACHE/f'features-s{seed}.json').read_text());boundprov=json.loads((BOUND/f'bound-s{seed}.json').read_text());assert freeprov['checkpoint_sha256']==boundprov['checkpoint_sha256']
    dest=OUT/f's{seed}{"-smoke" if smoke else ""}';save_json(dest/'backbone.json',{'free':freeprov,'bound':boundprov})
    teacher=fit(seed,'teacher',bt,bv,smoke=smoke);teacher_hash=fingerprint(teacher)
    # Produce fixed teaching targets on training data only; validation labels never become targets.
    targets={cid:np.asarray(predict(teacher,x,c,v)) for cid,x,c,v,y in bt}
    for weight in (.1,1.):
        fit(seed,f'student-lambda-{weight:g}',ft,fv,weight,targets,smoke)
        assert fingerprint(teacher)==teacher_hash
    save_json(dest/'verification.json',{'teacher_parameters_unchanged_during_student_training':True,'teacher_sha256':teacher_hash,'teaching_targets_training_only':True,'free_only_student_evaluation':True,'backbone_frozen':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:check()
    run(a.seed,a.smoke)
