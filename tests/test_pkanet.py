import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pkanet.graph import geometry
from pkanet.model import initialize,predict
from pkatrain.graph_data import pad
from pkatrain.trainer import ScalarEngine,save_checkpoint,load_checkpoint


def graph(backbone=None):
    rng=np.random.default_rng(42)
    if backbone is None:
        ca=rng.normal(size=(12,3))*3
        backbone=np.stack((ca+[-1,1,0],ca,ca+[1,0,0],ca+[1,1,0]),axis=1)
    g,valid=geometry(backbone,np.repeat([0,1],6))
    g.update(nodes=np.concatenate((np.eye(20,dtype=np.float32)[np.arange(12)],np.zeros((12,3)),valid[:,None]),axis=1).astype(np.float32),
        node_mask=np.ones(12,bool),query_residue=np.array([1,2,4,6],np.int32),query_group=np.array([0,1,2,3],np.int32))
    return g,backbone


def test_rigid_motion_padding_and_permutation():
    p=initialize(jax.random.PRNGKey(17));g,bb=graph();fn=jax.jit(predict)
    y=np.asarray(fn(p,g));rotation=np.linalg.qr(np.random.default_rng(3).normal(size=(3,3)))[0]
    rotated,_=graph(bb@rotation+np.array([7,-4,12]));np.testing.assert_allclose(fn(p,rotated),y,atol=3e-6)
    padded,_,_=pad(g,np.zeros(4),(32,32,16));np.testing.assert_allclose(fn(p,padded)[:4],y,atol=3e-6)
    order=np.array([7,2,9,0,11,3,8,5,1,4,10,6]);inverse=np.argsort(order)
    perm={k:(v[order] if k not in ('query_residue','query_group') else v) for k,v in g.items()}
    perm['neighbors']=inverse[g['neighbors'][order]];perm['query_residue']=inverse[g['query_residue']]
    np.testing.assert_allclose(fn(p,perm),y,atol=3e-6)


def test_train_gradient_checkpoint_and_mask(tmp_path):
    p=initialize(jax.random.PRNGKey(17));g,_=graph();g,y,m=pad(g,np.array([4,5,7,8],np.float32),(16,16,8))
    y[~m]=np.nan
    engine=ScalarEngine(predict);state=engine.optimizer.init(p)
    grad,initial=engine.audited_gradient(p,g,y,m)
    assert np.linalg.norm(np.asarray(grad['embed']['w']))>0
    assert 27000<sum(x.size for x in jax.tree.leaves(p))<33000
    for _ in range(12):
        grad,loss=engine.audited_gradient(p,g,y,m);p,state=engine.update(p,state,[grad])
    assert loss<initial
    save_checkpoint(tmp_path/'step',p,state,{'step':12})
    q,qs,meta=load_checkpoint(tmp_path/'step',(p,state))
    for a,b in zip(jax.tree.leaves((p,state)),jax.tree.leaves((q,qs))):np.testing.assert_array_equal(a,b)
    assert meta['step']==12


def test_degenerate_frame_is_finite():
    _,bb=graph();bb[0,0]=bb[0,1];bb[1,2]=bb[1,1]
    g,valid=geometry(bb,np.zeros(12,int));assert not valid[:2].any()
    assert np.isfinite(g['edge']).all()


@pytest.mark.parametrize('width,ff,count',[(20,32,10229),(32,72,28565),(44,88,49709)])
def test_model_sizes(width,ff,count):
    p=initialize(jax.random.PRNGKey(17),width=width,ff=ff)
    assert sum(x.size for x in jax.tree.leaves(p))==count
    g,_=graph();engine=ScalarEngine(predict)
    grad,loss=engine.audited_gradient(p,g,np.array([4,5,7,8],np.float32),np.ones(4,bool))
    assert np.isfinite(loss) and np.linalg.norm(np.asarray(grad['embed']['w']))>0


def test_original_model_equivalence():
    import os,runpy
    from pathlib import Path
    root=os.environ.get('PKABENCH_RUNTIME')
    if not root:pytest.skip('No original pilot snapshot in this environment')
    path=Path(root)/'pretraining/graph-pilot-v1/source-snapshot/pkanet/model.py'
    if not path.exists():pytest.skip('Original snapshot not available')
    old=runpy.run_path(str(path));key=jax.random.PRNGKey(17)
    p=initialize(key);previous=old['initialize'](key)
    for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(previous)):np.testing.assert_array_equal(a,b)
    g,_=graph()
    np.testing.assert_array_equal(jax.jit(predict)(p,g),jax.jit(old['predict'])(previous,g))


def test_sidechain_geometry_and_learning():
    from pkanet.graph import sidechain_features
    g,bb=graph();atoms=[{'CB':bb[i,1]+[.5,1.,1.]} for i in range(len(bb))]
    atoms[0]={}
    features=sidechain_features(bb,atoms);assert features.shape==(12,128) and not features[0].any()
    rotation=np.linalg.qr(np.random.default_rng(5).normal(size=(3,3)))[0];shift=np.array([9,-7,3])
    moved=[{k:v@rotation+shift for k,v in row.items()} for row in atoms]
    np.testing.assert_allclose(sidechain_features(bb@rotation+shift,moved),features,atol=1e-6)
    g['nodes']=np.concatenate((g['nodes'],features),axis=-1)
    params=initialize(jax.random.PRNGKey(17),width=44,ff=68,node_dim=152)
    assert sum(x.size for x in jax.tree.leaves(params))==50001
    engine=ScalarEngine(predict);grad,_=engine.audited_gradient(params,g,np.array([4,5,7,8],np.float32),np.ones(4,bool))
    assert np.linalg.norm(np.asarray(grad['embed']['w'][24:]))>0
    original=np.asarray(engine.forward(params,g))
    padded,_,_=pad(g,np.zeros(4),(32,32,16))
    np.testing.assert_allclose(engine.forward(params,padded)[:4],original,atol=3e-6)
    g['nodes'][:,24:]=0
    assert np.max(np.abs(np.asarray(engine.forward(params,g))-original))>1e-6


def test_schedule_and_strict_backbone_resume(tmp_path):
    from pkatrain.trainer import scalar_schedule
    from pkatrain.graph_data import mask_features
    cfg=dict(learning_rate=.001,schedule='constant_then_cosine',constant_epochs=20,epochs=100,end_learning_rate=1e-5)
    schedule=scalar_schedule(cfg,60)
    np.testing.assert_allclose([schedule(0),schedule(1199),schedule(1200)],[.001]*3,rtol=1e-6)
    np.testing.assert_allclose(schedule(6000),1e-5,rtol=1e-6)
    assert float(schedule(3600))<float(schedule(1200))
    g,_=graph();g['nodes'][:,22]=1
    before=g['nodes'].copy();mask_features(g,{'strict_backbone':True})
    assert not g['nodes'][:,22].any()
    np.testing.assert_array_equal(g['nodes'][:,:22],before[:,:22])
    p=initialize(jax.random.PRNGKey(17));engine=ScalarEngine(predict,schedule)
    state=engine.optimizer.init(p);grad,_=engine.audited_gradient(p,g,np.array([4,5,7,8],np.float32),np.ones(4,bool))
    p,state=engine.update(p,state,[grad]);save_checkpoint(tmp_path/'scheduled',p,state,{'step':1})
    restored,rs,_=load_checkpoint(tmp_path/'scheduled',(p,state))
    first=engine.update(p,state,[grad]);second=engine.update(restored,rs,[grad])
    for a,b in zip(jax.tree.leaves(first),jax.tree.leaves(second)):np.testing.assert_array_equal(a,b)
