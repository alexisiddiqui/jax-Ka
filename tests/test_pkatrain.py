from dataclasses import replace
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from jaxpropka.synthetic import synthetic_cache
from jaxpropka.model import _local_terms, PhysicalScales, one_hot, TitrationModel
from jaxpropka.parameters import ModelConfig
from jaxpropka.optx_solver import SolverConfig, active_channels, active_system, local_terms_curve_kernel
from pkatrain.branches import free_union, paired_inputs
from pkatrain.adapters.jaxka import local_terms
from pkatrain.forward import solve_branches
from pkatrain.losses import curve_loss, coverage

CFG=ModelConfig(steps=1024)
SCFG=SolverConfig(max_steps=512,rtol=1e-12,atol=1e-12)
PH=jnp.asarray([3.,5.,7.,9.,11.])

def fixture():
    a=synthetic_cache(n=3,neighbors=2,chains=1)
    b=replace(a,keys=tuple(replace(k,chain='chain_1') for k in a.keys),chain_ids=('chain_1',),
              bb_volume=a.bb_volume+.02)
    template=synthetic_cache(n=6,neighbors=2,chains=2)
    template=replace(template,native_index=np.concatenate([a.native_index,b.native_index]))
    # Free synthetic systems have independently summed environments.
    ab=free_union(template,a,b)
    ab=replace(ab,bb_volume=ab.bb_volume+.03)
    return ab,a,b

def single(cache,theta,ph=PH,mask=None):
    arrays=TitrationModel(cache)._d; p=jnp.asarray(one_hot(cache.native_index,np.float64))
    active=jnp.asarray(active_channels(cache,p))
    terms=local_terms(theta,arrays,p,CFG)
    return local_terms_curve_kernel(arrays,terms,ph,active,jnp.ones(active.shape,bool),
        config=CFG,solver_config=SCFG,gradient_mask=mask)[0]

def test_default_terms_and_three_parameter_gradients():
    cache=synthetic_cache(n=4,neighbors=2,chains=1)
    # Give the synthetic hydrogen-bond scale a nonzero local contribution.
    cache=replace(cache,bb_hbond=cache.bb_hbond+.1)
    d=TitrationModel(cache)._d; p=jnp.asarray(one_hot(cache.native_index,np.float64))
    a=_local_terms(d,p,CFG); b=_local_terms(d,p,CFG,PhysicalScales(1.,1.,1.))
    for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    fun=lambda t:jnp.sum(single(cache,t).total_charge)
    t=jnp.zeros(3); grad=np.asarray(jax.grad(fun)(t))
    for i in range(3):
        vals=[]
        for eps in (1e-3,1e-4,1e-5):
            e=np.eye(3)[i]*eps; vals.append(float((fun(t+e)-fun(t-e))/(2*eps)))
        assert any(np.isclose(grad[i],v,rtol=.01,atol=1e-7) for v in vals),(i,grad[i],vals)

def test_free_union_and_padding_match_separate_solutions_and_gradients():
    ab,a,b=fixture(); union=free_union(ab,a,b)
    np.testing.assert_array_equal(union.bb_volume[:3],a.bb_volume)
    np.testing.assert_array_equal(union.bb_volume[3:],b.bb_volume)
    theta=jnp.zeros(3)
    expected=np.concatenate([np.asarray(single(a,theta).protonated),np.asarray(single(b,theta).protonated)],axis=1)
    inputs,layout=paired_inputs(ab,a,b)
    def paired(t):
        terms=jax.vmap(lambda d,p:local_terms(t,d,p,CFG))(inputs['arrays'],inputs['probabilities'])
        return solve_branches(inputs,terms,PH,config=CFG,solver_config=SCFG)[0]
    got=paired(theta)
    np.testing.assert_allclose(got.protonated[1,:,:6],expected,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(got.protonated[:,:,6:],0)
    np.testing.assert_allclose(jax.grad(lambda t:jnp.sum(paired(t).total_charge[1]))(theta),
        jax.grad(lambda t:jnp.sum(single(a,t).total_charge)+jnp.sum(single(b,t).total_charge))(theta),rtol=.01,atol=1e-7)
    assert layout['M']%32==0 and len(set(inputs['active'][0]))==layout['M']
    # Same-residue terminal/side-chain terms remain present.
    assert any(union.pair_mask[i,k].any() for i in range(6) for k,j in enumerate(union.neighbors[i]) if i==j)

def test_independent_mean_field_root_and_uncoupled_enumeration():
    cache=synthetic_cache(n=3,neighbors=2,chains=1); d=TitrationModel(cache)._d
    p=jnp.asarray(one_hot(cache.native_index,np.float64)); active=active_channels(cache,p)
    terms=_local_terms(d,p,CFG); pk,f0,k=map(np.asarray,active_system(d,terms,jnp.asarray(active)))
    solved=single(cache,jnp.zeros(3),jnp.asarray([7.]))
    fn=lambda u:u-np.log(10)*(pk-7-f0-k@(1/(1+np.exp(-u))))
    independent=root(fn,np.log(10)*(pk-7-f0)); assert independent.success
    np.testing.assert_allclose(np.asarray(solved.protonated)[0].reshape(-1)[active],1/(1+np.exp(-independent.x)),atol=1e-7)
    # Enumeration is exact for uncoupled sites, not an equality oracle for mean field.
    import itertools
    pk=np.array([4.,7.,9.]); states=np.array(list(itertools.product((0.,1.),repeat=3)))
    w=np.exp(-np.log(10)*np.sum(states*(7-pk),axis=1)); exact=w@states/w.sum()
    np.testing.assert_allclose(exact,1/(1+10**(7-pk)),atol=1e-12)

def test_rejected_root_uses_safe_gradient_system_and_loss_masks():
    cache=synthetic_cache(n=3,neighbors=2,chains=1)
    mask=jnp.asarray([True,False,True,True,True])
    fun=lambda t:jnp.sum(jnp.where(mask,single(cache,t,mask=mask).total_charge,0))
    np.testing.assert_allclose(jax.grad(fun)(jnp.zeros(3)),
        jax.grad(lambda t:jnp.sum(single(cache,t,ph=PH[mask]).total_charge))(jnp.zeros(3)),atol=1e-7,rtol=.01)
    pred=jnp.ones((2,5,1,1)); ref=jnp.zeros_like(pred); eligible=jnp.ones((1,1),bool)
    accepted=jnp.stack([mask,mask]); loss,parts=curve_loss(pred,ref,eligible,accepted)
    assert np.isclose(loss,.8) and float(parts['paired'])==0
    assert coverage(eligible,accepted)['missing']==2


def test_optimizer_checkpoint_resume_and_shared_loss_identities(tmp_path):
    import optax
    from pkatrain.trainer import Engine,save_checkpoint,load_checkpoint
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.01))
    initial={'x':jnp.array([2.,-1.])}; target=jnp.array([.1,.2]); state=optimizer.init(initial)
    params=initial
    loss=lambda p:jnp.sum((p['x']-target)**2)
    for _ in range(5):
        update,state=optimizer.update(jax.grad(loss)(params),state,params); params=optax.apply_updates(params,update)
    save_checkpoint(tmp_path/'checkpoint',params,state,{'step':5})
    resumed,rs,meta=load_checkpoint(tmp_path/'checkpoint',(initial,optimizer.init(initial)))
    for _ in range(5):
        update,state=optimizer.update(jax.grad(loss)(params),state,params); params=optax.apply_updates(params,update)
        update,rs=optimizer.update(jax.grad(loss)(resumed),rs,resumed); resumed=optax.apply_updates(resumed,update)
    for a,b in zip(jax.tree.leaves((params,state)),jax.tree.leaves((resumed,rs))): np.testing.assert_array_equal(a,b)
    assert float(loss(params))<float(loss(initial))
    y=jnp.array([[[[.2]],[[.4]]],[[[.1]],[[.3]]]])
    mask=jnp.ones((1,1),bool); ok=jnp.ones((2,2),bool)
    value,_=curve_loss(y,y,mask,ok); assert value==0
    difference=y[0]-y[1]; np.testing.assert_array_equal(y[::-1][0]-y[::-1][1],-difference)


def test_distinct_complex_batch_gradient_matches_mean_of_single_gradients():
    from pkatrain.trainer import Engine
    ab,a,b=fixture()
    one,_=paired_inputs(ab,a,b)
    two,_=paired_inputs(replace(ab,bb_volume=ab.bb_volume+.02),a,b)
    batch=jax.tree.map(lambda x,y:np.stack([x,y]),one,two)
    eligible=np.stack([one['arrays']['group_mask'][0],two['arrays']['group_mask'][0]])
    labels=np.full((2,2,73,eligible.shape[1],9),.5)
    engine=Engine(local_terms,lambda t:jnp.sum(t*t)*.001,replace(CFG,steps=8),SCFG)
    theta=jnp.zeros(3)
    gradient,log,curves=engine.audited_batch_gradient(theta,batch,labels,eligible,1.)
    independent=[]
    for i,d in enumerate((one,two)):
        g,_=engine.audited_gradient(theta,d,labels[i],eligible[i],1.)
        independent.append(np.asarray(g))
    np.testing.assert_allclose(gradient,np.mean(independent,axis=0),rtol=1e-6,atol=1e-9)
    assert log['missing']==0 and curves.protonated.shape[:3]==(2,2,73)


def test_common_capacities_keep_unique_dummy_channels():
    ab,a,b=fixture()
    inputs,layout=paired_inputs(ab,a,b,capacities=(128,32,32,64))
    assert (layout['N'],layout['Ke'],layout['Kc'],layout['M'])==(128,32,32,64)
    for indices,valid in zip(inputs['active'],inputs['active_valid']):
        assert len(set(indices))==64
        assert np.all(indices[~valid]>=ab.n_residues*9)


def test_size_order_preserves_sampling_and_microbatch_gradient_weights():
    from collections import Counter
    from pkatrain.buckets import ordered_epoch,microbatches
    plan={'assignment':{'a':'small','b':'large','c':'small'},
        'buckets':[{'name':'small','batch_size':2},{'name':'large','batch_size':1}]}
    draws=['a','b','a','c','b'];ordered=ordered_epoch(draws,plan,np.random.default_rng(17))
    assert Counter(ordered)==Counter(draws)
    values={'a':1.,'b':4.,'c':7.};weighted=[]
    for ids,bucket in microbatches(ordered,plan):
        assert len(ids)<=bucket['batch_size']
        weighted.extend([np.mean([values[c] for c in ids])]*len(ids))
    assert np.mean(weighted)==np.mean([values[c] for c in draws])


def test_padding_prepared_inputs_matches_rebuilding_from_caches():
    from pkatrain.minibatch import pad_prepared
    ab,a,b=fixture();original,_=paired_inputs(ab,a,b)
    capacities=(128,32,32,64)
    expected,_=paired_inputs(ab,a,b,capacities=capacities)
    for dtype in (np.float32,np.float64):
        got=pad_prepared(original,ab.n_residues,capacities,dtype)
        reference=jax.tree.map(lambda x:x.astype(dtype) if np.issubdtype(x.dtype,np.floating) else x,expected)
        for x,y in zip(jax.tree.leaves(got),jax.tree.leaves(reference)):np.testing.assert_array_equal(x,y)
