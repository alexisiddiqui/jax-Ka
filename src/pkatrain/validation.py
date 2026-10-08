"""Real-complex numerical release gates and registered loss profiles."""
import json
import time
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from pkabench.runtime import atomic_json,digest
from .records import read,load
from .experiment import make_engine
from .adapters.jaxka import local_terms
from .losses import curve_loss,coverage


def real_gate(out,cid):
    from jaxpropka.cache import StructureCache
    from jaxpropka.model import TitrationModel,one_hot,_grid_pka_result
    from jaxpropka.optx_solver import active_channels,local_terms_curve_kernel
    from jaxpropka.batching import pack_inputs
    engine=make_engine(out); tight=replace(engine.solver_config,rtol=1e-12,atol=1e-12)
    inputs,reference,eligible,_=load(out,cid); theta=jnp.zeros(3,jnp.float64)
    began=time.monotonic(); forward=engine.forward(theta,inputs,jnp.ones((2,73),bool))
    cov=coverage(eligible,np.asarray(forward.converged)); assert cov['fraction']<=.05,cov
    ab=StructureCache.load(out/'prepared'/cid/'AB.npz')
    lookup={k:i for i,k in enumerate(ab.keys)}
    max_error=max_midpoint=0.; grad_separate=np.zeros(3); timings={}
    # Gate full-grid free union against independently prepared A/B solves.
    for state in ('A','B'):
        cache=StructureCache.load(out/'prepared'/cid/f'{state}.npz')
        d=TitrationModel(cache)._d; p=jnp.asarray(one_hot(cache.native_index,np.float64)); active=jnp.asarray(active_channels(cache,p))
        def solve(t):
            terms=local_terms(t,d,p,engine.config)
            return local_terms_curve_kernel(d,terms,engine.ph,active,jnp.ones(active.shape,bool),config=engine.config,solver_config=engine.solver_config,seed_steps=engine.seed_steps,seed_dtype=engine.seed_dtype)[0]
        separate=solve(theta); ids=np.asarray([lookup[k] for k in cache.keys])
        both=np.asarray(separate.converged)&np.asarray(forward.converged[1])
        assert both.any()
        max_error=max(max_error,float(np.max(abs(np.asarray(separate.protonated)[both]-np.asarray(forward.protonated)[1,both][:,ids]))))
        mid=_grid_pka_result(d,separate,engine.ph,engine.config)
        uarrays={k:v[1] for k,v in inputs['arrays'].items()}
        union_mid=_grid_pka_result(uarrays,jax.tree.map(lambda a:a[1],forward),engine.ph,engine.config)
        valid=np.asarray(mid.valid)&np.asarray(union_mid.valid)[ids]
        if valid.any(): max_midpoint=max(max_midpoint,float(np.max(abs(np.asarray(mid.value)[valid]-np.asarray(union_mid.value)[ids][valid]))))
    assert max_error<=1e-6 and max_midpoint<=1e-4,(max_error,max_midpoint)
    # Finite differences on several representative converged pH points. The root
    # must stay on the same branch; discontinuous perturbations are explicit.
    from .forward import solve_branches
    common=np.flatnonzero(np.asarray(forward.converged).all(axis=0))
    positions=common[np.unique(np.linspace(0,len(common)-1,min(3,len(common))).astype(int))]
    ph=engine.ph[positions]
    def objective(t,inputs):
        terms=jax.vmap(lambda d,p:local_terms(t,d,p,engine.config))(inputs['arrays'],inputs['probabilities'])
        curves,_=solve_branches(inputs,terms,ph,config=engine.config,solver_config=tight,seed_steps=engine.seed_steps,seed_dtype=engine.seed_dtype)
        return jnp.mean(curves.total_charge[0]-curves.total_charge[1])
    objective=jax.jit(objective); gradient=np.asarray(jax.grad(objective)(theta,inputs)); comparisons=[]
    for i in range(3):
        values=[]
        for eps in (1e-3,1e-4,1e-5):
            e=np.eye(3)[i]*eps
            values.append(float((objective(theta+e,inputs)-objective(theta-e,inputs))/(2*eps)))
        passed=any(np.isclose(gradient[i],v,rtol=.01,atol=1e-7) for v in values)
        comparisons.append({'parameter':i,'gradient':float(gradient[i]),'finite_differences':values,'passed':bool(passed)})
    assert all(c['passed'] for c in comparisons),comparisons
    grad,loss=engine.audited_gradient(theta,inputs,reference,eligible,1.)
    timings['first_gate_seconds']=time.monotonic()-began
    started=time.monotonic(); grad2,loss2=engine.audited_gradient(theta,inputs,reference,eligible,1.)
    timings['warm_step_seconds']=time.monotonic()-started
    import resource
    atomic_json(out/'gates'/f'{cid}.json',{'passed':True,'complex_id':cid,'free_union_max_curve_error':max_error,
        'free_union_max_midpoint_error':max_midpoint,'gradient_checks':comparisons,'coverage':cov,'loss':loss,
        'timings':timings,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'manifest_sha256':digest(out/'manifest.json')})
    print(json.dumps({'complex_id':cid,'passed':True,**timings}),flush=True)


def collect_gates(out):
    m=read(out/'manifest.json'); gates=[read(out/'gates'/f'{cid}.json') for cid in m['smoke']]
    if 'parent' in m and m['config'].get('seed_steps') is not None:
        speed=read(out/'speed_verification.json')
        assert speed['passed'] and speed['manifest_sha256']==digest(out/'manifest.json')
    assert all(g['passed'] and g['manifest_sha256']==digest(out/'manifest.json') for g in gates)
    missing=sum(g['coverage']['missing'] for g in gates); total=sum(g['coverage']['total'] for g in gates)
    assert missing/max(total,1)<=.01
    gradients=[read(out/'union_gradients'/f'{cid}.json') for cid in m['smoke']]
    assert all(g['passed'] and g['manifest_sha256']==digest(out/'manifest.json') for g in gradients)
    atomic_json(out/'real_gate.json',{'passed':True,'gates':gates,'union_gradients':gradients,'manifest_sha256':digest(out/'manifest.json')})


def collect_speed(out):
    m=read(out/'manifest.json');source=Path(m['parent']['path']);checks=[]
    for cid in ('18e1b5db56153c00','a779dcd261ce4357','d5334196bf4ab5e6'):
        pair=[]
        for threads in (1,8):
            path=source/'speed_checks'/f'{cid}-threads{threads}.json';r=read(path)
            assert r['complete'] and r['passed'],path
            assert r['manifest_sha256']==m['parent']['manifest_sha256']
            pair.append(r);checks.append({'path':str(path),'sha256':digest(path)})
        for a,b in zip(pair[0]['rows'],pair[1]['rows']):
            assert a['config']==b['config'] and a['theta']==b['theta']
            assert np.allclose(a['gradient'],b['gradient'],rtol=.01,atol=1e-7)
    atomic_json(out/'speed_verification.json',{'passed':True,'checks':checks,
        'manifest_sha256':digest(out/'manifest.json')})


def union_gradient(out,cid):
    """A real-data gate: free-union sensitivity equals independent A+B sensitivity."""
    from jaxpropka.cache import StructureCache
    from jaxpropka.model import TitrationModel,one_hot
    from jaxpropka.optx_solver import active_channels,local_terms_curve_kernel
    engine=make_engine(out); solver=replace(engine.solver_config,rtol=1e-12,atol=1e-12)
    inputs,_,_,_=load(out,cid); theta=jnp.zeros(3,jnp.float64)
    ph=jnp.asarray([4.,7.,10.])
    @jax.jit
    def objective(t,d,p,active,valid):
        terms=local_terms(t,d,p,engine.config)
        result,_=local_terms_curve_kernel(d,terms,ph,active,valid,config=engine.config,solver_config=solver,seed_steps=engine.seed_steps,seed_dtype=engine.seed_dtype)
        return jnp.mean(result.total_charge), result.converged
    value_grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
    (value,ok),gradient=value_grad(theta,{k:v[1] for k,v in inputs['arrays'].items()},inputs['probabilities'][1],inputs['active'][1],inputs['active_valid'][1])
    assert np.asarray(ok).all(), 'Free union did not converge at gradient-gate pH points'
    separate_value=0.; separate_gradient=np.zeros(3)
    for state in ('A','B'):
        cache=StructureCache.load(out/'prepared'/cid/f'{state}.npz')
        d=TitrationModel(cache)._d; p=jnp.asarray(one_hot(cache.native_index,np.float64))
        active=jnp.asarray(active_channels(cache,p))
        (v,ok),g=value_grad(theta,d,p,active,jnp.ones(active.shape,bool))
        assert np.asarray(ok).all(), (state,'gradient-gate convergence')
        separate_value+=float(v); separate_gradient+=np.asarray(g)
    actual=np.asarray(gradient)
    assert np.allclose(actual,separate_gradient,rtol=.01,atol=1e-7),(actual,separate_gradient)
    assert np.isclose(float(value),separate_value,rtol=0,atol=1e-6)
    atomic_json(out/'union_gradients'/f'{cid}.json',{'passed':True,'complex_id':cid,
        'union_gradient':actual.tolist(),'separate_gradient':separate_gradient.tolist(),
        'charge_error':abs(float(value)-separate_value),'ph':np.asarray(ph).tolist(),
        'manifest_sha256':digest(out/'manifest.json')})


def profile(out,cid):
    from jaxpropka.model import PhysicalScales,_local_terms
    from .forward import solve_branches
    engine=make_engine(out); inputs,reference,eligible,_=load(out,cid,dtype=np.dtype(read(out/'manifest.json')['config']['dtype'])); rows=[]
    config=read(out/'manifest.json')['config']
    @partial(jax.jit,static_argnames=('initialization',))
    def predict(scales,inputs,initialization='production'):
        terms=jax.vmap(lambda d,p:_local_terms(d,p,engine.config,PhysicalScales(*scales)))(inputs['arrays'],inputs['probabilities'])
        return solve_branches(inputs,terms,engine.ph,config=engine.config,solver_config=engine.solver_config,initialization=initialization,seed_steps=engine.seed_steps,seed_dtype=engine.seed_dtype)[0]
    for axis in range(3):
        for value in config['profile_scales']:
            scales=np.ones(3,dtype=np.dtype(config['dtype'])); scales[axis]=value
            c=predict(jnp.asarray(scales),inputs)
            cov=coverage(eligible,np.asarray(c.converged)); loss,parts=curve_loss(c.protonated,reference,eligible,c.converged,1.)
            up=predict(jnp.asarray(scales),inputs,initialization='up')
            down=predict(jnp.asarray(scales),inputs,initialization='down')
            valid=np.asarray(up.converged)&np.asarray(down.converged)
            gap=np.max(abs(np.asarray(up.protonated)-np.asarray(down.protonated)),axis=(-2,-1))
            rows.append({'axis':axis,'scale':value,'loss':float(loss)+1e-3*float(np.mean(np.log(scales)**2)),
                'absolute':float(parts['absolute']),'paired':float(parts['paired']),
                'hysteresis_max':float(gap[valid].max()) if valid.any() else None,
                'hysteretic_branch_ph_points':int(np.sum(valid & (gap>1e-4))),
                'invalid_up':int((~np.asarray(up.converged)).sum()),
                'invalid_down':int((~np.asarray(down.converged)).sum()),**cov})
    atomic_json(out/'profiles/complexes'/f'{cid}.json',{'complex_id':cid,'rows':rows,'manifest_sha256':digest(out/'manifest.json')})


def collect_profiles(out):
    m=read(out/'manifest.json'); groups={}; rows=[]
    for cid in m['train']:
        rec=read(out/'profiles/complexes'/f'{cid}.json'); assert rec['manifest_sha256']==digest(out/'manifest.json')
        group=read(out/'records'/f'{cid}.json')['component_id']
        for row in rec['rows']: groups.setdefault((row['axis'],row['scale'],group),[]).append(row)
    for axis in range(3):
        for value in m['config']['profile_scales']:
            gg=[rs for (a,s,g),rs in groups.items() if a==axis and s==value]
            rows.append({'axis':axis,'scale':value,'group_macro_loss':float(np.mean([np.mean([r['loss'] for r in rs]) for rs in gg])),
                'missing':sum(r['missing'] for rs in gg for r in rs),'total':sum(r['total'] for rs in gg for r in rs),
                'complexes_over_threshold':sum(r['fraction']>.05 for rs in gg for r in rs)})
    base=[r for r in rows if r['scale']==1.]
    assert all(r['complexes_over_threshold']==0 and r['missing']/max(r['total'],1)<=.01 for r in base)
    atomic_json(out/'profiles/verification.json',{'passed':True,'rows':rows,'interpretation':'Conditional full-training-set profiles, not global minima','manifest_sha256':digest(out/'manifest.json')})
