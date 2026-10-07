"""Model-independent JAX/Optax optimization, masked differentiation and checkpoints."""
import json
import os
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pkabench.runtime import atomic_json, digest
from .forward import solve_branches
from .losses import curve_loss, coverage, scalar_loss


class Engine:
    def __init__(self, terms_fn, prior_fn, config, solver_config, *, learning_rate=.001, clip=1., seed_steps=None, seed_dtype=None, dtype=jnp.float64):
        self.terms_fn=terms_fn; self.prior_fn=prior_fn; self.config=config; self.solver_config=solver_config
        self.seed_steps=seed_steps
        self.seed_dtype=seed_dtype
        self.optimizer=optax.chain(optax.clip_by_global_norm(clip),optax.adam(learning_rate))
        self.ph=jnp.linspace(-2.,16.,73,dtype=dtype)
        def forward(params,inputs,mask,initialization='production'):
            terms=jax.vmap(lambda d,p:terms_fn(params,d,p,config))(inputs['arrays'],inputs['probabilities'])
            return solve_branches(inputs,terms,self.ph,config=config,solver_config=solver_config,
                initialization=initialization,gradient_mask=mask,seed_steps=seed_steps,seed_dtype=seed_dtype)[0]
        self.forward=jax.jit(forward,static_argnames=('initialization',))
        def objective(params,inputs,reference,eligible,accepted,paired_weight):
            curves=forward(params,inputs,accepted)
            value,parts=curve_loss(curves.protonated,reference,eligible,accepted,paired_weight)
            prior=prior_fn(params)
            return value+prior,dict(parts,prior=prior)
        self.objective=jax.jit(objective)
        self.value_grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
        self.batch_forward=jax.jit(jax.vmap(self.forward,in_axes=(None,0,0)))
        def batch_objective(params,inputs,reference,eligible,accepted,paired_weight):
            values,_=jax.vmap(objective,in_axes=(None,0,0,0,0,None))(
                params,inputs,reference,eligible,accepted,paired_weight)
            return jnp.mean(values)
        self.batch_value_grad=jax.jit(jax.value_and_grad(batch_objective))

    def audited_batch_gradient(self,params,inputs,reference,eligible,paired_weight):
        curves=self.batch_forward(params,inputs,jnp.ones((len(eligible),2,73),bool))
        accepted=np.asarray(curves.converged)&np.asarray(jnp.all(jnp.isfinite(curves.protonated),axis=(-2,-1)))
        counts=[coverage(e,a) for e,a in zip(eligible,accepted)]
        if any(c['fraction']>.05 for c in counts):raise RuntimeError(f'Batch per-complex coverage failure: {counts}')
        loss,gradient=self.batch_value_grad(params,inputs,reference,eligible,jnp.asarray(accepted),paired_weight)
        if not np.isfinite(float(loss)) or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
            raise FloatingPointError('Nonfinite batched objective/gradient')
        return gradient,dict(loss=float(loss),missing=sum(c['missing'] for c in counts),
            total=sum(c['total'] for c in counts),per_complex=counts),curves

    def audited_gradient(self,params,inputs,reference,eligible,paired_weight):
        curves=self.forward(params,inputs,jnp.ones((2,73),bool))
        accepted=np.asarray(curves.converged)&np.asarray(jnp.all(jnp.isfinite(curves.protonated),axis=(-2,-1)))
        cov=coverage(eligible,accepted)
        if cov['fraction']>.05: raise RuntimeError(f'Per-complex missing-supervision threshold exceeded: {cov}')
        (loss,parts),gradient=self.value_grad(params,inputs,reference,eligible,jnp.asarray(accepted),paired_weight)
        if not np.isfinite(float(loss)) or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
            raise FloatingPointError('Nonfinite objective/gradient; no optimizer update was applied')
        return gradient,dict(loss=float(loss),**{k:float(v) for k,v in parts.items()},**cov,
            forward_compiled_signatures=self.forward._cache_size(),
            gradient_compiled_signatures=self.value_grad._cache_size())

    def update(self,params,state,gradients):
        gradient=jax.tree.map(lambda *g:jnp.mean(jnp.stack(g),axis=0),*gradients)
        updates,state=self.optimizer.update(gradient,state,params)
        return optax.apply_updates(params,updates),state


def scalar_schedule(config,updates_per_epoch):
    """Optimizer-update schedule; checkpointed Optax counters govern resumption."""
    if config.get('schedule') is None:return config['learning_rate']
    if config['schedule']!='constant_then_cosine':raise ValueError(config['schedule'])
    hold=config['constant_epochs']*updates_per_epoch
    decay=(config['epochs']-config['constant_epochs'])*updates_per_epoch
    assert hold>0 and decay>0
    return optax.join_schedules([optax.constant_schedule(config['learning_rate']),
        optax.cosine_decay_schedule(config['learning_rate'],decay,
            alpha=config['end_learning_rate']/config['learning_rate'])],[hold])


class ScalarEngine:
    """Shared output-label pretraining for any pytree model, without a physical solve."""
    update=Engine.update

    def __init__(self,predict,learning_rate=.001,*,training_predict=None):
        self.optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(learning_rate))
        self.forward=jax.jit(predict)
        self.value_grad=jax.jit(jax.value_and_grad(
            lambda p,x,y,m:scalar_loss(predict(p,x),y,m)))
        def batch_loss(p,x,y,m,valid,key=None):
            if training_predict is None:
                losses=jax.vmap(lambda a,b,c:scalar_loss(predict(p,a),b,c))(x,y,m)
            else:
                if key is None:raise ValueError('Stochastic training requires an RNG key')
                keys=jax.random.split(key,len(valid))
                losses=jax.vmap(lambda a,b,c,k:scalar_loss(training_predict(p,a,k),b,c))(x,y,m,keys)
            return jnp.sum(jnp.where(valid,losses,0.))/jnp.maximum(valid.sum(),1)
        self.batch_value_grad=jax.jit(jax.value_and_grad(batch_loss))
        self.batch_forward=jax.jit(jax.vmap(predict,in_axes=(None,0)))
        def batch_step(p,state,x,y,m,valid,key=None):
            loss,gradient=jax.value_and_grad(batch_loss)(p,x,y,m,valid,key)
            finite=jnp.isfinite(loss)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(gradient)]))
            updates,newstate=self.optimizer.update(gradient,state,p)
            return optax.apply_updates(p,updates),newstate,loss,finite
        self.batch_step=jax.jit(batch_step)

    def audited_batch_update(self,params,state,inputs,reference,eligible,valid,key=None):
        newparams,newstate,loss,finite=self.batch_step(params,state,inputs,reference,eligible,valid,key)
        if not bool(finite):raise FloatingPointError('Nonfinite batched scalar loss/gradient; update discarded')
        return newparams,newstate,float(loss)

    def audited_gradient(self,params,inputs,reference,eligible):
        loss,gradient=self.value_grad(params,inputs,reference,eligible)
        if not np.isfinite(float(loss)) or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)):
            raise FloatingPointError('Nonfinite scalar loss/gradient')
        return gradient,float(loss)


def save_checkpoint(folder,params,state,metadata):
    """Commit payload and metadata together; never overwrite a completed checkpoint."""
    folder=Path(folder); folder.parent.mkdir(parents=True,exist_ok=True)
    pending=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=folder.parent))
    leaves,tree=jax.tree.flatten((params,state))
    np.savez(pending/'state.npz',**{f'leaf{i}':np.asarray(v) for i,v in enumerate(leaves)})
    atomic_json(pending/'metadata.json',dict(metadata,tree=str(tree),sha256=digest(pending/'state.npz')))
    pending.rename(folder)
    atomic_json(folder.parent/'latest.json',{'checkpoint':folder.name})


def load_checkpoint(folder,template):
    folder=Path(folder); metadata=json.loads((folder/'metadata.json').read_text())
    assert digest(folder/'state.npz')==metadata['sha256']
    leaves,tree=jax.tree.flatten(template); assert str(tree)==metadata['tree']
    with np.load(folder/'state.npz',allow_pickle=False) as f:
        values=[jnp.asarray(f[f'leaf{i}']) for i in range(len(leaves))]
    for a,b in zip(leaves,values): assert np.shape(a)==np.shape(b)
    params,state=jax.tree.unflatten(tree,values)
    return params,state,metadata


def sample_epoch(records,rng):
    groups={}
    for r in records: groups.setdefault(r['component_id'],[]).append(r['complex_id'])
    names=sorted(groups)
    return [str(rng.choice(groups[str(rng.choice(names))])) for _ in records]
