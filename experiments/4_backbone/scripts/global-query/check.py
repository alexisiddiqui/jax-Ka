import os,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from pkatrain.production_train import make_model,model_inputs,initialize_aux,AuxEngine,_with_policy
from pkatrain.production_inputs import BackboneTable
from pkatrain.production_loading import PinderSource,normalization
from pkatrain.production_graphs import output,read
from pkatrain.production_global_query import query_tokens,edge_features,allowed
from pkanet.site_model import _site_tokens
root=Path(os.environ['PKABENCH_RUNTIME']);out=Path(os.environ['CODE'])
p,_,_=make_model(17,width=64,orientation=True,dihedrals=True,variant='global-query')
pbase,_,_=make_model(17,width=64,orientation=True,dihedrals=True)
for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pbase)):np.testing.assert_array_equal(a,b)
# Compare chunked global query against the original attention on a complete graph.
n=8;table=np.zeros((n,14),np.float32)
for i in range(n):
 x=2.*i;table[i]=[x-1,1,0,x,0,0,x+1,0,0,x+1,1,0,int(i>=4),i%4]
table=jnp.asarray(table);sr=jnp.arange(n);nb=jnp.broadcast_to(sr,(n,n))
g={'global_backbone':table,'node_mask':jnp.ones(n,bool),'allow_cross_chain':jnp.array(False),'site_residue':sr,'site_type':jnp.arange(n)%9,'site_mask':jnp.ones(n,bool),'neighbors':nb,'edge':edge_features(table,sr),'switch':jnp.ones((n,n))}
g['edge_mask']=allowed(g,sr)
h=jax.random.normal(jax.random.PRNGKey(7),(n,64))
a=jax.jit(query_tokens)(p,g,h);b=jax.jit(_site_tokens)(p,g,h)
np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
ga=jax.grad(lambda h:jnp.sum(query_tokens(p,g,h)**2))(h);gb=jax.grad(lambda h:jnp.sum(_site_tokens(p,g,h)**2))(h)
np.testing.assert_allclose(ga,gb,atol=2e-4,rtol=2e-4)
# Free query outputs have exactly zero dependence on partner values; bound mask includes every valid node.
grad=jax.grad(lambda x:query_tokens(p,g,x)[0].sum())(h)
np.testing.assert_array_equal(grad[4:],jnp.zeros_like(grad[4:]))
assert int(allowed({**g,'allow_cross_chain':jnp.array(True)},sr).sum())==n*n
print('PASS dense-query output/gradient equivalence and zero cross-chain free-query gradient',flush=True)
manifest=read(output(root,'pinder')/'manifest-v1.json');bt=BackboneTable(root,'pinder')
rows=sorted((r for r in manifest['records'] if r['split']=='train'),key=lambda r:r['n'])
row=next(r for r in rows if len(np.unique(bt(r['id'])[:,12]))>1)
source=PinderSource(_with_policy(manifest,2),norms=normalization(manifest,.5),sidechains=bt)
graphs,t,m,_,_,meta,v=source.load([row['id']]);graphs=jax.tree.map(jnp.asarray,graphs)
assert np.asarray(graphs['allow_cross_chain'])[0].tolist()==[True,False]
raw=jax.tree.map(lambda x:x[0,1],graphs);fn=model_inputs(orientation=True,dihedrals=True,variant='global-query')
chain=np.asarray(raw['nodes'][:,-2]);chain0=chain[0];partner=(chain!=chain0)&np.asarray(raw['node_mask'],bool)
changed={**raw,'nodes':raw['nodes'].at[partner,:24].add(3.).at[partner,-14:-2].add(100.)}
base=jax.jit(fn[1])(p,raw);changed_pred=jax.jit(fn[1])(p,changed)
own=(chain[np.asarray(raw['query_residue'])]==chain0)&np.asarray(m[0],bool)
assert own.any();np.testing.assert_allclose(np.asarray(base)[own],np.asarray(changed_pred)[own],atol=2e-5,rtol=2e-5)
print('PASS end-to-end free-branch independence after perturbing partner features and coordinates',flush=True)
source.close();jax.clear_caches()
# Largest actual training bucket at the production batch size, with a finite update.
row=max(rows,key=lambda r:r['n']);source=PinderSource(_with_policy(manifest,16),norms=normalization(manifest,.5),sidechains=BackboneTable(root,'pinder'))
graphs,t,m,_,_,meta,v=source.load([row['id']]);graphs=jax.tree.map(jnp.asarray,graphs)
p=initialize_aux(p,17,'ce','burial');engine=AuxEngine(p,1.,.1,heads='burial',site_embeddings=fn[0]);start=time.time()
(loss,aux),grad=engine.paired_value_grad(p,graphs,jnp.asarray(t),jnp.asarray(m),jnp.ones_like(m,dtype=jnp.float32)*.5,jnp.zeros_like(m),jnp.asarray(v),jax.random.PRNGKey(17))
assert np.isfinite(loss) and all(np.isfinite(a).all() for a in jax.tree.leaves(grad))
p2,state,finite=engine.apply(p,engine.optimizer.init(p),grad,jax.tree.map(jnp.zeros_like,grad),jnp.asarray(.001))
assert bool(finite) and all(np.isfinite(a).all() for a in jax.tree.leaves(p2))
result={'dense_reference_equivalence':True,'gradient_equivalence':True,'free_partner_gradient_zero':True,'free_end_to_end_independence':True,'largest_finite_update':True,'batch_shape':list(graphs['nodes'].shape),'largest_seconds':time.time()-start,'test_data_included':False}
(out/'check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);source.close()
