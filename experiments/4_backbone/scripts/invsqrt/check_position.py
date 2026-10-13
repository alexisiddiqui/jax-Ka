import os,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pkatrain.production_train import make_model,model_inputs,initialize_aux,AuxEngine,_with_policy
from pkatrain.production_inputs import BackboneTable
from pkatrain.production_loading import PinderSource,normalization
from pkatrain.production_graphs import output,read
from pkatrain.production_position import annotate,delta_correction,initialize
from pkatrain.loading import LoaderConfig
root=Path(os.environ['PKABENCH_RUNTIME']);manifest=read(output(root,'pinder')/'manifest-v1.json')
rows=sorted((r for r in manifest['records'] if r['split']=='train'),key=lambda r:r['n'])
source=PinderSource(_with_policy(manifest,2),norms=normalization(manifest,.5),sidechains=BackboneTable(root,'pinder'),config=LoaderConfig())
graphs,targets,mask,_,_,meta,valid=source.load([rows[0]['id']]);graphs=jax.tree.map(jnp.asarray,graphs)
g=jax.tree.map(lambda x:x[0,0],graphs);gf=jax.tree.map(lambda x:x[0,1],graphs)
params,_,_=make_model(17,width=64,orientation=True,dihedrals=True)
base=model_inputs(orientation=True,dihedrals=True)[1];expected=jax.jit(base)(params,g)
results={}
for variant in ('fourier','rope','shared-projection','relations','paired-delta'):
 p,_,_=make_model(17,width=64,orientation=True,dihedrals=True,variant=variant)
 fn=model_inputs(orientation=True,dihedrals=True,variant=variant)
 pred=jax.jit(fn[1])(p,g)
 # Renaming chain IDs must never alter any output.
 renamed={**g,'nodes':g['nodes'].at[:,-2].set(101-7*g['nodes'][:,-2])}
 np.testing.assert_allclose(pred,jax.jit(fn[1])(p,renamed),rtol=2e-5,atol=2e-5)
 if variant in ('relations','shared-projection'):
  ag=annotate(g,variant);af=annotate(gf,variant)
  assert not bool(jnp.any(af['edge_mask']&(af['position_chain'][:,None]!=af['position_chain'][af['neighbors']])))
  assert int(ag['edge_mask'][:,-4:].sum())==0, 'unexpected absent local edge in intact tiny graph'
  np.testing.assert_allclose(pred,expected,rtol=2e-5,atol=2e-5)
 if variant=='fourier':
  pzero={**p,'position_projection':jax.tree.map(jnp.zeros_like,p['position_projection'])}
  np.testing.assert_allclose(jax.jit(fn[1])(pzero,g),expected,rtol=2e-5,atol=2e-5)
 if variant=='rope':
  shifted={**g,'nodes':g['nodes'].at[:,-1].add(10)}
  np.testing.assert_allclose(pred,jax.jit(fn[1])(p,shifted),rtol=2e-5,atol=2e-5)
 if variant=='paired-delta':
  head=jax.tree.map(lambda x:jax.random.normal(jax.random.PRNGKey(x.size),x.shape),p['delta_head'])
  a=jax.random.normal(jax.random.PRNGKey(4),(3,64));b=a*.7
  np.testing.assert_allclose(delta_correction(head,a,b),-delta_correction(head,b,a),rtol=1e-6,atol=1e-6)
  np.testing.assert_array_equal(delta_correction(head,a,a),jnp.zeros(3))
 p=initialize_aux(p,17,'ce','burial');engine=AuxEngine(p,1.,.1,heads='burial',site_embeddings=fn[0],shift_weights={'weights':[.6,2.3,2.8,3.2,4,5,5]})
 (_,aux),grad=engine.paired_value_grad(p,graphs,jnp.asarray(targets),jnp.asarray(mask),jnp.ones_like(mask,dtype=jnp.float32)*.5,jnp.zeros_like(mask),jnp.asarray(valid),jax.random.PRNGKey(7))
 assert all(np.isfinite(a).all() for a in jax.tree.leaves(grad))
 state=engine.optimizer.init(p);p2,state,finite=engine.apply(p,state,grad,jax.tree.map(jnp.zeros_like,grad),jnp.asarray(.001))
 assert bool(finite) and all(np.isfinite(a).all() for a in jax.tree.leaves(p2))
 results[variant]={'finite_update':True,'chain_relabeling':True,'paired_loss':float(aux[1])}
 print('PASS',variant,results[variant],flush=True)
source.close()
Path(os.environ['POSITION_CHECK_OUT']).write_text(json.dumps(results,indent=2)+'\n')
