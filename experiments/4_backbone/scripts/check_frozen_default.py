import os,json
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from pkatrain.production_train import export_model,selected_model,_with_policy,load_exported
from pkatrain.production_defaults import DEFAULT_RUN,DEFAULT_EXPORT,load_frozen_default
from pkatrain.production_loading import PinderSource
from pkatrain.production_inputs import BackboneTable
from pkatrain.production_graphs import output,read
root=Path(os.environ['PKABENCH_RUNTIME']);code=Path(os.environ['CODE'])
export_model(root,DEFAULT_RUN,code/'models'/DEFAULT_EXPORT)
params,fn=selected_model(root);embed,predict,info=load_frozen_default()
loaded=load_exported(code/"models"/DEFAULT_EXPORT)
for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(loaded)):np.testing.assert_array_equal(a,b)
m=read(output(root,'pinder')/'manifest-v1.json');r=min((r for r in m['records'] if r['split']=='train'),key=lambda r:r['n'])
s=PinderSource(_with_policy(m,1),fraction=.5,sidechains=BackboneTable(root,'pinder'))
graphs,*_=s.load([r['id']])
for branch in (0,1):
 g=jax.tree.map(lambda x:jnp.asarray(x[0,branch]),graphs)
 np.testing.assert_allclose(jax.jit(predict)(g),jax.jit(lambda graph:fn[1](params,graph))(g),rtol=1e-6,atol=1e-6)
 a=embed(g);b=fn[0](params,g)
 for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)
 grad=jax.grad(lambda nodes:embed({**g,'nodes':nodes})[0].sum())(g['nodes']);np.testing.assert_array_equal(grad,jnp.zeros_like(grad))
s.close()
result={'default_run':info['run'],'epoch':info['selected_epoch'],'parameters':info['parameter_count'],'export_matches_bound_and_free':True,'frozen_gradients_zero':True}
(code/'default-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
