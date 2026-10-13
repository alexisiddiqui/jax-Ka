import os,sys,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pkatrain.production_train import make_model,model_inputs,initialize_aux,AuxEngine,_with_policy
from pkatrain.production_inputs import BackboneTable
from pkatrain.production_loading import PinderSource,normalization
from pkatrain.production_graphs import output,read
root=Path(os.environ['PKABENCH_RUNTIME']);variant=sys.argv[1];manifest=read(output(root,'pinder')/'manifest-v1.json')
# Exercise genuinely missing local edges and a peptide break on a controlled graph.
from pkatrain.production_position import annotate
nodes=np.zeros((3,38),np.float32)
for i in range(3):
 x=3.8*i; nodes[i,-14:]=[x-1.2,1,0,x,0,0,x+1.3,0,0,x+1.5,1,0,0,i]
gapgraph={'nodes':jnp.asarray(nodes),'node_mask':jnp.ones(3,bool),'neighbors':jnp.arange(3)[:,None],'edge':jnp.zeros((3,1,20)),'edge_mask':jnp.ones((3,1),bool),'switch':jnp.ones((3,1))}
added=annotate(gapgraph,'relations');assert int(added['edge_mask'][:,-4:].sum())==6
broken={**gapgraph,'nodes':gapgraph['nodes'].at[1,-14:-11].set(jnp.array([100.,100.,100.]))}
added=annotate(broken,'relations');assert int(added['edge_mask'][:,-4:].sum())==2

row=max((r for r in manifest['records'] if r['split']=='train'),key=lambda r:r['n'])
source=PinderSource(_with_policy(manifest,16),norms=normalization(manifest,.5),sidechains=BackboneTable(root,'pinder'))
g,t,m,_,_,meta,v=source.load([row['id']]);g=jax.tree.map(jnp.asarray,g)
p,_,_=make_model(17,width=64,orientation=True,dihedrals=True,variant=variant);p=initialize_aux(p,17,'ce','burial')
fn=model_inputs(orientation=True,dihedrals=True,variant=variant)
w=read(root/'training/gqt-production-v2/runs/invsqrt-weights-0.500-v1.json')
e=AuxEngine(p,1.,.1,heads='burial',site_embeddings=fn[0],shift_weights=w)
start=time.time();(loss,aux),grad=e.paired_value_grad(p,g,jnp.asarray(t),jnp.asarray(m),jnp.ones_like(m,dtype=jnp.float32)*.5,jnp.zeros_like(m),jnp.asarray(v),jax.random.PRNGKey(17))
assert np.isfinite(loss) and all(np.isfinite(a).all() for a in jax.tree.leaves(grad))
p2,state,finite=e.apply(p,e.optimizer.init(p),grad,jax.tree.map(jnp.zeros_like,grad),jnp.asarray(.001))
assert bool(finite) and all(np.isfinite(a).all() for a in jax.tree.leaves(p2))
result={'variant':variant,'complex_id':row['id'],'n':row['n'],'batch_shape':list(g['nodes'].shape),'finite_update':True,'loss':float(loss),'seconds':time.time()-start}
Path(os.environ['CODE'],f'largest-{variant}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);source.close()
