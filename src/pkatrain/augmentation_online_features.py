"""Compare cached edges with GPU construction from compact backbone/topology records."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_cache_pilot import FeatureSource, FEATURE_SOURCE_FILES, feature_shapes
from .augmentation_graphs import INPUT_FIELDS
from .augmentation_store import AugmentationStore, ReleaseWriter
from .loading import LoaderConfig, Prefetcher

DERIVED = frozenset(('edge','switch','site_edge','site_switch'))


def rebuild_features(graph, xp=np):
    """One graph; exact supplied topology, 20-A basis, float32 arithmetic.

    The final fourteen node columns hold N/CA/C/O, chain and sequence. Masked
    slots are zeroed; equality is required on active slots and model outputs.
    Radius searching is deliberately outside this function.
    """
    table=graph['nodes'][:,-14:];ca=table[:,3:6]
    x=table[:,6:9]-ca;xn=xp.linalg.norm(x,axis=-1,keepdims=True);x=x/xp.maximum(xn,1e-6)
    y=table[:,:3]-ca;y=y-(y*x).sum(-1,keepdims=True)*x
    yn=xp.linalg.norm(y,axis=-1,keepdims=True);y=y/xp.maximum(yn,1e-6)
    valid=(xn[:,0]>1e-5)&(yn[:,0]>1e-5)
    frame=xp.stack((x,y,xp.cross(x,y)),axis=-1)
    chain=table[:,12];sequence=table[:,13]
    # Default GPU dot precision can round 3-D geometry through TF32 inputs.
    # Keep these tiny products at float32 precision without changing the model.
    precision={} if xp is np else dict(precision='highest')
    centers=xp.linspace(0.,20.,16,dtype=table.dtype)
    def radial(distance):return xp.exp(-((distance[...,None]-centers)/1.5)**2)
    def switch(distance):
        return xp.where(distance<18.,1.,.5*(1+xp.cos(xp.pi*xp.clip((distance-18.)/2.,0.,1.))))
    neighbors=graph['neighbors'];delta=ca[neighbors]-ca[:,None]
    distance=xp.sqrt((delta*delta).sum(-1)+1e-8)
    direction=xp.einsum('nkj,njl->nkl',delta,frame,**precision)/distance[...,None]*valid[:,None,None]
    edge=xp.concatenate((radial(distance),direction,(chain[neighbors]==chain[:,None])[...,None]),axis=-1)
    result=dict(graph)
    result['edge']=xp.where(graph['edge_mask'][...,None],edge,0.).astype(xp.float32)
    result['switch']=xp.where(graph['edge_mask'],switch(distance),0.).astype(xp.float32)
    ri=graph['site_residue'];rj=ri[graph['site_neighbors']]
    delta=ca[rj]-ca[ri,None];distance=xp.sqrt((delta*delta).sum(-1)+1e-8)
    fi=frame[ri];fj=frame[rj]
    di=xp.einsum('skc,scl->skl',delta,fi,**precision)/distance[...,None]*valid[ri,None,None]
    dj=xp.einsum('skc,skcl->skl',-delta,fj,**precision)/distance[...,None]*valid[rj][...,None]
    orientation=xp.matmul(xp.swapaxes(fi[:,None],-1,-2),fj,**precision).reshape((*rj.shape,9))
    orientation=orientation*(valid[ri,None]&valid[rj])[...,None]
    same=chain[ri,None]==chain[rj]
    separation=xp.where(same,xp.clip((sequence[rj]-sequence[ri,None])/32.,-1.,1.),0.)
    edge=xp.concatenate((radial(distance),di,dj,orientation,same[...,None],
                         (ri[:,None]==rj)[...,None],separation[...,None]),axis=-1)
    result['site_edge']=xp.where(graph['site_edge_mask'][...,None],edge,0.).astype(xp.float32)
    result['site_switch']=xp.where(graph['site_edge_mask'],switch(distance),0.).astype(xp.float32)
    return result


def build_compact(source, destination):
    provenance=dict(kind='compact-backbone-topology-pilot-v1',
        source_manifest_sha256=digest(Path(source)/'features/manifest.json'),
        removed=sorted(DERIVED),neighbor_search_precomputed=True)
    if (Path(destination)/'features/manifest.json').exists():
        with AugmentationStore(Path(source)/'features') as store,AugmentationStore(Path(destination)/'features') as compact:
            if compact.manifest['provenance']!=provenance or set(store.by_id)!=set(compact.by_id):
                raise ValueError('compact pilot belongs to another source')
            compact.verify_files()
            return dict(records=len(compact.rows),expanded_bytes=store.manifest['compressed_bytes'],
                compact_bytes=compact.manifest['compressed_bytes'],ratio=compact.manifest['compressed_bytes']/store.manifest['compressed_bytes'])
    with AugmentationStore(Path(source)/'features') as store, \
            ReleaseWriter(Path(destination)/'features',provenance,zstd_level=1) as writer:
        expected=[]
        for row in store.rows:
            record=store.raw(row['id']);arrays={k:v for k,v in record['arrays'].items() if k not in DERIVED}
            writer.add(row['id'],record['metadata'],arrays,summary=row['summary']);expected.append(row['id'])
        manifest=writer.finish(expected_ids=expected)
        return dict(records=len(expected),expanded_bytes=store.manifest['compressed_bytes'],
            compact_bytes=manifest['compressed_bytes'],ratio=manifest['compressed_bytes']/store.manifest['compressed_bytes'])


class CompactSource(FeatureSource):
    """Same live native inputs, but compact variant reads and host-to-device batches."""
    def __init__(self,path,compact,*args,**kwargs):
        super().__init__(path,*args,**kwargs)
        self.store.close();self.store=AugmentationStore(Path(compact)/'features',verify_reads=False)
        self.layouts=None

    def load(self,spec):
        bucket,kind,ids=spec;n,k,q,s,sk=self.capacities[bucket]
        branches=2 if self.dataset=='pinder' else 1
        shapes={name:shape for name,shape in feature_shapes(n,k,q,s,sk).items()
                if name in INPUT_FIELDS and name not in DERIVED}
        integers={'neighbors','query_residue','query_group','site_residue','site_type','site_neighbors','query_site'}
        graph={name:np.zeros((len(ids),branches,*shape),bool if name.endswith('mask') else
            np.int32 if name in integers else np.float32) for name,shape in shapes.items()}
        mask=np.zeros((len(ids),q),bool)
        def one(slot):
            parent,rid=ids[slot];meta=self.store.by_id[rid]['summary'];variant=self._variant(rid)
            touched=sum(x.nbytes for x in variant.values())
            if self.dataset=='pkpdb':views=(variant,);queries=variant['query_site']
            else:
                control=self._control(meta['control_key']);bound={name:control[name] for name in graph};free=dict(bound)
                free['edge_mask']=control['free_edge_mask'];free['site_edge_mask']=control['free_site_edge_mask']
                views=(bound,free);queries=bound['query_site']
                # Existing native stores still decode expanded arrays; do not hide this cost.
                touched+=sum(x.nbytes for x in control.values())
            for branch,g in enumerate(views):
                for name,value in g.items():
                    if name not in graph:continue
                    if name=='query_site':value=np.maximum(value,0)
                    graph[name][(slot,branch,*(slice(0,x) for x in value.shape))]=value
            if self.dataset=='pinder':
                residues=variant['residue_ids'];sites=variant['site_ids']
                for name in ('nodes','node_mask'):graph[name][slot,1,residues]=variant[name]
                for name in ('neighbors','edge_mask'):
                    target=graph[name][slot,1];target[residues]=0
                    value=residues[variant[name]] if name=='neighbors' else variant[name]
                    target[residues,:value.shape[1]]=value
                for name in ('site_neighbors','site_edge_mask'):
                    target=graph[name][slot,1];target[sites]=0
                    value=sites[variant[name]] if name=='site_neighbors' else variant[name]
                    target[sites,:value.shape[1]]=value
            mask[slot,:len(queries)]=queries>=0
            return touched
        touched=sum(self.pool.map(one,range(len(ids))))
        if self.pad_transform is not None:graph=self.pad_transform(graph)
        return graph,mask,touched


def benchmark(args):
    import jax
    import jax.numpy as jnp
    from pkanet.site_model import initialize_site_auxiliary
    from pkanet.model import linear
    from pkanet.attention_backend import set_backend,backend
    from .production_inputs import add_orientation,add_dihedral_inputs,input_functions
    from .production_depth import model_functions
    from .production_train import micro_size
    from .production_loading import spread_padding
    cpus=int(os.environ['SLURM_CPUS_PER_TASK']);require_compute(threads=cpus,gpu_benchmark=True)
    devices=jax.devices()
    if len(devices)!=1 or devices[0].platform!='gpu':raise RuntimeError('exactly one GPU required')
    set_backend('triton')
    if backend()!='triton':raise RuntimeError('pilot requires the production Triton attention backend')
    args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.pilot/'build-summary.json').read_text())
    import pkanet
    source_root=Path(pkanet.__file__).parents[1]
    for name in FEATURE_SOURCE_FILES:
        if digest(source_root/name)!=summary['feature_source_sha256'][name]:raise ValueError('feature source differs: '+name)
    started=time.perf_counter();storage=build_compact(args.pilot,args.compact)
    atomic_json(args.output/'storage.json',storage);print(json.dumps(dict(stage='compact-built',**storage)),flush=True)
    params=initialize_site_auxiliary(jax.random.PRNGKey(17),width=64,ff=128)
    params=add_dihedral_inputs(add_orientation(params,17),17)
    embeddings,_=input_functions(model_functions(),orientation=True,dihedrals=True)
    def flatten(graph):
        b,branches=graph['nodes'].shape[:2]
        return jax.tree.map(lambda a:a.reshape((b*branches,*a.shape[2:])),graph)
    def restore(flat,graph):
        b,branches=graph['nodes'].shape[:2]
        return jax.tree.map(lambda a:a.reshape((b,branches,*a.shape[1:])),flat)
    def construct(graph):return restore(jax.vmap(lambda g:rebuild_features(g,jnp))(flatten(graph)),graph)
    construct_jit=jax.jit(construct)
    def make_step(online):
        def objective(p,graph,mask,key):
            b,branches=graph['nodes'].shape[:2];flat=flatten(graph)
            if online:flat=jax.vmap(lambda g:rebuild_features(g,jnp))(flat)
            keys=jnp.repeat(jax.random.split(key,b),branches,axis=0)
            def single(g,draw):
                tokens,query=embeddings(p,g,key=draw,dropout_rate=.1)
                return (8*jnp.tanh(linear(p['head'],tokens)[:,0])[query],
                        linear(p['auxiliary']['burial'],tokens)[:,0][query])
            prediction,burial=jax.vmap(single)(flat,keys)
            prediction=prediction.reshape(b,branches,-1);burial=burial.reshape(b,branches,-1)
            paired=jnp.sum((prediction[:,0]-prediction[:,1])**2*mask) if branches==2 else 0.
            return (jnp.sum(prediction**2*mask[:,None,:])+jnp.sum(jax.nn.softplus(burial)*mask[:,None,:])+paired)/jnp.maximum(mask.sum(),1)
        return jax.jit(jax.value_and_grad(objective))
    steps={False:make_step(False),True:make_step(True)};results=[];checks=[];resident=[]
    def to_device(batch):
        graph,mask,touched=batch
        return jax.device_put(graph),jax.device_put(mask),touched
    for dataset,buckets in [('pkpdb',('128','512','1024','1280')),('pinder',('128','512','1024','1536'))]:
        sources={False:FeatureSource(args.pilot,args.workers,16,dataset=dataset,runtime=args.runtime,pad_transform=spread_padding),
            True:CompactSource(args.pilot,args.compact,args.workers,16,dataset=dataset,runtime=args.runtime,pad_transform=spread_padding)}
        try:
            for bucket in buckets:
                if bucket not in sources[False].by_bucket:continue
                micro=micro_size(16,int(bucket));spec=sources[False].plans(bucket,'augmented',1)[0]
                print(json.dumps(dict(stage='bucket',dataset=dataset,bucket=bucket,micro=micro)),flush=True)
                hosts={};on_device={};values={};gradients={}
                for online,source in sources.items():
                    g,m,_=source.load((*spec[:-1],spec[-1][:micro]));hosts[online]=(g,m)
                    on_device[online]=(jax.device_put(g),jax.device_put(m))
                    value,grad=steps[online](params,*on_device[online],jax.random.PRNGKey(17))
                    jax.block_until_ready((value,grad));values[online]=float(value);gradients[online]=jax.device_get(grad)
                expected=hosts[False][0];actual=jax.device_get(construct_jit(on_device[True][0]))
                maxima={}
                for field in DERIVED:
                    mask=expected['site_edge_mask' if field.startswith('site_') else 'edge_mask']
                    maxima[field]=float(np.max(np.abs(actual[field][mask]-expected[field][mask]),initial=0.))
                square_error=0.;square_reference=0.;finite=True
                for a,b in zip(jax.tree.leaves(gradients[False]),jax.tree.leaves(gradients[True])):
                    a=a.astype(np.float64);b=b.astype(np.float64)
                    square_error+=float(((a-b)**2).sum());square_reference+=float((a*a).sum())
                    finite=finite and np.isfinite(a).all() and np.isfinite(b).all()
                grad_relative=float(np.sqrt(square_error/max(square_reference,1e-30)))
                loss_relative=abs(values[True]-values[False])/max(abs(values[False]),1e-12)
                same_inputs=all(np.array_equal(hosts[True][0][k],v) for k,v in expected.items() if k not in DERIVED)
                check=dict(dataset=dataset,bucket=bucket,active_feature_max_abs=maxima,
                    loss=values,loss_relative=loss_relative,gradient_relative_l2=grad_relative,
                    finite=bool(finite and np.isfinite(list(values.values())).all()),same_topology_backbone_and_masks=same_inputs)
                check['passed']=bool(check['finite'] and same_inputs and max(maxima.values())<=2e-4 and loss_relative<=1e-4 and grad_relative<=1e-3)
                checks.append(check);print(json.dumps(dict(stage='agreement',**check)),flush=True)
                for _ in range(2):jax.block_until_ready(construct_jit(on_device[True][0]))
                begin=time.perf_counter()
                for _ in range(20):jax.block_until_ready(construct_jit(on_device[True][0]))
                geometry_ms=(time.perf_counter()-begin)*1000/20
                for online in (False,True):
                    for _ in range(2):jax.block_until_ready(steps[online](params,*on_device[online],jax.random.PRNGKey(17)))
                    begin=time.perf_counter()
                    for draw in range(20):jax.block_until_ready(steps[online](params,*on_device[online],jax.random.PRNGKey(17+draw)))
                    resident.append(dict(dataset=dataset,bucket=bucket,kind='online' if online else 'cached',
                        micro=micro,milliseconds=(time.perf_counter()-begin)*1000/20,geometry_only_ms=geometry_ms,
                        host_to_device_bytes=sum(a.nbytes for a in hosts[online][0].values())))
                del actual,expected,gradients,on_device,hosts
                for repeat in range(2):
                    for online in ((False,True) if repeat==0 else (True,False)):
                        source=sources[online];stream=Prefetcher(source,source.plans(bucket,'augmented',args.steps),
                            LoaderConfig(workers=args.workers,prefetch=2),to_device=to_device if args.device_prefetch else None)
                        begin=time.perf_counter();losses=[];read_bytes=0;transfer_bytes=0
                        for _,(g,m,touched) in stream:
                            read_bytes+=touched;transfer_bytes+=sum(a.nbytes for a in g.values())
                            for start in range(0,16,micro):
                                dg=jax.device_put(jax.tree.map(lambda a:a[start:start+micro],g));dm=jax.device_put(m[start:start+micro])
                                value,grad=steps[online](params,dg,dm,jax.random.PRNGKey(17+len(losses)))
                                jax.block_until_ready((value,grad));losses.append(float(value))
                        wall=time.perf_counter()-begin
                        if not np.isfinite(losses).all():raise ValueError('nonfinite timed objective')
                        result=dict(dataset=dataset,bucket=bucket,kind='online' if online else 'cached',repeat=repeat,
                            seconds=wall,structures_per_second=args.steps*16/wall,telemetry=stream.telemetry.summary(),
                            materialised_read_bytes=read_bytes,host_to_device_bytes=transfer_bytes)
                        results.append(result);print(json.dumps(dict(stage='timed',**result)),flush=True)
                        atomic_json(args.output/'progress.json',dict(results=results,checks=checks,resident=resident,storage=storage))
        finally:
            for source in sources.values():source.close()
    comparisons=[]
    for c in checks:
        group=[r for r in results if (r['dataset'],r['bucket'])==(c['dataset'],c['bucket'])]
        speeds={kind:float(np.median([r['structures_per_second'] for r in group if r['kind']==kind])) for kind in ('cached','online')}
        comparisons.append(dict(dataset=c['dataset'],bucket=c['bucket'],**speeds,
            throughput_ratio=speeds['online']/speeds['cached'],
            online_worst_wait_fraction=max(r['telemetry']['wait_fraction'] for r in group if r['kind']=='online')))
    report=dict(kind='online-backbone-edge-construction-pilot-v1',passed=bool(checks) and all(c['passed'] for c in checks),
        checks=checks,resident=resident,results=results,comparisons=comparisons,storage=storage,
        wall_s=time.perf_counter()-started,gpu=str(devices[0]),job=os.environ['SLURM_JOB_ID'],
        source_manifest_sha256=digest(args.pilot/'features/manifest.json'),
        feature_source_sha256=summary['feature_source_sha256'],snapshot_sha256=digest(Path(__file__)),
        training_source_sha256={name:digest(source_root/name) for name in
            ('pkanet/model.py','pkanet/attention_backend.py','pkanet/triton_attention.py','pkatrain/production_depth.py')},
        attention_backend='triton',
        device_prefetch=args.device_prefetch,
        training_enabled=False,notes=[
            'Warm pilot, not full-dataset cold random I/O; compact payload is on the same shared filesystem as cached inputs',
            'Precomputed exact neighbor lists; timing includes GPU feature construction but no radius search',
            'Both paths read existing expanded live native PINDER graphs; compact path discards native derived fields before H2D',
            'Online reconstructs both bound and free PINDER branches, so measures more arithmetic than augmented-partner-only reconstruction',
            'Base-A width64 ff128 orientation+dihedrals dropout0.1 synthetic paired/burial forward-backward; no optimizer or model training'])
    atomic_json(args.output/'report.json',report)
    print(json.dumps(dict(stage='complete',passed=report['passed'],wall_s=report['wall_s'],comparisons=comparisons)),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--compact',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--device-prefetch',action='store_true')
    args=p.parse_args();benchmark(args)


if __name__=='__main__':main()
