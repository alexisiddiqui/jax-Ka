"""Size-stratified, packed free-partner feature-cache pilot; no weight updates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import struct
import time
import traceback

import numpy as np
import zstandard

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_store import AugmentationStore, ReleaseWriter, verify_records
from .augmentation_graphs import INPUT_FIELDS, backbone_table, pair_inputs, same_padded

BOUNDS = (128, 256, 384, 512, 640, 768, 1024, 1280, 1536)
FEATURE_SOURCE_FILES = ('pkanet/graph.py', 'pkatrain/site_graph_data.py',
                        'pkatrain/production_graphs.py', 'pkatrain/production_inputs.py')


def stratum(row):
    s = row['summary']; n = s['n']
    b = next(i for i, limit in enumerate((150, 300, 500, 750, 1500)) if n <= limit)
    return '|'.join(map(str, (s['dataset'], s['split'], s['state'], b)))


def choose_sample(rows, count=256, stress_per_kind=4):
    """Uniform random sampling within every nonempty stratum; stress weights zero."""
    groups = defaultdict(list)
    for row in rows: groups[stratum(row)].append(row)
    if count < len(groups): raise ValueError('sample too small to cover every stratum')
    allocation = {k: 1 for k in groups}
    while sum(allocation.values()) < min(count, len(rows)):
        eligible = [k for k in sorted(groups) if allocation[k] < len(groups[k])]
        for k in eligible:
            if sum(allocation.values()) >= count: break
            allocation[k] += 1
    rng = np.random.default_rng(20261027); sample = []; seen = set()
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda r:r['id'])
        for i in rng.choice(len(group), allocation[key], replace=False):
            row = group[int(i)]; seen.add(row['id'])
            sample.append(dict(parent=row['id'], stratum=key, weight=len(group)/allocation[key],
                               population=len(group), sample_count=allocation[key], stress_only=False,
                               summary=row['summary']))
    stress = defaultdict(list)
    for row in rows:
        s = row['summary']
        if s['quota_met']: stress[s['dataset'], s['state']].append(row)
    for kind in sorted(stress):
        group=stress[kind]
        for row in sorted(group, key=lambda r:(-r['summary']['n'], r['id']))[:stress_per_kind]:
            if row['id'] in seen: continue
            seen.add(row['id']); sample.append(dict(parent=row['id'], stratum=stratum(row), weight=0.,
                population=len(groups[stratum(row)]), sample_count=allocation[stratum(row)],
                stress_only=True, summary=row['summary']))
    # Rare failure classes are explicit stress controls rather than dominating the probability sample.
    for dataset in ('pkpdb','pinder'):
        for status in ('partial_quota','preparation_failed','zero_control_failed'):
            group=sorted((r for r in rows if r['summary']['dataset']==dataset and r['summary']['status']==status),key=lambda r:r['id'])
            if group and not any(x['summary']['dataset']==dataset and x['summary']['status']==status for x in sample):
                row=group[0];seen.add(row['id']);sample.append(dict(parent=row['id'],stratum=stratum(row),weight=0.,
                    population=len(groups[stratum(row)]),sample_count=allocation[stratum(row)],stress_only=True,summary=row['summary']))
    return sample


def unit_graph(native, bb, chain, sequence, residue_ids):
    """A free partner in canonical order, with explicit maps back to its complex."""
    from pkanet.graph import geometry
    from .site_graph_data import build_site_graph
    residue_ids = np.asarray(residue_ids, np.int32)
    local = np.full(len(native['nodes']), -1, np.int32); local[residue_ids] = np.arange(len(residue_ids))
    query_ids = np.flatnonzero(local[native['query_residue']] >= 0).astype(np.int32)
    site_ids = np.flatnonzero(local[native['site_residue']] >= 0).astype(np.int32)
    nodes = native['nodes'][residue_ids].copy(); graph, valid = geometry(bb, chain)
    nodes[:, 23] = valid; graph.update(nodes=nodes, node_mask=native['node_mask'][residue_ids].copy())
    qr = local[native['query_residue'][query_ids]]; qg = native['query_group'][query_ids]
    eligible = native['query_site'][query_ids] >= 0
    site = build_site_graph(bb, chain, nodes, qr[eligible], qg[eligible])
    qs = np.full(len(qr), -1, np.int32); qs[eligible] = site['query_site']; site['query_site'] = qs
    graph.update(site, query_residue=qr, query_group=qg)
    if not np.array_equal(residue_ids[site['site_residue']], native['site_residue'][site_ids]):
        raise ValueError('partner site order changed')
    if not np.array_equal(site['site_type'], native['site_type'][site_ids]): raise ValueError('site type changed')
    if not np.array_equal(site_ids[qs[eligible]], native['query_site'][query_ids][eligible]):
        raise ValueError('partner query order changed')
    graph['nodes'] = np.concatenate((nodes, backbone_table(bb, chain, sequence)), axis=-1)
    graph.update(residue_ids=residue_ids, site_ids=site_ids, query_ids=query_ids)
    return graph


def replace_partner(control, variant):
    """Assemble the full free branch without copying native bound inputs into storage."""
    native = {k:control[k] for k in INPUT_FIELDS}
    n = len(native['nodes']); s = len(native['site_residue'])
    k = max(native['neighbors'].shape[1], variant['neighbors'].shape[1])
    sk = max(native['site_neighbors'].shape[1], variant['site_neighbors'].shape[1])
    free = {name:np.array(value, copy=True) for name,value in native.items()}
    for key, shape in dict(neighbors=(n,k), edge=(n,k,20), edge_mask=(n,k), switch=(n,k),
                           site_neighbors=(s,sk), site_edge=(s,sk,34), site_edge_mask=(s,sk), site_switch=(s,sk)).items():
        old = free[key]; out = np.zeros(shape, old.dtype)
        out[tuple(slice(0,x) for x in old.shape)] = old; free[key] = out
    free['edge_mask'][:,:native['neighbors'].shape[1]] = control['free_edge_mask']
    free['site_edge_mask'][:,:native['site_neighbors'].shape[1]] = control['free_site_edge_mask']
    residues = variant['residue_ids']; sites = variant['site_ids']
    for name in ('nodes','node_mask'): free[name][residues] = variant[name]
    for name in ('neighbors','edge','edge_mask','switch'):
        free[name][residues] = 0
        value = residues[variant[name]] if name == 'neighbors' else variant[name]
        free[name][residues,:value.shape[1]] = value
    for name in ('site_neighbors','site_edge','site_edge_mask','site_switch'):
        free[name][sites] = 0
        value = sites[variant[name]] if name == 'site_neighbors' else variant[name]
        free[name][sites,:value.shape[1]] = value
    partners = control['partners']; mask = free['edge_mask']
    if np.any(partners[free['neighbors']][mask] != np.broadcast_to(partners[:,None],mask.shape)[mask]):
        raise ValueError('free residue edge crosses partners')
    sp = partners[free['site_residue']]; mask = free['site_edge_mask']
    if np.any(sp[free['site_neighbors']][mask] != np.broadcast_to(sp[:,None],mask.shape)[mask]):
        raise ValueError('free site edge crosses partners')
    return free


def feature_shapes(n,k,q,s,sk):
    return dict(nodes=(n,38),node_mask=(n,),neighbors=(n,k),edge=(n,k,20),edge_mask=(n,k),switch=(n,k),
        query_residue=(q,),query_group=(q,),query_site=(q,),site_residue=(s,),site_type=(s,),site_mask=(s,),
        site_neighbors=(s,sk),site_edge=(s,sk,34),site_edge_mask=(s,sk),site_switch=(s,sk),
        residue_ids=(n,),query_ids=(q,),site_ids=(s,))


def build_layouts(root):
    """Pin compact unit dimensions to immutable frames; decode headers once offline."""
    root=Path(root);dims=[];dtypes=None
    with AugmentationStore(root/'features') as store:
        for row in store.rows:
            a=store.raw(row['id'])['arrays']
            d=(len(a['nodes']),a['neighbors'].shape[1],len(a['query_residue']),len(a['site_residue']),a['site_neighbors'].shape[1])
            shapes=feature_shapes(*d)
            if set(a)!=set(shapes) or any(a[k].shape!=v for k,v in shapes.items()):raise ValueError('feature schema changed')
            types={k:v.dtype.str for k,v in a.items()}
            if dtypes is not None and types!=dtypes:raise ValueError('feature dtypes differ')
            dtypes=types;dims.append(d)
        np.save(root/'feature-layouts.npy',np.asarray(dims,np.int32))
        result=dict(feature_manifest_sha256=digest(root/'features/manifest.json'),
            layout_sha256=digest(root/'feature-layouts.npy'),records=len(dims),dtypes=dtypes,
            shape_definition='base-A free partner, sorted arrays in KAAR1 record body; header skipped after full payload verification')
        atomic_json(root/'feature-layouts.json',result)
    return result


def _init_worker(cpus,coordinates):
    slot = multiprocessing.current_process()._identity[0]-1
    os.sched_setaffinity(0, {cpus[slot % len(cpus)]})
    global COORDINATE_STORE
    COORDINATE_STORE=AugmentationStore(coordinates)


def _build_parent(task):
    coordinate_path, runtime, item = task; started = time.monotonic()
    try:
        record = COORDINATE_STORE.raw(item['parent'])
        m, a = record['metadata'], record['arrays']
        if not a or not m.get('zero_quality',{}).get('geometry_valid',False):
            return dict(item=item, status='coordinate_unavailable', source_status=m['status'], wall_s=time.monotonic()-started)
        from .augmentation_sources import canonical_source
        from .production_inputs import _sequence_index
        native, bb, chain, partners, atoms, mapping = canonical_source(runtime, record)
        sequence = _sequence_index(chain); control = {k:native[k].copy() for k in INPUT_FIELDS}
        control['nodes'] = np.concatenate((control['nodes'],backbone_table(bb,chain,sequence)),axis=-1)
        control.update(partners=np.asarray(partners,np.int32),
            free_edge_mask=native.get('branch_edge_mask', np.stack((native['edge_mask'],)*2))[1],
            free_site_edge_mask=native.get('branch_site_edge_mask', np.stack((native['site_edge_mask'],)*2))[1])
        order = np.argsort(mapping); residues = np.asarray(mapping[order],np.int32); views=[]
        for candidate, xyz in [(0,a['zero_coordinates'])]+list(zip(m['selected'],a['coordinates'])):
            t = time.monotonic(); moved = xyz[a['backbone_atoms']][order].astype(bb.dtype,copy=False)
            graph = unit_graph(native,moved,chain[residues],sequence[residues],residues)
            free = replace_partner(control,graph)
            # A separate full-complex construction checks partition/remapping, not just dimensions.
            from .augmentation_graphs import free_graph
            all_bb=bb.copy();all_bb[residues]=moved
            reference=free_graph(native,all_bb,chain,partners)
            for name in INPUT_FIELDS:
                got=free[name][:,:24] if name=='nodes' else free[name]
                expected=reference[name]
                if name in ('neighbors','edge','edge_mask','switch','site_neighbors','site_edge','site_edge_mask','site_switch'):
                    # Native opposite-partner rows retain their original masked slots.
                    ids=residues if name in ('neighbors','edge','edge_mask','switch') else graph['site_ids']
                    if not same_padded(got[ids],expected[ids]):raise ValueError('independent free graph differs: '+name)
                elif name=='nodes':
                    if not np.array_equal(got,expected):raise ValueError('independent free nodes differ')
            views.append(dict(candidate=int(candidate),arrays=graph,build_and_verify_s=time.monotonic()-t))
        row=m['source'];key=f"{row['dataset']}/{row['id']}"
        return dict(item=item,status='built',control=control,control_key=key,views=views,
            n=len(bb),wall_s=time.monotonic()-started,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    except Exception as exc:
        return dict(item=item,status='feature_failed',error=type(exc).__name__+': '+str(exc),
                    traceback=traceback.format_exc()[-4000:],wall_s=time.monotonic()-started)


def build(args):
    cpus=int(os.environ['SLURM_CPUS_PER_TASK']);require_compute(threads=cpus,allow_comp1400=args.allow_comp1400)
    if (args.output/'features/manifest.json').exists():raise FileExistsError('pilot already built')
    args.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with AugmentationStore(args.coordinates) as store:
        sample=choose_sample(store.rows,args.sample);source_hash=digest(args.coordinates/'manifest.json')
    atomic_json(args.output/'sample.json',dict(parents=sample,coordinate_manifest_sha256=source_hash,
        population=114457,representative=sum(not x['stress_only'] for x in sample),stress=sum(x['stress_only'] for x in sample)))
    provenance=dict(kind='free-partner-feature-cache-pilot-v1',coordinate_manifest_sha256=source_hash,
        sample_sha256=digest(args.output/'sample.json'),training_enabled=False,
        features='base-A: 24 node columns plus 14-column backbone; orientation and dihedrals remain device-computed',
        source_sha256={f:digest(Path(__file__).parents[1]/f) for f in FEATURE_SOURCE_FILES},
        storage='free partner only; native control stored once per complex for transport and independent audit')
    results=[];expected=[];controls=[];native_ids=set();parent_rows=[]
    with ReleaseWriter(args.output/'features',provenance,zstd_level=1) as writer, ReleaseWriter(args.output/'controls',provenance,zstd_level=1) as native_writer:
        with ProcessPoolExecutor(args.workers,initializer=_init_worker,initargs=(sorted(os.sched_getaffinity(0)),args.coordinates)) as pool:
            tasks=[(args.coordinates,args.runtime,item) for item in sample]
            for result in pool.map(_build_parent,tasks,chunksize=1):
                item=result['item'];parent=item['parent'];entry=dict(item,status=result['status'],wall_s=result['wall_s'],feature_bytes=0,raw_bytes=0,views=0)
                if result['status']=='built':
                    key=result['control_key'];dataset=item['summary']['dataset'];n=result['n']
                    if key not in native_ids:
                        meta=dict(dataset=dataset,source_parent=key,n=n,split=item['summary']['split'])
                        native_writer.add(key,meta,result['control'],summary=meta);native_ids.add(key);controls.append(key)
                    for view in result['views']:
                        c=view['candidate'];rid=parent+f'/c{c:02d}';g=view['arrays']
                        dims=[n,max(result['control']['neighbors'].shape[1],g['neighbors'].shape[1]),
                              len(result['control']['query_residue']),len(result['control']['site_residue']),
                              max(result['control']['site_neighbors'].shape[1],g['site_neighbors'].shape[1])]
                        meta=dict(source_parent=parent,control_key=key,dataset=dataset,split=item['summary']['split'],candidate=c,
                            kind='zero' if c==0 else 'augmented',dims=dims,weight=item['weight'],stress_only=item['stress_only'],
                            stratum=item['stratum'],geometry_labels_reused=False)
                        row=writer.add(rid,meta,g,summary=meta);expected.append(rid)
                        entry['feature_bytes']+=row['bytes'];entry['raw_bytes']+=row['raw_bytes'];entry['views']+=1
                    entry['peak_rss_GiB']=result['peak_rss_GiB'];entry['n_complex']=n
                else:
                    entry.update({k:result[k] for k in ('error','traceback','source_status') if k in result})
                parent_rows.append(entry)
                if result['status']=='feature_failed' and sum(x['status']=='feature_failed' for x in parent_rows)<=3:
                    print(json.dumps(dict(parent=parent,error=result.get('traceback'))),flush=True)
                if len(parent_rows)%20==0:print(json.dumps(dict(parents=len(parent_rows),records=len(expected),counts=dict(Counter(x['status'] for x in parent_rows)))),flush=True)
        feature_manifest=writer.finish(expected_ids=expected);native_writer.finish(expected_ids=controls)
    # Verify every stored record and publish capacities before any timing.
    capacities={};native_capacities={}
    with AugmentationStore(args.output/'features') as features,AugmentationStore(args.output/'controls') as control_store:
        def check(record,row):
            m=record['metadata'];g=record['arrays'];control=control_store.raw(m['control_key'])['arrays']
            free=replace_partner(control,g)
            if not np.array_equal(free['nodes'][g['residue_ids'],-14:],g['nodes'][:,-14:]):raise ValueError('backbone table changed')
        verify_records(args.output/'features',features.rows,check_record=check,expected_files=feature_manifest['files'])
        for row in features.rows:
            m=row['summary'];dataset=m['dataset'];dims=m['dims'];b=str(next(b for b in BOUNDS if dims[0]<=b))
            key=dataset+':'+b;v=[int(b),*dims[1:]]
            capacities[key]=[max(x,y) for x,y in zip(capacities.get(key,[0]*5),v)]
        for row in control_store.rows:
            m=row['summary'];g=control_store.raw(row['id'])['arrays'];n=m['n'];b=str(next(b for b in BOUNDS if n<=b));key=m['dataset']+':'+b
            v=[int(b),g['neighbors'].shape[1],len(g['query_residue']),len(g['site_residue']),g['site_neighbors'].shape[1]]
            native_capacities[key]=[max(x,y) for x,y in zip(native_capacities.get(key,[0]*5),v)]
    projected_bytes=sum(x['weight']*x['feature_bytes'] for x in parent_rows)
    projected_s=sum(x['weight']*x['wall_s'] for x in parent_rows)
    file_stats=[p for p in args.output.rglob('*') if p.is_file()]
    summary=dict(passed=not any(x['status']=='feature_failed' for x in parent_rows),parents=len(sample),representative=sum(not x['stress_only'] for x in sample),
        counts=dict(Counter(x['status'] for x in parent_rows)),records=len(expected),controls=len(controls),
        zstd_level=1,shard_bytes=2*1024**3,feature_bytes=feature_manifest['compressed_bytes'],
        projected_full_feature_bytes=projected_bytes,projected_full_payload_shards=math.ceil(projected_bytes/(2*1024**3)),
        projected_consolidated_files=math.ceil(projected_bytes/(2*1024**3))+5,
        projected_build_worker_hours=projected_s/3600,wall_s=time.monotonic()-start,
        capacities=capacities,native_capacities=native_capacities,parents_detail=parent_rows,
        snapshot_sha256=digest(Path(__file__)),feature_source_sha256=provenance['source_sha256'],
        feature_manifest_sha256=digest(args.output/'features/manifest.json'),
        files_at_build=len(file_stats),bytes_at_build=sum(p.stat().st_size for p in file_stats),training_enabled=False,
        limitations=['Native controls are pilot transport only; full production reuses native stores',
            'Build timing includes independent full-graph verification and IPC/writing is additional',
            'Size-stratified projection includes unavailable coordinates and failed feature mappings as explicit zero-cache coverage',
            'Full-dataset random I/O and future model inputs are not established by this pilot'])
    atomic_json(args.output/'build-summary.json',summary)
    if any(x['status']=='feature_failed' for x in parent_rows):
        raise RuntimeError('Feature failures retained in build-summary.json; repair the adapter before benchmarking')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('parents_detail','capacities','native_capacities')}),flush=True)


class FeatureSource:
    """Native production reads plus one selected partner feature record; bounded prefetch."""
    def __init__(self,path,workers,batch,*,dataset='pinder',runtime=None,pad_transform=None):
        self.path=Path(path);self.dataset=dataset;self.batch=batch;self.pad_transform=pad_transform
        self.store=AugmentationStore(self.path/'features',verify_reads=False)
        compact=(self.path/'partitions/manifest.json').exists()
        if compact and runtime is None:
            self.store.close()
            raise ValueError('production feature cache requires live native training stores')
        self.controls=AugmentationStore(self.path/('partitions' if compact else 'controls'),verify_reads=False)
        self.pool=ThreadPoolExecutor(workers)
        summary=json.loads((self.path/'build-summary.json').read_text());prefix=dataset+':'
        self.capacities={k.split(':')[1]:v for k,v in summary['capacities'].items() if k.startswith(prefix)}
        self.native_capacities={k.split(':')[1]:v for k,v in summary['native_capacities'].items() if k.startswith(prefix)}
        self.by_bucket=defaultdict(dict);self.live=None
        self.layouts=None;self.layout_types=None
        if dataset=='pinder' and (self.path/'feature-layouts.json').exists():
            layout=json.loads((self.path/'feature-layouts.json').read_text())
            if layout['feature_manifest_sha256']!=digest(self.path/'features/manifest.json') or layout['layout_sha256']!=digest(self.path/'feature-layouts.npy'):
                raise ValueError('feature layout provenance differs')
            dimensions=np.load(self.path/'feature-layouts.npy',mmap_mode='r')
            if dimensions.shape!=(len(self.store.rows),5):raise ValueError('feature layout shape differs')
            self.layouts={row['id']:tuple(map(int,d)) for row,d in zip(self.store.rows,dimensions)}
            self.layout_types={name:np.dtype(t) for name,t in layout['dtypes'].items()}
        self.partitions={row['id']:self.controls.raw(row['id'])['arrays']['partners'].copy()
                         for row in self.controls.rows if row['summary']['dataset']==dataset}
        if runtime is not None:
            from .production_graphs import ProductionStore,output
            from .production_inputs import BackboneTable
            self.live=ProductionStore(output(runtime,dataset)/'store-v2');self.backbone=BackboneTable(runtime,dataset)
            live_manifest=json.loads((output(runtime,dataset)/'manifest-v1.json').read_text())
            for bucket in self.capacities:
                capacity=live_manifest['capacities'][bucket]
                self.native_capacities[bucket]=list(capacity)
                self.capacities[bucket]=[max(a,b) for a,b in zip(self.capacities[bucket],capacity)]
        for row in self.store.rows:
            m=row['summary']
            if m['dataset']!=dataset:continue
            b=str(next(b for b in BOUNDS if m['dims'][0]<=b))
            self.by_bucket[b].setdefault(m['source_parent'],{})[m['candidate']]=row['id']
        for b in list(self.by_bucket):
            self.by_bucket[b]={p:v for p,v in self.by_bucket[b].items() if any(c>0 for c in v)}
            if not self.by_bucket[b]:del self.by_bucket[b];self.capacities.pop(b,None);self.native_capacities.pop(b,None)

    def plans(self,bucket,kind,steps):
        parents=sorted(self.by_bucket[bucket]);rng=np.random.default_rng(20261027+int(bucket));order=[]
        while len(order)<steps*self.batch:order.extend(rng.permutation(parents).tolist())
        plans=[]
        for step in range(steps):
            ids=[]
            for parent in order[step*self.batch:(step+1)*self.batch]:
                choices=self.by_bucket[bucket][parent];cs=sorted(c for c in choices if c>0)
                c=0 if kind=='native' else cs[int(rng.integers(len(cs)))]
                ids.append((parent,choices[c]))
            plans.append((bucket,kind,ids))
        return plans

    def _variant(self,rid):
        if self.layouts is None:return self.store.raw(rid)['arrays']
        row=self.store.by_id[rid];blob=os.pread(self.store.fds[row['shard']],row['bytes'],row['offset'])
        raw=zstandard.ZstdDecompressor().decompress(blob,max_output_size=row['raw_bytes'],allow_extra_data=False)
        if len(raw)!=row['raw_bytes'] or raw[:6]!=b'KAAR1\0':raise ValueError('invalid feature frame')
        position=14+struct.unpack_from('<Q',raw,6)[0];out={}
        for name,shape in sorted(feature_shapes(*self.layouts[rid]).items()):
            dtype=self.layout_types[name];size=math.prod(shape)
            out[name]=np.frombuffer(raw,dtype=dtype,count=size,offset=position).reshape(shape);position+=size*dtype.itemsize
        if position!=len(raw):raise ValueError('feature body differs from verified layout')
        return out

    def _control(self,key):
        if self.live is None:return self.controls.raw(key)['arrays']
        cid=key.split('/',1)[1];raw=self.live.raw(cid);table=self.backbone(cid)
        g={k:raw[k] for k in INPUT_FIELDS};g['nodes']=np.concatenate((g['nodes'],table),axis=-1)
        g['free_edge_mask']=raw.get('branch_edge_mask',np.stack((raw['edge_mask'],)*2))[1]
        g['free_site_edge_mask']=raw.get('branch_site_edge_mask',np.stack((raw['site_edge_mask'],)*2))[1]
        # Isolation validation uses the pinned canonical partition, not a guessed chain alias.
        g['partners']=self.partitions[key]
        return g

    def load(self,spec):
        bucket,kind,ids=spec;n,k,q,s,sk=(self.native_capacities if kind=='native' else self.capacities)[bucket]
        branches=2 if self.dataset=='pinder' else 1
        shapes=dict(nodes=(n,38),node_mask=(n,),neighbors=(n,k),edge=(n,k,20),edge_mask=(n,k),switch=(n,k),
            query_residue=(q,),query_group=(q,),site_residue=(s,),site_type=(s,),site_mask=(s,),
            site_neighbors=(s,sk),site_edge=(s,sk,34),site_edge_mask=(s,sk),site_switch=(s,sk),query_site=(q,))
        integers={'neighbors','query_residue','query_group','site_residue','site_type','site_neighbors','query_site'}
        graph={name:np.zeros((len(ids),branches,*shape),bool if name.endswith('mask') else np.int32 if name in integers else np.float32) for name,shape in shapes.items()}
        mask=np.zeros((len(ids),q),bool)
        def one(slot):
            parent,rid=ids[slot];meta=self.store.by_id[rid]['summary']
            variant=self._variant(rid) if kind!='native' else None
            if self.dataset=='pkpdb' and variant is not None:
                # The cached assembly already contains the complete single-branch input.
                views=(variant,);touched=sum(x.nbytes for x in variant.values());queries=variant['query_site']
            else:
                control=self._control(meta['control_key'])
                bound={name:control[name] for name in INPUT_FIELDS};free=dict(bound)
                free['edge_mask']=control['free_edge_mask'];free['site_edge_mask']=control['free_site_edge_mask']
                touched=sum(x.nbytes for x in control.values());queries=bound['query_site']
                views=(bound,free) if branches==2 else (free,)
            for branch,g in enumerate(views):
                for name,value in g.items():
                    if name not in graph:continue
                    if name=='query_site':value=np.maximum(value,0)
                    graph[name][(slot,branch,*(slice(0,x) for x in value.shape))]=value
            if self.dataset=='pinder' and variant is not None:
                # Update the final padded buffer directly; the preflight already verified mappings/isolation.
                residues=variant['residue_ids'];sites=variant['site_ids']
                for name in ('nodes','node_mask'):graph[name][slot,1,residues]=variant[name]
                for name in ('neighbors','edge','edge_mask','switch'):
                    target=graph[name][slot,1];target[residues]=0
                    value=residues[variant[name]] if name=='neighbors' else variant[name]
                    target[residues,:value.shape[1]]=value
                for name in ('site_neighbors','site_edge','site_edge_mask','site_switch'):
                    target=graph[name][slot,1];target[sites]=0
                    value=sites[variant[name]] if name=='site_neighbors' else variant[name]
                    target[sites,:value.shape[1]]=value
                touched+=sum(x.nbytes for x in variant.values())
            mask[slot,:len(queries)]=queries>=0
            return touched
        touched=sum(self.pool.map(one,range(len(ids))))
        if self.pad_transform is not None:graph=self.pad_transform(graph)
        return graph,mask,touched

    def provenance(self):return self.store.manifest
    def close(self):
        self.pool.shutdown();self.store.close();self.controls.close()
        if self.live is not None:self.live.close()


def verify_live(args):
    from .production_graphs import ProductionStore,output
    from .production_inputs import BackboneTable
    stores={};tables={};checked=0;max_error=0.
    import pkanet
    source_root=Path(pkanet.__file__).parents[1]
    summary=json.loads((args.output/'build-summary.json').read_text())
    with AugmentationStore(args.output/'features') as features:
        if digest(args.output/'features/manifest.json')!=summary['feature_manifest_sha256']:
            raise ValueError('feature manifest changed after the build')
        features.verify_files()
    for name,expected in summary['feature_source_sha256'].items():
        if digest(source_root/name)!=expected:raise ValueError('live feature source differs: '+name)
    with AugmentationStore(args.output/'controls') as controls:
        controls.verify_files()
        for row in controls.rows:
            dataset,cid=row['id'].split('/',1)
            if dataset not in stores:
                stores[dataset]=ProductionStore(output(args.runtime,dataset)/'store-v2');tables[dataset]=BackboneTable(args.runtime,dataset)
            raw=stores[dataset].raw(cid);g=controls.raw(row['id'])['arrays']
            for key in INPUT_FIELDS:
                expected=raw[key];got=g[key][:,:24] if key=='nodes' else g[key]
                if key in ('edge','site_edge'):
                    err=float(np.max(np.abs(got-expected),initial=0));max_error=max(max_error,err)
                    if err>2e-5:raise ValueError('live native geometry differs: '+row['id']+'/'+key)
                elif not np.array_equal(got,expected):raise ValueError('live native identity differs: '+row['id']+'/'+key)
            if not np.array_equal(g['nodes'][:,-14:],tables[dataset](cid)):raise ValueError('live backbone differs')
            for name,field in (('free_edge_mask','branch_edge_mask'),('free_site_edge_mask','branch_site_edge_mask')):
                if field in raw and not np.array_equal(g[name],raw[field][1]):raise ValueError('free branch mask differs')
            checked+=1
    for s in stores.values():s.close()
    result=dict(passed=True,checked=checked,max_geometry_error=max_error,
        feature_manifest_sha256=digest(args.output/'features/manifest.json'),
        live_verification_sha256={d:digest(output(args.runtime,d)/'store-v2/verification.json') for d in stores})
    atomic_json(args.output/'live-compatibility.json',result);print(json.dumps(result),flush=True)


def benchmark(args):
    import pkatrain.augmentation_preflight as probe
    cpus=int(os.environ['SLURM_CPUS_PER_TASK']);require_compute(threads=cpus,gpu_benchmark=True)
    if not args.loader_workers or min(args.loader_workers)<1 or max(args.loader_workers)>cpus:
        raise ValueError('loader workers must fit the CPU allocation')
    if args.steps<20:raise ValueError('GPU preflight requires at least twenty timed batches')
    compatibility=json.loads((args.output/'live-compatibility.json').read_text())
    if not compatibility['passed'] or compatibility['feature_manifest_sha256']!=digest(args.output/'features/manifest.json'):
        raise ValueError('live compatibility not verified for this cache')
    live=json.loads(args.protocol.read_text());arch=live['architecture']
    if any(arch.get(k)!=v for k,v in dict(width=64,ff=128,orientation_edges=13,dihedral_features=6).items()):
        raise ValueError('unsupported production architecture')
    if not live['dropout'].startswith('oGQT residual dropout 0.1 '):raise ValueError('unsupported dropout')
    results=args.results or args.output;results.mkdir(parents=True,exist_ok=True)
    for dataset in args.datasets:
        def make_source(path,workers,batch,pad_transform=None):
            source=FeatureSource(path,workers,batch,dataset=dataset,runtime=args.runtime,pad_transform=pad_transform)
            if args.buckets:
                wanted=set(args.buckets)
                for b in list(source.capacities):
                    if b not in wanted:
                        source.capacities.pop(b);source.native_capacities.pop(b);source.by_bucket.pop(b,None)
            if not source.capacities:
                source.close();raise ValueError('requested buckets contain no eligible parents')
            return source
        probe.PilotSource=make_source
        report=probe.gpu_benchmark(args.output,results/f'gpu-{dataset}.json',workers=tuple(args.loader_workers),
            prefetch=(2,4),steps=args.steps,repeats=3,batch=live['batch'],protocol=args.protocol)
        report.update(dataset=dataset,branches=2 if dataset=='pinder' else 1,
            native_reader='live ProductionStore and BackboneTable',
            augmentation_reader='one separately indexed free-partner record; no native bound duplication per view',
            candidate_sampling='deterministic shuffled parent cycle and uniformly sampled saved augmentation',
            scope='matched forward/backward compute proxy with synthetic loss; no optimiser or geometry labels',
            pilot_source_sha256=digest(Path(__file__)),layout_index_sha256=digest(args.output/'feature-layouts.json') if (args.output/'feature-layouts.json').exists() else None)
        atomic_json(results/f'gpu-{dataset}.json',report)
        print(json.dumps(dict(dataset=dataset,passed=report['passed'],viable_settings=report['viable_settings'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','layouts','verify-live','benchmark'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--coordinates',type=Path)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--sample',type=int,default=256)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--allow-comp1400',action='store_true')
    p.add_argument('--protocol',type=Path);p.add_argument('--loader-workers',type=int,nargs='+',default=[8,16,32])
    p.add_argument('--results',type=Path)
    p.add_argument('--datasets',nargs='+',choices=('pkpdb','pinder'),default=['pkpdb','pinder'])
    p.add_argument('--buckets',nargs='+')
    p.add_argument('--steps',type=int,default=20);args=p.parse_args()
    if args.command=='build':build(args)
    elif args.command=='layouts':
        require_compute(threads=1,allow_comp1400=args.allow_comp1400);print(json.dumps(build_layouts(args.output)),flush=True)
    elif args.command=='verify-live':
        require_compute(threads=1,gpu_benchmark=True);verify_live(args)
    else:benchmark(args)


if __name__=='__main__':main()
