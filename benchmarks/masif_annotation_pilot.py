"""Size-stratified dense MaSIF annotation throughput; no ANM or model training."""
from __future__ import annotations

import argparse
import csv
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import gzip
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
import sys

import numpy as np
import psutil

from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import ReleaseWriter,AugmentationStore,finalize_workers
from augmentation_masif_audit import write_pdb

BOUNDS=(150,300,600,10**9)


def select_units(units,runtime,*,dataset='all',split='all',separate_chains=False,include_bound=False):
    units=[u for u in units if (dataset=='all' or u['dataset']==dataset) and
           (split=='all' or u['split']==split) and (include_bound or u['family']!='pinder_bound')]
    if not separate_chains:return units
    counts={}
    for name in ['pkpdb.tsv','pkpdb-val.tsv']:
        with (runtime/'training/pool-v4'/name).open() as stream:
            counts.update({r['id']:int(r['n_chains']) for r in csv.DictReader(stream,delimiter='\t')})
    result=[]
    for u in units:
        if u['dataset']!='pkpdb':result.append(u);continue
        count=counts[u['cid']]
        for index in range(count):
            result.append(dict(u,id=f'pkpdb/{u["cid"]}/chain-{index:04d}',state=f'chain-{index:04d}',
                family='pkpdb_chain',chain_index=index,expected_chains=count,assembly_n=u['n'],
                n=max(1,int(np.ceil(u['n']/count))),size_is_metadata_estimate=True))
    return result


def cell(row):return row['family']+'|'+str(next(i for i,b in enumerate(BOUNDS) if row['n']<=b))


def load_cohort(units,path,runtime,coordinate_manifest_sha256):
    """Select a pinned node cohort without renumbering canonical chain IDs."""
    data=json.loads(Path(path).read_text())
    if (data.get('format')!='masif-native-cohort-v1' or data.get('dataset')!='pkpdb'
            or data.get('split')!='train'):
        raise ValueError('cohort must contain native pKPDB training chains')
    if (data.get('coordinate_manifest_sha256')!=coordinate_manifest_sha256
            or data.get('training_tsv_sha256')!=digest(runtime/'training/pool-v4/pkpdb.tsv')):
        raise ValueError('cohort source inventory changed')
    lookup={u['id']:u for u in units};result=[];seen=set()
    for row in data['units']:
        rid=row['id'];unit=lookup.get(rid)
        if rid in seen or unit is None:raise ValueError('duplicate or absent cohort unit: '+rid)
        if unit['dataset']!='pkpdb' or unit['split']!='train' or 'chain_index' not in unit:
            raise ValueError('cohort unit is not a separate training chain')
        if type(row['n']) is not int or row['n']<=0:raise ValueError('cohort contains an empty protein chain')
        cost=float(row.get('scheduling_cost',row['n']))
        if not np.isfinite(cost) or cost<=0:raise ValueError('invalid cohort scheduling cost')
        result.append(dict(unit,n=row['n'],source_chain_label=row['source_chain_label'],
                           scheduling_cost=cost,size_is_metadata_estimate=True))
        seen.add(rid)
    if not result or not data.get('global_cohort_sha256'):raise ValueError('empty or unpinned cohort')
    return result,dict(cohort_manifest_sha256=digest(path),global_cohort_sha256=data['global_cohort_sha256'])


def inventory(coordinates):
    # Metadata only; failed preparation records stay in the source population.
    manifest=json.loads((coordinates/'manifest.json').read_text())
    rows=json.loads((coordinates/'index.json').read_text())['records'];units=[];pairs=defaultdict(list)
    for row in rows:
        dataset,cid,state=row['id'].split('/',2);s=row['summary']
        unit=dict(id=row['id'],dataset=dataset,cid=cid,state=state,split=s['split'],n=s['n'],
            family='pkpdb_assembly' if dataset=='pkpdb' else 'pinder_partner')
        units.append(unit)
        if dataset=='pinder':pairs[cid].append(unit)
    for cid,partners in pairs.items():
        if {u['state'] for u in partners}!={'A','B'}:raise ValueError('PINDER partner inventory incomplete')
        units.append(dict(id='pinder/'+cid+'/AB',dataset='pinder',cid=cid,state='AB',split=partners[0]['split'],
            n=sum(u['n'] for u in partners),family='pinder_bound'))
    return units,manifest


def choose_sample(units):
    groups=defaultdict(list)
    for row in units:groups[cell(row)].append(row)
    rng=np.random.default_rng(20261115);sample=[];seen=set()
    for key,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['id']);count=min(2,len(rows))
        for index in rng.choice(len(rows),count,replace=False):
            row=rows[int(index)];sample.append(dict(row,stratum=key,population=len(rows),weight=len(rows)/count,stress=False));seen.add(row['id'])
    for family in sorted({r['family'] for r in units}):
        row=min((r for r in units if r['family']==family),key=lambda r:(-r['n'],r['id']))
        if row['id'] not in seen:sample.append(dict(row,stratum=cell(row),population=len(groups[cell(row)]),weight=0.,stress=True))
    return sample


def canonical_chain_indices(chain,index,expected_chains):
    """Keep the training chain labels even when another chain has no protein nodes."""
    chain=np.asarray(chain)
    if not 0<=index<expected_chains or np.any((chain<0)|(chain>=expected_chains)):
        raise ValueError('pKPDB chain inventory differs from training metadata')
    canonical=np.flatnonzero(chain==index).astype(np.int32)
    if not len(canonical):raise ValueError('empty pKPDB chain: no canonical protein residues')
    return canonical


def canonical_partner_indices(chain,indices,expected_chains):
    """Keep all chains of one partner in stable training-residue order."""
    if not indices or len(indices)!=len(set(indices)):
        raise ValueError('empty or duplicate partner chain selection')
    for index in indices:canonical_chain_indices(chain,index,expected_chains)
    return np.flatnonzero(np.isin(chain,indices)).astype(np.int32)


def export_native(runtime,item,path,*,surface_policy=None):
    from jaxpropka.parameters import THREE_TO_INDEX
    from scipy.spatial import cKDTree
    if item['dataset']=='pkpdb':
        from pkatrain.production_sidechains import _pkpdb_atoms
        from pkatrain.production_inputs import _residues
        bb,atoms,identity=_pkpdb_atoms(runtime,item['cid']);other,chain,other_identity=_residues('pkpdb',runtime,item['cid'])
        if not np.array_equal(identity,other_identity) or not np.allclose(bb,other,atol=.001,rtol=0):raise ValueError('pKPDB canonical atom order differs')
        canonical=np.arange(len(bb),dtype=np.int32)
        if item.get('source_files_sha256'):
            from pkatrain.production_graphs import source,PKPDB,PKPDB_STRUCTURES
            src=source(runtime,f'{PKPDB}/entries/{item["cid"]}')
            for name,expected in item['source_files_sha256'].items():
                file=(source(runtime,f'{PKPDB_STRUCTURES}/{item["cid"][1:3]}/{item["cid"]}.cif.gz')
                      if name=='structure' else src/name)
                if digest(file)!=expected:raise ValueError('pinned pKPDB partner input changed: '+name)
        if 'chain_indices' in item:
            from pkatrain.production_graphs import source,PKPDB
            src=source(runtime,f'{PKPDB}/entries/{item["cid"]}')
            labels=json.loads((src/'defects.json').read_text())['sequences']
            if len(labels)!=item['expected_chains'] or [labels[i]['chain'] for i in item['chain_indices']]!=item['source_chain_labels']:
                raise ValueError('pKPDB partner chain labels differ from inventory')
            canonical=canonical_partner_indices(chain,item['chain_indices'],item['expected_chains'])
            if canonical.tolist()!=item['canonical_residue_ids']:raise ValueError('pKPDB partner canonical mapping changed')
            bb=bb[canonical];identity=identity[canonical];chain=chain[canonical]
            atoms=[atoms[int(i)] for i in canonical]
        if 'chain_index' in item:
            canonical=canonical_chain_indices(chain,item['chain_index'],item['expected_chains'])
            bb=bb[canonical];identity=identity[canonical];chain=chain[canonical]
            atoms=[atoms[int(i)] for i in canonical]
            from pkatrain.production_graphs import source,PKPDB
            labels=json.loads((source(runtime,f'{PKPDB}/entries/{item["cid"]}')/'defects.json').read_text())['sequences']
            if len(labels)!=item['expected_chains']:raise ValueError('pKPDB chain inventory differs from training metadata')
            source_chain_label=labels[item['chain_index']]['chain']
            if item.get('source_chain_label',source_chain_label)!=source_chain_label:
                raise ValueError('pKPDB source chain label differs from pinned cohort')
    elif item['dataset']=='sabdab':
        from pkabench.sabdab_units import load_prepared
        topology,metadata=load_prepared(item['prepared_unit'])
        canonical=np.asarray([i for i,k in enumerate(topology.keys)
            if item['state']=='AB' or metadata['side_by_chain'][k.chain]==item['state']],np.int32)
        if canonical.tolist()!=item['canonical_residue_ids'] or not len(canonical):
            raise ValueError('SAbDab biological-partner residue map changed')
        bb=topology.backbone[canonical];chain=topology.chain_index[canonical];identity=topology.native_index[canonical]
        atoms=[{str(a.atom_name):a.coord for a in topology.residue(int(i))
            if str(a.element).upper() not in ('H','D')} for i in canonical]
    else:
        from jaxpropka.topology import load_topology
        from biotite.structure.io import pdbx
        from pkatrain.production_graphs import source,PINDER
        src=source(runtime,f'{PINDER}/entries/{item["cid"]}')
        for name,expected in item.get('source_files_sha256',{}).items():
            if digest(src/name)!=expected:raise ValueError('pinned partner input changed: '+name)
        with gzip.open(src/'AB.cif.gz','rt') as stream:file=pdbx.CIFFile.read(stream)
        atom_array=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
        topology=load_topology(atom_array,gap_policy='cap',freeze_disulfides=True)
        canonical=np.arange(topology.n_residues,dtype=np.int32)
        if item['state']!='AB':
            lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(topology.keys)}
            mapping=json.loads((src/f'map_{item["state"]}.json').read_text())
            canonical=np.array(sorted(lookup[c,int(n),ins] for _,_,c,n,ins in mapping if (c,int(n),ins) in lookup),np.int32)
        if not len(canonical):raise ValueError('empty PINDER partner')
        bb=topology.backbone[canonical];chain=topology.chain_index[canonical];identity=topology.native_index[canonical]
        atoms=[{str(a.atom_name):a.coord for a in topology.residue(int(i)) if str(a.element).upper() not in ('H','D')} for i in canonical]
        if item.get('unit_policy'):
            keys={(str(c),int(n),str(ins)) for _,_,c,n,ins in mapping}
            if len(canonical)!=len(keys):raise ValueError('partner map contains residues absent from canonical protein topology')
    segments=np.zeros(len(bb),np.int32)
    for i in range(1,len(bb)):
        segments[i]=segments[i-1]+int(chain[i]!=chain[i-1] or np.linalg.norm(bb[i-1,2]-bb[i,0])>2.)
    added=[]
    if surface_policy:
        from masif_surface_quality import SURFACE_POLICY,complete_terminal_oxygens
        if surface_policy!=SURFACE_POLICY:raise ValueError('unknown surface preparation policy')
        atoms,added=complete_terminal_oxygens(atoms,segments)
    inverse={i:name for name,i in THREE_TO_INDEX.items()}
    names=[];xyz=[];owners=[]
    for r,atom in enumerate(atoms):
        for name,position in sorted(atom.items()):
            if name[0] in ('H','D'):continue
            names.append(name);xyz.append(position);owners.append(r)
    sg=[(r,atom['SG']) for r,atom in enumerate(atoms) if 'SG' in atom]
    bonds=[]
    if len(sg)>1:
        for a,b in sorted(cKDTree(np.array([x for _,x in sg])).query_pairs(2.3)):bonds.append([sg[a][0],sg[b][0]])
    meta=dict(atom_names=names,residue_names=[inverse[int(i)] for i in identity],disulfides=bonds,
        residue_keys=[[int(i)] for i in canonical])
    write_pdb(path,np.asarray(xyz),meta,dict(segments=segments,atom_residue=np.asarray(owners,np.int32)))
    return dict(canonical_residue_ids=canonical.tolist(),actual_residues=len(bb),heavy_atoms=len(xyz),
        surface_policy=surface_policy,added_terminal_oxygen_canonical_ids=[int(canonical[i]) for i in added],
        source_chain_label=source_chain_label if item['dataset']=='pkpdb' and 'chain_index' in item else None,
        pdb_sha256=digest(path),source='native canonical training structure; no ANM/minimisation')


def worker(task):
    args,slot,cpus,items=task;os.sched_setaffinity(0,set(cpus));started=time.monotonic()
    output=args.output;runtime=args.masif;records=[];expected=[];peak=0
    packed=output/'fingerprints/workers'/f'{slot:04d}'
    provenance=dict(kind='dense-native-masif-pilot-v1',sample_sha256=digest(output/'sample.json'),
        checkpoint_sha256=args.checkpoint_sha256,
        feature_adapter_sha256=digest(Path(__file__).with_name('augmentation_masif_bridge.py')),
        checkpoint_validator_sha256=digest(Path(__file__).with_name('masif_checkpoint.py')),
        all_molecular_components=args.all_components,
        bridge_sha256=digest(Path(__file__).with_name('masif_annotation_bridge.py')))
    with tempfile.TemporaryDirectory(prefix=f'masif-annotation-{os.environ["SLURM_JOB_ID"]}-{slot}-',dir='/tmp') as temporary, \
            ReleaseWriter(packed,provenance,zstd_level=1) as writer:
        temporary=Path(temporary);jobs=[];by_id={}
        for index,item in enumerate(items):
            root=temporary/f'{index:03d}';root.mkdir();t=time.monotonic()
            try:
                info=export_native(args.runtime,item,root/'input.pdb')
                job=dict(id=item['id'],input=f'/audit/{index:03d}/input.pdb',output=f'/audit/{index:03d}/features.npz',
                    receipt=f'/audit/{index:03d}/receipt.json',seed=20261115,canonical_residue_ids=info['canonical_residue_ids'],
                    all_components=args.all_components)
                jobs.append(job);by_id[item['id']]=(item,root,dict(info,source_export_s=time.monotonic()-t))
            except Exception as exc:records.append(dict(item,status='failed',stage='source_export',error=str(exc),
                traceback=traceback.format_exc()[-4000:],wall_s=time.monotonic()-t))
        atomic_json(temporary/'jobs.json',jobs)
        command=[str(runtime/'setup-env/bin/udocker'),'--quiet','run','--env=LD_LIBRARY_PATH=/compat',
            '--volume='+str(runtime/'compat')+':/compat','--env=OMP_NUM_THREADS=1','--env=OPENBLAS_NUM_THREADS=1',
            '--env=MKL_NUM_THREADS=1','--env=CUDA_VISIBLE_DEVICES=',
            '--volume='+str(temporary)+':/audit','--volume='+str(Path(__file__).resolve().parent)+':/code',
            '--volume='+str(args.checkpoint.resolve().parent)+':/checkpoint',
            '--env=MASIF_CHECKPOINT_PREFIX=/checkpoint/'+args.checkpoint.name,
            'masif-audit','python','/code/masif_annotation_bridge.py','/audit/jobs.json']
        env=dict(os.environ,UDOCKER_DIR=str(runtime/'udocker'))
        with (output/f'worker-{slot:02d}.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            while process.poll() is None:
                rss=0
                try:
                    tree=[psutil.Process(process.pid)]+psutil.Process(process.pid).children(recursive=True)
                    for p in tree:
                        try:rss+=p.memory_info().rss
                        except psutil.Error:pass
                except psutil.Error:pass
                peak=max(peak,rss+psutil.Process(os.getpid()).memory_info().rss)
                atomic_json(output/f'progress-{slot:02d}.json',dict(status='running',worker=slot,peak_tree_rss_bytes=peak,
                    receipts=sum((root/'receipt.json').exists() for _,root,_ in by_id.values()),total=len(items),wall_s=time.monotonic()-started))
                time.sleep(1)
        for rid,(item,root,info) in by_id.items():
            if (root/'receipt.json').exists():receipt=json.loads((root/'receipt.json').read_text())
            else:receipt=dict(id=rid,status='failed',stage='container',error=f'container exit {process.returncode} without receipt')
            record=dict(item,**{k:v for k,v in receipt.items() if k!='id'},**info)
            if receipt['status']=='complete':
                if receipt.get('checkpoint_sha256')!=args.checkpoint_sha256:
                    raise ValueError('annotation checkpoint differs from release provenance')
                with np.load(root/'features.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
                t=time.monotonic();row=writer.add(rid,dict(item,receipt=receipt,source=info),arrays,summary=item)
                record.update(compressed_bytes=row['bytes'],raw_bytes=row['raw_bytes'],pack_s=time.monotonic()-t);expected.append(rid)
            records.append(record)
        writer.finish(expected_ids=expected)
    result=dict(worker=slot,parents=records,peak_tree_rss_bytes=peak,wall_s=time.monotonic()-started)
    atomic_json(output/f'worker-{slot:02d}.json',result)
    atomic_json(output/f'progress-{slot:02d}.json',dict(status='complete',worker=slot,total=len(items),peak_tree_rss_bytes=peak,wall_s=result['wall_s']))
    return result


def summarize(output,sample,results,wall_s):
    parents=[r for w in results for r in w['parents']];families={};stages=defaultdict(float)
    for family in sorted({r['family'] for r in parents}):
        rows=[r for r in parents if r['family']==family and not r['stress']];complete=[r for r in rows if r['status']=='complete']
        projected=sum(r['weight']*(r['annotation_s']-r['timings'].get('model_restore_s',0)+r['source_export_s']+r.get('pack_s',0)) for r in complete)
        families[family]=dict(population=sum(r['weight'] for r in rows),sampled=len(rows),complete=len(complete),
            projected_worker_hours_successful_strata=projected/3600,
            projected_compressed_bytes=sum(r['weight']*r['compressed_bytes'] for r in complete),
            failed=[dict(id=r['id'],stage=r['stage'],error=r['error']) for r in rows if r['status']!='complete'])
        for r in complete:
            for key,value in r['timings'].items():
                if key not in ('validation_s','model_restore_s'):stages[key]+=r['weight']*value
    peak=max(r['peak_tree_rss_bytes'] for r in results)
    reservations=int(np.ceil(peak*1.25/(2*1024**3)))
    workers48=min(47//max(1,reservations),int(48*2*1024**3*.9/(peak*1.25)))
    return dict(status='complete',families=families,stage_projected_worker_hours={k:v/3600 for k,v in stages.items()},
        model_restore_s=[r['timings']['model_restore_s'] for r in parents if r['status']=='complete' and r['timings'].get('model_restore_s',0)>1],
        memory_aware_48cpu_projection=dict(cpus_per_worker=max(1,reservations),workers=workers48,worker_margin=1.25,memory_reserve=.1,
            limitation='Extrapolation from four reserved CPUs per single-threaded worker; high-concurrency throughput unmeasured'),
        counts=dict(Counter(r['status'] for r in parents)),peak_tree_rss_bytes=peak,wall_s=wall_s,parents=parents,
        model='frozen MaSIF-search all-features 80D; 12A geodesic radius; 200 vertices/patch; CPU inference',
        surface_coverage='one fingerprint per mesh vertex',training_enabled=False,
        limitations=['Small stratified pilot; failures remain explicit and projections do not impute successful runtimes for them',
            'Native annotation only; no ANM/reconstruction/minimisation',
            'Native source heavy atoms remain fixed; PDB2PQR atom additions/coordinate-preserving aliases are recorded',
            'Projected worker-hours need a memory-aware worker count; queueing and full-dataset I/O are excluded'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--coordinates',type=Path)
    p.add_argument('--v5-antibody-root',type=Path,help='v5 pool root: mandatory Ab/Ag partner assemblies, no per-chain expansion')
    p.add_argument('--pkpdb-abag-inventory',type=Path,help='pinned pKPDB Ab/Ag assembly inventory; all chains of each partner stay together')
    p.add_argument('--native-unit-inventory',type=Path,help='source-pinned pKPDB recovery units, including intact antibody-only assemblies')
    p.add_argument('--sabdab-native-inventory',type=Path,help='pinned native SAbDab assemblies; joint Ab, joint Ag and bound surfaces')
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--masif',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true')
    p.add_argument('--checkpoint',type=Path,required=True,help='validated TensorFlow checkpoint prefix')
    p.add_argument('--preprocessing',choices=['paper','legacy-python'],default='paper')
    p.add_argument('--paper-runtime',type=Path);p.add_argument('--matlab',type=Path,default=Path(os.environ.get('MATLAB_BIN','/usr/local/MATLAB/R2021b/bin/matlab')))
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--resume',action='store_true');p.add_argument('--full',action='store_true')
    p.add_argument('--external-command-timeout',type=int,default=1800,help='seconds per external tool; larger values require a dedicated resource allocation')
    p.add_argument('--dataset',choices=['all','pkpdb','pinder','sabdab'],default='all');p.add_argument('--split',choices=['all','train','val'],default='all')
    p.add_argument('--separate-chains',action='store_true');p.add_argument('--whole-assemblies',dest='separate_chains',action='store_false')
    p.add_argument('--include-bound',action='store_true');p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--gpu-python',type=Path,help='launch one local persistent GPU consumer using this Python')
    p.add_argument('--gpu-spool',type=Path,help='shared spool with independently allocated GPU consumers')
    p.add_argument('--smoke-family',choices=('both','pkpdb_assembly','pkpdb_chain','pinder_partner'),default='both')
    p.add_argument('--all-components',action='store_true',default=True)
    p.add_argument('--single-component',dest='all_components',action='store_false',help='legacy MSMS comparison only')
    p.add_argument('--only-id',action='append',help='explicit unit IDs; repeat for a non-probability integration panel')
    p.add_argument('--cohort-file',type=Path,help='source-pinned native training-chain cohort for this node')
    args=p.parse_args();cpus=int(os.environ['SLURM_CPUS_PER_TASK']);require_compute(threads=cpus,allow_comp1400=True)
    from masif_surface_quality import SURFACE_POLICY
    args.surface_policy=SURFACE_POLICY if args.preprocessing=='paper' else None
    if sum(bool(x) for x in [args.coordinates,args.v5_antibody_root,args.pkpdb_abag_inventory,args.native_unit_inventory,args.sabdab_native_inventory])!=1:
        raise ValueError('choose coordinate inventory, v5 PINDER antibody pool, or pKPDB Ab/Ag inventory')
    if args.sabdab_native_inventory and (args.dataset not in ('all','sabdab') or args.separate_chains or args.cohort_file or args.preprocessing!='paper'):
        raise ValueError('SAbDab native inventory requires paper preprocessing and complete biological assemblies')
    if (args.v5_antibody_root or args.pkpdb_abag_inventory) and (args.separate_chains or args.include_bound or args.cohort_file):
        raise ValueError('v5 antibody annotations require complete separate Ab/Ag partner assemblies')
    if args.v5_antibody_root and args.dataset=='pkpdb' or args.pkpdb_abag_inventory and args.dataset=='pinder':
        raise ValueError('partner inventory and requested dataset differ')
    if (args.v5_antibody_root or args.pkpdb_abag_inventory) and args.preprocessing!='paper':raise ValueError('partner builds require the validated paper backend')
    if args.native_unit_inventory and (args.dataset=='pinder' or args.separate_chains or args.include_bound or args.cohort_file or args.preprocessing!='paper'):
        raise ValueError('native recovery inventories require paper pKPDB units without expansion')
    if not 1<=args.batch_size<=32 or args.workers<1 or args.workers*4>cpus:raise ValueError('each worker needs four CPUs; batch size must be 1..32')
    if not 60<=args.external_command_timeout<=7200:raise ValueError('external command timeout must be 60..7200 seconds')
    if not 0<=args.shard_index<args.shards:raise ValueError('invalid cohort shard')
    if args.gpu_python and args.gpu_spool:raise ValueError('choose local or externally managed GPU consumers')
    if args.preprocessing!='paper' and (args.resume or args.gpu_python or args.gpu_spool):raise ValueError('resume/GPU support requires the paper backend')
    args.checkpoint_sha256={name:digest(str(args.checkpoint)+suffix) for name,suffix in
        [('model.data-00000-of-00001','.data-00000-of-00001'),('model.index','.index'),('model.meta','.meta')]}
    backend=worker;profile='legacy-python-diagnostic'
    if args.preprocessing=='paper':
        from masif_paper_runtime import verify,PROFILE
        from masif_paper_batch import worker as paper_worker
        args.paper_runtime=args.paper_runtime or args.masif.parent/'masif-paper-runtime-v1'
        verify(args.paper_runtime,args.checkpoint);args.paper_runtime_sha256=digest(args.paper_runtime/'manifest.json')
        backend=paper_worker;profile=PROFILE
    args.output.mkdir(parents=True,exist_ok=args.resume);started=time.monotonic()
    if args.sabdab_native_inventory:
        from pkabench.sabdab_units import surface_inventory
        units,manifest=surface_inventory(args.sabdab_native_inventory)
        units=[u for u in units if args.split=='all' or u['split']==args.split]
        coordinate_sha=manifest['inventory_sha256']
    elif args.native_unit_inventory:
        from masif_pkpdb_recovery import load_inventory
        units,manifest=load_inventory(args.native_unit_inventory,args.runtime,args.split)
        coordinate_sha=manifest['inventory_sha256']
    elif args.pkpdb_abag_inventory:
        from masif_pkpdb_abag import load_inventory
        units,manifest=load_inventory(args.pkpdb_abag_inventory,args.runtime,args.split)
        coordinate_sha=manifest['inventory_sha256']
    elif args.v5_antibody_root:
        from masif_partner_units import inventory_v5
        units,manifest=inventory_v5(args.v5_antibody_root,args.runtime,args.split)
        coordinate_sha=manifest['inventory_sha256']
    else:
        units,manifest=inventory(args.coordinates)
        units=select_units(units,args.runtime,dataset=args.dataset,split=args.split,separate_chains=args.separate_chains,include_bound=args.include_bound)
        coordinate_sha=digest(args.coordinates/'manifest.json')
    cohort_metadata={}
    if args.cohort_file:
        units,cohort_metadata=load_cohort(units,args.cohort_file,args.runtime,digest(args.coordinates/'manifest.json'))
    units=sorted(units,key=lambda r:r['id'])[args.shard_index::args.shards]
    if args.only_id:
        requested=set(args.only_id);sample=[dict(u,stress=False,weight=0.,stratum=cell(u)) for u in units if u['id'] in requested]
        if {r['id'] for r in sample}!=requested:raise ValueError('explicit units absent from the requested cohort')
    elif args.full:sample=[dict(u,stress=False,weight=1.,stratum=cell(u)) for u in units]
    else:sample=choose_sample(units)
    if args.smoke:
        families=sorted({r['family'] for r in sample})
        if args.smoke_family!='both':families=[args.smoke_family]
        sample=[min((r for r in sample if r['family']==family),key=lambda r:r['n']) for family in families]
    if not sample:raise ValueError('empty annotation cohort')
    sample_metadata=dict(parents=sample,population=dict(Counter(r['family'] for r in units)),
        coordinate_manifest_sha256=coordinate_sha,source=manifest.get('source',f'native {args.dataset}, split={args.split}'),
        unit_policy=manifest.get('unit_policy','coordinate-inventory-v1'),surface_policy=args.surface_policy,
        source_inventory=manifest if args.v5_antibody_root or args.pkpdb_abag_inventory or args.native_unit_inventory or args.sabdab_native_inventory else None,
        checkpoint_sha256=args.checkpoint_sha256,preprocessing_profile=profile,
        paper_runtime_sha256=getattr(args,'paper_runtime_sha256',None),workers=args.workers,batch_size=args.batch_size,
        builder_sha256={name:digest(Path(__file__).with_name(name)) for name in
            ['masif_annotation_pilot.py','masif_paper_batch.py','masif_paper_bridge.py','masif_paper_matlab.m',
             'masif_checkpoint.py','masif_gpu_queue.py','masif_gpu_model.py','masif_partner_units.py','masif_surface_quality.py','masif_pkpdb_abag.py','masif_pkpdb_recovery.py']},
        inference_backend='jax-gpu' if args.gpu_python or args.gpu_spool else 'tensorflow-cpu',
        all_molecular_components=args.all_components,external_command_timeout_s=args.external_command_timeout)
    sample_metadata.update(cohort_metadata)
    sample_path=args.output/'sample.json'
    if sample_path.exists():
        if json.loads(sample_path.read_text())!=sample_metadata:raise ValueError('resume cohort/configuration differs')
    else:atomic_json(sample_path,sample_metadata)
    if args.resume and (args.output/'fingerprints/manifest.json').exists():
        previous=json.loads((args.output/'report.json').read_text())
        if previous['counts'].get('failed',0):raise RuntimeError('published partial release has explicit failures; use a new output for repaired inputs')
        print(json.dumps(dict(status='already_complete',report=str(args.output/'report.json'))));return
    consumer=None;consumer_log=None;local_spool=None
    try:
        if args.gpu_python:
            if not os.environ.get('SLURM_JOB_GPUS'):raise RuntimeError('a GPU allocation is required')
            from masif_gpu_queue import initialize
            local_spool=tempfile.TemporaryDirectory(prefix='masif-gpu-',dir='/tmp');args.gpu_spool=Path(local_spool.name)
            initialize(args.gpu_spool,args.checkpoint_sha256)
            environment=os.environ.copy();environment.pop('JAX_PLATFORMS',None);environment['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
            consumer_log=(args.output/'gpu-consumer.log').open('a')
            consumer=subprocess.Popen([str(args.gpu_python),str(Path(__file__).with_name('masif_gpu_queue.py')),
                '--runtime',str(args.paper_runtime),'--spool',str(args.gpu_spool)],env=environment,stdout=consumer_log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+180
            while not list(args.gpu_spool.glob('ready-*.json')):
                if consumer.poll() is not None or time.monotonic()>deadline:raise RuntimeError('GPU consumer failed to start; see gpu-consumer.log')
                time.sleep(.2)
        if args.gpu_spool:
            from masif_gpu_queue import initialize
            initialize(args.gpu_spool,args.checkpoint_sha256)
            if not list(args.gpu_spool.glob('ready-*.json')):raise RuntimeError('no GPU consumer is ready')
        groups=[[] for _ in range(args.workers)];loads=[0]*args.workers
        for item in sorted(sample,key=lambda r:(-r.get('scheduling_cost',r['n']),r['id'])):
            i=min(range(args.workers),key=lambda k:loads[k]);groups[i].append(item);loads[i]+=item.get('scheduling_cost',item['n'])
        allowed=sorted(os.sched_getaffinity(0));tasks=[(args,i,allowed[4*i:4*i+4],group) for i,group in enumerate(groups)];results=[]
        with ProcessPoolExecutor(args.workers) as pool:
            for future in as_completed([pool.submit(backend,task) for task in tasks]):
                result=future.result();results.append(result);print(json.dumps(dict(worker=result['worker'],wall_s=result['wall_s'],counts=dict(Counter(r['status'] for r in result['parents'])))),flush=True)
        expected=[r['id'] for w in results for r in w['parents'] if r['status']=='complete']
        finalize_workers(args.output/'fingerprints',[args.output/'fingerprints/workers'/f'{i:04d}' for i in range(args.workers)],
            dict(kind='dense-native-masif-paper-v1' if args.preprocessing=='paper' else 'dense-native-masif-pilot-v1',
                 sample_sha256=digest(sample_path),checkpoint_sha256=args.checkpoint_sha256,
                 preprocessing_profile=profile,paper_runtime_sha256=getattr(args,'paper_runtime_sha256',None),
                 surface_policy=args.surface_policy,unit_policy=sample_metadata['unit_policy'],
                 requested_units=len(sample),completed_units=len(expected)),expected_ids=expected)
        report=summarize(args.output,sample,results,time.monotonic()-started)
        report.update(checkpoint_sha256=args.checkpoint_sha256,preprocessing_profile=profile,
            paper_runtime_sha256=getattr(args,'paper_runtime_sha256',None),
            inference_backend=sample_metadata['inference_backend'],native_heavy_coordinates_fixed=args.preprocessing!='paper')
        report['status']='partial' if report['counts'].get('failed',0) else 'complete'
        report['selection_mode']='explicit' if args.only_id else 'full' if args.full else 'stratified'
        report['model']='MaSIF-search, original paper, five features, 80D' if args.preprocessing=='paper' else report['model']
        if args.preprocessing=='paper':
            report['limitations']=[s for s in report['limitations'] if not s.startswith('Native source heavy atoms remain fixed')]
            report['limitations']+=['Original Reduce/protonation can adjust side chains; canonical residue mapping is retained',
                'MATLAB startup per batch and multi-worker license concurrency require a throughput pilot']
        atomic_json(args.output/'report.json',report)
        if report['counts'].get('failed',0):raise RuntimeError('annotation contains explicit failures; inspect report.json')
    finally:
        if consumer is not None:
            (args.gpu_spool/'stop').touch()
            try:consumer.wait(timeout=30)
            except subprocess.TimeoutExpired:consumer.terminate();consumer.wait(timeout=15)
        if consumer_log:consumer_log.close()
        if local_spool:local_spool.cleanup()


if __name__=='__main__':main()
