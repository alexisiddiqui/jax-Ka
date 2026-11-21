"""Host-side original-paper batches; only packed fingerprints leave scratch."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

import numpy as np
import psutil

from pkabench.runtime import atomic_json,digest
from pkatrain.augmentation_store import ReleaseWriter
from masif_paper_runtime import PROFILE
from masif_surface_quality import SURFACE_POLICY


def run_process(command,args,slot,stage,log,env):
    started=time.monotonic();peak=0;published=0.
    with log.open('a') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-started>args.batch_size*getattr(args,'external_command_timeout',1800)+300:raise TimeoutError('paper batch stage timeout')
                rss=psutil.Process(os.getpid()).memory_info().rss
                try:
                    parent=psutil.Process(process.pid)
                    for child in [parent]+parent.children(recursive=True):
                        try:rss+=child.memory_info().rss
                        except psutil.Error:pass
                except psutil.Error:pass
                peak=max(peak,rss)
                if time.monotonic()-published>=5:
                    atomic_json(args.output/f'progress-{slot:02d}.json',dict(status='running',stage=stage,worker=slot,peak_tree_rss_bytes=peak))
                    published=time.monotonic()
                time.sleep(.2)
            if process.returncode:raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')
        except BaseException:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            process.wait(timeout=15);raise
    return peak,time.monotonic()-started


def matlab_literal(value):return "'"+str(value).replace("'","''")+"'"


def worker_provenance(args):
    return dict(kind='dense-native-masif-paper-v1',sample_sha256=digest(args.output/'sample.json'),
        preprocessing_profile=PROFILE,paper_runtime_sha256=args.paper_runtime_sha256,
        surface_policy=getattr(args,'surface_policy',SURFACE_POLICY),
        checkpoint_sha256=args.checkpoint_sha256,inference_backend='jax-gpu' if args.gpu_spool else 'tensorflow-cpu',
        host_sha256=digest(__file__),bridge_sha256=digest(Path(__file__).with_name('masif_paper_bridge.py')),
        matlab_driver_sha256=digest(Path(__file__).with_name('masif_paper_matlab.m')))


def worker(task):
    from masif_annotation_pilot import export_native
    args,slot,cpus,items=task;os.sched_setaffinity(0,set(cpus));started=time.monotonic()
    packed=args.output/'fingerprints/workers'/f'{slot:04d}';state=args.output/f'worker-{slot:02d}.json'
    provenance=worker_provenance(args)
    if (packed/'manifest.json').exists():
        if not args.resume:raise FileExistsError('worker release exists')
        if json.loads((packed/'manifest.json').read_text())['provenance']!=provenance:raise ValueError('resume preprocessing profile differs')
        return json.loads(state.read_text())
    records=[];peak=0;matlab_wall=0.;batches=0
    with ReleaseWriter(packed,provenance,zstd_level=1) as writer:
        if state.exists():
            old=json.loads(state.read_text());records=[r for r in old['parents'] if r['status']=='complete' and r['id'] in writer.ids]
            peak=old['peak_tree_rss_bytes'];matlab_wall=old.get('matlab_process_wall_s',0.);batches=old.get('batches',0)
        remaining=[item for item in items if item['id'] not in writer.ids]
        for offset in range(0,len(remaining),args.batch_size):
            with tempfile.TemporaryDirectory(prefix=f'masif-paper-{os.environ["SLURM_JOB_ID"]}-{slot}-',dir='/tmp') as directory:
                temporary=Path(directory);jobs=[];by_id={}
                for i,item in enumerate(remaining[offset:offset+args.batch_size]):
                    root=temporary/f'{i:04d}';root.mkdir();t=time.monotonic()
                    try:
                        info=export_native(args.runtime,item,root/'input.pdb',surface_policy=args.surface_policy)
                        info['source_export_s']=time.monotonic()-t
                        item=dict(item,selection_n=item['n'],n=info['actual_residues'],size_is_metadata_estimate=False)
                        job=dict(id=item['id'],input=f'/audit/{i:04d}/input.pdb',work=f'/audit/{i:04d}',host_work=str(root),
                            output=f'/audit/{i:04d}/features.npz',receipt=f'/audit/{i:04d}/receipt.json',
                            canonical_residue_ids=info['canonical_residue_ids'],paper_runtime_sha256=args.paper_runtime_sha256,
                            surface_policy=args.surface_policy,external_command_timeout_s=getattr(args,'external_command_timeout',1800))
                        jobs.append(job);by_id[item['id']]=(item,root,info)
                    except Exception as exc:records.append(dict(item,status='failed',stage='source_export',error=str(exc),traceback=traceback.format_exc()[-2000:]))
                atomic_json(temporary/'jobs.json',jobs)
                if jobs:
                    container=[str(args.masif/'setup-env/bin/udocker'),'--quiet','run','--env=LD_LIBRARY_PATH=/compat',
                        '--volume='+str(args.masif/'compat')+':/compat','--env=OMP_NUM_THREADS=1','--env=OPENBLAS_NUM_THREADS=1',
                        '--env=MKL_NUM_THREADS=1','--env=CUDA_VISIBLE_DEVICES=','--env=PYTHONPATH=/code',
                        '--env=PYTHONDONTWRITEBYTECODE=1','--volume='+str(temporary)+':/audit',
                        '--volume='+str(Path(__file__).resolve().parent)+':/code',
                        '--volume='+str(args.paper_runtime.resolve())+':/paper-runtime',
                        '--volume='+str(args.checkpoint.resolve().parent)+':/checkpoint',
                        '--env=MASIF_CHECKPOINT_PREFIX=/checkpoint/'+args.checkpoint.name]
                    if args.gpu_spool:container+=['--volume='+str(args.gpu_spool.resolve())+':/gpu-spool']
                    container+=['masif-audit','python','/code/masif_paper_bridge.py']
                    env=dict(os.environ,UDOCKER_DIR=str(args.masif/'udocker'));log=args.output/f'worker-{slot:02d}.log'
                    command=container+['surface','--jobs','/audit/jobs.json','--runtime','/paper-runtime']
                    memory,_=run_process(command,args,slot,'surface',log,env);peak=max(peak,memory)
                    if any(json.loads((root/'surface-receipt.json').read_text())['status']=='complete' for _,root,_ in by_id.values()):
                        expression='addpath('+matlab_literal(Path(__file__).resolve().parent)+');masif_paper_matlab('+matlab_literal(temporary/'jobs.json')+','+matlab_literal(args.paper_runtime.resolve())+');'
                        memory,elapsed=run_process([str(args.matlab),'-singleCompThread','-batch',expression],args,slot,'matlab',log,env)
                        peak=max(peak,memory);matlab_wall+=elapsed
                    command=container+['describe','--jobs','/audit/jobs.json','--runtime','/paper-runtime']
                    if args.gpu_spool:command+=['--gpu-spool','/gpu-spool']
                    memory,_=run_process(command,args,slot,'descriptors',log,env);peak=max(peak,memory)
                    for rid,(item,root,info) in by_id.items():
                        receipt=json.loads((root/'receipt.json').read_text()) if (root/'receipt.json').exists() else dict(status='failed',stage='container',error='missing paper receipt')
                        record={**item,**{k:v for k,v in receipt.items() if k!='id'},**info}
                        if receipt['status']=='complete':
                            if receipt['preprocessing_profile']!=PROFILE or receipt['paper_runtime_sha256']!=args.paper_runtime_sha256 or receipt['checkpoint_sha256']!=args.checkpoint_sha256:
                                raise ValueError('paper output provenance differs from release')
                            if receipt.get('surface_policy')!=args.surface_policy:raise ValueError('surface preparation policy differs from release')
                            with np.load(root/'features.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
                            t=time.monotonic();row=writer.add(rid,dict(item,receipt=receipt,source=info),arrays,summary=item)
                            record.update(compressed_bytes=row['bytes'],raw_bytes=row['raw_bytes'],pack_s=time.monotonic()-t)
                        records.append(record)
                batches+=1
                # Batch size <64 prevents the store's automatic checkpoint from
                # racing this metadata state: write state before committing data.
                result=dict(worker=slot,parents=records,peak_tree_rss_bytes=peak,wall_s=time.monotonic()-started,
                            matlab_process_wall_s=matlab_wall,batches=batches)
                atomic_json(state,result);writer.checkpoint()
        if not remaining:
            result=dict(worker=slot,parents=records,peak_tree_rss_bytes=peak,wall_s=time.monotonic()-started,
                        matlab_process_wall_s=matlab_wall,batches=batches);atomic_json(state,result)
        writer.finish(expected_ids=[r['id'] for r in records if r['status']=='complete'])
    atomic_json(args.output/f'progress-{slot:02d}.json',dict(status='complete',worker=slot,units=len(items),peak_tree_rss_bytes=peak))
    return result


def merge_releases(output,children):
    """Publish one reader index over disjoint node-local writer directories."""
    from pkatrain.augmentation_store import finalize_workers
    output=Path(output);directories=[];expected=[];common=None;contract=None;sources={};requested=0
    if (output/'manifest.json').exists():raise FileExistsError('immutable merged release already published')
    for child in children:
        child=Path(child)
        if not child.resolve().is_relative_to(output.resolve()):raise ValueError('node output must be inside merged release root')
        report=json.loads((child/'report.json').read_text())
        if report['status']!='complete' or report['counts'].get('failed',0):raise ValueError('cannot merge incomplete annotation node')
        root=child/'fingerprints';manifest=json.loads((root/'manifest.json').read_text());provenance=manifest['provenance']
        scientific={k:provenance[k] for k in ['kind','preprocessing_profile','paper_runtime_sha256','checkpoint_sha256']}
        scientific['surface_policy']=provenance.get('surface_policy','original-v1')
        if scientific['preprocessing_profile']!=PROFILE:raise ValueError('not an original-paper release')
        if common is not None and common!=scientific:raise ValueError('cannot merge different preprocessing/model profiles')
        sample=json.loads((child/'sample.json').read_text())
        cohort={k:sample[k] for k in ['source','coordinate_manifest_sha256','all_molecular_components']}
        cohort['global_cohort_sha256']=sample.get('global_cohort_sha256')
        cohort['unit_policy']=sample.get('unit_policy','coordinate-inventory-v1')
        if contract is not None and contract!=cohort:raise ValueError('cannot merge different input cohorts')
        contract=cohort
        common=scientific;sources[str(child.resolve())]=digest(root/'manifest.json')
        directories.extend(sorted((root/'workers').glob('[0-9][0-9][0-9][0-9]')))
        expected.extend(r['id'] for r in json.loads((root/'index.json').read_text())['records'])
        requested+=provenance['requested_units']
    if not children or len(expected)!=len(set(expected)):raise ValueError('empty or overlapping annotation shards')
    return finalize_workers(output,directories,dict(common,cohort=contract,node_manifests=sources,requested_units=requested,completed_units=len(expected)),expected_ids=expected)


if __name__=='__main__':
    import argparse
    from pkabench.runtime import require_compute
    p=argparse.ArgumentParser(__doc__);p.add_argument('--merge',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require_compute(threads=1,allow_comp1400=True);merge_releases(a.output,a.merge)
