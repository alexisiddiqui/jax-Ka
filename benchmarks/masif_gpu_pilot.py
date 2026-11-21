"""Extract frozen MaSIF CPU references, then benchmark a numerically checked JAX GPU forward pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from masif_checkpoint import validate_arrays


def atomic(path,value):
    path=Path(path);pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');pending.replace(path)


def prepare(args):
    from pkatrain.augmentation_store import AugmentationStore
    from pkabench.runtime import require_compute
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),gpu_benchmark=True,allow_comp1400=True)
    args.output.mkdir(parents=True,exist_ok=False)
    ids=['pkpdb/5xmz/assembly','pkpdb/4lcv/assembly','pkpdb/5kxd/assembly',
         'pinder/6beb__A1_P68440--6beb__B1_P68440/A',
         'pinder/7d58__A1_O14802--7d58__I1_Q9Y2Y1/A',
         'pinder/2vjq__A1_O06644--2vjq__B1_O06644/AB']
    with tempfile.TemporaryDirectory(prefix='masif-gpu-reference-',dir='/tmp') as temporary,AugmentationStore(args.native/'fingerprints') as store:
        temporary=Path(temporary);jobs=[]
        for i,rid in enumerate(ids):
            data=store.raw(rid)['arrays'];filename=f'mesh-{i}.npz'
            np.savez_compressed(temporary/filename,**{k:data[k] for k in ('vertices','faces','normals','raw_features')})
            jobs.append(dict(id=rid,file=filename))
        atomic(temporary/'reference-jobs.json',jobs)
        r=args.masif;command=[str(r/'setup-env/bin/udocker'),'--quiet','run','--env=LD_LIBRARY_PATH=/compat',
            '--volume='+str(r/'compat')+':/compat','--env=OMP_NUM_THREADS=1','--env=OPENBLAS_NUM_THREADS=1',
            '--env=MKL_NUM_THREADS=1','--env=CUDA_VISIBLE_DEVICES=',
            '--volume='+str(temporary)+':/audit','--volume='+str(Path(__file__).resolve().parent)+':/code',
            '--volume='+str(args.checkpoint.resolve().parent)+':/checkpoint',
            '--env=MASIF_CHECKPOINT_PREFIX=/checkpoint/'+args.checkpoint.name,
            'masif-audit','python','/code/masif_gpu_reference.py']
        with (args.output/'cpu-reference.log').open('w') as log:
            subprocess.run(command,env=dict(os.environ,UDOCKER_DIR=str(r/'udocker')),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        for name in ('reference-bundle.npz','reference-bundle.json'):shutil.copy2(temporary/name,args.output/name)


def gpu(args):
    import jax
    import jax.numpy as jnp
    if not os.environ.get('SLURM_JOB_GPUS'):raise RuntimeError('allocated GPU required')
    devices=jax.devices()
    if len(devices)!=1 or devices[0].platform!='gpu':raise RuntimeError('exactly one GPU required')
    started=time.perf_counter()
    metadata=json.loads((args.output/'reference-bundle.json').read_text())
    if hashlib.sha256((args.output/'reference-bundle.npz').read_bytes()).hexdigest()!=metadata['bundle_sha256']:
        raise ValueError('CPU reference bundle changed')
    # CPU surface-preparation costs may be reused across checkpoint profiles;
    # inference references and weights must come from the same validated bundle.
    if not metadata.get('checkpoint_validation',{}).get('parameters_finite'):
        raise ValueError('CPU reference lacks strict checkpoint validation')
    data=np.load(args.output/'reference-bundle.npz',allow_pickle=False)
    host_weights={key[len('weight__'):]:data[key] for key in data.files if key.startswith('weight__')}
    validate_arrays(host_weights)
    weights={key:jnp.asarray(value) for key,value in host_weights.items()}
    features=np.concatenate([data['features_'+str(i)] for i in range(len(metadata['cases']))])
    rho=np.concatenate([data['rho_'+str(i)] for i in range(len(metadata['cases']))])
    theta=np.concatenate([data['theta_'+str(i)] for i in range(len(metadata['cases']))])
    mask=np.concatenate([data['mask_'+str(i)] for i in range(len(metadata['cases']))])
    reference=np.concatenate([data['reference_'+str(i)] for i in range(len(metadata['cases']))])
    angle_steps=jnp.asarray([np.float32(k*2*np.pi/16) for k in range(16)])
    two_pi=jnp.float32(2*np.pi);eps=jnp.float32(1e-5)
    def forward(p,feat,r,t,m):
        out=[]
        for f in range(5):
            gr=jnp.exp(-(r[...,None]-p[f'mu_rho_{f}'])**2/(p[f'sigma_rho_{f}']**2+eps))
            def rotate(k,best):
                angle=jnp.mod(t+angle_steps[k],two_pi)
                gt=jnp.exp(-(angle[...,None]-p[f'mu_theta_{f}'])**2/(p[f'sigma_theta_{f}']**2+eps))
                act=gr*gt*m[...,None];act=act/(jnp.sum(act,axis=1,keepdims=True)+eps)
                desc=jnp.sum(act*feat[:,:,f,None],axis=1)
                conv=jnp.matmul(desc,p[f'W_conv_{f}'],precision=jax.lax.Precision.HIGHEST)+p[f'b_conv_{f}']
                return jnp.maximum(best,conv)
            result=jax.lax.fori_loop(0,16,rotate,jnp.full((len(feat),80),-jnp.inf))
            out.append(jax.nn.relu(result))
        return jnp.matmul(jnp.concatenate(out,axis=-1),p['fully_connected/weights'],precision=jax.lax.Precision.HIGHEST)+p['fully_connected/biases']
    compiled=jax.jit(forward);results=[];checks=[]
    if args.debug:
        stats={key:dict(shape=list(value.shape),finite=bool(np.isfinite(np.asarray(value)).all()),
            minimum=float(np.min(np.asarray(value))),maximum=float(np.max(np.asarray(value)))) for key,value in weights.items()}
        print(json.dumps(dict(stage='debug-weights',weights=stats)),flush=True)
        inputs=(features[:2],rho[:2],theta[:2],mask[:2])
        print(json.dumps(dict(stage='debug-inputs',inputs=[dict(shape=list(x.shape),finite=bool(np.isfinite(x).all()),minimum=float(x.min()),maximum=float(x.max())) for x in inputs])),flush=True)
        with jax.disable_jit():value=np.asarray(forward(weights,*jax.device_put(inputs)))
        print(json.dumps(dict(stage='debug-eager',finite=bool(np.isfinite(value).all()),max_abs=float(np.max(np.abs(value-reference[:2]))),values=value[0,:8].tolist(),reference=reference[0,:8].tolist())),flush=True)
        value=np.asarray(compiled(weights,*jax.device_put(inputs)))
        print(json.dumps(dict(stage='debug-compiled',finite=bool(np.isfinite(value).all()),max_abs=float(np.max(np.abs(value-reference[:2]))),values=value[0,:8].tolist())),flush=True)
        return
    for batch in (32,128,256,512,1024):
        # Every native reference patch is checked, not merely a synthetic batch.
        predictions=[];compile_started=time.perf_counter()
        for start in range(0,len(features),batch):
            xs=[np.pad(a[start:start+batch],[(0,max(0,batch-len(a[start:start+batch])))]+[(0,0)]*(a.ndim-1)) for a in (features,rho,theta,mask)]
            value=compiled(weights,*jax.device_put(tuple(xs)));predictions.append(np.asarray(value)[:min(batch,len(features)-start)])
        predictions=np.concatenate(predictions);error=np.abs(predictions-reference)
        check=dict(batch=batch,patches=len(reference),max_abs=float(error.max()),
            relative_l2=float(np.linalg.norm(predictions-reference)/max(np.linalg.norm(reference),1e-12)),
            finite=bool(np.isfinite(predictions).all()),passed=bool(np.isfinite(predictions).all() and error.max()<=1e-4))
        checks.append(check);print(json.dumps(dict(stage='agreement',**check)),flush=True)
        if not check['passed']:raise ValueError('GPU forward pass does not match frozen TensorFlow reference')
        host=tuple(np.asarray(a[:batch],np.float32) for a in (features,rho,theta,mask));device=jax.device_put(host)
        for _ in range(5):jax.block_until_ready(compiled(weights,*device))
        t=time.perf_counter()
        for _ in range(30):jax.block_until_ready(compiled(weights,*device))
        resident_s=(time.perf_counter()-t)/30
        # Include the original float64->float32 host conversion, H2D, inference
        # and descriptor return; compilation and checkpoint restore are separate.
        original=tuple(a.astype(np.float64) for a in host)
        t=time.perf_counter()
        for _ in range(30):
            cast=tuple(a.astype(np.float32) for a in original)
            value=compiled(weights,*jax.device_put(cast));np.asarray(value)
        transfer_s=(time.perf_counter()-t)/30
        result=dict(batch=batch,resident_ms=1000*resident_s,transfer_inference_return_ms=1000*transfer_s,
            resident_patches_s=batch/resident_s,end_to_end_patches_s=batch/transfer_s,
            compile_and_check_s=time.perf_counter()-compile_started)
        results.append(result);print(json.dumps(dict(stage='timing',**result)),flush=True)
        atomic(args.output/'gpu-progress.json',dict(checks=checks,results=results))
    best=max(results,key=lambda r:r['end_to_end_patches_s']);batch=best['batch'];step_s=best['transfer_inference_return_ms']/1000
    native=json.loads((args.native/'report.json').read_text());projection={}
    for family in native['families']:
        rows=[r for r in native['parents'] if r['family']==family and not r['stress'] and r['status']=='complete']
        cpu_h=0.;gpu_h=0.;old_h=0.
        for r in rows:
            old=(r['annotation_s']-r['timings'].get('model_restore_s',0)+r['source_export_s']+r.get('pack_s',0))
            cpu_h+=r['weight']*(old-r['timings']['inference_s'])/3600
            gpu_h+=r['weight']*int(np.ceil(r['vertices']/batch))*step_s/3600
            old_h+=r['weight']*old/3600
        projection[family]=dict(cpu_preprocessing_worker_hours=cpu_h,gpu_inference_hours=gpu_h,old_cpu_worker_hours=old_h)
    total={}
    workers=native['memory_aware_48cpu_projection']['workers']
    for name,families in [('pkpdb_plus_separated_pinder',('pkpdb_assembly','pinder_partner')),
                          ('with_bound_pinder',('pkpdb_assembly','pinder_partner','pinder_bound'))]:
        cpu=sum(projection[f]['cpu_preprocessing_worker_hours'] for f in families)/workers
        device_hours=sum(projection[f]['gpu_inference_hours'] for f in families)
        total[name]=dict(cpu_workers=workers,cpu_preprocessing_wall_hours=cpu,
            one_gpu_hours=device_hours,four_gpu_hours=device_hours/4,
            one_gpu_overlap_lower_bound_hours=max(cpu,device_hours),one_gpu_serial_upper_hours=cpu+device_hours,
            four_gpu_overlap_lower_bound_hours=max(cpu,device_hours/4),four_gpu_serial_upper_hours=cpu+device_hours/4)
    report=dict(passed=all(c['passed'] for c in checks),gpu=devices[0].device_kind,job=os.environ['SLURM_JOB_ID'],
        checkpoint_parameters_finite=all(bool(np.isfinite(np.asarray(v)).all()) for v in weights.values()),
        nonfinite_parameter_names=[k for k,v in weights.items() if not np.isfinite(np.asarray(v)).all()],
        activation_semantics='Strict finite five-feature model; ordinary rotation maximum and ReLU',
        checkpoint_usable=True,
        production_ready=False,
        checks=checks,results=results,best_batch=batch,projection_by_family=projection,totals=total,
        reference=metadata,wall_s=time.perf_counter()-started,training_enabled=False,
        limitations=['A40 inference measured; full CPU/GPU producer-consumer pipeline not implemented or timed',
            '1/4 GPU full-run times are capacity projections; four-GPU execution not benchmarked',
            'CPU preprocessing, memory reservations and known curvature failures remain from the native pilot',
            'Identical validated weights and float32 real patch inputs; no model retraining',
            'Checkpoint numerical validation does not establish task accuracy under a changed preprocessing profile',
            'JAX full-float32 dense products; no reduced-precision tensor-core approximation'])
    atomic(args.output/'report.json',report);print(json.dumps(dict(stage='complete',best=best,totals=total)),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=('prepare','gpu'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--masif',type=Path);p.add_argument('--debug',action='store_true')
    p.add_argument('--checkpoint',type=Path,help='validated TensorFlow checkpoint prefix, required for prepare')
    args=p.parse_args()
    if args.command=='prepare' and args.checkpoint is None:p.error('prepare requires --checkpoint')
    (prepare if args.command=='prepare' else gpu)(args)


if __name__=='__main__':main()
