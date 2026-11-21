"""Bounded file requests for persistent A40 inference, including remote CPU nodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    temporary.write_text(json.dumps(value,sort_keys=True));os.replace(temporary,path)


def initialize(spool,checkpoint):
    spool=Path(spool);spool.mkdir(parents=True,exist_ok=True,mode=0o700)
    value=dict(checkpoint_sha256=checkpoint,batch=1024);path=spool/'config.json'
    temporary=spool/(uuid.uuid4().hex+'.config.pending');temporary.write_text(json.dumps(value,sort_keys=True))
    try:
        try:os.link(temporary,path)
        except FileExistsError:pass
    finally:temporary.unlink()
    if json.loads(path.read_text())!=value:raise ValueError('GPU spool configuration mismatch')
    if (spool/'stop').exists():raise ValueError('GPU spool was stopped; use a fresh spool')


def validate_names(request,key):
    expected=dict(input=key+'.npz',output=key+'.output.npy',response=key+'.response')
    if any(request.get(k)!=value for k,value in expected.items()):raise ValueError('unsafe queue path or request identity')


def submit(spool,arrays,checkpoint,timeout=600):
    """One outstanding bounded chunk per producer; clean all data after reply."""
    spool=Path(spool);config=json.loads((spool/'config.json').read_text())
    if config['checkpoint_sha256']!=checkpoint:raise ValueError('GPU queue model mismatch')
    key=uuid.uuid4().hex;source=spool/(key+'.npz');request=spool/(key+'.request')
    response=spool/(key+'.response');output=spool/(key+'.output.npy');claim=spool/(key+'.working')
    try:
        np.savez(source,**arrays)
        # Remote NFS clients can cache a missing name for tens of seconds.
        # Publish an existing mailbox before the request; close-to-open reads
        # then discover the consumer's atomic replacement without that delay.
        atomic(response,dict(status='pending'))
        atomic(request,dict(input=source.name,output=output.name,response=response.name,
                            checkpoint_sha256=checkpoint,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        deadline=time.monotonic()+timeout
        while True:
            try:result=json.loads(response.read_text())
            except FileNotFoundError:result=None
            if result is not None and result['status']!='pending':break
            if time.monotonic()>deadline:raise TimeoutError('GPU inference response timed out')
            time.sleep(.05)
        if result['status']!='complete':raise RuntimeError(result['error'])
        if result.get('checkpoint_sha256')!=checkpoint:raise ValueError('GPU response model mismatch')
        value=np.load(output,allow_pickle=False)
        if value.shape!=(len(arrays['features']),80) or not np.isfinite(value).all():raise ValueError('invalid GPU fingerprints')
        return value
    finally:
        for path in [request,source,response,output]:
            # NFS can report the old request name after a consumer renamed it.
            # A missing cleanup target must not discard successful inference.
            try:path.unlink()
            except FileNotFoundError:pass
        # An in-flight timed-out request remains claimed; consumers report the
        # error, and job-local scratch cleanup removes it after consumers stop.


def serve(args):
    import jax
    from masif_checkpoint import validate_arrays
    from masif_gpu_model import make_forward
    from masif_paper_runtime import verify,sha
    if not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')):raise RuntimeError('GPU allocation required')
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise RuntimeError('one visible GPU required per consumer')
    manifest=verify(args.runtime);spool=args.spool;spool.mkdir(parents=True,exist_ok=True)
    initialize(spool,manifest['checkpoint_sha256'])
    with np.load(args.runtime/'weights.npz',allow_pickle=False) as data:weights={k:data[k] for k in data.files}
    validate_arrays(weights);weights=jax.device_put(weights);forward=make_forward()
    zeros=[np.zeros(shape,np.float32) for shape in [(1024,200,5),(1024,200),(1024,200),(1024,200)]]
    jax.block_until_ready(forward(weights,*jax.device_put(tuple(zeros))))
    ready=spool/('ready-'+str(os.getpid())+'.json')
    metrics=dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,
                 gpu=jax.devices()[0].device_kind,started_s=time.time(),status='ready',
                 requests=0,failed=0,patches=0,service_s=0.,device_s=0.)
    published=0.
    def heartbeat(force=False):
        nonlocal published
        if force or time.monotonic()-published>=5:
            atomic(ready,dict(metrics,updated_s=time.time()));published=time.monotonic()
    heartbeat(True)
    while not (spool/'stop').exists():
        heartbeat()
        available=list(spool.glob('*.request'))
        if not available:time.sleep(.05);continue
        for path in available:
            claim=path.with_suffix('.working')
            try:path.rename(claim)
            except FileNotFoundError:continue
            response=spool/(path.stem+'.response')
            began=time.monotonic()
            try:
                request=json.loads(claim.read_text());validate_names(request,path.stem)
                if request['checkpoint_sha256']!=manifest['checkpoint_sha256']:raise ValueError('request model mismatch')
                source=spool/request['input']
                if sha(source)!=request['sha256']:raise ValueError('GPU input changed')
                with np.load(source,allow_pickle=False) as data:arrays=[data[k] for k in ['features','rho','theta','mask']]
                n=len(arrays[0]);shapes=[(n,200,5)]+[(n,200)]*3
                if not 1<=n<=1024 or any(x.shape!=s or not np.isfinite(x).all() for x,s in zip(arrays,shapes)):
                    raise ValueError('invalid GPU input shape or values')
                padded=[np.pad(x.astype(np.float32),[(0,1024-n)]+[(0,0)]*(x.ndim-1)) for x in arrays]
                device_started=time.monotonic()
                value=np.asarray(forward(weights,*jax.device_put(tuple(padded))))[:n]
                metrics['device_s']+=time.monotonic()-device_started
                if not np.isfinite(value).all():raise ValueError('nonfinite GPU fingerprints')
                np.save(spool/request['output'],value)
                result=dict(status='complete',checkpoint_sha256=manifest['checkpoint_sha256'])
                metrics['patches']+=n
            except Exception as exc:result=dict(status='failed',error=type(exc).__name__+': '+str(exc))
            atomic(response,result);claim.unlink()
            metrics['requests']+=1;metrics['failed']+=int(result['status']!='complete')
            metrics['service_s']+=time.monotonic()-began;heartbeat()
    metrics['status']='stopped';heartbeat(True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--spool',type=Path,required=True)
    serve(p.parse_args())
