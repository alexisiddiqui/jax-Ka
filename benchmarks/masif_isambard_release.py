"""Transfer a native release and run exactly one GH200 loader allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from augmentation_cache_transfer import transfer,ssh,atomic_json


def run(root,stage,host):
    root=Path(root);base='/scratch/u6xq/hussain.u6xq';submission=base+'/_submission/masif-native-release-v1'
    name='masif-native-pilot-v1' if stage=='pilot' else 'masif-native-release-v1'
    source=root/('pilot-release' if stage=='pilot' else 'release');destination=base+'/_runtime/jax-Ka/anm-pilot-v1/'+name
    work=root/('transfer-'+stage);work.mkdir(parents=True,exist_ok=True)
    transfer(SimpleNamespace(source=source,work=work,host=host,destination=destination,project=815102))
    ssh(host,['mkdir','-p',submission])
    for filename in ['loader-code.tgz','masif-loader-isambard.sbatch']:
        local=root/'submission'/filename
        subprocess.run(['rsync','-rt','--ignore-existing','-e','ssh -o BatchMode=yes -o ConnectTimeout=30',str(local),host+':'+submission+'/'+filename],check=True)
        expected=hashlib.sha256(local.read_bytes()).hexdigest()
        got=ssh(host,['sha256sum',submission+'/'+filename]).stdout.decode().split()[0]
        if got!=expected:raise ValueError('remote immutable loader source differs: '+filename)
    report=submission+'/loader-'+stage+'.json';expected=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
    control=work/'remote-job.json'
    if control.exists():
        state=json.loads(control.read_text())
        if state['source_manifest_sha256']!=expected:raise ValueError('remote loader job belongs to another release')
        job=state['job']
    else:
        result=ssh(host,['sbatch','--parsable','--output='+submission+'/loader-'+stage+'-%j.log',submission+'/masif-loader-isambard.sbatch',
            destination,report,'0' if stage=='pilot' else '128'])
        job=result.stdout.decode().strip().split(';')[0]
        if not job.isdigit():raise ValueError('unexpected remote Slurm submission response')
        atomic_json(control,dict(job=job,source_manifest_sha256=expected,report=report,requested_gpus=1,requested_cpus=72))
    began=time.monotonic();last=None
    while True:
        rows=ssh(host,['sacct','-j',job,'-X','-n','--format=State']).stdout.decode().strip().split()
        state=rows[0].rstrip('+') if rows else 'PENDING_ACCOUNTING'
        if state!=last:
            print(json.dumps(dict(remote_job=job,state=state,stage=stage)),flush=True);last=state
            atomic_json(work/'loader-status.json',dict(status=state,job=job,stage=stage))
        if state=='COMPLETED':break
        if state in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL']:
            raise RuntimeError('Isambard loader job '+job+' ended '+state+'; log '+submission+'/loader-'+stage+'-'+job+'.log')
        if time.monotonic()-began>5400:raise TimeoutError('remote loader did not finish within 90 minutes; job remains tracked in '+str(control))
        time.sleep(15)
    result=json.loads(ssh(host,['cat',report]).stdout)
    if result['status']!='complete' or result['source_manifest_sha256']!=expected:raise ValueError('remote loader report failed or source differs')
    atomic_json(root/('isambard-loader-'+stage+'.json'),result)
    atomic_json(work/'loader-status.json',dict(status='complete',job=job,stage=stage,destination=destination,source_manifest_sha256=expected,
        device=result['device'],sampled_assemblies=result['sampled_assemblies']))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['pilot','full'],required=True);p.add_argument('--host',default='u6xq.aip2.isambard');a=p.parse_args()
    run(a.root,a.stage,a.host)
