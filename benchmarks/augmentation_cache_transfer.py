"""Manifest-only feature-cache transfer, with resumable staging and SHA publication."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    pending.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def manifest_files(root, manifest):
    root=Path(root).resolve()
    if manifest.get('format')!='augmentation-feature-cache-v1' or manifest.get('verified') is not True:
        raise ValueError('not a verified feature release')
    files=manifest['files']
    if 'manifest.json' in files:raise ValueError('outer manifest cannot hash itself')
    for name,expected in files.items():
        relative=Path(name); path=root/relative
        if relative.is_absolute() or '..' in relative.parts or '\n' in name or '\r' in name:
            raise ValueError('unsafe manifest path')
        if not path.resolve().is_relative_to(root):raise ValueError('manifest path escapes root')
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('invalid SHA-256')
    return files


# This preflight runs on the remote login host as transfer bookkeeping. It
# creates only the manifest allowlist's directories, never an unpacked tree.
REMOTE_PREPARE = r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
destination=Path(sys.argv[1]); project=sys.argv[2]
raw=sys.stdin.buffer.read(); manifest=json.loads(raw); files=manifest['files']
if manifest.get('format')!='augmentation-feature-cache-v1' or not manifest.get('verified'):raise RuntimeError('invalid release')
stage=Path(str(destination)+'.incoming');control=Path(str(destination)+'.transfer')
if destination.exists():
 p=destination/'manifest.json'
 if p.is_file() and p.read_bytes()==raw:
  print(json.dumps(dict(status='already-published',destination=str(destination))));sys.exit(0)
 raise RuntimeError('destination already exists with a different release')
quota=subprocess.check_output(['lfs','quota','-p',project,'/scratch'],text=True)
line=next(x.split() for x in quota.splitlines() if x.strip().startswith('/scratch '))
used_bytes=int(line[1].rstrip('*'))*1024;limit_bytes=int(line[3])*1024
used_files=int(line[5].rstrip('*'));limit_files=int(line[7])
dirs=set()
for name in files:
 p=Path(name)
 if p.is_absolute() or '..' in p.parts or '\n' in name:raise RuntimeError('unsafe manifest path')
 dirs.update(str(q) for q in p.parents if str(q)!='.')
required_entries=len(files)+1+len(dirs)+128
required_bytes=int(sys.argv[3])+2*2*1024**3+1024**3
if not limit_files or used_files+required_entries>limit_files:raise RuntimeError('insufficient inode headroom including transfer reserve')
if not limit_bytes or used_bytes+required_bytes>limit_bytes:raise RuntimeError('insufficient byte headroom including transfer reserve')
if stage.is_symlink() or control.is_symlink():raise RuntimeError('staging/control may not be symlinks')
control.mkdir(parents=True,exist_ok=True);stage.mkdir(exist_ok=True)
expected=control/'expected-manifest.json'
if expected.exists() and expected.read_bytes()!=raw:raise RuntimeError('staging belongs to another release')
if not expected.exists():expected.write_bytes(raw)
for name in sorted(dirs):
 p=stage/name
 if not p.resolve().is_relative_to(stage.resolve()):raise RuntimeError('destination directory escapes staging')
 p.mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(status='ready',destination=str(destination),stage=str(stage),control=str(control),
 manifest_sha256=hashlib.sha256(raw).hexdigest(),payload_files=len(files)+1,directories=len(dirs),
 used_bytes=used_bytes,limit_bytes=limit_bytes,used_files=used_files,limit_files=limit_files,
 reserved_added_entries=required_entries,remaining_inodes_after_reserve=limit_files-used_files-required_entries)))
'''


def ssh(host, argv, *, input=None):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=30',host,shlex.join(argv)],
        input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)


def verify(args):
    """Only transfer integrity work; no model, featurisation or GPU allocation."""
    raw=args.expected_manifest.read_bytes(); manifest=json.loads(raw)
    files=manifest_files(args.source,manifest); started=time.monotonic()
    progress=dict(status='verifying',files_total=len(files),files_checked=0,bytes_checked=0)
    atomic_json(args.progress,progress)
    def check(item):
        name,expected=item;path=args.source/name
        if not path.is_file() or path.is_symlink():raise ValueError('missing/linked destination: '+name)
        if sha256(path)!=expected:raise ValueError('destination SHA-256 mismatch: '+name)
        return name,path.stat().st_size
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(check,item) for item in files.items()]
        for future in as_completed(futures):
            name,size=future.result();progress['files_checked']+=1;progress['bytes_checked']+=size
            progress.update(last_file=name,wall_s=time.monotonic()-started)
            atomic_json(args.progress,progress)
            if progress['files_checked']%64==0:print(json.dumps(progress),flush=True)
    # A crash between installing the marker and renaming the directory may
    # leave the exact source marker in staging; it is safe to verify again.
    marker=args.source/'manifest.json'
    allowed=set(files)
    if marker.exists() and marker.read_bytes()==raw:allowed.add('manifest.json')
    extras=[str(p.relative_to(args.source)) for p in args.source.rglob('*')
        if p.is_file() and str(p.relative_to(args.source)) not in allowed]
    if extras:raise ValueError('unlisted files remain in staging: '+str(extras[:5]))
    if args.destination.exists():raise FileExistsError('published destination already exists')
    # Make the unchanged source manifest the marker only after all payloads pass.
    pending=args.source/'.manifest.pending'
    with pending.open('wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pending.replace(args.source/'manifest.json')
    args.source.rename(args.destination)
    progress.update(status='complete',destination=str(args.destination),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),reader_files=len(files)+1,
        wall_s=time.monotonic()-started)
    atomic_json(args.progress,progress);print(json.dumps(progress),flush=True)


def transfer(args):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURMD_NODENAME','').split('.')[0]!='comp1400':
        raise RuntimeError('transfer must run in the authorised comp1400 allocation')
    args.work.mkdir(parents=True,exist_ok=True);status=args.work/'status.json'
    raw=(args.source/'manifest.json').read_bytes();manifest=json.loads(raw)
    files=manifest_files(args.source,manifest)
    sizes={name:(args.source/name).stat().st_size for name in files}
    total=sum(sizes.values())+len(raw)
    preparation=ssh(args.host,['python3','-c',REMOTE_PREPARE,args.destination,str(args.project),str(total)],input=raw)
    remote=json.loads(preparation.stdout);atomic_json(args.work/'remote-preflight.json',remote)
    if remote['status']=='already-published':atomic_json(status,remote);print(json.dumps(remote));return
    stage=remote['stage'];control=remote['control']
    atomic_json(status,dict(status='copying',job_id=os.environ['SLURM_JOB_ID'],bytes=total,files=len(files)+1,
        destination=args.destination,manifest_sha256=hashlib.sha256(raw).hexdigest()))
    rsync_base=['rsync','-rt','--omit-dir-times','--partial','--partial-dir=.rsync-partial',
        '-e','ssh -o BatchMode=yes -o ConnectTimeout=30']
    # Freeze the verifier and compact source audit artifacts beside the release.
    subprocess.run(rsync_base+[str(Path(__file__).resolve()),args.host+':'+control+'/transfer.py'],check=True)
    for name in ('launch.json',):
        if (args.source/name).exists():subprocess.run(rsync_base+[str(args.source/name),args.host+':'+control+'/'+name],check=True)
    archive=args.source.parent/'feature-cache-release-v1-code.tgz'
    if archive.exists():subprocess.run(rsync_base+[str(archive),args.host+':'+control+'/frozen-source.tgz'],check=True)
    bins=[[],[]];loads=[0,0]
    for name in sorted(files,key=lambda k:(-sizes[k],k)):
        slot=min(range(2),key=lambda i:(loads[i],i));bins[slot].append(name);loads[slot]+=sizes[name]
    lists=[]
    for i,names in enumerate(bins):
        path=args.work/f'files-{i}.txt';path.write_text('\n'.join(sorted(names))+'\n');lists.append(path)
    def copy(i):
        log=args.work/f'rsync-{i}.log'
        with log.open('ab') as output:
            for attempt in range(3):
                command=rsync_base+['--stats','--out-format=%t %l %n','--files-from='+str(lists[i]),
                    str(args.source)+'/',args.host+':'+stage+'/']
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
                if result.returncode==0:return dict(stream=i,status='complete',bytes=loads[i])
                output.write(f'\nrsync exit {result.returncode}, attempt {attempt+1}\n'.encode());output.flush()
                if attempt<2:time.sleep(10)
            raise RuntimeError(f'transfer stream {i} failed; see {log}')
    try:
        with ThreadPoolExecutor(2) as pool:
            for future in as_completed([pool.submit(copy,i) for i in range(2)]):print(json.dumps(future.result()),flush=True)
        atomic_json(status,dict(status='verifying',destination=args.destination,bytes=total,files=len(files)+1))
        command=['python3',control+'/transfer.py','verify','--source',stage,
            '--expected-manifest',control+'/expected-manifest.json','--destination',args.destination,
            '--progress',control+'/verification-progress.json']
        # Keep progress on the remote filesystem and stdout in the Slurm log.
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=30',args.host,shlex.join(command)],check=True)
        report=json.loads(ssh(args.host,['cat',control+'/verification-progress.json']).stdout)
        atomic_json(args.work/'verification.json',report);atomic_json(status,report)
    except Exception as exc:
        atomic_json(status,dict(status='failed',error=str(exc),destination=args.destination));raise


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('transfer');t.add_argument('--source',type=Path,required=True)
    t.add_argument('--work',type=Path,required=True);t.add_argument('--host',required=True)
    t.add_argument('--destination',required=True);t.add_argument('--project',type=int,default=815102)
    v=sub.add_parser('verify');v.add_argument('--source',type=Path,required=True)
    v.add_argument('--destination',type=Path,required=True);v.add_argument('--expected-manifest',type=Path,required=True)
    v.add_argument('--progress',type=Path,required=True)
    args=p.parse_args();(transfer if args.command=='transfer' else verify)(args)


if __name__=='__main__':main()
