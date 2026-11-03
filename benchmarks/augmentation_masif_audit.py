"""Frozen MaSIF audit of existing ANM coordinates; CPU Slurm workloads only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import shutil
import string
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.augmentation_store import AugmentationStore, ReleaseWriter, fingerprint
from augmentation_surface_audit import fit, frames, rms, surface, anchors

SEED = 20261013
DATASETS = ("atlas", "pinder", "pkpdb")
CONFIG = dict(version=1, seed=SEED, geodesic_radius_A=12., max_patch_vertices=200,
              mesh_resolution_A=1., msms_probe_A=1.5, mesh_anchor_tolerance_A=2.5,
              max_anchors=32, dropout=False, flip=False, inference_device="CPU",
              features=["shape_index", "distance_dependent_curvature", "hbond", "electrostatics", "hydropathy"])


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def git_revision(directory):
    git=Path(directory)/'.git'
    head=(git/'HEAD').read_text().strip()
    if not head.startswith('ref: '):return head
    reference=head[5:]
    if (git/reference).exists():return (git/reference).read_text().strip()
    for line in (git/'packed-refs').read_text().splitlines():
        if line and not line.startswith(('#','^')):
            value,name=line.split()
            if name==reference:return value
    raise ValueError('bundled source revision missing')


def choose_sample(eligible):
    """One median and one tail parent per dataset/size bin; stable tie breaking."""
    result = []
    for dataset in DATASETS:
        for size_bin in range(5):
            sub = [r for r in eligible if r["dataset"] == dataset and r["size_bin"] == size_bin]
            if len(sub) < 2: raise ValueError(f"fewer than two eligible parents: {dataset}/{size_bin}")
            median = float(np.median([r["mean_local_surface_distance_A"] for r in sub]))
            typical = min(sub, key=lambda r: (abs(r["mean_local_surface_distance_A"]-median), r["parent"]))
            tail = min([r for r in sub if r["parent"] != typical["parent"]],
                       key=lambda r: (-r["p95_local_surface_distance_A"], r["parent"]))
            result.extend([dict(typical, sample_kind="typical"), dict(tail, sample_kind="tail")])
    assert len({r["parent"] for r in result}) == 30
    return result


def select(args):
    require_compute(threads=1)
    values = defaultdict(list)
    with (args.previous_audit/"patches.csv").open() as f:
        for row in csv.DictReader(f):
            if row["group"] == "selected_vs_zero" and row["zero_valid"] == "True":
                values[row["parent"]].append(float(row["patch_after_local_distance_A"]))
    eligible = []; excluded = []
    with AugmentationStore(args.store) as store:
        for row in store.rows:
            record = store.raw(row["id"]); meta = record["metadata"]
            source = meta["source"]; old = args.historical/f'{source["index"]:03d}'
            reason = None
            if len(meta["selected"]) != 4 or not meta["zero_quality"]["geometry_valid"]:
                reason = "current zero/selection failed"
            elif row["id"] not in values: reason = "no previous surface patches"
            elif not (old/"report.json").exists(): reason = "historical report absent"
            else:
                previous = json.loads((old/"report.json").read_text())
                metrics = {r["candidate"]: r for r in previous["metrics"]}
                if not all(metrics.get(c, {}).get("geometry_valid", False) for c in (0, 1, 9, 17)):
                    reason = "historical direction/zero failed"
            if reason:
                excluded.append(dict(parent=row["id"], reason=reason)); continue
            eligible.append(dict(parent=row["id"], dataset=source["dataset"], split=source["split"],
                                 size_bin=int(source["bin"]), source_index=int(source["index"]),
                                 source_group=source["group"], n_residues=len(meta["residue_names"]),
                                 mean_local_surface_distance_A=float(np.mean(values[row["id"]])),
                                 p95_local_surface_distance_A=float(np.quantile(values[row["id"]], .95))))
    result = dict(config=CONFIG, parents=choose_sample(eligible), exclusions=excluded,
                  eligible_counts={f"{d}/{b}": sum(r["dataset"]==d and r["size_bin"]==b for r in eligible)
                                   for d in DATASETS for b in range(5)},
                  store=str(args.store.resolve()), historical=str(args.historical.resolve()),
                  previous_audit=str(args.previous_audit.resolve()),
                  input_manifest_sha256=digest(args.store/"manifest.json"),
                  previous_patch_csv_sha256=digest(args.previous_audit/"patches.csv"),
                  sampling="one typical and one tail per dataset/length bin; enriched, not population representative")
    atomic_json(args.output/"sample.json", result)
    print(json.dumps(dict(parents=30, output=str(args.output/"sample.json"))), flush=True)


def write_pdb(path, xyz, metadata, arrays):
    """Sequential residue IDs, one chain per segment, reversible identity map."""
    segments = arrays["segments"]
    unique = list(dict.fromkeys(map(int, segments)))
    codes = string.ascii_uppercase+string.ascii_lowercase+string.digits
    if len(unique)>len(codes) or len(metadata["residue_names"])>9999 or len(xyz)>99999:
        raise ValueError("unit exceeds reversible PDB export limits")
    chain = {s: codes[i] for i,s in enumerate(unique)}
    lines = []; last = None
    for j,(position,r,name) in enumerate(zip(xyz,arrays["atom_residue"],metadata["atom_names"]),1):
        r = int(r); ch = chain[int(segments[r])]
        if last is not None and last != ch: lines.append("TER")
        atomfield = f" {name:<3s}" if len(name)<4 else name
        x,y,z = position
        lines.append(f"ATOM  {j:5d} {atomfield} {metadata['residue_names'][r]:3s} {ch}{r+1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          {name[0]:>2s}")
        last = ch
    lookup = {(int(r),n):j+1 for j,(r,n) in enumerate(zip(arrays["atom_residue"],metadata["atom_names"]))}
    for a,b in metadata["disulfides"]:
        lines.append(f"CONECT{lookup[a,'SG']:5d}{lookup[b,'SG']:5d}")
        lines.append(f"CONECT{lookup[b,'SG']:5d}{lookup[a,'SG']:5d}")
    path.write_text("\n".join(lines+["TER","END"])+"\n")
    return dict(chain_by_segment=chain, residue_keys=metadata["residue_keys"])


def load_views(sample, parent):
    with AugmentationStore(sample["store"]) as store: record = store.raw(parent["parent"])
    m,a = record["metadata"],record["arrays"]
    old = Path(sample["historical"])/f'{m["source"]["index"]:03d}'
    prep = json.loads((old/"preparation.json").read_text())
    heavy = np.flatnonzero(np.asarray(prep["is_native"]) & np.asarray(prep["is_heavy"]))
    if [prep["atom_names"][i] for i in heavy] != m["atom_names"]:
        raise ValueError("historical atom names differ")
    if not np.array_equal(np.asarray(prep["residue"])[heavy], a["atom_residue"]):
        raise ValueError("historical residue correspondence differs")
    with np.load(old/"prepared.npz") as z:
        if not np.allclose(z["xyz"][heavy], a["native_coordinates"], atol=.001,rtol=0):
            raise ValueError("historical source differs")
    views = [("zero", a["zero_coordinates"]), ("native", a["native_coordinates"])]
    views += [(f"selected_{c}",x) for c,x in zip(m["selected"],a["coordinates"])]
    views += [("historical_zero", np.load(old/"native/c00/min.npy")[heavy])]
    views += [(f"amplitude_{amp}",np.load(old/f"native/c{c:02d}/min.npy")[heavy]) for c,amp in ((1,"0.5"),(9,"1"),(17,"2"))]
    his = {}
    for r,resname in enumerate(m["residue_names"]):
        if resname == "HIS":
            his[str(r+1)] = sorted({name for name,owner,native in zip(prep["atom_names"],prep["residue"],prep["is_native"])
                                    if native and owner==r and name in {"HD1","HE2"}})
    return m,a,[(role,np.asarray(x,np.float32)) for role,x in views],his


def container_command(runtime, work, job):
    return [str(runtime/"setup-env/bin/udocker"), "--quiet", "run",
            "--env=LD_LIBRARY_PATH=/compat", "--volume="+str(runtime/"compat")+":/compat",
            "--env=OMP_NUM_THREADS=1", "--env=OPENBLAS_NUM_THREADS=1", "--env=MKL_NUM_THREADS=1",
            "--env=CUDA_VISIBLE_DEVICES=", "--volume="+str(work)+":/audit",
            "--volume="+str(Path(__file__).resolve().parent)+":/code",
            "masif-audit", "python", "/code/augmentation_masif_bridge.py", "/audit/"+job]


def bridge(runtime, work, job):
    env = dict(os.environ, UDOCKER_DIR=str(runtime/"udocker"))
    with (work/(job+".log")).open("w") as log:
        subprocess.run(container_command(runtime,work,job), env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=3600)


def execute_frame(runtime,work,job):
    """Large MSMS/APBS intermediates stay on node-local job scratch."""
    temporary=os.environ.get('SLURM_TMPDIR') or '/tmp'
    with tempfile.TemporaryDirectory(prefix='ka-masif-'+os.environ['SLURM_JOB_ID']+'-',dir=temporary) as name:
        staging=Path(name)
        shutil.copyfile(work/'input.pdb',staging/'input.pdb')
        atomic_json(staging/'job.json',job)
        atomic_json(work/'job.json',job)
        try:
            bridge(runtime,staging.resolve(),'job.json')
            for filename in ('features.npz','receipt.json'):
                pending=work/(filename+'.pending')
                shutil.copyfile(staging/filename,pending);os.replace(pending,work/filename)
        finally:
            log=staging/'job.json.log'
            if log.exists():shutil.copyfile(log,work/log.name)


def preflight(args):
    require_compute(threads=1)
    work = args.output/"preflight"; work.mkdir(parents=True,exist_ok=True)
    atomic_json(work/"job.json", dict(command="preflight", result="/audit/result.json", config=CONFIG))
    bridge(args.runtime.resolve(),work.resolve(),"job.json")
    result = json.loads((work/"result.json").read_text())
    result["image_manifest_sha256"] = digest(args.runtime/"image-manifest.txt")
    result["bridge_sha256"] = digest(Path(__file__).with_name("augmentation_masif_bridge.py"))
    result["udocker_freeze_sha256"] = digest(args.runtime/"setup-freeze.txt")
    result["compat_sha256"] = {p.name:digest(p) for p in (args.runtime/"compat").iterdir()}
    roots={p.resolve() for p in (args.runtime/'udocker/containers').glob('*/ROOT/masif')}
    if len(roots)!=1:raise ValueError('ambiguous installed MaSIF source')
    installed=roots.pop()
    result['bundled_git_revision']=git_revision(installed)
    for name in ['default_config/chemistry.py','triangulation/fixmesh.py','triangulation/compute_normal.py',
                 'triangulation/xyzrn.py','input_output/save_ply.py','masif_modules/train_ppi_search.py']:
        result['source_sha256'][name]=digest(installed/'source'/name)
    atomic_json(args.output/"environment.json", result)
    print(json.dumps(result),flush=True)


def parent_task(task):
    sample,parent,runtime,output,smoke = task
    started=time.monotonic(); metadata={**parent}; saved={}; failures=[]; roles={}
    env=json.loads((Path(output)/"environment.json").read_text())
    parent_hash=hashlib.sha256(parent["parent"].encode()).hexdigest()[:16]
    cache=Path(output)/"cache"/parent_hash; cache.mkdir(parents=True,exist_ok=True)
    try:
        m,a,views,his=load_views(sample,parent)
        ref=a["zero_coordinates"].astype(float); bb=a["backbone_atoms"]; owner=a["atom_residue"]
        radii=np.array([{"C":1.70,"N":1.55,"O":1.52,"S":1.80}[n[0]]+1.4 for n in m["atom_names"]])
        s=surface(ref,bb,owner,radii,256)
        aa=anchors(ref,bb,owner,s,10.,32)
        anchor_residue=np.array([r for r,_,_ in aa],np.int32)
        centers=np.array([c for _,c,_ in aa])
        if not len(aa): raise ValueError("no eligible corresponding anchors")
        ref_frames=frames(ref,bb); refca=ref[bb[:,1]]
        byhash={}; viewmeta={}
        if smoke:
            rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
            views += [("rigid_zero",np.asarray(ref@rotation+[10.,-7.,3.],np.float32)),
                      ("repeat_zero",np.asarray(ref,np.float32))]
        for role,xyz in views:
            key=array_hash(xyz)
            if key in byhash and role != "repeat_zero":
                roles[role]=byhash[key]; continue
            frameid=f"v{len(viewmeta):02d}"; roles[role]=frameid; byhash[key]=frameid
            current=frames(xyz.astype(float),bb)
            targetcenters=np.stack([xyz[bb[r,1]]+(centers[i]-refca[r])@ref_frames[r].T@current[r] for i,r in enumerate(anchor_residue)])
            work=cache/frameid; work.mkdir(exist_ok=True)
            signature=fingerprint(dict(input=key,environment=env,config=CONFIG,his=his,
                                        host_sha256=digest(__file__),
                                        bridge_sha256=digest(Path(__file__).with_name("augmentation_masif_bridge.py")),
                                        anchor_centers=array_hash(targetcenters),role=role,
                                        seed=SEED+1 if role=="repeat_zero" else SEED))
            result=work/"features.npz"; receipt=work/"receipt.json"
            if not (receipt.exists() and result.exists() and json.loads(receipt.read_text()).get("signature")==signature
                    and json.loads(receipt.read_text()).get("npz_sha256")==digest(result)):
                write_pdb(work/"input.pdb",xyz,m,a)
                job=dict(command="frame", config=CONFIG, signature=signature,
                    input="/audit/input.pdb", output="/audit/features.npz", receipt="/audit/receipt.json",
                    his_states=his, anchor_residue=anchor_residue.tolist(), anchor_centers=targetcenters.tolist(),
                    seed=SEED+1 if role=="repeat_zero" else SEED, rigid_mesh_control=smoke and role=="zero")
                execute_frame(Path(runtime),work,job)
            receipt_data=json.loads(receipt.read_text())
            if receipt_data["signature"]!=signature or receipt_data["npz_sha256"]!=digest(result):
                raise ValueError("cached frame signature/checksum mismatch")
            with np.load(result) as z:
                data={k:z[k].copy() for k in z.files}
            viewmeta[frameid]=dict(role=role,coordinates_sha256=key,receipt=receipt_data)
            for name,value in data.items(): saved[frameid+"_"+name]=value
            saved[frameid+"_coordinates"]=xyz
            if role=="zero":
                valid=data["anchor_valid"].astype(bool)
                centers[valid]=data["anchor_centers"][valid]
            # Keep only packed frame cache and receipts; temporary grids and
            # whole-surface patch tensors are disposable.
            for child in work.iterdir():
                if child.name not in {"features.npz","receipt.json","job.json","job.json.log","input.pdb"}:
                    if child.is_dir(): shutil.rmtree(child)
                    else: child.unlink()
        # Charge/protonation identities must match across all views. Charges
        # may differ only by rounding in an old PARSE assignment.
        base=viewmeta[roles["zero"]]["receipt"]["charge_identity_sha256"]
        if any(v["receipt"]["charge_identity_sha256"]!=base for v in viewmeta.values()):
            raise ValueError("protonation/charge assignment changed between views")
        saved["backbone_atoms"]=bb; saved["atom_residue"]=owner; saved["completed"]=a["completed"]
        saved["anchor_residue"]=anchor_residue
        metadata.update(status="complete",roles=roles,views=viewmeta,residue_keys=m["residue_keys"],
                        atom_names=m["atom_names"],failures=failures,wall_s=time.monotonic()-started)
    except Exception as e:
        import traceback
        failures.append(dict(error=str(e),traceback=traceback.format_exc()))
        metadata.update(status="failed",roles=roles,failures=failures,wall_s=time.monotonic()-started)
        saved={}
    return metadata,saved


def worker_affinity(queue):
    if hasattr(os,"sched_setaffinity"):os.sched_setaffinity(0,set(queue.get()))


def staging_task(task):
    parent,runtime,output=task
    token=hashlib.sha256(parent['parent'].encode()).hexdigest()[:16]
    golden=Path(output)/'cache'/token/'v00'
    result=Path(output)/'staging-validation'/token;result.mkdir(parents=True,exist_ok=True)
    job=json.loads((golden/'job.json').read_text())
    shutil.copyfile(golden/'input.pdb',result/'input.pdb')
    execute_frame(Path(runtime),result,job)
    errors={}
    with np.load(golden/'features.npz') as a,np.load(result/'features.npz') as b:
        if set(a.files)!=set(b.files):raise ValueError('staging changed output fields')
        for name in a.files:
            if a[name].shape!=b[name].shape or a[name].dtype!=b[name].dtype:raise ValueError('staging changed array layout')
            if a[name].dtype.kind=='f':
                errors[name]=float(np.max(np.abs(a[name]-b[name]))) if a[name].size else 0.
                if not np.allclose(a[name],b[name],atol=1e-6,rtol=1e-6):raise ValueError('staging changed '+name)
            elif not np.array_equal(a[name],b[name]):raise ValueError('staging changed '+name)
    old=json.loads((golden/'receipt.json').read_text());new=json.loads((result/'receipt.json').read_text())
    if old['charge_identity_sha256']!=new['charge_identity_sha256']:raise ValueError('staging changed charge identity')
    return dict(parent=parent['parent'],passed=True,max_absolute_errors=errors,
                golden_npz_sha256=digest(golden/'features.npz'),staged_npz_sha256=digest(result/'features.npz'),
                golden_wall_s=old['wall_s'],staged_wall_s=new['wall_s'])


def validate_staging(args):
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    sample=json.loads((args.output/'sample.json').read_text())
    parents=[next(r for r in sample['parents'] if r['dataset']==d and r['size_bin']==2 and r['sample_kind']=='typical') for d in DATASETS]
    environment=json.loads((args.output/'environment.json').read_text())
    if environment['bridge_sha256']!=digest(Path(__file__).with_name('augmentation_masif_bridge.py')):
        raise ValueError('scientific adapter changed since golden outputs')
    queue=multiprocessing.Queue()
    for subset in np.array_split(sorted(os.sched_getaffinity(0)),3):queue.put(list(map(int,subset)))
    with ProcessPoolExecutor(max_workers=3,initializer=worker_affinity,initargs=(queue,)) as pool:
        results=list(pool.map(staging_task,[(r,str(args.runtime.resolve()),str(args.output.resolve())) for r in parents]))
    atomic_json(args.output/'staging-validation.json',dict(passed=True,results=results,host_code_sha256=digest(__file__),
        bridge_sha256=environment['bridge_sha256'],slurm_job=os.environ['SLURM_JOB_ID'],
        scope='same three real zero views: node-local scratch versus original shared scratch'))
    print(json.dumps(results),flush=True)


def run(args):
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    sample=json.loads((args.output/"sample.json").read_text())
    environment=json.loads((args.output/"environment.json").read_text())
    if not environment["passed"]: raise ValueError("environment preflight failed")
    if digest(Path(sample["store"])/"manifest.json")!=sample["input_manifest_sha256"]:
        raise ValueError("input release changed after selection")
    parents=sample["parents"]
    if args.smoke:
        parents=[next(r for r in parents if r["dataset"]==d and r["size_bin"]==2 and r["sample_kind"]=="typical") for d in DATASETS]
    else:
        smoke=json.loads((args.output/"smoke/summary.json").read_text())
        if not smoke["passed"]: raise ValueError("three-parent smoke test did not pass")
    dest=args.output/("smoke" if args.smoke else "full")
    provenance=dict(kind="frozen-masif-audit-v1",config=CONFIG,environment=environment,
                    sample_sha256=digest(args.output/"sample.json"),
                    code_sha256=digest(__file__),bridge_sha256=digest(Path(__file__).with_name("augmentation_masif_bridge.py")))
    if (dest/"coordinates/manifest.json").exists():
        print("immutable completed audit already exists",flush=True); return
    with ReleaseWriter(dest/"coordinates",provenance) as writer:
        tasks=[(sample,r,str(args.runtime.resolve()),str(args.output.resolve()),args.smoke) for r in parents if r["parent"] not in writer.ids]
        tasks.sort(key=lambda t:(-t[1]['n_residues'],t[1]['parent']))
        queue=multiprocessing.Queue()
        for subset in np.array_split(sorted(os.sched_getaffinity(0)),args.workers):queue.put(list(map(int,subset)))
        with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_affinity,initargs=(queue,)) as pool:
            for future in as_completed([pool.submit(parent_task,t) for t in tasks]):
                metadata,arrays=future.result()
                writer.add(metadata["parent"],metadata,arrays,summary=dict(dataset=metadata["dataset"],status=metadata["status"]))
                writer.checkpoint()
                print(json.dumps(dict(parent=metadata["parent"],status=metadata["status"],wall_s=metadata["wall_s"],failures=metadata["failures"])),flush=True)
        writer.finish(expected_ids=[r["parent"] for r in parents])
    atomic_json(dest/"execution.json",dict(parents=len(parents),workers=args.workers,allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']),
                                         slurm_job=os.environ["SLURM_JOB_ID"],node=os.environ["SLURMD_NODENAME"]))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["select","preflight","run","report","validate-staging"])
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--store",type=Path)
    p.add_argument("--historical",type=Path)
    p.add_argument("--previous-audit",type=Path)
    p.add_argument("--runtime",type=Path)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--smoke",action="store_true")
    return p


if __name__=="__main__":
    args=parser().parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command=="report":
        from augmentation_masif_report import report
        report(args)
    else: globals()[args.command.replace('-','_')](args)
