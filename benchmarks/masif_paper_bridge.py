"""Original-paper surfaces and chunked feature assembly in the TF1 container."""
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import types

import numpy as np
from masif_surface_quality import SURFACE_POLICY,clean_mesh,validate_coverage,validate_reduce_warning

PROFILE='masif-paper-2019-v1'
SURFACE_WARNINGS=[]
SURFACE_TIMEOUT=1800


def save(path,data):
    path=Path(path);temporary=Path(str(path)+'.pending')
    temporary.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(str(temporary),str(path))


def setup_surface(runtime):
    sys.path.insert(0,str(runtime/'dependencies'));sys.path.insert(0,str(runtime/'python3/source'))
    debugger=types.ModuleType('ipdb')
    def debug(*args,**kwargs):raise RuntimeError('interactive debugger in batch preprocessing')
    debugger.set_trace=debug;sys.modules['ipdb']=debugger
    original=subprocess.Popen
    class TextPopen(original):
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('universal_newlines',True);original.__init__(self,*args,**kwargs)
        def communicate(self,*args,**kwargs):
            kwargs.setdefault('timeout',SURFACE_TIMEOUT)
            try:
                out,err=original.communicate(self,*args,**kwargs)
                if (self.returncode==1 and isinstance(self.args,(list,tuple)) and len(self.args)==3
                        and Path(self.args[0]).name=='reduce' and self.args[1]=='-HIS'):
                    warning=validate_reduce_warning(Path(self.args[2]).read_text(),out,err)
                    SURFACE_WARNINGS.append(dict(warning,stderr_tail=err[-2000:]));return out,err
                if self.returncode:
                    raise RuntimeError('external command failed '+str(self.args)+' exit='+str(self.returncode)+
                                       ' stderr='+str(err)[-2000:]+' stdout='+str(out)[-1000:])
                return out,err
            except subprocess.TimeoutExpired:
                self.kill();original.communicate(self);raise
    subprocess.Popen=TextPopen
    os.environ.setdefault('PDB2XYZRN','/bin/false')


def component_mesh(pdb,compute_msms):
    from triangulation.xyzrn import output_pdb_as_xyzrn
    from input_output.read_msms import read_msms
    from default_config.global_vars import msms_bin
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    spheres=pdb.parent/'spheres.xyzrn';output_pdb_as_xyzrn(str(pdb),str(spheres))
    lines=spheres.read_text().splitlines();fields=[x.split() for x in lines]
    xyz=np.array([[float(v) for v in row[:3]] for row in fields]);radii=np.array([float(row[3])+1.5 for row in fields])
    tree=cKDTree(xyz);distances=tree.sparse_distance_matrix(tree,2*float(radii.max()),output_type='coo_matrix')
    keep=distances.data<=radii[distances.row]+radii[distances.col]
    graph=csr_matrix((np.ones(int(keep.sum()),np.int8),(distances.row[keep],distances.col[keep])),shape=(len(xyz),len(xyz)))
    count,labels=connected_components(graph,directed=False)
    if count==1:
        v,f,n,names,_=compute_msms(str(pdb),protonate=True)
        return v,f,n,names,[len(xyz)],np.zeros(len(v),np.int32)
    vertices=[];faces=[];normals=[];names=[];components=[];counts=[];offset=0
    for component in range(count):
        selected=np.flatnonzero(labels==component);counts.append(len(selected))
        prefix=pdb.parent/('component-'+str(component));spherefile=Path(str(prefix)+'.xyzrn')
        spherefile.write_text('\n'.join(lines[int(i)] for i in selected)+'\n')
        result=subprocess.run([msms_bin,'-density','3.0','-hdensity','3.0','-probe','1.5',
            '-if',str(spherefile),'-of',str(prefix),'-af',str(prefix)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)
        if result.returncode:raise RuntimeError('MSMS component failed: '+str(result.stderr)[-1500:])
        v,f,n,ids=read_msms(str(prefix));vertices.append(v);faces.append(f+offset);normals.append(n);names.extend(ids)
        offset+=len(v);components.append(np.full(len(v),component,np.int32))
    return np.concatenate(vertices),np.concatenate(faces),np.concatenate(normals),names,counts,np.concatenate(components)


def checked_apbs(vertices,pdb,prefix):
    """Paper PARSE/APBS commands with checked exits and complete vertex output."""
    from default_config.global_vars import apbs_bin,pdb2pqr_bin,multivalue_bin
    def run(command):
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=str(pdb.parent))
        process.communicate()
    run([pdb2pqr_bin,'--ff=parse','--whitespace','--noopt','--apbs-input',pdb.name,prefix.name])
    if not prefix.with_suffix('.in').exists():raise ValueError('PDB2PQR did not write APBS input')
    run([apbs_bin,prefix.name+'.in'])
    if not prefix.with_suffix('.dx').exists():raise ValueError('APBS did not write potential grid')
    np.savetxt(str(prefix)+'.csv',vertices,delimiter=',',fmt='%.17g')
    run([multivalue_bin,prefix.name+'.csv',prefix.name+'.dx',prefix.name+'_out.csv'])
    rows=np.loadtxt(str(prefix)+'_out.csv',delimiter=',',ndmin=2)
    if rows.shape!=(len(vertices),4) or not np.isfinite(rows).all():raise ValueError('incomplete or nonfinite APBS vertex potentials')
    if not np.allclose(rows[:,:3],vertices,atol=.002,rtol=0):raise ValueError('APBS vertex coordinates changed')
    return rows[:,3]


def surface(job,runtime):
    global SURFACE_TIMEOUT
    from scipy.io import savemat
    from scipy.spatial import cKDTree
    import pymesh
    from default_config.masif_opts import masif_opts
    from input_output.protonate import protonate
    from input_output.extractPDB import extractPDB
    from input_output.save_ply import save_ply
    from input_output.read_ply import read_ply
    from triangulation.computeMSMS import computeMSMS
    from triangulation.computeCharges import computeCharges,assignChargesToNewMesh
    from triangulation.computeHydrophobicity import computeHydrophobicity
    from triangulation.computeAPBS import computeAPBS
    from triangulation.compute_normal import compute_normal
    from triangulation.fixmesh import fix_mesh
    for function in [protonate,computeMSMS,computeCharges,assignChargesToNewMesh,computeAPBS,fix_mesh]:
        if not Path(inspect.getfile(function)).resolve().as_posix().startswith(str(runtime/'python3/source')+'/'):
            raise ValueError('surface function came from a different source profile')
    work=Path(job['work']);started=time.time();times={};stage='protonation'
    try:
        SURFACE_TIMEOUT=int(job.get('external_command_timeout_s',1800))
        if not 60<=SURFACE_TIMEOUT<=7200:raise ValueError('invalid external command timeout')
        SURFACE_WARNINGS[:]=[]
        if job.get('surface_policy')!=SURFACE_POLICY:raise ValueError('unknown surface preparation policy')
        masif_opts['tmp_dir']=str(work);t=time.time()
        protonate(job['input'],str(work/'protonated.pdb'))
        chains=''.join(sorted({line[21] for line in (work/'protonated.pdb').read_text().splitlines() if line.startswith('ATOM')}))
        pdb=work/'unit.pdb';extractPDB(str(work/'protonated.pdb'),str(pdb),chains);times['hydrogenation_s']=time.time()-t
        stage='surface';t=time.time();v,f,n,names,counts,components=component_mesh(pdb,computeMSMS)
        hb=computeCharges(str(pdb.with_suffix('')),v,names);hydro=computeHydrophobicity(names)
        stage='mesh_quality'
        atoms=np.array([[float(line[k:k+8]) for k in (30,38,46)] for line in pdb.read_text().splitlines()
                        if line.startswith('ATOM') and line[76:78].strip() not in ('H','D')])
        mesh=fix_mesh(pymesh.form_mesh(v,f),1.)
        mesh_path='paper-cleanup';mesh_retry_error=None
        try:
            vertices,faces,cleanup=clean_mesh(mesh.vertices,mesh.faces)
            coverage=validate_coverage(v,vertices,atoms)
        except ValueError as error:
            # The paper outer-hull step can discard almost the entire molecule.
            # Retry from raw MSMS without that operation; validate the result.
            mesh_retry_error=str(error);mesh_path='msms-remesh-without-outer-hull'
            mesh=pymesh.form_mesh(v,f);mesh,_=pymesh.remove_duplicated_vertices(mesh,.001)
            mesh,_=pymesh.split_long_edges(mesh,1.)
            for iteration in range(10):
                before=mesh.num_vertices
                mesh,_=pymesh.collapse_short_edges(mesh,1.,preserve_feature=True)
                mesh,_=pymesh.remove_obtuse_triangles(mesh,150.,100)
                if before==mesh.num_vertices:break
            mesh,_=pymesh.remove_duplicated_vertices(mesh,.001)
            vertices,faces,cleanup=clean_mesh(mesh.vertices,mesh.faces)
            coverage=validate_coverage(v,vertices,atoms)
        mesh=pymesh.form_mesh(vertices,faces);normals=compute_normal(mesh.vertices,mesh.faces)
        if not np.isfinite(normals).all():raise ValueError('nonfinite mesh normals')
        hb=assignChargesToNewMesh(mesh.vertices,v,hb,masif_opts);hydro=assignChargesToNewMesh(mesh.vertices,v,hydro,masif_opts)
        nearest=cKDTree(v).query(mesh.vertices)[1];owner=np.array([int(name.split('_')[1])-1 for name in names],np.int32)[nearest]
        represented=np.unique(components[nearest]).tolist()
        if len(represented)!=len(counts):raise ValueError('mesh cleanup lost a molecular component')
        if owner.min()<0 or owner.max()>=len(job['canonical_residue_ids']):raise ValueError('surface residue mapping outside canonical unit')
        times['surface_s']=time.time()-t;stage='electrostatics';t=time.time()
        potential=checked_apbs(mesh.vertices,pdb,work/'unit');times['electrostatics_s']=time.time()-t
        save_ply(str(work/'surface.ply'),mesh.vertices,mesh.faces,normals=normals,charges=potential,
            normalize_charges=True,hbond=hb,hphob=hydro)
        vv,ff,_,charge,_,hh,hy=read_ply(str(work/'surface.ply'))
        if not all(np.isfinite(x).all() for x in [vv,charge,hh,hy]):raise ValueError('nonfinite surface inputs')
        displacement,indices=cKDTree(mesh.vertices).query(vv)
        if len(vv)!=len(mesh.vertices) or not np.array_equal(indices,np.arange(len(vv))) or not np.array_equal(ff,mesh.faces):
            raise ValueError('PLY vertex/face correspondence changed')
        savemat(str(work/'surface.mat'),dict(vertices=vv,faces=ff,charge=charge,hbond=hh,hydrophobicity=hy))
        np.savez(str(work/'surface.npz'),vertices=vv,faces=ff,owner=owner)
        result=dict(status='complete',id=job['id'],stage=stage,timings=times,vertices=len(vv),
            surface_policy=SURFACE_POLICY,mesh_path=mesh_path,mesh_retry_error=mesh_retry_error,
            external_tool_exit_policy='checked-exits-reduce323-validated-warning-v2',external_command_timeout_s=SURFACE_TIMEOUT,hydrogenation_warnings=list(SURFACE_WARNINGS),
            mesh_cleanup=cleanup,mesh_coverage=coverage,
            molecular_component_atoms=counts,represented_molecular_components=represented,
            ply_coordinate_rounding_max_A=float(displacement.max()),
            surface_residues=int(len(np.unique(owner))),apbs_input=(work/'unit.in').read_text(),wall_s=time.time()-started)
    except Exception as exc:
        result=dict(status='failed',id=job['id'],stage=stage,timings=times,error=type(exc).__name__+': '+str(exc),
                    traceback=traceback.format_exc()[-3000:],wall_s=time.time()-started)
    save(work/'surface-receipt.json',result)


def original_functions(path,names=None):
    import h5py
    import scipy.sparse as sp
    tree=ast.parse(path.read_text());namespace=dict(np=np,h5py=h5py,sp=sp)
    definitions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and (names is None or x.name in names)]
    exec(compile(ast.Module(body=definitions),str(path),'exec'),namespace)
    return namespace


def describe(jobs,runtime,manifest,gpu_spool):
    import augmentation_masif_bridge as base
    from masif_gpu_queue import submit
    source=runtime/'original/source/masif_modules'
    read=original_functions(source/'read_data_from_matfile.py',{'load_matlab_file','extract_patch_and_coord'})
    extract=original_functions(source/'extract_features.py');pack=original_functions(source/'compute_input_feat.py')
    model=None;restore=0.
    for job in jobs:
        work=Path(job['work']);started=time.time();stage='surface';times={}
        try:
            receipt=json.loads((work/'surface-receipt.json').read_text())
            if receipt['status']!='complete':save(work/'receipt.json',receipt);continue
            times.update(receipt['timings']);stage='paper_matlab'
            matlab=json.loads((work/'matlab-receipt.json').read_text())
            if matlab['status']!='complete':raise ValueError(matlab['error'])
            if model is None:
                began=time.time();model=base.model();restore=time.time()-began
                if model.checkpoint_sha256!=manifest['checkpoint_sha256']:raise ValueError('restored model differs from preprocessing profile')
            times.update(curvature_s=matlab['curvature_s'],geodesics_s=matlab['coordinates_s'])
            stage='paper_features';t=time.time()
            shape=read['load_matlab_file'](str(work/'shapes/mesh.mat'),'p1',True)
            coords=read['load_matlab_file'](str(work/'coords/mesh.mat'),('all_patch_coord','p1'))
            count=int(np.asarray(shape['X']).size);descriptors=np.empty((count,80),np.float32)
            feature_s=0.;inference_s=0.;reference_error=0.;reference_checks=0
            patch_support=[]
            for start in range(0,count,1024):
                t=time.time();descriptions=[];polar=[]
                for center in range(start,min(count,start+1024)):
                    patch,coord=read['extract_patch_and_coord'](center,shape,coords,12.,200);descriptions.append(patch);polar.append(coord)
                r,theta,si,ddc,electro,hb,hyd=extract['extract_features'](descriptions,polar)
                rho,angles,features,mask=pack['compute_input_feat'](r,theta,si,ddc,hb,electro,200,list_hphob=hyd)
                arrays=dict(features=features.astype(np.float32),rho=rho.astype(np.float32),theta=angles.astype(np.float32),mask=mask[:,:,0].astype(np.float32))
                if not all(np.isfinite(x).all() for x in arrays.values()):raise ValueError('nonfinite paper features')
                if job.get('patch_diagnostics'):
                    patch_support.extend(np.sum(arrays['mask']>0,axis=1).astype(int).tolist())
                feature_s+=time.time()-t;t=time.time()
                if gpu_spool:
                    value=submit(gpu_spool,arrays,model.checkpoint_sha256)
                    # Every structure checks real patches against independent TF1 inference.
                    if start==0:
                        expected=base.infer(model,*[arrays[k][:3] for k in ['features','rho','theta','mask']])
                        reference_error=float(np.max(np.abs(value[:3]-expected)));reference_checks=len(expected)
                        if reference_error>1e-4:raise ValueError('GPU descriptors differ from TensorFlow reference')
                else:value=base.infer(model,*[arrays[k] for k in ['features','rho','theta','mask']])
                descriptors[start:start+len(value)]=value;inference_s+=time.time()-t
            times.update(patches_s=feature_s,inference_s=inference_s,model_restore_s=restore);restore=0.
            if job.get('patch_diagnostics'):
                save(job['patch_diagnostics'],dict(centers=count,support=patch_support,
                    empty_patches=sum(n==0 for n in patch_support),patches_below_three_vertices=sum(n<3 for n in patch_support),
                    finite_features=True,radius_A=12.,max_patch_vertices=200))
            with np.load(work/'surface.npz',allow_pickle=False) as mesh:
                vertices=mesh['vertices'];faces=mesh['faces'];owner=mesh['owner']
            normals=np.asarray(shape['normal']).T
            raw=np.stack([np.asarray(shape['shape_index']).ravel(),np.asarray(shape['hbond']).ravel(),
                extract['normalize_electrostatics']([np.asarray(shape['charge']).ravel()])[0],np.asarray(shape['hphob']).ravel()/4.5],axis=1)
            if not np.isfinite(descriptors).all():raise ValueError('nonfinite final descriptors')
            stage='save';t=time.time()
            np.savez_compressed(job['output'],vertices=vertices.astype(np.float32),faces=faces.astype(np.int32),
                normals=normals.astype(np.float32),owner=owner.astype(np.int32),raw_features=raw.astype(np.float32),
                descriptors=descriptors,canonical_residue_ids=np.asarray(job['canonical_residue_ids'],np.int32))
            times['save_s']=time.time()-t
            result=dict(receipt,status='complete',stage='complete',timings=times,descriptor_dimensions=80,
                checkpoint_sha256=model.checkpoint_sha256,checkpoint_validation=model.checkpoint_validation,
                preprocessing_profile=PROFILE,paper_runtime_sha256=job['paper_runtime_sha256'],
                descriptor_max_reference_error=reference_error,reference_patches_checked=reference_checks,
                inference_backend='jax-gpu' if gpu_spool else 'tensorflow-cpu',
                annotation_s=sum(times.values()),wall_s=time.time()-started)
        except Exception as exc:
            result=dict(status='failed',id=job['id'],stage=stage,timings=times,error=type(exc).__name__+': '+str(exc),
                        traceback=traceback.format_exc()[-3000:],wall_s=time.time()-started)
        save(job['receipt'],result);print(json.dumps(dict(id=job['id'],status=result['status'],stage=result['stage'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=['surface','describe'])
    parser.add_argument('--jobs',type=Path,required=True);parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--gpu-spool',type=Path);args=parser.parse_args()
    jobs=json.loads(args.jobs.read_text());manifest=json.loads((args.runtime/'manifest.json').read_text())
    if manifest['profile']!=PROFILE:raise ValueError('unknown preprocessing profile')
    if args.command=='surface':
        setup_surface(args.runtime)
        for job in jobs:surface(job,args.runtime)
    else:describe(jobs,args.runtime,manifest,args.gpu_spool)
