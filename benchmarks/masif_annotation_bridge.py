"""Python-3.6 author-container runner for dense frozen MaSIF-search annotation.

Uses bounded shortest-path blocks and sparse distance storage instead of the
upstream dense N-by-N angle matrix. MDS, features and checkpoint are unchanged.
"""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.manifold import MDS
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

import augmentation_masif_bridge as base

MODEL = None
DISTANCES = None
REFERENCE_POLAR = base.polar_at_centers


def prepare_distances(mesh):
    global DISTANCES
    start=time.time();vertices=mesh.vertices;faces=mesh.faces;n=len(vertices)
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]])),axis=1),axis=0)
    a,b=edges.T;weights=np.linalg.norm(vertices[a]-vertices[b],axis=1)
    graph=csr_matrix((np.concatenate((weights,weights)),(np.r_[a,b],np.r_[b,a])),shape=(n,n))
    data=[];indices=[];indptr=[0]
    for start_row in range(0,n,32):
        block=dijkstra(graph,directed=False,indices=np.arange(start_row,min(start_row+32,n)),limit=12.)
        for offset,row in enumerate(block):
            valid=np.flatnonzero(np.isfinite(row));values=row[valid].copy()
            values[valid==start_row+offset]=1e-8
            data.append(values);indices.append(valid.astype(np.int32));indptr.append(indptr[-1]+len(valid))
    DISTANCES=csr_matrix((np.concatenate(data),np.concatenate(indices),np.asarray(indptr,np.int64)),shape=(n,n))
    faceidx={}
    for i,face in enumerate(faces):
        for v in face:faceidx.setdefault(int(v),[]).append(i)
    DISTANCES.sort_indices()
    return faceidx,time.time()-start,int(DISTANCES.data.nbytes+DISTANCES.indices.nbytes+DISTANCES.indptr.nbytes)


def sparse_polar(mesh,centers,seed,radius=12.,max_vertices=200):
    from geometry.compute_polar_coordinates import compute_thetas
    if DISTANCES is None:raise RuntimeError('geodesic distances not prepared')
    vertices=mesh.vertices;faces=mesh.faces
    normals=np.stack([mesh.get_attribute('vertex_n'+a) for a in 'xyz'],axis=1)
    faceidx={}
    for i,face in enumerate(faces):
        for v in face:faceidx.setdefault(int(v),[]).append(i)
    rho=np.zeros((len(centers),max_vertices));theta=rho.copy();mask=rho.copy()
    indices=np.full((len(centers),max_vertices),-1,np.int32)
    for row,center in enumerate(centers):
        if row and row%1000==0:print(json.dumps(dict(stage='patches',completed=row,total=len(centers))),flush=True)
        dd=DISTANCES.getrow(int(center));order=np.lexsort((dd.indices,dd.data))[:max_vertices]
        members=dd.indices[order];inner=dd.indices[dd.data<radius/2]
        if len(inner)<3:raise ValueError('insufficient inner patch at vertex '+str(center))
        matrix=DISTANCES[inner,:][:,inner].toarray()
        plane=MDS(n_components=2,n_init=1,eps=.1,max_iter=50,dissimilarity='precomputed',n_jobs=1,
                  random_state=seed+int(center)).fit_transform(matrix)
        angle=compute_thetas(plane,int(center),vertices,faces,normals,inner,faceidx)
        outer=np.setdiff1d(members,inner)
        if len(outer):
            distances=DISTANCES[outer,:][:,inner].toarray();distances[distances==0]=np.inf
            angle[outer]=angle[inner[np.argmin(distances,axis=1)]]
        width=len(members);rho[row,:width]=dd.data[order];theta[row,:width]=angle[members]%(2*np.pi)
        mask[row,:width]=1;indices[row,:width]=members
    return rho,theta,mask,indices


def protonate_native(job):
    """Keep Reduce's native histidine choice; heavy coordinates must stay fixed."""
    source=Path(job['input']);out=source.with_name('protonated.pdb')
    text=base.checked(['reduce','-HIS','-NOFLIP',str(source)])
    atomlines=[line for line in text.splitlines() if line.startswith('ATOM  ')]
    atomlines.sort(key=lambda line:int(line[22:26]));ordered=[];last=None;serial_map={}
    for j,line in enumerate(atomlines,1):
        if last is not None and line[21]!=last:ordered.append('TER')
        if line[12:16].strip()[0]!='H':serial_map[int(line[6:11])]=j
        ordered.append(line[:6]+'{:5d}'.format(j)+line[11:]);last=line[21]
    for line in source.read_text().splitlines():
        if line.startswith('CONECT'):
            ids=[int(line[i:i+5]) for i in range(6,len(line),5) if line[i:i+5].strip()]
            ordered.append('CONECT'+''.join('{:5d}'.format(serial_map[i]) for i in ids))
    out.write_text('\n'.join(ordered+['TER','END'])+'\n')
    a,b=base.heavy_pdb(source),base.heavy_pdb(out)
    if set(a)!=set(b):raise ValueError('Reduce changed heavy atom inventory')
    error=max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
    if error>.000501:raise ValueError('Reduce moved native heavy atoms')
    return out,error


def electrostatics_native(vertices,pdb):
    """Native MaSIF chemistry may complete atoms; original coordinates stay fixed."""
    from default_config.global_vars import pdb2pqr_bin,apbs_bin,multivalue_bin
    directory=str(pdb.parent);name='potential'
    base.checked([pdb2pqr_bin,'--ff=parse','--whitespace','--noopt','--nodebump','--apbs-input',pdb.name,name],cwd=directory)
    original=base.heavy_pdb(pdb);actual={};identity=[]
    for line in (pdb.parent/name).read_text().splitlines():
        if not line.startswith(('ATOM','HETATM')):continue
        f=line.split();resid=int(f[-6]);atom=f[2];xyz=np.array(list(map(float,f[-5:-2])))
        identity.append((resid,atom,f[3],float(f[-2]),float(f[-1])))
        if atom[0]!='H':actual[resid,atom]=xyz
    missing=set(original)-set(actual);added=set(actual)-set(original);renamed=[]
    # Some terminal force fields rename equivalent oxygens. Require a unique
    # same-residue/element coordinate match rather than losing correspondence.
    for key in sorted(missing):
        choices=[candidate for candidate in added if candidate[0]==key[0] and candidate[1][0]==key[1][0]
                 and np.max(np.abs(actual[candidate]-original[key]))<=.000501]
        if len(choices)!=1:raise ValueError('PDB2PQR lost native heavy atom '+str(key)+'; added '+str(sorted(added)[:15]))
        renamed.append([list(key),list(choices[0])]);actual[key]=actual[choices[0]];added.remove(choices[0])
    movement=max(float(np.max(np.abs(actual[key]-xyz))) for key,xyz in original.items())
    if movement>.000501:raise ValueError('PDB2PQR moved a native heavy atom: '+str(movement))
    charge_hash=hashlib.sha256(json.dumps(sorted(identity),separators=(',',':')).encode()).hexdigest()
    base.checked([apbs_bin,name+'.in'],cwd=directory)
    np.savetxt(str(pdb.parent/(name+'.csv')),vertices,delimiter=',')
    base.checked([multivalue_bin,name+'.csv',name+'.dx',name+'_out.csv'],cwd=directory)
    potentials=np.loadtxt(str(pdb.parent/(name+'_out.csv')),delimiter=',')[:,3]
    if len(potentials)!=len(vertices) or not np.isfinite(potentials).all():raise ValueError('invalid APBS field')
    return potentials,charge_hash,dict(added_heavy_atoms=[list(k) for k in sorted(added)],
        renamed_heavy_atoms=renamed,native_heavy_max_movement_A=movement)


def compute_all_components(pdb):
    """MSMS's default follows one component; explicitly mesh each sphere union."""
    from triangulation.xyzrn import output_pdb_as_xyzrn
    from input_output.read_msms import read_msms
    from default_config.global_vars import msms_bin
    xyzrn=pdb.parent/'all-spheres.xyzrn';output_pdb_as_xyzrn(str(pdb),str(xyzrn))
    lines=xyzrn.read_text().splitlines();fields=[line.split() for line in lines]
    xyz=np.array([[float(v) for v in row[:3]] for row in fields]);radii=np.array([float(row[3])+1.5 for row in fields])
    tree=cKDTree(xyz);distances=tree.sparse_distance_matrix(tree,2*float(np.max(radii)),output_type='coo_matrix')
    keep=distances.data<=radii[distances.row]+radii[distances.col]
    graph=csr_matrix((np.ones(int(keep.sum()),np.int8),(distances.row[keep],distances.col[keep])),shape=(len(xyz),len(xyz)))
    count,labels=connected_components(graph,directed=False)
    vertices=[];faces=[];normals=[];names=[];areas={};offset=0;counts=[];vertex_components=[]
    for component in range(count):
        selected=np.flatnonzero(labels==component);counts.append(len(selected))
        prefix=pdb.parent/('surface-{:04d}'.format(component));path=Path(str(prefix)+'.xyzrn')
        path.write_text('\n'.join(lines[int(i)] for i in selected)+'\n')
        base.checked([msms_bin,'-density','3.0','-hdensity','3.0','-probe','1.5','-if',str(path),'-of',str(prefix),'-af',str(prefix)])
        v,f,n,ids=read_msms(str(prefix));vertices.append(v);faces.append(f+offset);normals.append(n);names.extend(ids);offset+=len(v)
        vertex_components.append(np.full(len(v),component,np.int32))
        with open(str(prefix)+'.area') as stream:
            next(stream)
            for line in stream:
                row=line.split();areas[row[3]]=row[1]
    return np.concatenate(vertices),np.concatenate(faces),np.concatenate(normals),names,areas,counts,np.concatenate(vertex_components)


def annotate(job):
    global MODEL,DISTANCES
    started=time.time();times={};stage='imports'
    from default_config.masif_opts import masif_opts
    from triangulation.computeMSMS import computeMSMS
    from triangulation.fixmesh import fix_mesh
    from triangulation.computeCharges import computeCharges,assignChargesToNewMesh
    from triangulation.computeHydrophobicity import computeHydrophobicity
    from triangulation.compute_normal import compute_normal
    from scipy.spatial import cKDTree
    import pymesh
    masif_opts['tmp_dir']=str(Path(job['input']).parent)
    try:
        stage='hydrogenation';t=time.time();pdb,heavy_error=protonate_native(job);times[stage+'_s']=time.time()-t
        stage='surface';t=time.time()
        if job.get('all_components'):
            v,f,n,names,areas,component_atoms,raw_components=compute_all_components(pdb)
        else:
            v,f,n,names,areas=computeMSMS(str(pdb),protonate=True);component_atoms=None;raw_components=np.zeros(len(v),np.int32)
        hbond=computeCharges(str(pdb.with_suffix('')),v,names);hyd=computeHydrophobicity(names)
        mesh=fix_mesh(pymesh.form_mesh(v,f),1.);normals=compute_normal(mesh.vertices,mesh.faces)
        for j,axis in enumerate('xyz'):
            mesh.add_attribute('vertex_n'+axis);mesh.set_attribute('vertex_n'+axis,normals[:,j])
        hb=assignChargesToNewMesh(mesh.vertices,v,hbond,masif_opts);hyd=assignChargesToNewMesh(mesh.vertices,v,hyd,masif_opts)
        raw_owner=np.array([int(name.split('_')[1])-1 for name in names],int)
        nearest=cKDTree(v).query(mesh.vertices)[1];owner=raw_owner[nearest]
        represented_components=np.unique(raw_components[nearest]).tolist();times[stage+'_s']=time.time()-t
        stage='electrostatics';t=time.time();potential,identity,completion=electrostatics_native(mesh.vertices,pdb);times[stage+'_s']=time.time()-t
        stage='geodesics';_,elapsed,sparse_bytes=prepare_distances(mesh);times[stage+'_s']=elapsed
        stage='patches';t=time.time();base.polar_at_centers=sparse_polar
        centers=np.arange(len(mesh.vertices));features,rho,theta,mask,indices,raw,flat=base.features_at(mesh,hb,potential,hyd,centers,job['seed'])
        times[stage+'_s']=time.time()-t
        stage='model_restore';t=time.time()
        if MODEL is None:MODEL=base.model()
        times[stage+'_s']=time.time()-t
        stage='inference';t=time.time();descriptors=base.infer(MODEL,features,rho,theta,mask);times[stage+'_s']=time.time()-t
        # Compare random real-surface centers with the independently validated
        # adapter. Its geodesic paths are recomputed, not read from our CSR.
        stage='validation';t=time.time();chosen=np.random.RandomState(job['seed']).choice(len(centers),min(3,len(centers)),replace=False)
        base.polar_at_centers=REFERENCE_POLAR
        ref,rr,rt,rm,ri,_,_=base.features_at(mesh,hb,potential,hyd,chosen,job['seed'])
        error=float(np.max(np.abs(base.infer(MODEL,ref,rr,rt,rm)-descriptors[chosen])))
        if error>1e-4:raise ValueError('sparse dense-annotation descriptor differs: '+str(error))
        base.polar_at_centers=sparse_polar;times[stage+'_s']=time.time()-t
        stage='save';t=time.time()
        np.savez_compressed(job['output'],vertices=np.asarray(mesh.vertices,np.float32),faces=np.asarray(mesh.faces,np.int32),
            normals=np.asarray(normals,np.float32),owner=owner.astype(np.int32),
            raw_features=raw.astype(np.float32),descriptors=descriptors.astype(np.float32),
            canonical_residue_ids=np.asarray(job['canonical_residue_ids'],np.int32))
        times[stage+'_s']=time.time()-t
        result=dict(status='complete',id=job['id'],vertices=len(mesh.vertices),faces=len(mesh.faces),
            descriptor_dimensions=80,descriptor_max_reference_error=error,heavy_coordinate_max_error_A=heavy_error,
            checkpoint_sha256=MODEL.checkpoint_sha256,checkpoint_validation=MODEL.checkpoint_validation,
            preprocessing_profile='Python geometry with original-paper electrostatics clipping',
            charge_identity_sha256=identity,sparse_geodesic_bytes=sparse_bytes,timings=times,wall_s=time.time()-started,
            pqr_completion=completion,
            molecular_component_atoms=component_atoms,surface_residues=int(len(np.unique(owner))),
            represented_molecular_components=represented_components,raw_mesh_vertices=len(v),
            annotation_s=sum(value for key,value in times.items() if key!='validation_s'))
    except Exception as exc:
        result=dict(status='failed',id=job['id'],stage=stage,error=type(exc).__name__+': '+str(exc),
            traceback=traceback.format_exc()[-4000:],timings=times,wall_s=time.time()-started)
    finally:
        base.polar_at_centers=REFERENCE_POLAR;DISTANCES=None
    base.savejson(job['receipt'],result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    import sys
    from masif_checkpoint import resolve_checkpoint,validate_checkpoint
    validate_checkpoint(resolve_checkpoint())
    jobs=json.loads(Path(sys.argv[1]).read_text())
    def timeout(signum,frame):raise TimeoutError('annotation exceeded 30-minute per-structure limit')
    signal.signal(signal.SIGALRM,timeout)
    for job in jobs:
        signal.alarm(1800)
        try:annotate(job)
        finally:signal.alarm(0)
