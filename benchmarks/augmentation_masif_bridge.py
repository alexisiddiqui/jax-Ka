"""Python 3.6-compatible adapter executed inside the frozen author container.

Uses upstream MSMS/PyMesh/features/network. Computes upstream fast geodesic
coordinates only at requested centers, avoiding its quadratic all-center theta
allocation. No training operations are executed.
"""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/masif/source")


def sha(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def savejson(path, value):
    pending = str(path)+".pending"
    with open(pending,"w") as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
    os.rename(pending,str(path))


def checked(args,cwd=None):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=cwd,timeout=1800)
    # Legacy Reduce can return a positive adjustment count on successful PDB
    # output. Its product is checked atom-by-atom below; signal/fatal exits fail.
    reduce_product=(args[0]=="reduce" and p.returncode>=0 and "ATOM  " in p.stdout.decode()
                    and "FATAL" not in p.stderr.decode().upper())
    if p.returncode and not reduce_product:
        raise RuntimeError("command failed: {}\n{}\n{}".format(args,p.stdout.decode()[-3000:],p.stderr.decode()[-3000:]))
    return p.stdout.decode()


def model():
    import tensorflow as tf
    original=tf.ConfigProto
    def config(*args,**kwargs):
        cfg=original(*args,**kwargs)
        cfg.intra_op_parallelism_threads=1;cfg.inter_op_parallelism_threads=1
        cfg.device_count["GPU"]=0
        return cfg
    tf.ConfigProto=config
    from masif_modules.MaSIF_ppi_search import MaSIF_ppi_search
    obj=MaSIF_ppi_search(12.,n_thetas=16,n_rhos=5,n_rotations=16,idx_gpu="/cpu:0",feat_mask=[1.]*5)
    checkpoint="/masif/data/masif_ppi_search/nn_models/sc05/all_feat/model_data/model"
    obj.saver.restore(obj.session,checkpoint)
    return obj


def infer(obj,features,rho,theta,mask):
    from masif_modules.train_ppi_search import compute_val_test_desc
    result=compute_val_test_desc(obj,np.arange(len(features)),rho,theta,features,mask,batch_size=128,flip=False)
    if result.shape!=(len(features),80) or not np.isfinite(result).all():
        raise ValueError("invalid frozen fingerprints")
    return result


def heavy_pdb(path):
    atoms={}
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM  ","HETATM")) and line[12:16].strip()[0] != "H":
            key=(int(line[22:26]),line[12:16].strip())
            if key in atoms: raise ValueError("duplicate exported heavy atom")
            atoms[key]=np.array([float(line[k:k+8]) for k in (30,38,46)])
    return atoms


def protonate(job):
    source=Path(job["input"]); out=source.with_name("protonated.pdb")
    text=checked(["reduce","-HIS","-NOFLIP",str(source)])
    # Fix histidine inventory to the already prepared source's pH-7 state.
    # Ring H positions follow bonded heavy atoms; heavy coordinates never move.
    lines=text.splitlines(); atoms={}
    for line in lines:
        if line.startswith("ATOM  "):
            atoms[(int(line[22:26]),line[12:16].strip())]=(line,np.array([float(line[k:k+8]) for k in (30,38,46)]))
    kept=[l for l in lines if not (l.startswith("ATOM  ") and l[17:20]=="HIS" and l[12:16].strip() in {"HD1","HE2"})]
    serial=max([int(l[6:11]) for l in lines if l.startswith("ATOM  ")]+[0])
    for resid,hydrogens in job["his_states"].items():
        resid=int(resid)
        for hydrogen in hydrogens:
            nitrogen,neighbors=("ND1",("CG","CE1")) if hydrogen=="HD1" else ("NE2",("CD2","CE1"))
            template,n=atoms[resid,nitrogen]
            direction=np.zeros(3)
            for neighbor in neighbors:
                v=n-atoms[resid,neighbor][1]; direction+=v/np.linalg.norm(v)
            length=np.linalg.norm(direction)
            if length<1e-6:raise ValueError("degenerate histidine ring")
            direction/=length; xyz=n+direction
            serial+=1
            kept.append("ATOM  {:5d}  {:<3s} {}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}           H".format(
                serial,hydrogen,template[17:27],xyz[0],xyz[1],xyz[2],1.,0.))
    # PDB2PQR groups consecutive residue records. Histidine H must be emitted
    # beside its own residue, not appended after all chains.
    atomlines=[l for l in kept if l.startswith("ATOM  ")]
    atomlines.sort(key=lambda l:int(l[22:26]))
    ordered=[];last=None;serial_map={}
    for j,line in enumerate(atomlines,1):
        if last is not None and line[21]!=last:ordered.append("TER")
        if line[12:16].strip()[0]!="H":serial_map[int(line[6:11])]=j
        ordered.append(line[:6]+"{:5d}".format(j)+line[11:]);last=line[21]
    for line in source.read_text().splitlines():
        if line.startswith("CONECT"):
            ids=[int(line[i:i+5]) for i in range(6,len(line),5) if line[i:i+5].strip()]
            ordered.append("CONECT"+"".join("{:5d}".format(serial_map[i]) for i in ids))
    out.write_text("\n".join(ordered+["TER","END"])+"\n")
    a,b=heavy_pdb(source),heavy_pdb(out)
    if set(a)!=set(b): raise ValueError("Reduce changed heavy atom inventory")
    error=max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
    if error>.000501: raise ValueError("Reduce moved heavy atoms")
    return out,error


def electrostatics(vertices,pdb):
    from default_config.global_vars import pdb2pqr_bin,apbs_bin,multivalue_bin
    directory=str(pdb.parent); base="potential"
    checked([pdb2pqr_bin,"--ff=parse","--whitespace","--noopt","--nodebump","--apbs-input",pdb.name,base],cwd=directory)
    pqr=Path(directory)/base; original=heavy_pdb(pdb); pqrheavy={}; identity=[]
    for line in pqr.read_text().splitlines():
        if not line.startswith(("ATOM","HETATM")):continue
        fields=line.split(); resid=int(fields[-6]); name=fields[2]
        xyz=np.array(list(map(float,fields[-5:-2])))
        identity.append((resid,name,fields[3],float(fields[-2]),float(fields[-1])))
        if name[0]!="H":pqrheavy[resid,name]=xyz
    if set(pqrheavy)!=set(original):raise ValueError("PDB2PQR changed heavy atom inventory")
    if max(np.max(np.abs(pqrheavy[k]-original[k])) for k in original)>.000501:
        raise ValueError("PDB2PQR moved heavy atoms despite --noopt")
    charge_identity=hashlib.sha256(json.dumps(sorted(identity),separators=(",",":")).encode()).hexdigest()
    checked([apbs_bin,base+".in"],cwd=directory)
    np.savetxt(str(Path(directory)/(base+".csv")),vertices,delimiter=",")
    checked([multivalue_bin,base+".csv",base+".dx",base+"_out.csv"],cwd=directory)
    potentials=np.loadtxt(str(Path(directory)/(base+"_out.csv")),delimiter=",")[:,3]
    if len(potentials)!=len(vertices) or not np.isfinite(potentials).all():raise ValueError("invalid APBS field")
    return potentials,charge_identity


def vertex_area(vertices,faces):
    areas=np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)/2
    result=np.zeros(len(vertices))
    for j in range(3):np.add.at(result,faces[:,j],areas/3)
    return result


def polar_at_centers(mesh,centers,seed,radius=12.,max_vertices=200):
    """Upstream fast MDS construction with bounded center/distance storage."""
    import networkx as nx
    from sklearn.manifold import MDS
    from geometry.compute_polar_coordinates import compute_thetas
    vertices=mesh.vertices;faces=mesh.faces
    normals=np.stack([mesh.get_attribute("vertex_n"+a) for a in "xyz"],axis=1)
    graph=nx.Graph();graph.add_nodes_from(range(len(vertices)))
    edges=np.concatenate([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]])
    graph.add_weighted_edges_from((int(a),int(b),float(np.linalg.norm(vertices[a]-vertices[b]))) for a,b in edges)
    faceidx={}
    for i,face in enumerate(faces):
        for v in face:faceidx.setdefault(int(v),[]).append(i)
    rho=np.zeros((len(centers),max_vertices));theta=rho.copy();mask=rho.copy()
    indices=np.full((len(centers),max_vertices),-1,int)
    for row,center in enumerate(centers):
        d=nx.single_source_dijkstra_path_length(graph,int(center),cutoff=radius)
        ordered=sorted(d.items(),key=lambda kv:kv[1])[:max_vertices]
        members=np.array([k for k,v in ordered],int)
        inner=np.array(sorted(k for k,v in d.items() if v<radius/2),int)
        if len(inner)<3: raise ValueError("insufficient geodesic inner patch")
        distances={int(i):nx.single_source_dijkstra_path_length(graph,int(i),cutoff=radius) for i in inner}
        matrix=np.array([[distances[int(i)].get(int(j),0.) for j in inner] for i in inner])
        np.fill_diagonal(matrix,1e-8)
        mds=MDS(n_components=2,n_init=1,eps=.1,max_iter=50,dissimilarity="precomputed",n_jobs=1,random_state=seed+int(center))
        plane=mds.fit_transform(matrix)
        angle=compute_thetas(plane,int(center),vertices,faces,normals,inner,faceidx)
        for i,v in enumerate(members):
            if v not in inner:
                # Undirected distance lookup equals the upstream D[v,inner]
                # with missing paths (>radius) treated as infinity.
                dd=np.array([distances[int(k)].get(int(v),np.inf) for k in inner])
                angle[v]=angle[inner[np.argmin(dd)]]
            rho[row,i]=max(d[int(v)],1e-8);theta[row,i]=angle[v]%(2*np.pi)
            mask[row,i]=1;indices[row,i]=v
    return rho,theta,mask,indices


def features_at(mesh,hbond,potential,hydropathy,centers,seed,raw_only=False):
    from masif_modules.read_data_from_surface import compute_ddc,normalize_electrostatics
    mesh.add_attribute("vertex_mean_curvature"); mesh.add_attribute("vertex_gaussian_curvature")
    h=mesh.get_attribute("vertex_mean_curvature");k=mesh.get_attribute("vertex_gaussian_curvature")
    elem=h*h-k;elem[elem<0]=1e-8
    with np.errstate(divide="ignore",invalid="ignore"):
        si=np.arctan(h/np.sqrt(elem))*2/np.pi
    flat=(h==0)&(elem==0);si[flat]=0.
    if not np.isfinite(si).all():raise ValueError("nonfinite surface curvature")
    # save_ply normalizes potentials /10, then read_data normalizes/clips.
    raw=np.stack([si,hbond,normalize_electrostatics(potential/10),hydropathy/4.5],axis=1)
    if raw_only:return raw,int(flat.sum())
    rho,theta,mask,indices=polar_at_centers(mesh,centers,seed)
    normals=np.stack([mesh.get_attribute("vertex_n"+a) for a in "xyz"],axis=1)
    feats=np.zeros((len(centers),200,5))
    for row,center in enumerate(centers):
        members=indices[row,mask[row].astype(bool)]
        cp=np.flatnonzero(members==center)[0]
        ddc=compute_ddc(mesh.vertices[members],normals[members],cp,rho[row,mask[row].astype(bool)])
        feats[row,:len(members),0]=si[members];feats[row,:len(members),1]=ddc
        feats[row,:len(members),2:]=raw[members,1:]
    if not np.isfinite(feats).all():raise ValueError("nonfinite patch features")
    return feats,rho,theta,mask,indices,raw,int(flat.sum())


def preflight(job):
    import tensorflow as tf
    import pymesh
    obj=model()
    rng=np.random.RandomState(0)
    feat=rng.normal(size=(4,200,5));rho=rng.uniform(0,12,(4,200));theta=rng.uniform(0,2*np.pi,(4,200));mask=np.ones((4,200))
    one=infer(obj,feat,rho,theta,mask);two=infer(obj,feat,rho,theta,mask)
    error=float(np.max(np.abs(one-two)))
    if error>1e-6:raise ValueError("frozen inference is not repeatable")
    # Compare the bounded-center adapter against the unmodified upstream
    # all-center pipeline on a small closed mesh, including neural output.
    from triangulation.compute_normal import compute_normal
    from input_output.save_ply import save_ply
    from masif_modules.read_data_from_surface import read_data_from_surface
    vv=2*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
    ff=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]],int)
    nn=compute_normal(vv,ff);mesh=pymesh.form_mesh(vv,ff)
    for j,axis in enumerate("xyz"):
        mesh.add_attribute("vertex_n"+axis);mesh.set_attribute("vertex_n"+axis,nn[:,j])
    hb=np.linspace(-1,1,6);pot=np.linspace(-2,2,6);hyd=np.linspace(-4.5,4.5,6)
    filename=Path(job["result"]).with_name("reference.ply")
    save_ply(str(filename),vv,ff,normals=nn,charges=pot,hbond=hb,hphob=hyd,normalize_charges=True)
    np.random.seed(0)
    uf,ur,ut,um,ui,labels,coords=read_data_from_surface(str(filename),dict(max_distance=12.,max_shape_size=200))
    cf,cr,ct,cm,ci,raw,flat=features_at(mesh,hb,pot,hyd,np.array([0]),0)
    # Match vertex order explicitly: tied geodesic distances may be returned
    # in a different order without changing the surface convolution.
    uids=list(ui[0]);cids=list(ci[0,cm[0].astype(bool)])
    ix=[uids.index(v) for v in cids]
    feature_error=float(np.max(np.abs(cf[0,:len(ix)]-uf[0,ix])))
    radius_error=float(np.max(np.abs(cr[0,:len(ix)]-ur[0,ix])))
    descriptor_error=float(np.max(np.abs(infer(obj,uf[:1],ur[:1],ut[:1],um[:1])-infer(obj,cf,cr,ct,cm))))
    if feature_error>1e-5 or radius_error>1e-5 or descriptor_error>1e-4:
        raise ValueError("selected-center adapter differs from upstream: {}".format((feature_error,radius_error,descriptor_error)))
    files=list(Path("/masif/data/masif_ppi_search/nn_models/sc05/all_feat/model_data").glob("model*"))
    source_files=["masif_modules/MaSIF_ppi_search.py","masif_modules/read_data_from_surface.py","geometry/compute_polar_coordinates.py","triangulation/computeMSMS.py","triangulation/computeCharges.py","triangulation/computeHydrophobicity.py"]
    result=dict(passed=True,python=sys.version,numpy=np.__version__,tensorflow=tf.__version__,
                selected_center_upstream_check=dict(feature_error=feature_error,radius_error_A=radius_error,descriptor_error=descriptor_error),
                checkpoint_sha256={f.name:sha(f) for f in files},repeat_max_abs_error=error,
                descriptor_dimensions=80,source_sha256={f:sha(Path("/masif/source")/f) for f in source_files},
                environment={k:os.environ.get(k) for k in ("MSMS_BIN","APBS_BIN","MULTIVALUE_BIN","PDB2PQR_BIN")})
    for name,binary in result["environment"].items():
        if not binary or not Path(binary).exists():raise ValueError("missing binary: "+name)
    result["binary_sha256"]={name:sha(binary) for name,binary in result["environment"].items()}
    savejson(job["result"],result)


def frame(job):
    started=time.time();timings={}
    from default_config.masif_opts import masif_opts
    masif_opts["tmp_dir"]=str(Path(job["input"]).parent)
    from triangulation.computeMSMS import computeMSMS
    from triangulation.fixmesh import fix_mesh
    from triangulation.computeCharges import computeCharges,assignChargesToNewMesh
    from triangulation.computeHydrophobicity import computeHydrophobicity
    from triangulation.compute_normal import compute_normal
    import pymesh
    t=time.time();pdb,heavy_error=protonate(job);timings["hydrogenation_s"]=time.time()-t
    t=time.time();v,f,n,names,areas=computeMSMS(str(pdb),protonate=True)
    hbond=computeCharges(str(pdb.with_suffix("")),v,names);hphob=computeHydrophobicity(names)
    regular=fix_mesh(pymesh.form_mesh(v,f),1.)
    normals=compute_normal(regular.vertices,regular.faces)
    for j,axis in enumerate("xyz"):
        regular.add_attribute("vertex_n"+axis);regular.set_attribute("vertex_n"+axis,normals[:,j])
    hb=assignChargesToNewMesh(regular.vertices,v,hbond,masif_opts)
    hyd=assignChargesToNewMesh(regular.vertices,v,hphob,masif_opts)
    raw_owner=np.array([int(name.split("_")[1])-1 for name in names],int)
    owners=raw_owner[cKDTree(v).query(regular.vertices)[1]]
    timings["surface_s"]=time.time()-t
    if job.get("geometry_only"):
        if not job.get("raw_only"):raise ValueError("geometry-only mode requires raw-only features")
        potential=np.zeros(len(regular.vertices));identity=None
    else:
        t=time.time();potential,identity=electrostatics(regular.vertices,pdb);timings["electrostatics_s"]=time.time()-t
    anchors=np.array(job["anchor_residue"],int);targets=np.array(job["anchor_centers"],float)
    vertex_ids=np.full(len(anchors),-1,int);valid=np.zeros(len(anchors),bool);distance=np.full(len(anchors),999.,float)
    for i,r in enumerate(anchors):
        choices=np.flatnonzero(owners==r)
        if len(choices):
            distances=np.linalg.norm(regular.vertices[choices]-targets[i],axis=1)
            pick=np.argmin(distances);vertex_ids[i]=choices[pick];distance[i]=distances[pick]
            valid[i]=distance[i]<=job["config"]["mesh_anchor_tolerance_A"]
    if valid.sum()<3:raise ValueError("fewer than three corresponding surface anchors")
    t=time.time()
    if job.get("raw_only"):
        raw,flat_count=features_at(regular,hb,potential,hyd,vertex_ids[valid],job["seed"],raw_only=True)
        timings["raw_features_s"]=time.time()-t
        repeat_error=None
    else:
        feats,rho,theta,mask,indices,raw,flat_count=features_at(regular,hb,potential,hyd,vertex_ids[valid],job["seed"])
        timings["patches_s"]=time.time()-t
        t=time.time();obj=model();desc=infer(obj,feats,rho,theta,mask);again=infer(obj,feats,rho,theta,mask)
        repeat_error=float(np.max(np.abs(desc-again)))
        if repeat_error>1e-6:raise ValueError("fingerprints changed on cached repeat")
        timings["inference_and_restore_s"]=time.time()-t
    arrays=dict(vertices=np.asarray(regular.vertices,np.float32),faces=np.asarray(regular.faces,np.int32),
        normals=np.asarray(normals,np.float32),owner=np.asarray(owners,np.int32),raw_features=np.asarray(raw,np.float32),
        vertex_area=np.asarray(vertex_area(regular.vertices,regular.faces),np.float32),
        anchor_valid=valid,anchor_vertex_ids=vertex_ids.astype(np.int32),anchor_mapping_distance=distance.astype(np.float32))
    if job.get("geometry_only"):arrays["electrostatics_vertices"]=np.asarray(regular.vertices,np.float64)
    centers=targets.copy();centers[valid]=regular.vertices[vertex_ids[valid]]
    arrays["anchor_centers"]=centers.astype(np.float32)
    for name,value in ([] if job.get("raw_only") else [("descriptors",desc),("features",feats),("rho",rho),("theta",theta),("mask",mask),("patch_indices",indices)]):
        target=np.zeros((len(anchors),)+value.shape[1:],dtype=np.int32 if name=="patch_indices" else np.float32)
        if name=="patch_indices":target.fill(-1)
        target[valid]=value;arrays[name]=target
    # Fixed-mesh rigid transform exercises geodesic/normal handling independently
    # of triangulation and APBS grid noise.
    rigid_error=None
    if job.get("rigid_mesh_control") and not job.get("raw_only"):
        rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        rotated=pymesh.form_mesh(regular.vertices.dot(rot)+[10.,-7.,3.],regular.faces)
        for j,axis in enumerate("xyz"):
            rotated.add_attribute("vertex_n"+axis);rotated.set_attribute("vertex_n"+axis,normals.dot(rot)[:,j])
        rf,rr,rt,rm,ri,rraw,rflat=features_at(rotated,hb,potential,hyd,vertex_ids[valid],job["seed"])
        rd=infer(obj,rf,rr,rt,rm)
        rigid_error=float(np.median(np.linalg.norm(rd-desc,axis=1)))
        arrays["rigid_mesh_descriptor_distances"]=np.linalg.norm(rd-desc,axis=1).astype(np.float32)
    if any(not np.isfinite(x).all() for x in arrays.values()):raise ValueError("nonfinite saved frame")
    np.savez_compressed(job["output"],**arrays)
    savejson(job["receipt"],dict(signature=job["signature"],npz_sha256=sha(job["output"]),
        geometry_only=bool(job.get("geometry_only")),charge_identity_sha256=identity,heavy_coordinate_max_error_A=heavy_error,
        valid_anchors=int(valid.sum()),total_anchors=len(valid),mesh_vertices=len(regular.vertices),
        flat_curvature_vertices=flat_count,repeat_descriptor_max_error=repeat_error,
        rigid_mesh_median_descriptor_distance=rigid_error,timings=timings,wall_s=time.time()-started,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        coordinate_algorithm="upstream fast MDS at selected centers; per-center deterministic seed"))


def deferred_electrostatics(job):
    """Evaluate charges on the retained mesh; never remesh or reprotonate."""
    from masif_modules.read_data_from_surface import normalize_electrostatics
    started=time.time()
    with np.load(job["mesh"]) as z:vertices=z["electrostatics_vertices" if "electrostatics_vertices" in z else "vertices"].copy()
    t=time.time();potentials,identity=electrostatics(vertices,Path(job["protonated"]))
    apbs_s=time.time()-t
    values=np.asarray(normalize_electrostatics(potentials/10),np.float32)
    np.savez_compressed(job["output"],electrostatics=values)
    savejson(job["receipt"],dict(signature=job["signature"],npz_sha256=sha(job["output"]),
        charge_identity_sha256=identity,timings=dict(electrostatics_s=apbs_s),wall_s=time.time()-started))


if __name__=="__main__":
    job=json.loads(Path(sys.argv[1]).read_text())
    np.random.seed(job.get("seed",0))
    if job["command"]=="preflight":preflight(job)
    elif job["command"]=="frame":frame(job)
    elif job["command"]=="electrostatics":deferred_electrostatics(job)
    else:raise ValueError("unknown bridge command")
