"""Sparse geometric kernels, built with NumPy/SciPy outside JIT.

Atom-pair intermediates exist only for one residue edge at a time; no global
N*N*atoms*atoms tensors. Graphs are unions over every candidate identity and
are never silently truncated to a user-specified K.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial import cKDTree
from .cache import StructureCache
from .parameters import (ALPHABET, CLASSES, AA_CLASSES, NEUTRAL_AA, AA_TO_INDEX, BB_RANGES,
                         hbond_range)


def radial_summary(centers, xyz, volumes):
    """PROPKA 3.0 atomistic radial volume and 15-A heavy-atom count for one source."""
    if len(xyz)==0:
        return np.zeros(len(centers)),np.zeros(len(centers))
    r2=np.sum((centers[:,None,:]-xyz[None,:,:])**2,axis=-1)
    volume=np.sum(np.where(r2<400.,volumes[None,:]/np.maximum(2.75**4,r2*r2),0),axis=-1)
    mass=np.sum(r2<225.,axis=-1)
    return volume,mass


def hbond_strength(donor, acceptor, ranges, amplitude=.85):
    """Strongest directional donor--acceptor atom pair for one group pair.

    An O/S/Lys donor uses its heavy-atom coordinate and no angular term, while
    His/Arg/amide/Trp/backbone NH use virtual H and a positive cosine factor.
    The virtual-H construction and protonation state energies are approximations.
    """
    if ranges is None or len(donor.point)==0 or len(acceptor.acceptor)==0:
        return 0.
    delta=acceptor.acceptor[None,:,:]-donor.point[:,None,:]
    distance=np.linalg.norm(delta,axis=-1)
    direction=delta/np.maximum(distance[:,:,None],1e-8)
    cosine=np.sum(direction*donor.axis[:,None,:],axis=-1)
    factor=np.where(donor.angular[:,None],np.maximum(cosine,0),1)
    lo,hi=ranges
    if hi<=lo:
        return 0.
    return float(amplitude*np.max(np.clip((hi-distance)/(hi-lo),0,1)*factor))


def radius_graph(anchors,reach,cutoff,*,include_self=False,max_neighbors=None):
    anchors=np.asarray(anchors,float);reach=np.asarray(reach,float)
    n=len(anchors)
    if anchors.shape!=(n,3) or reach.shape!=(n,) or not np.isfinite(anchors).all() or np.any(reach<0):
        raise ValueError("invalid anchors/reaches")
    tree=cKDTree(anchors);rows=[]
    for i in range(n):
        candidate=tree.query_ball_point(anchors[i],cutoff+reach[i]+float(reach.max()))
        row=sorted(j for j in candidate if (include_self or j!=i)
                   and np.linalg.norm(anchors[i]-anchors[j])<=cutoff+reach[i]+reach[j]+1e-7)
        rows.append(row)
    needed=max(1,max(map(len,rows)))
    if max_neighbors is not None:
        if not isinstance(max_neighbors,int) or max_neighbors<needed:
            raise ValueError(f"neighbor overflow: need K>={needed}, requested {max_neighbors}; truncation is forbidden")
        capacity=max_neighbors
    else:
        capacity=needed
    indices=np.zeros((n,capacity),np.int32);mask=np.zeros((n,capacity),bool)
    for i,row in enumerate(rows):
        indices[i,:len(row)]=row;mask[i,:len(row)]=True
    return indices,mask


def native_identities(topology):
    """Allowed-identity mask for native-sequence-only evaluation."""
    return np.eye(20,dtype=bool)[np.asarray(topology.native_index)]


def build_cache(topology, candidates, *, max_env_neighbors=None, max_pair_neighbors=None,
                dtype=np.float32, identities=None):
    """Structural cache. ``identities``: optional bool [N,20] allowed-identity mask.

    Environmental terms are computed and STORED only for allowed neighbor
    identities (compact [N,Ke,9,A] tensors with ``identity_columns``), so the
    cache is valid only for sequences P with no mass elsewhere (checked by
    TitrationModel). Graphs and
    pair kernels are identity-independent and unchanged. ``native_identities``
    gives the benchmark case, removing ~19/20 of the radial and neutral H-bond work.
    """
    start=time.perf_counter()
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype("float32"),np.dtype("float64")):
        raise ValueError("kernel dtype must be float32 or float64")
    lib=candidates;n=topology.n_residues
    if identities is None:
        identity_mask=np.ones((n,20),bool)
    else:
        identity_mask=np.asarray(identities)
        if identity_mask.shape!=(n,20) or identity_mask.dtype!=bool or not identity_mask.any(-1).all():
            raise ValueError("identities must be a bool [N,20] mask with one allowed identity per row")
        frozen=np.asarray(topology.disulfide)
        if not identity_mask[frozen,np.asarray(topology.native_index)[frozen]].all():
            raise ValueError("fixed-covalent residues must allow their native identity")
    # column_of[j,a]: env column holding identity a of source residue j (-1 = not stored).
    allowed_count=identity_mask.sum(-1)
    width=20 if identities is None else int(allowed_count.max())
    columns=np.zeros((n,width),np.int32);column_of=np.full((n,20),-1,np.int64)
    for j in range(n):
        ids=np.flatnonzero(identity_mask[j]) if identities is not None else np.arange(20)
        columns[j,:len(ids)]=ids;columns[j,len(ids):]=ids[0];column_of[j,ids]=np.arange(len(ids))
    reach=np.zeros(n)
    for i in range(n):
        points=[lib.centers[i],lib.backbone_xyz[i]]
        for candidate in lib.residues[i]:
            points.extend((candidate.side_xyz,candidate.polar.point,candidate.polar.acceptor))
        points=np.concatenate(points)
        reach[i]=np.linalg.norm(points-lib.anchors[i],axis=-1).max()+1e-5
    env_idx,env_mask=radius_graph(lib.anchors,reach,20,max_neighbors=max_env_neighbors)
    idx,row_mask=radius_graph(lib.anchors,reach,10,include_self=True,max_neighbors=max_pair_neighbors)
    ke=env_idx.shape[1];kc=idx.shape[1]
    volume=np.zeros((n,ke,9,width),dtype);mass=np.zeros_like(volume);hb=np.zeros_like(volume)
    bbv=np.zeros((n,9),dtype);bbm=np.zeros_like(bbv);bbhb=np.zeros_like(bbv);reorg=np.zeros_like(bbv)
    localhb=np.zeros((n,9,20),dtype)
    pro=AA_TO_INDEX["P"]
    for i in range(n):
        # Same-residue backbone H bonds: sidechain identity conditions out Pro;
        # for a terminal site the backbone donor depends on P[i,Pro].
        for g in range(9):
            if not lib.group_mask[i,g]:
                continue
            bb_self_donor=hbond_strength(lib.polar[i][g],lib.backbone_co[i],BB_RANGES[g])
            bb_self_accept=hbond_strength(lib.backbone_nh[i],lib.polar[i][g],BB_RANGES[g])
            bbhb[i,g]+=bb_self_donor
            if g<7:
                bbhb[i,g]-=bb_self_accept
            else:
                localhb[i,g,:]-=bb_self_accept;localhb[i,g,pro]+=bb_self_accept
        for k,j in enumerate(env_idx[i]):
            if not env_mask[i,k]:
                continue
            v,m=radial_summary(lib.centers[i],lib.backbone_xyz[j],lib.backbone_volume[j])
            bbv[i]+=v;bbm[i]+=m
            for a,candidate in enumerate(lib.residues[j]):
                if not identity_mask[j,a]:
                    continue
                v,m=radial_summary(lib.centers[i],candidate.side_xyz,candidate.side_volume)
                volume[i,k,:,column_of[j,a]]=v;mass[i,k,:,column_of[j,a]]=m
            for g in range(9):
                if not lib.group_mask[i,g]:
                    continue
                target=lib.polar[i][g]
                bbhb[i,g]+=hbond_strength(target,lib.backbone_co[j],BB_RANGES[g])
                bbn=hbond_strength(lib.backbone_nh[j],target,BB_RANGES[g])
                hb[i,k,g,:]-=bbn
                if column_of[j,pro]>=0:
                    hb[i,k,g,column_of[j,pro]]+=bbn
                for a in NEUTRAL_AA:
                    if not identity_mask[j,a]:
                        continue
                    neutral=lib.residues[j][a].polar
                    ranges=hbond_range(CLASSES[g],AA_CLASSES[a])
                    hb[i,k,g,column_of[j,a]]+=hbond_strength(target,neutral,ranges)-hbond_strength(neutral,target,ranges)
        # Source PROPKA 3.0 reorganization: a geometric factor, scaled by burial later.
        for j in [i]+env_idx[i,env_mask[i]].tolist():
            c,o=topology.backbone[j,2:4]
            axis=(o-c)/max(np.linalg.norm(o-c),1e-8)
            for g in (0,1):
                delta=lib.centers[i,g]-o;distance=np.linalg.norm(delta)
                cosine=np.dot(delta,axis)/max(distance,1e-8)
                if distance<6 and cosine>.001:
                    reorg[i,g]+=.8*min(1.,(6-distance)/3.)
    pair_mask=np.zeros((n,kc,9,9),bool)
    cg=np.zeros((n,kc,9,9),dtype);hd=np.zeros_like(cg);hr=np.zeros_like(cg)
    edge_lookup={(i,int(j)):k for i in range(n) for k,j in enumerate(idx[i]) if row_mask[i,k]}
    for (i,j),k in edge_lookup.items():
        if j<i:
            continue
        kr=edge_lookup[j,i]
        allowed=lib.group_mask[i,:,None]&lib.group_mask[j,None,:]
        if i==j:
            allowed &= ~np.eye(9,dtype=bool);allowed[:7,:7]=False
        distance=np.linalg.norm(lib.centers[i,:,None,:]-lib.centers[j,None,:,:],axis=-1)
        r=np.maximum(distance,4.)
        block=np.where(allowed,244.12/r*np.clip((10-r)/6,0,1),0)
        forward=np.zeros((9,9));reverse=np.zeros((9,9))
        for g,t in zip(*np.nonzero(allowed)):
            ranges=hbond_range(CLASSES[g],CLASSES[t])
            forward[g,t]=hbond_strength(lib.polar[i][g],lib.polar[j][t],ranges)
            reverse[g,t]=hbond_strength(lib.polar[j][t],lib.polar[i][g],ranges)
        # Numerical reciprocity is exact, including different type centers.
        active=allowed & ((block>0)|(forward>0)|(reverse>0))
        pair_mask[i,k]=active;cg[i,k]=block;hd[i,k]=forward;hr[i,k]=reverse
        pair_mask[j,kr]=active.T;cg[j,kr]=block.T;hd[j,kr]=reverse.T;hr[j,kr]=forward.T
    metadata={**topology.metadata,**lib.metadata,"kernel_schema":1,"parameter_family":"PROPKA 3.0 Nov30",
              "coupling":"fractional mean field with binary donor/acceptor state energies",
              "environment_cutoff_angstrom":20.,"burial_cutoff_angstrom":15.,
              "coulomb_cutoff_angstrom":[4.,10.],"env_K":ke,"pair_K":kc,
              "precompute_seconds":time.perf_counter()-start}
    if identities is not None:
        # Backbone-NH terms fill every column; padded columns must hold exact zeros.
        hb*=(np.arange(width)[None,:]<allowed_count[:,None])[env_idx][:,:,None,:]
        # Only added when restricted, so unrestricted cache fingerprints are unchanged.
        metadata["allowed_identities"]=["".join(ALPHABET[a] for a in np.flatnonzero(row)) for row in identity_mask]
    return StructureCache(keys=topology.keys,chain_ids=topology.chain_ids,
                          native_index=topology.native_index,chain_index=topology.chain_index,
                          group_mask=lib.group_mask,frozen=topology.disulfide,
                          env_neighbors=env_idx,env_mask=env_mask,volume=volume,mass=mass,hbond=hb,
                          local_hbond=localhb,bb_volume=bbv,bb_mass=bbm,bb_hbond=bbhb,reorganization=reorg,
                          neighbors=idx,pair_mask=pair_mask,coulomb_geometry=cg,hb_donor=hd,hb_reverse=hr,
                          metadata=metadata,
                          identity_columns=None if identities is None else columns).validate()
