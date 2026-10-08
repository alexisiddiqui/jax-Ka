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


def build_cache_reference(topology, candidates, *, max_env_neighbors=None, max_pair_neighbors=None,
                          dtype=np.float32, identities=None):
    """Loop implementation, retained verbatim as the equivalence oracle for build_cache.

    Structural cache. ``identities``: optional bool [N,20] allowed-identity mask.

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


# ---------------------------------------------------------------------------
# Vectorized implementation, bitwise-equivalent to build_cache_reference:
# per-element arithmetic is evaluated with the same expressions;
# reductions keep the reference lengths (radial sums are grouped by atom count,
# so NumPy's pairwise summation is identical); accumulated arrays are updated
# one neighbor slot at a time in the reference order and dtype.
# ---------------------------------------------------------------------------

def _pad_polar(sets, d, a):
    """Stack PolarGeometry objects into padded arrays (donor point/axis/angular/mask, acceptor/mask)."""
    m=len(sets)
    pt=np.zeros((m,d,3));ax=np.zeros((m,d,3));ang=np.zeros((m,d),bool);dm=np.zeros((m,d),bool)
    acc=np.zeros((m,a,3));am=np.zeros((m,a),bool)
    for s,p in enumerate(sets):
        nd=len(p.point);na=len(p.acceptor)
        pt[s,:nd]=p.point;ax[s,:nd]=p.axis;ang[s,:nd]=p.angular;dm[s,:nd]=True
        acc[s,:na]=p.acceptor;am[s,:na]=True
    return pt,ax,ang,dm,acc,am


def _hbond_batch(pt, ax, ang, dm, acc, am, lo, hi, amplitude=.85):
    """Batched hbond_strength: donors [...,D,3], acceptors [...,A,3], ranges lo/hi [...] (nan = none)."""
    delta=acc[...,None,:,:]-pt[...,:,None,:]
    distance=np.linalg.norm(delta,axis=-1)
    direction=delta/np.maximum(distance[...,None],1e-8)
    cosine=np.sum(direction*ax[...,:,None,:],axis=-1)
    factor=np.where(ang[...,:,None],np.maximum(cosine,0),1)
    ok=np.isfinite(lo)&(hi>lo)
    lo_=np.where(ok,lo,0.)[...,None,None];hi_=np.where(ok,hi,1.)[...,None,None]
    val=np.where(dm[...,:,None]&am[...,None,:],np.clip((hi_-distance)/(hi_-lo_),0,1)*factor,0.)
    best=val.max(axis=(-2,-1)) if val.shape[-1] and val.shape[-2] else np.zeros(val.shape[:-2])
    return np.where(ok&dm.any(-1)&am.any(-1),amplitude*best,0.)


def _ranges(rows, cols):
    """lo/hi arrays [len(rows),len(cols)] from hbond_range; nan where no interaction."""
    lo=np.full((len(rows),len(cols)),np.nan);hi=np.full_like(lo,np.nan)
    for r,x in enumerate(rows):
        for c,y in enumerate(cols):
            v=hbond_range(x,y) if x is not None and y is not None else None
            if v is not None:
                lo[r,c],hi[r,c]=v
    return lo,hi


def _chunks(n, size):
    for start in range(0,n,size):
        yield slice(start,min(n,start+size))


def build_cache(topology, candidates, *, max_env_neighbors=None, max_pair_neighbors=None,
                dtype=np.float32, identities=None, chunk=4096):
    """Structural cache (vectorized). ``identities``: optional bool [N,20] allowed-identity mask.

    Environmental terms are computed and STORED only for allowed neighbor
    identities (compact [N,Ke,9,A] tensors with ``identity_columns``), so the
    cache is valid only for sequences P with no mass elsewhere (checked by
    TitrationModel). Graphs and pair kernels are identity-independent.
    ``native_identities`` gives the benchmark case.

    Equivalent to build_cache_reference: every array is bitwise identical (see
    tests/test_cache_vectorized.py). ``chunk`` bounds the edge batch size and so
    the transient memory; it does not affect results.
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
    pro=AA_TO_INDEX["P"];gm=np.asarray(lib.group_mask,bool);centers=np.asarray(lib.centers,float)
    # Same-residue backbone H bonds: O(9N) scalar calls, kept identical to the reference.
    for i in range(n):
        for g in range(9):
            if not gm[i,g]:
                continue
            bb_self_donor=hbond_strength(lib.polar[i][g],lib.backbone_co[i],BB_RANGES[g])
            bb_self_accept=hbond_strength(lib.backbone_nh[i],lib.polar[i][g],BB_RANGES[g])
            bbhb[i,g]+=bb_self_donor
            if g<7:
                bbhb[i,g]-=bb_self_accept
            else:
                localhb[i,g,:]-=bb_self_accept;localhb[i,g,pro]+=bb_self_accept
    # Padded polar geometry: targets [N,9], backbone NH/CO [N], neutral candidates [N,5].
    neutral_sets=[lib.residues[j][a].polar for j in range(n) for a in NEUTRAL_AA]
    target_sets=[lib.polar[i][g] for i in range(n) for g in range(9)]
    every=target_sets+list(lib.backbone_nh)+list(lib.backbone_co)+neutral_sets
    dmax=max(1,max(len(p.point) for p in every));amax=max(1,max(len(p.acceptor) for p in every))
    tgt=[x.reshape((n,9)+x.shape[1:]) for x in _pad_polar(target_sets,dmax,amax)]
    nhp=_pad_polar(list(lib.backbone_nh),dmax,amax);cop=_pad_polar(list(lib.backbone_co),dmax,amax)
    neu=[x.reshape((n,len(NEUTRAL_AA))+x.shape[1:]) for x in _pad_polar(neutral_sets,dmax,amax)]
    bb_lo,bb_hi=BB_RANGES[:,0].astype(float),BB_RANGES[:,1].astype(float)
    neu_lo,neu_hi=_ranges(CLASSES,[AA_CLASSES[a] for a in NEUTRAL_AA])      # [9,5]
    pair_lo,pair_hi=_ranges(CLASSES,CLASSES)                                 # [9,9]
    # Flattened environment edges in reference order (i ascending, slot ascending).
    ei,ek=np.nonzero(env_mask);ej=env_idx[ei,ek];edges=len(ei)
    # Backbone radial terms, grouped by backbone atom count (reduction lengths as reference).
    nbb=np.array([len(x) for x in lib.backbone_xyz]);bbmax=int(nbb.max())
    bbxyz=np.zeros((n,bbmax,3));bbvol=np.zeros((n,bbmax))
    for j in range(n):
        bbxyz[j,:nbb[j]]=lib.backbone_xyz[j];bbvol[j,:nbb[j]]=lib.backbone_volume[j]
    slot_v=np.zeros((n,ke,9));slot_m=np.zeros((n,ke,9),np.int64)
    for count in np.unique(nbb):
        if count==0:
            continue
        sel=np.flatnonzero(nbb[ej]==count)
        for part in _chunks(len(sel),chunk):
            e=sel[part];c=centers[ei[e]];x=bbxyz[ej[e],:count];w=bbvol[ej[e],:count]
            r2=np.sum((c[:,:,None,:]-x[:,None,:,:])**2,axis=-1)
            slot_v[ei[e],ek[e]]=np.sum(np.where(r2<400.,w[:,None,:]/np.maximum(2.75**4,r2*r2),0),axis=-1)
            slot_m[ei[e],ek[e]]=np.sum(r2<225.,axis=-1)
    # Sidechain radial terms for allowed (edge, identity) pairs, grouped by atom count.
    nside=np.array([[len(lib.residues[j][a].side_xyz) for a in range(20)] for j in range(n)])
    smax=max(1,int(nside.max()))
    sxyz=np.zeros((n,20,smax,3));svol=np.zeros((n,20,smax))
    for j in range(n):
        for a in range(20):
            if identity_mask[j,a] and nside[j,a]:
                sxyz[j,a,:nside[j,a]]=lib.residues[j][a].side_xyz;svol[j,a,:nside[j,a]]=lib.residues[j][a].side_volume
    pe,pa=np.nonzero(identity_mask[ej])
    pcount=nside[ej[pe],pa]
    for count in np.unique(pcount):
        if count==0:
            continue
        sel=np.flatnonzero(pcount==count)
        for part in _chunks(len(sel),chunk):
            q=sel[part];e=pe[q];a=pa[q];j=ej[e]
            c=centers[ei[e]];x=sxyz[j,a,:count];w=svol[j,a,:count]
            r2=np.sum((c[:,:,None,:]-x[:,None,:,:])**2,axis=-1)
            col=column_of[j,a]
            volume[ei[e],ek[e],:,col]=np.sum(np.where(r2<400.,w[:,None,:]/np.maximum(2.75**4,r2*r2),0),axis=-1)
            mass[ei[e],ek[e],:,col]=np.sum(r2<225.,axis=-1)
    # Edge H bonds: target to backbone CO/NH and to neutral candidates.
    slot_co=np.zeros((n,ke,9))
    zero=np.zeros((),dtype)
    for part in _chunks(edges,chunk):
        i=ei[part];k=ek[part];j=ej[part];m=len(i)
        t=[x[i] for x in tgt]                                         # [m,9,...]
        co=[x[j][:,None] for x in cop];nh=[x[j][:,None] for x in nhp] # broadcast over g
        co_val=_hbond_batch(t[0],t[1],t[2],t[3],co[4],co[5],bb_lo,bb_hi)
        bbn=_hbond_batch(nh[0],nh[1],nh[2],nh[3],t[4],t[5],bb_lo,bb_hi)
        slot_co[i,k]=co_val
        # Reference arithmetic in the stored dtype: 0 - bbn, then (+bbn) for Pro, (+diff) for neutrals.
        row=np.broadcast_to((zero-bbn.astype(dtype))[:,:,None],(m,9,width)).copy()
        pcol=column_of[j,pro];has=pcol>=0
        if has.any():
            r=np.flatnonzero(has)
            row[r,:,pcol[r]]=row[r,:,pcol[r]]+bbn[r].astype(dtype)
        for q,a in enumerate(NEUTRAL_AA):
            ok=identity_mask[j,a]
            if not ok.any():
                continue
            r=np.flatnonzero(ok);nq=[x[j[r],q][:,None] for x in neu];tr=[x[r] for x in t]
            fwd=_hbond_batch(tr[0],tr[1],tr[2],tr[3],nq[4],nq[5],neu_lo[:,q],neu_hi[:,q])
            rev=_hbond_batch(nq[0],nq[1],nq[2],nq[3],tr[4],tr[5],neu_lo[:,q],neu_hi[:,q])
            col=column_of[j[r],a]
            row[r,:,col]=row[r,:,col]+(fwd-rev).astype(dtype)
        hb[i,k]=np.where(gm[i][:,:,None],row,0)
    # Ordered accumulation per neighbor slot, matching the reference's in-place updates.
    for k in range(ke):
        on=env_mask[:,k]
        if not on.any():
            continue
        bbv[on]=bbv[on]+slot_v[on,k];bbm[on]=bbm[on]+slot_m[on,k]
        sel=on[:,None]&gm
        bbhb[:]=np.where(sel,bbhb+slot_co[:,k].astype(dtype),bbhb)
    # Reorganization (PROPKA 3.0): own backbone first, then neighbors in slot order.
    # A vectorized screen selects candidate (i, j, g) within a small margin of the
    # cutoffs; candidates are then evaluated with the reference's scalar code
    # (1-D BLAS norm/dot and its float32/float64 update semantics), so the result
    # is bitwise identical. Candidates are few (backbone O within ~6 A).
    co_xyz=np.asarray(topology.backbone[:,3],float);c_xyz=np.asarray(topology.backbone[:,2],float)
    axis_ref=[None]*n
    def axis_of(j):
        if axis_ref[j] is None:
            c,o=topology.backbone[j,2:4]
            axis_ref[j]=(o-c)/max(np.linalg.norm(o-c),1e-8)
        return axis_ref[j]
    bond=co_xyz-c_xyz;axis_v=bond/np.maximum(np.linalg.norm(bond,axis=-1),1e-8)[:,None]
    def reorg_step(js,on):
        for g in (0,1):
            delta=centers[:,g]-co_xyz[js];distance=np.linalg.norm(delta,axis=-1)
            cosine=np.sum(delta*axis_v[js],axis=-1)/np.maximum(distance,1e-8)
            for i in np.flatnonzero(on&(distance<6+1e-6)&(cosine>.001-1e-6)):
                j=int(js[i]);o=topology.backbone[j,3]
                d=lib.centers[i,g]-o;dist=np.linalg.norm(d)
                cos=np.dot(d,axis_of(j))/max(dist,1e-8)
                if dist<6 and cos>.001:
                    reorg[i,g]+=.8*min(1.,(6-dist)/3.)
    reorg_step(np.arange(n),np.ones(n,bool))
    for k in range(ke):
        if env_mask[:,k].any():
            reorg_step(env_idx[:,k],env_mask[:,k])
    # Pair kernels for j >= i, then reciprocal (transposed) slots, as in the reference.
    pair_mask=np.zeros((n,kc,9,9),bool)
    cg=np.zeros((n,kc,9,9),dtype);hd=np.zeros_like(cg);hr=np.zeros_like(cg)
    edge_lookup={(i,int(j)):k for i in range(n) for k,j in enumerate(idx[i]) if row_mask[i,k]}
    pairs=np.array([(i,j,k,edge_lookup[j,i]) for (i,j),k in edge_lookup.items() if j>=i],np.int64).reshape(-1,4)
    eye=np.eye(9,dtype=bool);selfmask=~eye;selfmask[:7,:7]=False
    results=[]
    for part in _chunks(len(pairs),max(1,chunk//8)):
        pi,pj,pk,pr=pairs[part].T
        allowed=gm[pi][:,:,None]&gm[pj][:,None,:]
        same=pi==pj
        allowed[same]&=selfmask
        distance=np.linalg.norm(centers[pi][:,:,None,:]-centers[pj][:,None,:,:],axis=-1)
        r=np.maximum(distance,4.)
        block=np.where(allowed,244.12/r*np.clip((10-r)/6,0,1),0)
        ti=[x[pi] for x in tgt];tj=[x[pj] for x in tgt]
        gi=[x[:,:,None] for x in ti];gj=[x[:,None,:] for x in tj]   # donor/acceptor over (g,t)
        forward=np.where(allowed,_hbond_batch(gi[0],gi[1],gi[2],gi[3],gj[4],gj[5],pair_lo,pair_hi),0.)
        reverse=np.where(allowed,_hbond_batch(gj[0],gj[1],gj[2],gj[3],gi[4],gi[5],pair_lo,pair_hi),0.)
        active=allowed&((block>0)|(forward>0)|(reverse>0))
        pair_mask[pi,pk]=active;cg[pi,pk]=block;hd[pi,pk]=forward;hr[pi,pk]=reverse
        results.append((pj,pr,active,block,forward,reverse))
    for pj,pr,active,block,forward,reverse in results:
        tr=lambda x:np.swapaxes(x,-1,-2)
        pair_mask[pj,pr]=tr(active);cg[pj,pr]=tr(block);hd[pj,pr]=tr(reverse);hr[pj,pr]=tr(forward)
    metadata={**topology.metadata,**lib.metadata,"kernel_schema":1,"parameter_family":"PROPKA 3.0 Nov30",
              "coupling":"fractional mean field with binary donor/acceptor state energies",
              "environment_cutoff_angstrom":20.,"burial_cutoff_angstrom":15.,
              "coulomb_cutoff_angstrom":[4.,10.],"env_K":ke,"pair_K":kc,
              "precompute_seconds":time.perf_counter()-start}
    if identities is not None:
        hb*=(np.arange(width)[None,:]<allowed_count[:,None])[env_idx][:,:,None,:]
        metadata["allowed_identities"]=["".join(ALPHABET[a] for a in np.flatnonzero(row)) for row in identity_mask]
    return StructureCache(keys=topology.keys,chain_ids=topology.chain_ids,
                          native_index=topology.native_index,chain_index=topology.chain_index,
                          group_mask=lib.group_mask,frozen=topology.disulfide,
                          env_neighbors=env_idx,env_mask=env_mask,volume=volume,mass=mass,hbond=hb,
                          local_hbond=localhb,bb_volume=bbv,bb_mass=bbm,bb_hbond=bbhb,reorganization=reorg,
                          neighbors=idx,pair_mask=pair_mask,coulomb_geometry=cg,hb_donor=hd,hb_reverse=hr,
                          metadata=metadata,
                          identity_columns=None if identities is None else columns).validate()
