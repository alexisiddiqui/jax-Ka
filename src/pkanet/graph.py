"""Raw backbone radius graphs with invariant local-frame edge features."""
import numpy as np
from scipy.spatial import cKDTree


def geometry(backbone, chain, radius=20.):
    ca=backbone[:,1]; x=backbone[:,2]-ca
    y=backbone[:,0]-ca
    xnorm=np.linalg.norm(x,axis=-1,keepdims=True); x=x/np.maximum(xnorm,1e-6)
    y=y-(y*x).sum(-1,keepdims=True)*x
    ynorm=np.linalg.norm(y,axis=-1,keepdims=True); y=y/np.maximum(ynorm,1e-6)
    valid=(xnorm[:,0]>1e-5)&(ynorm[:,0]>1e-5)
    frames=np.stack((x,y,np.cross(x,y)),axis=-1)
    rows=cKDTree(ca).query_ball_point(ca,radius)
    k=max(map(len,rows)); neighbors=np.zeros((len(ca),k),np.int32); mask=np.zeros((len(ca),k),bool)
    for i,row in enumerate(rows):
        row=sorted(row); neighbors[i,:len(row)]=row; mask[i,:len(row)]=True
    delta=ca[neighbors]-ca[:,None]; distance=np.sqrt((delta**2).sum(-1)+1e-8)
    direction=np.einsum('nkj,njl->nkl',delta,frames)/distance[...,None]
    direction*=valid[:,None,None]
    rbf=np.exp(-((distance[...,None]-np.linspace(0,radius,16))/1.5)**2)
    edge=np.concatenate((rbf,direction,(chain[neighbors]==chain[:,None])[...,None]),axis=-1)
    # Smooth attention weight goes to zero at the radius boundary.
    switch=np.where(distance<radius-2,1.,.5*(1+np.cos(np.pi*np.clip((distance-radius+2)/2,0,1))))
    return dict(neighbors=neighbors,edge=edge.astype(np.float32),edge_mask=mask,
                switch=switch.astype(np.float32)),valid
