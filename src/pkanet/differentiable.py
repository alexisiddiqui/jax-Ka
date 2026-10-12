"""Continuous oGQT features; discrete residue/site identities remain fixed."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from .graph import SIDECHAIN_ATOMS
from .features import featurise
from jaxpropka.topology import load_topology

ATOM_NAMES = ('N', 'CA', 'C', 'O') + SIDECHAIN_ATOMS


def _frames(backbone):
    ca = backbone[:, 1]
    def normalise(v):
        length = jnp.sqrt(jnp.maximum(jnp.sum(v*v, -1, keepdims=True), 1e-12))
        return v/length, length[:, 0] > 1e-5
    x, xv = normalise(backbone[:, 2]-ca)
    y = backbone[:, 0]-ca
    y, yv = normalise(y-jnp.sum(y*x, -1, keepdims=True)*x)
    return ca, jnp.stack((x,y,jnp.cross(x,y)), -1), xv & yv


def _switch(distance):
    return jnp.where(distance < 18., 1., .5*(1+jnp.cos(jnp.pi*jnp.clip((distance-18.)/2., 0., 1.))))


@dataclass(frozen=True)
class StructureFeatures:
    """Fixed discrete template, rebuilding every continuous feature per call.

    Dense candidate pairs avoid stale neighbor lists when coordinates move.
    Memory therefore scales quadratically; this is not the sparse inference CLI.
    Sequence values affect the 20 residue channels, not discrete site creation,
    query types, atom presence or topology. Reprepare after discrete mutations.
    """
    graph: dict
    chain: np.ndarray
    sequence_index: np.ndarray
    atom_present: np.ndarray
    sidechains: bool = False

    @classmethod
    def from_file(cls, path, *, sidechains=False, **options):
        options = {**dict(gap_policy='cap', ignore_nonprotein=False), **options}
        graph, metadata = featurise(path, **options)
        top = load_topology(path, freeze_disulfides=True, **options)
        n = top.n_residues
        present = np.zeros((n, len(SIDECHAIN_ATOMS)), bool)
        xyz = np.zeros((n, len(SIDECHAIN_ATOMS), 3), np.float32)
        for i in range(n):
            atoms = {str(a.atom_name): a.coord for a in top.residue(i)}
            for j, name in enumerate(SIDECHAIN_ATOMS):
                if name in atoms: present[i,j] = True; xyz[i,j] = atoms[name]
        seq_index = np.zeros(n, np.int32); counts = {}
        for i, ch in enumerate(top.chain_index):
            seq_index[i] = counts.get(int(ch), 0); counts[int(ch)] = seq_index[i]+1
        template = cls(graph, np.asarray(top.chain_index), seq_index, present, sidechains)
        inputs = (np.eye(20,dtype=np.float32)[top.native_index], np.asarray(top.backbone,dtype=np.float32), xyz)
        return template, inputs, metadata

    def __call__(self, sequence, backbone, sidechain):
        n = len(self.chain)
        if sequence.shape != (n,20) or backbone.shape != (n,4,3) or sidechain.shape != (n,len(SIDECHAIN_ATOMS),3):
            raise ValueError('Expected sequence [N,20], backbone [N,4,3], sidechain [N,32,3]')
        ca, frame, valid = _frames(backbone)
        chain = jnp.asarray(self.chain); pos = jnp.asarray(self.sequence_index)
        neighbors = jnp.broadcast_to(jnp.arange(n,dtype=jnp.int32), (n,n))
        delta = ca[None]-ca[:,None]
        distance = jnp.sqrt(jnp.sum(delta*delta,-1)+1e-8)
        rbf = jnp.exp(-((distance[...,None]-jnp.linspace(0.,20.,16,dtype=backbone.dtype))/1.5)**2)
        direction = jnp.einsum('nkj,njl->nkl',delta,frame)/distance[...,None]
        direction = direction*valid[:,None,None]
        edge = jnp.concatenate((rbf,direction,(chain[None]==chain[:,None])[...,None]),-1)
        nodes = jnp.concatenate((sequence,jnp.asarray(self.graph['nodes'][:,20:23]),valid[:,None]),-1)
        if self.sidechains:
            local = jnp.einsum('nkj,njl->nkl',sidechain-ca[:,None],frame)/10.
            local = jnp.where((jnp.asarray(self.atom_present)&valid[:,None])[...,None],local,0.)
            side = jnp.concatenate((local,jnp.asarray(self.atom_present)[...,None]),-1)
            nodes = jnp.concatenate((nodes,side.reshape(n,-1)),-1)
        graph = {k:jnp.asarray(v) for k,v in self.graph.items()}
        graph.update(nodes=nodes,neighbors=neighbors,edge=edge,edge_mask=distance<=20.,switch=_switch(distance))
        ri = graph['site_residue']; s = len(ri)
        sn = jnp.broadcast_to(jnp.arange(s,dtype=jnp.int32),(s,s)); rj = ri[sn]
        d = ca[rj]-ca[ri,None]; dist = jnp.sqrt(jnp.sum(d*d,-1)+1e-8)
        fi = frame[ri]; fj = frame[rj]
        di = jnp.einsum('skc,scl->skl',d,fi)/dist[...,None]*valid[ri,None,None]
        dj = jnp.einsum('skc,skcl->skl',-d,fj)/dist[...,None]*valid[rj][...,None]
        ori = jnp.matmul(jnp.swapaxes(fi[:,None],-1,-2),fj).reshape(s,s,9)
        ori = ori*(valid[ri,None]&valid[rj])[...,None]
        same = chain[ri,None]==chain[rj]
        sep = jnp.where(same,jnp.clip((pos[rj]-pos[ri,None]).astype(backbone.dtype)/32.,-1.,1.),0.)
        se = jnp.concatenate((jnp.exp(-((dist[...,None]-jnp.linspace(0.,20.,16,dtype=backbone.dtype))/1.5)**2),
            di,dj,ori,same[...,None],(ri[:,None]==rj)[...,None],sep[...,None]),-1)
        graph.update(site_neighbors=sn,site_edge=se,site_edge_mask=dist<=20.,site_switch=_switch(dist))
        return graph
