"""Matched missing-context views; independent of labels and model RNG streams."""
import hashlib
import numpy as np


def residue_mask(n, protected, *, seed, epoch, complex_id, probability=.05):
    assert 0 <= probability < 1 and epoch >= 1
    token=f'context-v1:{seed}:{epoch}:{complex_id}'.encode()
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'))
    mask=rng.random(n)<probability
    mask[np.asarray(protected,dtype=np.int64)]=False
    return mask


def epoch_mask(plan, seed, epoch, probability=.05):
    result=np.zeros(plan['residues'],bool)
    for cid,r in plan['structures'].items():
        result[r['start']:r['stop']]=residue_mask(r['stop']-r['start'],r['protected'],
            seed=seed,epoch=epoch,complex_id=cid,probability=probability)
    return result


def mask_digest(mask):return hashlib.sha256(np.asarray(mask,dtype=np.uint8).tobytes()).hexdigest()


def mask_graph(graph, masked):
    """Remove incident geometry while retaining residue identity and node order."""
    assert len(masked)==len(graph['nodes'])
    assert not masked[graph['query_residue']].any(), 'Supervised centres must remain observed'
    affected=masked[:,None] | masked[graph['neighbors']]
    graph['edge'][affected,:19]=0.  # All radial and directional channels.
    graph['edge_mask'][affected]=False
    graph['switch'][affected]=0.
    graph['nodes'][masked,23]=0.  # Existing explicit frame/geometry-valid flag.
    # Optional columns are side-chain local coordinates and atom-presence bits.
    # They are geometric context and must not leak a masked residue.
    if graph['nodes'].shape[1]>24:graph['nodes'][masked,24:]=0.
    return graph


def encode_candidates(classes, values, owners, masked, residue_onehot):
    """Filter before selecting 250; source candidates must already be stably sorted."""
    keep=np.flatnonzero(~masked[owners])[:250]
    result=np.zeros(4008,np.float32)
    result[np.arange(len(keep))*16+classes[keep]]=values[keep]
    result[4000:]=residue_onehot
    return result


class PKAIContext:
    def __init__(self,path):
        from pathlib import Path
        import json
        from pkabench.runtime import digest
        path=Path(path);receipt=json.loads((path/'verification.json').read_text())
        assert receipt['passed']
        for name,sha in receipt['files'].items():assert digest(path/name)==sha
        self.plan=json.loads((path/'plan.json').read_text())
        self.offsets=np.load(path/'offsets.npy',mmap_mode='r')
        self.classes=np.load(path/'classes.npy',mmap_mode='r')
        self.values=np.load(path/'values.npy',mmap_mode='r')
        self.owners=np.load(path/'owners.npy',mmap_mode='r')
        self.available=np.load(path/'available.npy',mmap_mode='r')
        self.mask=None

    def set_epoch(self,seed,epoch,probability=.05):
        self.mask=epoch_mask(self.plan,seed,epoch,probability)
        return mask_digest(self.mask)

    def verify_reference(self,ids,features):
        """Audit packed global row offsets against unchanged dense native inputs."""
        self.mask=np.zeros(self.plan['residues'],bool)
        np.testing.assert_allclose(self.augment(ids,features),features,rtol=2e-6,atol=1e-8)
        self.mask=None

    def augment(self,ids,features):
        assert self.mask is not None and np.all(self.available[ids])
        result=np.empty_like(features)
        for j,i in enumerate(ids):
            start,stop=self.offsets[i:i+2]
            result[j]=encode_candidates(self.classes[start:stop],self.values[start:stop],
                self.owners[start:stop],self.mask,features[j,4000:])
        return result
