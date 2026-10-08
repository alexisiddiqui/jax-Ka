"""AB/free layouts assembled from independently prepared structural caches."""
from dataclasses import fields
import numpy as np
from jaxpropka.cache import StructureCache
from jaxpropka.batching import pack_inputs, _ENV_FIELDS, _PAIR_FIELDS
from jaxpropka.model import one_hot
from jaxpropka.optx_solver import active_channels


def free_union(ab, a, b):
    """Reindex a disjoint union without borrowing any AB environment summaries."""
    for cache in (ab, a, b): cache.validate()
    lookup = {key: i for i, key in enumerate(ab.keys)}
    if set(a.keys) & set(b.keys) or set(a.keys) | set(b.keys) != set(ab.keys):
        raise ValueError('A/B must partition AB residue identities exactly')
    if any((c.identity_columns is None) != (ab.identity_columns is None) for c in (a,b)):
        raise ValueError('Cache identity storage differs')
    ke = max(c.env_neighbors.shape[1] for c in (a,b))
    kc = max(c.neighbors.shape[1] for c in (a,b))
    values = {}
    for field in fields(StructureCache):
        name = field.name; value = getattr(ab,name)
        if isinstance(value,np.ndarray):
            shape = list(value.shape)
            if name in _ENV_FIELDS: shape[1] = ke
            elif name in _PAIR_FIELDS: shape[1] = kc
            values[name] = np.zeros(shape,dtype=value.dtype)
    for cache in (a,b):
        dest = np.asarray([lookup[key] for key in cache.keys])
        for name in values:
            if name == 'chain_index': continue
            source = getattr(cache,name)
            if name in ('neighbors','env_neighbors'): source = dest[source]
            slices = (dest,) + tuple(slice(0,n) for n in source.shape[1:])
            values[name][slices] = source
    values['chain_index'] = ab.chain_index.copy()
    union = StructureCache(keys=ab.keys,chain_ids=ab.chain_ids,metadata={'source':'independent A/B union'},
        **values,**({'identity_columns':None} if ab.identity_columns is None else {}))
    union.validate()
    for name in ('native_index','frozen','group_mask'):
        if not np.array_equal(getattr(union,name),getattr(ab,name)):
            raise ValueError(f'AB/free {name} differs; do not coerce channel semantics')
    return union


def paired_inputs(ab,a,b,capacities=None):
    free = free_union(ab,a,b); caches=(ab,free)
    p = one_hot(ab.native_index,np.float64)
    active = [active_channels(c,p) for c in caches]
    m = max(len(x) for x in active); m = max(32,((m+31)//32)*32)
    dummy_rows = (max(m-len(x) for x in active)+8)//9
    n = ((ab.n_residues+dummy_rows+63)//64)*64
    ke = ((max(c.env_neighbors.shape[1] for c in caches)+15)//16)*16
    kc = ((max(c.neighbors.shape[1] for c in caches)+15)//16)*16
    if capacities is not None:
        nn,ee,cc,mm=capacities
        if ee<ke or cc<kc or mm<m or nn<ab.n_residues+(max(mm-len(x) for x in active)+8)//9:
            raise ValueError('Batch capacities cannot hold this complex and distinct dummy channels')
        n,ke,kc,m=nn,ee,cc,mm
    packed = [pack_inputs(c,p,capacities=(n,ke,kc)) for c in caches]
    arrays = {k:np.stack([d[0][k] for d in packed]) for k in packed[0][0]}
    indices=[]; masks=[]
    for act in active:
        dummy=np.arange(ab.n_residues*9,ab.n_residues*9+m-len(act),dtype=np.int32)
        indices.append(np.concatenate([act,dummy])); masks.append(np.arange(m)<len(act))
    return {'arrays':arrays,'probabilities':np.stack([x[1] for x in packed]),
            'active':np.stack(indices),'active_valid':np.stack(masks)}, {'N':n,'Ke':ke,'Kc':kc,'M':m,'real_residues':ab.n_residues}
