"""Stack distinct records after rebuilding their paired layouts to common capacities."""
import numpy as np
import jax
from jaxpropka.batching import _ENV_FIELDS,_PAIR_FIELDS
from .records import load

def pad_prepared(inputs,real_n,capacities,dtype):
    """Pad the already validated AB/free union; never rebuild its geometry."""
    n,ke,kc,m=capacities;arrays={}
    for name,value in inputs['arrays'].items():
        shape=list(value.shape);shape[1]=n
        if name in _ENV_FIELDS:shape[2]=ke
        elif name in _PAIR_FIELDS:shape[2]=kc
        if any(a>b for a,b in zip(value.shape,shape)):raise ValueError('Batch capacity would truncate prepared inputs')
        target=np.zeros(shape,dtype=dtype if np.issubdtype(value.dtype,np.floating) else value.dtype)
        target[tuple(slice(0,s) for s in value.shape)]=value;arrays[name]=target
    p=np.zeros((2,n,20),dtype);p[:,:,0]=1;p[:,:inputs['probabilities'].shape[1]]=inputs['probabilities']
    active=[];valid=[]
    for indices,mask in zip(inputs['active'],inputs['active_valid']):
        actual=indices[mask];count=m-len(actual)
        if count<0 or real_n*9+count>n*9:raise ValueError('Insufficient distinct dummy channels')
        active.append(np.concatenate([actual,np.arange(real_n*9,real_n*9+count,dtype=indices.dtype)]))
        valid.append(np.arange(m)<len(actual))
    return dict(arrays=arrays,probabilities=p,active=np.stack(active),active_valid=np.stack(valid))

def pack_records(out,ids,dtype=np.float64,capacities=None):
    loaded=[load(out,cid) for cid in ids]
    m=max(r[3]['layout']['M'] for r in loaded)
    needed=max(r[3]['layout']['real_residues']+(m-int(r[0]['active_valid'].sum(axis=1).min())+8)//9 for r in loaded)
    n=max(max(r[3]['layout']['N'] for r in loaded),((needed+63)//64)*64)
    ke=max(r[3]['layout']['Ke'] for r in loaded);kc=max(r[3]['layout']['Kc'] for r in loaded)
    if capacities is not None:n,ke,kc,m=capacities
    inputs=[];refs=[];masks=[]
    for cid,(prepared,ref,eligible,receipt) in zip(ids,loaded):
        d=pad_prepared(prepared,receipt['layout']['real_residues'],(n,ke,kc,m),dtype)
        inputs.append(d)
        padded=np.zeros((2,73,n,9),dtype);padded[:,:,:ref.shape[2]]=ref;refs.append(padded)
        mask=np.zeros((n,9),bool);mask[:eligible.shape[0]]=eligible;masks.append(mask)
    return jax.tree.map(lambda *x:np.stack(x),*inputs),np.stack(refs),np.stack(masks),dict(N=n,Ke=ke,Kc=kc,M=m)
