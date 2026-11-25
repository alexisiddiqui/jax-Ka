"""Occupied-slot rank inversion shared by CPU spatial queries."""
import jax.numpy as jnp

def _compact_slots(count,rank):
    """Invert offset-major occupied-slot ranks (including runs of empty cells)."""
    end=jnp.cumsum(count,axis=1); lo=jnp.zeros(rank.shape,jnp.int32); hi=jnp.full(rank.shape,count.shape[1],jnp.int32)
    for _ in range(count.shape[1].bit_length()):
        mid=(lo+hi)//2
        edge=jnp.take_along_axis(end,jnp.minimum(mid,count.shape[1]-1),axis=1)
        right=(mid<count.shape[1]) & (rank>=edge)
        lo=jnp.where(right,mid+1,lo); hi=jnp.where(right,hi,mid)
    off=jnp.minimum(lo,count.shape[1]-1); prior=jnp.take_along_axis(end-count,off,axis=1)
    return off,rank-prior,rank<end[:,-1:]
