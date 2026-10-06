"""Curve objectives shared by all LocalTerms adapters; explicit availability."""
import numpy as np
import jax.numpy as jnp


def curve_loss(predicted, reference, eligible, accepted, paired_weight=1.):
    """Shapes: [2,H,N,G], [2,H,N,G], [N,G], [2,H].

    Normalize each objective by its originally eligible observations, not the
    changing solver-valid count. Coverage thresholds are checked outside AD.
    """
    mask = eligible[None,None] & accepted[:,:,None,None]
    delta = jnp.where(mask, predicted-reference, 0)
    denom = jnp.maximum(1, 2*predicted.shape[1]*jnp.sum(eligible))
    absolute = jnp.sum(delta**2)/denom
    pairmask = eligible[None] & (accepted[0]&accepted[1])[:,None,None]
    pair = jnp.where(pairmask, (predicted[0]-predicted[1])-(reference[0]-reference[1]), 0)
    paired = jnp.sum(pair**2)/jnp.maximum(1,predicted.shape[1]*jnp.sum(eligible))
    return absolute+paired_weight*paired, {'absolute':absolute,'paired':paired}


def coverage(eligible, accepted):
    eligible = np.asarray(eligible,bool); accepted = np.asarray(accepted,bool)
    count = int(eligible.sum())
    missing = int((~accepted).sum())*count
    total = accepted.size*count
    return {'missing':missing,'total':total,'fraction':missing/max(total,1)}
