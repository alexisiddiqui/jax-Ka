"""Three-scale JAX-Ka adapter; no solver or loss in this module."""
import jax.numpy as jnp
from jaxpropka.model import PhysicalScales, _local_terms

SCALE_NAMES = ('desolvation', 'hydrogen_bond_and_reorganization', 'coulomb')

def physical_scales(theta):
    value = jnp.exp(jnp.log(4.) * jnp.tanh(theta))
    return PhysicalScales(*value)

def local_terms(theta, arrays, probabilities, config):
    return _local_terms(arrays, probabilities, config, physical_scales(theta))
