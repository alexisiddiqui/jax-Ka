"""Shared two-branch solve, independent of the source of LocalTerms."""
from functools import partial
import jax
import jax.numpy as jnp
from jaxpropka.optx_solver import local_terms_curve_kernel


@partial(jax.jit, static_argnames=('config','solver_config','initialization','seed_steps','seed_dtype'))
def solve_branches(inputs, terms, ph, *, config, solver_config,
                   initialization='production', gradient_mask=None, seed_steps=None, seed_dtype=None):
    if gradient_mask is None:
        gradient_mask = jnp.ones((2,len(ph)),bool)
    def one(arrays, branch_terms, active, valid, mask):
        return local_terms_curve_kernel(arrays,branch_terms,ph,active,valid,
            config=config,solver_config=solver_config,initialization=initialization,
            gradient_mask=mask,seed_steps=seed_steps,seed_dtype=seed_dtype)
    return jax.vmap(one)(inputs['arrays'],terms,inputs['active'],inputs['active_valid'],gradient_mask)
