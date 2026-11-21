"""The validated finite MaSIF-search forward pass, shared by GPU workers."""
import numpy as np


def make_forward():
    import jax
    import jax.numpy as jnp
    steps=jnp.asarray([np.float32(k*2*np.pi/16) for k in range(16)])
    two_pi=jnp.float32(2*np.pi);eps=jnp.float32(1e-5)
    def forward(p,feat,r,t,m):
        outputs=[]
        for f in range(5):
            radial=jnp.exp(-(r[...,None]-p[f'mu_rho_{f}'])**2/(p[f'sigma_rho_{f}']**2+eps))
            def rotate(k,best):
                angle=jnp.mod(t+steps[k],two_pi)
                angular=jnp.exp(-(angle[...,None]-p[f'mu_theta_{f}'])**2/(p[f'sigma_theta_{f}']**2+eps))
                act=radial*angular*m[...,None];act=act/(jnp.sum(act,axis=1,keepdims=True)+eps)
                value=jnp.sum(act*feat[:,:,f,None],axis=1)
                value=jnp.matmul(value,p[f'W_conv_{f}'],precision=jax.lax.Precision.HIGHEST)+p[f'b_conv_{f}']
                return jnp.maximum(best,value)
            value=jax.lax.fori_loop(0,16,rotate,jnp.full((len(feat),80),-jnp.inf))
            outputs.append(jax.nn.relu(value))
        return jnp.matmul(jnp.concatenate(outputs,axis=-1),p['fully_connected/weights'],precision=jax.lax.Precision.HIGHEST)+p['fully_connected/biases']
    return jax.jit(forward)
