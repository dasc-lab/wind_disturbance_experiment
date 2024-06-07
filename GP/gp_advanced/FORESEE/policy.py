import jax.numpy as jnp
from jax import jit

def policy(params, state, state_ref):
    m = 0.641 #kg
    g = 9.8066
    kx = params[0]
    kv = params[1]
    ex = state[0:3] - state_ref[0:3]
    ev = state[3:6] - state_ref[3:6]
    return -kx * ex -kv * ev + m*(acc_ref) - m*g 