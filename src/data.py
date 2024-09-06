import jax.numpy as jnp
from jax import vmap
from jax.random import normal, randint, split

def get_signal(L):
    t = jnp.linspace(0, 2*jnp.pi, L+1)[:-1]
    x = jnp.exp(jnp.sin(t)) / 4
    return x, t


def get_samples(key, x, noise_std, N):
    L = len(x)
    shiftkey, noisekey = split(key, 2)
    shift = randint(shiftkey, (N,), 0, L)
    noise = normal(noisekey, (N, L)) * noise_std
    y = vmap(lambda s, z: jnp.roll(x + z, s), in_axes=(0, 0))(shift, noise)
    return y, noise, shift