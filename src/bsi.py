import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import ifft
from utils import compl_sign, circulant, symmetrize, bispectrum, loss_fft
from typing import Callable, Tuple
from jaxtyping import Array, Float, Complex, Integer
from jax.random import PRNGKey, split, normal


def _power_iters(M: int, x0: Float[Array, "L"], tol: float, maxiter: int=100) -> Tuple[Float[Array, "d"], Float[Array, "d"], Integer]:
    val = (x0, tol + 1., 0)
    
    def bdy_func(val):
        x, res, it = val
        z = M @ x
        z = compl_sign(z)
        z = z * jnp.sign(z[0])
        res = jnp.linalg.norm(z - x) / len(x)**0.5
        x = z
        it = it + 1
        return (x, res, it)
    
    def cond_func(val):
        _, res, it = val
        return (res > tol) & (it < maxiter)
    
    while cond_func(val):
        val = bdy_func(val)
    return val[0]

def _iterative_phase_synch(B: Complex[Array, "L L"], 
                         z0: Complex[Array, "L"]=None, 
                         tol: float=1e-6, maxiter = None, 
                         callback: Callable=None,
                         prngkey=None) -> Complex[Array, "L"]:
    if isinstance(maxiter, int):
        inner_maxiter = maxiter
        outer_maxiter = 45
    elif isinstance(maxiter, tuple):
        if (len(maxiter) == 2):
            outer_maxiter, inner_maxiter = maxiter
        else:
            raise ValueError('maxiter should be an integer or a tuple of two integers')
    else:
        inner_maxiter = 1000
        outer_maxiter = 45
        
    
    if prngkey is None:
        prngkey = PRNGKey(0)
        
    if z0 is None: 
        z0 = compl_sign(B[0,0])
    elif z0 == 0:
        z0 = 1
    else:
        z0 = compl_sign(z0)
        
    L = B.shape[0]
    B = (B + jnp.conj(B).T) / 2
    
    Mfun = lambda z: jnp.multiply(B, jnp.conj(circulant(z)))
    it = 0
    key1, key2 = split(prngkey, 2)
    #z = normal(key1, (L,)) + 1j*normal(key2, (L,))
    #z = z.at[0].set(z0)
    z = np.random.randn(L) + 1j * np.random.randn(L)
    z[0] = z0
    z = compl_sign(symmetrize(z))
     
    res = tol+1.
    it = 0
    while (it < outer_maxiter) and (res > tol):
        M = Mfun(z)
        znew = _power_iters(M, z, tol=tol, maxiter=inner_maxiter)
        znew = znew*compl_sign(z0) / znew[0]
        znew = compl_sign(symmetrize(znew))
        res = jnp.linalg.norm(z - znew) / L ** 0.5
        z = znew
        it=it+1
        if callback is not None:
            callback(z, res, it)
    return z

def bsi(y: Float[Array, "N L"], sigma: float, **kwargs) -> Float[Array, "L"]:
    """Bispectrum inversion method with iterative phase synchronization using power iterations.
    This code is largely based on the description and code from https://arxiv.org/abs/1705.00641.
    Instead of solving the synchronization problem with riemannian trust regions, we use power iterations.
    Args:
        y (N x L array of floats): N Noisy shifted signals of length L
        sigma (float): Noise level

    Returns:
        L size array of floats: signal reconstruction
    """
    y_mean, y_auto_fft, B, yfft = bispectrum(y, sigma)
    
    L = len(y_auto_fft)
    z = _iterative_phase_synch(B, y_mean, **kwargs)
    
    xfft = (y_auto_fft ** 0.5) * z
    xfft = xfft.at[0].set(y_mean * L)
    
    # Hotfix, take best out of x and x2:
    xfft2 = (y_auto_fft ** 0.5) * (-z)
    xfft2 = xfft2.at[0].set(y_mean * L)

    if loss_fft(xfft, yfft) < loss_fft(xfft2, yfft):
        return ifft(xfft).real
    else:
        return ifft(xfft2).real
