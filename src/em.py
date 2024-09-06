import jax.numpy as jnp
from jax import jit, random
from jax.numpy.fft import fft, ifft
from utils import relative_error
from typing import Callable, Any, Tuple
from jaxtyping import Array, Float, Complex, PyTree, jaxtyped, Integer

def em(X: Float[Array, "N L"], 
       sigma: float, 
       x0: Float[Array, "L"], 
       tol: float=1e-10, 
       batch_niter: int=3000, 
       full_niter: int=10000, 
       callback: Callable=None) -> Float[Array, "L"]:
    X = X.T
    d, N = X.shape
    
    x0fft = fft(x0)
    assert x0fft.shape[0] == d, 'Initial guess x must have length N.'

    # In practice, we iterate with the DFT of the signal x
    xfft = x0fft

    # Precomputations on the observations
    Xfft = fft(X, axis=0)
    sqnormX = jnp.sum(jnp.abs(X)**2, axis=0)[None, :]

    # If the number of observations is large, get started with iterations
    # over only a sample of the observations
    if N >= 3000:
        batch_size = 1000
        for _ in range(batch_niter):
            sample = random.randint(random.PRNGKey(2), (batch_size,), 0, N)
            xfft_new = em_iteration(xfft, Xfft[:, sample], sqnormX[:, sample], sigma)
            xfft = xfft_new
            
    for iter in range(full_niter):
        xfft_new = em_iteration(xfft, Xfft, sqnormX, sigma)
        if relative_error(ifft(xfft), ifft(xfft_new)) < tol:
            break
        xfft = xfft_new
        if callback is not None:
            callback(xfft, jnp.linalg.norm(xfft-xfft_new), iter)

    x = ifft(xfft).real
    return x


@jit
def em_iteration(fftx: Complex[Array, "L"], fftX: Complex[Array, "N L"], sqnormX: Float[Array, "L"], sigma: float) -> Complex[Array, "L"]:
    C = ifft(fftx.conj()[:, None] * fftX, axis=0)
    T = (2 * C - sqnormX) / (2 * sigma**2)
    T = T - jnp.max(T, axis=0, keepdims=True)
    W = jnp.exp(T)
    W = W / jnp.sum(W, axis=0, keepdims=True)
    fftx_new = jnp.mean(fft(W, axis=0).conj() * fftX, axis=1)
    return fftx_new
