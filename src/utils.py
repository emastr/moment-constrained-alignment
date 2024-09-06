import jax.numpy as jnp
from jax import jit, vmap
from jax.numpy.fft import fft, ifft
from jax.scipy.linalg import toeplitz
from typing import Callable, Any, Tuple
from jaxtyping import Array, Float, Complex, PyTree, jaxtyped, Integer

def resize(xfft: Complex[Array, "L"], n: int) -> Complex[Array, "L"]:
    n_old = len(xfft)
    k = (min(n, n_old)-1) // 2 
    xfft_new = jnp.zeros(n, dtype=xfft.dtype)
    xfft_new = xfft_new.at[:1+k].set(xfft[:1+k])
    xfft_new = xfft_new.at[-k:].set(xfft[-k:])
    return xfft_new

@jit
def autocorr(x: Float[Array, "L"]) -> Float[Array, "L"]:
    return ifft(autocorr_fft(fft(x)))


def autocorr_fft(xfft: Complex[Array, "L"]) -> Float[Array, "L"]:
    return jnp.abs(xfft) ** 2


def compl_sign(x: Complex[Array, "L"]) -> Complex[Array, "L"]:
    return x / jnp.abs(x)


def flip_modes(z: Complex[Array, "L"]) -> Complex[Array, "L"]:
    return jnp.roll(jnp.flip(z), 1)    

@jit
def circulant(v: Float[Array, "L"]) -> Float[Array, "L L"]:
    v_flip = flip_modes(v)
    return toeplitz(v_flip, v)

@jit
def symmetrize(u: Complex[Array, "L"]) -> Complex[Array, "L"]:
    u_rev = flip_modes(u)
    return (jnp.conjugate(u_rev) + u)/2


@jit
def find_shift(xfft: Complex[Array, "L"], yfft: Complex[Array, "L"]) -> Integer:
    return jnp.argmax(ifft(xfft * yfft.conj()).real)

@jit
def roll_fft(yfft: Complex[Array, "L"], shift: Integer[Array, "L"]) -> Complex[Array, "L"]:
    """Shift signals according to shifts. Shifts should be complex numbers on the unit circle."""
    d = len(yfft)
    powers = jnp.arange(0, d, 1)
    return yfft * jnp.exp(-2 * jnp.pi * 1j * shift/d) ** powers

@jit
def align(xfft: Complex[Array, "L"], yfft: Complex[Array, "L"], y: Float[Array, "L"]):
    shift = find_shift(xfft, yfft)
    return jnp.roll(y, shift)


@jit
def align_to_signal(z: Float[Array, "L"], x: Float[Array, "L"]) -> Float[Array, "L"]:
    zfft, xfft = fft(z), fft(x)
    return align(xfft, zfft, z)

@jit
def align_fft(xfft: Complex[Array, "L"], yfft: Complex[Array, "L"]) -> Complex[Array, "L"]:
    shift = find_shift(xfft, yfft)
    return roll_fft(yfft, shift)


@jit
def project_moments(xfft: Complex[Array, "L"], acf_fft: Float[Array, "L"], mean: float) -> Complex[Array, "L"]:
    #xfft = xfft.at[0].set(mean*(len(xfft)-1))
    xfft = xfft.at[1:].set(jnp.where(xfft[1:] != 0, \
                           xfft[1:]*acf_fft[1:]**0.5 / jnp.abs(xfft[1:]), 0))
    return xfft

@jit
def align_average(xfft: Complex[Array, "L"], yfft: Complex[Array, "N L"]) -> Complex[Array, "L"]:
    return jnp.mean(vmap(align_fft, (None, 0))(xfft, yfft), axis=0)

@jit
def align_average_and_project(xfft: Complex[Array, "L"], yfft: Complex[Array, "N L"], acf_fft: Float[Array, "L"], mean: float) -> Complex[Array, "L"]:
    xfft = align_average(xfft, yfft)
    xfft = project_moments(xfft, acf_fft, mean)
    return xfft


def relative_error(x: Float[Array, "L"], y: Float[Array, "L"]) -> float:
    return jnp.linalg.norm(x - y) / jnp.linalg.norm(x)


@jit
def loss_fft(xfft: Complex[Array, "L"], yfft: Complex[Array, "L"]) -> float:
    """
    Calculates the loss between two complex-valued FFT signals.
    Parameters:
    xfft (Complex[Array, "L"]): The first complex-valued FFT signal.
    yfft (Complex[Array, "L"]): The second complex-valued FFT signal.
    Returns:
    float: The loss between the two FFT signals.
    """

    yfft_align = vmap(align_fft, (None, 0))(xfft, yfft)
    return jnp.mean(jnp.abs(yfft_align - xfft)**2)


@jit
def powerspectrum(X: Float[Array, "N L"], sigma: float) -> Tuple[float, Float[Array, "L"], Complex[Array, "N L"]]:
    """Calculates the power spectrum of a given input array.

    Args:
        X (Float[Array, "N L"]): The input array of shape (N, L).
        sigma (float): The standard deviation.

        Tuple[float, Float[Array, "L"], Complex[Array, "N L"]]: A tuple containing the mean value of X, the power spectrum, and the Fourier transform of X.

    """
    (N, L) = X.shape
    xmean = jnp.mean(jnp.mean(X))
    Xc = X - xmean
    Xc_fft = fft(Xc, axis=1)
    Xc_auto_fft = jnp.clip(jnp.mean(jnp.abs(Xc_fft)**2., axis=0) - sigma**2 * L, 0, None)
    return xmean, Xc_auto_fft, fft(X)
    
@jit
def bispectrum(X: Float[Array, "N L"], sigma: float) -> Tuple[float, Float[Array, "L"], Complex[Array, "L L"], Complex[Array, "N L"]]:
    """Calculate the bispectrum of the given input array.
    Args:
        X (Float[Array, "N L"]): The input array of shape (N, L).
        sigma (float): The value of sigma.
    Returns:
        Tuple[float, Float[Array, "L"], Complex[Array, "L L"], Complex[Array, "N L"]]: A tuple containing the following:
            - xmean (float): The mean value of the input array.
            - Xc_auto_fft (Float[Array, "L"]): The auto-correlation of the input array after FFT.
            - B_est (Complex[Array, "L L"]): The estimated bispectrum.
            - Xc_fft (Complex[Array, "N L"]): The FFT of the input array after mean subtraction.
    """
    (N, L) = X.shape
    xmean = jnp.mean(jnp.mean(X))
    Xc = X - xmean
    Xc_fft = fft(Xc, axis=1)
    Xc_auto_fft = jnp.clip(jnp.mean(jnp.abs(Xc_fft)**2., axis=0) - sigma**2 * L, 0, None)

    B_est = jnp.zeros((L, L))
    
    def B_row(xm_fft):
        Bm = jnp.multiply((xm_fft[:, None] * jnp.conjugate(xm_fft[None, :])), circulant(xm_fft))
        return Bm
    
    B_est = jnp.mean(vmap(B_row, 0)(Xc_fft), axis=0)
    return xmean, Xc_auto_fft, B_est, Xc_fft

def bisection_search(function: Callable, target: float, a: float, b: float, bitol: float=1e-6, verbose: bool=True, **kwargs) -> Tuple[float, float, int]:
    ## First, make sure function(a) < target < function(b)
    fa = function(a, **kwargs)
    fb = function(b, **kwargs)
    if fa < target:
        return a, fa, -1
    if fb > target:
        return b, fb, -1
    ## Now, do the bisection search
    while b - a > bitol:
        c = int((a + b) / 2)
        fc = function(c, **kwargs)
        if fc > target:
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        if verbose:
            print(f"Interval [{a}, {b}]", end="\r")
    c = int((a + b) / 2)
    fc = function(c, **kwargs)
    return c, fc, 0