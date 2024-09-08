from jax.numpy.fft import fft, ifft
from utils import relative_error, align_average_and_project, align_average, powerspectrum
from jaxtyping import Array, Float

def mca(y: Float[Array, "N L"], 
        sigma: float, 
        x0: Float[Array, "L"], 
        tol: float=1e-6, 
        maxiter: int=100000, 
        alpha: float=1.0, 
        callback: int=None, 
        project: bool=True) -> Float[Array, "L"]:
    
    x0fft = fft(x0)
    mean, acf_fft, yfft = powerspectrum(y, sigma)
    res = tol + 1.
    i = 0
    xfft = x0fft
    while (res > tol) and i < maxiter:
        if project:
            xfft_new = (1 - alpha) * xfft + alpha * align_average_and_project(xfft, yfft, acf_fft, mean)
        else:
            xfft_new = (1 - alpha) * xfft + alpha * align_average(xfft, yfft)
        res = relative_error(ifft(xfft), ifft(xfft_new))
        xfft = xfft_new
        i += 1
        if callback is not None:
            callback(xfft, res, i)
    return ifft(xfft).real