"""Various data smoothing methods"""
import numpy as np
from .kernels import gaussian

def kde(x, y, delta, k=gaussian):
    """Kernel density estimation.

    Estimates a continuous probability distribution from discrete data using
    kernel density estimation.

    Args:
        x (numpy.array): Values at which to evaluate the estimated PDF.
        y (numpy.array): Data to estimate the PDE from.
        delta (float): Smoothing parameter.
        f (optional): The kernel function to use. Should be positive semi-definite
            and normalised (i.e. a valid PDF). Defaults to a Gaussian distribution.
    """
    N = len(y)
    M = len(x)

    out = np.zeros(M)
    for i in range(M):
        out[i] = k((x[i] - y) / delta).sum()

    out *= (1.0 / (N * delta))

    return out

def sma(y, n):
    """Simple moving average, SMA(n).

    Args:
        y (numpy.array): Data to calculate moving average from.
        n (int): Number of points to use when calculating moving avg.

    Returns:
        An array of size `len(y) - n`.

    Raises:
        ValueError: If `n > len(y)`.
    """
    N = len(y) - n
    if n < 0:
        raise ValueError("Input doesn't contain enough data for moving average.")

    out = [y[i:i+n].mean() for i in range(len(y) - n)]
    out = np.array(out)

    return out

def ewma(y, alpha):
    """Exponentialy weighted moving average.

    Args:
        y (numpy.array): Data to calculate the moving average from.
        alpha (float): Smoothing parameter.

    Returns:
        An array of size `len(y)`.
    """
    avg = np.zeros(len(y))
    avg[0] = y[0]
    for i in range(1, len(y)):
        avg[i] = alpha * y[i] + (1 - alpha) * avg[i - 1]

    return avg
