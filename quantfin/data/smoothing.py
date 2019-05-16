"""Various data smoothing methods"""
import numpy as np
from .dist import gaussian

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