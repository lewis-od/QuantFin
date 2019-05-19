"""Kernel functions for use in kernel density estimation."""
import numpy as np

def gaussian(x):
    """Gaussian distribution with mean 0 and standard deviation 1."""
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-(x ** 2) / 2)

def epanechnikov(x):
    """Epanechnikov kernel.

    Defined by:

        k(x) = (3/4)*(1 - x^2) for -1 < x < 1
        k(x) = 0               Otherwise
    """
    out = (3.0 / 4.0) * (1.0 - (x ** 2))
    out[x < -1] = 0.0
    out[x > 1] = 0.0

    return out
