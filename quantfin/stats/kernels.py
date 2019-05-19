"""Kernel functions for use in kernel density estimation."""
import numpy as np
from .pdfs import gaussian as gaussian_pde
# NOTE: Could just `from .pdfs import gaussian` and this would work, but then
# we'd have no documentation that the Gaussian kernel can be accessed in this
# way.

def gaussian(x):
    """Gaussian distribution with mean 0 and standard deviation 1."""
    return gaussian_pde(x)

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
