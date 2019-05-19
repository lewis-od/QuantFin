"""Probability density functions."""

import numpy as np

def gaussian(x, mu=0.0, sigma=1.0):
    """Gaussian distribution.

    Args:
        x: Point(s) to evaluate PDF at.
        mu (float, optional): Mean of distribution.
        sigma (float, optional): Stdev of distribution.

    Returns:
        The value(s) of the Gaussian PDF at point(s) `x`.
    """
    N = np.sqrt(2 * np.pi) # Normalisation const
    return np.exp(-(x - mu)**2 / sigma) / N

def mixed_gaussian(x, mus, sigmas, rs):
    """Mixed Gaussian distribution.

    Args:
        x: Point(s) to evaluate the PDF at.
        mus (np.array): Means of each Gaussian dist. in the mixture.
        sigmas (np.array): Stdevs of each Gaussian in the mixture.
        rs (np.array): Weights of each Gaussian.

    Returns:
        The value(s) of the mixed Gaussian PDF at point(s) `x`.

    Raises:
        ValueError: If `mus`, `sigmas`, and `rs` don't have the same length.
    """
    if len(mus) != len(sigmas) or len(sigmas) != len(rs) or len(rs) != len(mus):
        raise ValueError("mus, sigmas, and rs must all be the same length.")

    # If x is a scalar, convert it to an array
    if np.isscalar(x):
        x = np.array([x])

    K = len(mus) # Number of Gaussians in the mixture
    value = np.zeros(len(x)) # Value to return
    for k in range(K):
        # Weighted sum of Gaussians
        value += rs[k] * gaussian(x, mu=mus[k], sigma=sigmas[k])

    # If x is a scalar, return a scalar
    if len(value) == 1:
        value = value[0]

    return value
