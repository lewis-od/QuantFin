"""Sample from different probability distributions"""

import numpy as np

def mixed_gaussian(mu1, sigma1, mu2, sigma2, r, shape=(1,)):
    """Sample numbers from a mixed Gaussian distribution.

    The distribution has the probability density function:

        P(x) = r*phi(x; mu_1, sigma_1) + (1 - r)*phi(x; mu_2, sigma_2)

    Where `phi(x; mu, sigma)` is the univariate Gaussian distribution with mean
    `mu` and standard deviation `sigma`.

    Args:
        mu1 (float): Mean of 1st Guassian distribution.
        sigma1 (float): Standard deviation of 1st Gaussian distribution.
        mu2 (float): Mean of 2nd Gaussian distribution.
        sigma2 (float): Standard deviation of 2nd Gaussian distribution.
        r (float): Mixture ratio. Should be in the range [0, 1].
        shape (tuple, optional): Shape of the output array. Defualts to (1,).

    Returns:
        A numpy array of shape `shape` containing numbers sampled from the
        distribution.
    """
    n1 = np.random.normal(mu1, sigma1, size=shape)
    n2 = np.random.normal(mu2, sigma2, size=shape)
    rs = np.random.uniform(0, 1, shape)
    n1[rs > r] = n2[rs > r]

    return n1
