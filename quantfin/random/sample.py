"""Sample from different probability distributions"""

import numpy as np

def mixed_gaussian(mus, sigmas, rs, size=1):
    """Sample numbers from a mixed Gaussian distribution.

    The distribution has the probability density function:

        P(x) = sum_{k} r_k*phi(x; mu_k, sigma_k)

        provided sum_{k} r_k = 1

    Where `phi(x; mu, sigma)` is the univariate Gaussian distribution with mean
    `mu` and standard deviation `sigma`.

    Args:
        mus (numpy.array): Means of each Gaussian in the mixture.
        sigmas (numpy.array): Stdevs of each Gaussian.
        r (numpy.array): Weights of each Gaussian. Should sum to 1.
        size (int, optional): Size of the output array.

    Returns:
        A numpy array containing numbers sampled from the distribution.

    Raises:
        ValueError: If `mus`, `sigmas`, and `rs` aren't all the same length.
    """
    if len(mus) != len(sigmas) or len(sigmas) != len(rs):
        raise ValueError("mus, sigmas, and rs must all be the same size.")

    if len(mus) == 1:
        raise ValueError("Can't create mixed Gaussian dist. from 1 Gaussian.")

    if not isinstance(rs, np.ndarray):
        rs = np.array(rs)

    intervals = rs.cumsum() # Split [0,1] into appropriately weighted intervals
    vals = np.zeros(size) # Output values
    # import pdb; pdb.set_trace()
    for n in range(size):
        # Generate a random number on [0,1]
        r = np.random.uniform(0, 1)
        for k in range(len(intervals)):
            if r < intervals[k]:
                # If it's in the kth interval, sample from a Gaussian dist.
                # with mean mu_k and stdev sigma_k
                vals[n] = np.random.normal(mus[k], sigmas[k])
                break

    return vals
