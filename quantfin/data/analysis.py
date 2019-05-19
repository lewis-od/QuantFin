"""Tools for fitting distributions to data."""

import numpy as np
from scipy.stats import norm

def standardise(x):
    """Transforms data so that it has mean 0 and standard deviation 1.

    The transform is given by:

        z_i = (x_i - mu) / sigma

    where `mu` is the mean and `sigma` is the standard deviation of the data.

    Args:
        x (numpy.array): Data to transform.
    """
    mu = x.mean()
    sigma = x.std()

    return (x - mu) / sigma

def quantile_quantile(x, dist=norm):
    """Calculates theoretical and actual quantiles of a data set for a QQ plot.

    Args:
        x (numpy.array): The data to analyse.
        dist (scipy.stats.rv_continuous, optional): The distribution to use
            when calculating theoretical quantiles. Must have a `ppf` method.
            Defaults to `scipy.stats.norm`.

    Returns:
        A tuple `(z_theoretical, z_ordered)`, where `z_theoretical` are the
        quantile values calculated from the theoretical distribution, and
        `z_ordered` are the quantiles calculated from the ordered data.
    """
    N = len(x)
    z = standardise(x)
    z.sort()

    p = np.array([(i - 0.5)/N for i in range(1, N+1)])
    z_pred = dist.ppf(p)

    return (z_pred, z)
