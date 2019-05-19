"""Tools for fitting distributions to data."""

import numpy as np
from scipy.stats import norm
from .kernels import gaussian

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

def fit_mixed_gaussian(x, K, epsilon=1e-4, maxiter=500):
    """Fits a mixed Gaussian distribution to the data.

    Fits a weighted sum of Gaussian distributions to data using the maximum
    expectation method.

    Args:
        x (numpy.array): Data to fit to.
        K (int): Number of Gaussians to use in the mixture.
        epsilon (float, optional): Accuracy to solve to.
        maxiter (int, optional): Max number of iterations.

    Returns:
        A tuple of numpy arrays `(mu, sigma, r)`. The kth entry of each array
        contains the mean/standard deviation/weight of the kth Gaussian
        distribution in the mixture.

    Raises:
        RuntimeError: If the iteration fails to converge.
    """
    N_x = len(x) # Number of data points
    r = np.ones(K) / N_x # Weights of each Gaussian
    mu = np.zeros(K) # Means of each Gaussian
    sigma = np.ones(K) # Variance of each Gaussian
    def g(y_n, k):
        # g(x_n; mu_k, sigma_k) =
        # r_k * phi(x_n; mu_k sigma_k) / sum_{j=1}^{K} r_j phi(x_n; mu_k, sigma_j)
        denom = (r * gaussian((y_n - mu) / sigma)).sum()
        return (r[k] * gaussian((y_n - mu[k]) / sigma[k])) / denom

    def N(k):
        # N_k = sum_{n=1}^{N} g(x_n; mu_k, sigma_k)
        g_vals = np.array([g(x_n, k) for x_n in x])
        return g_vals.sum()

    def update_mu(k):
        # mu_k = (1/N_k) * sum_{n=1}^{N} g(x_n; mu_k, sigma_k) * x_n
        summand = np.array([g(x_n, k) * x_n for x_n in x])
        return summand.sum() / N(k)

    def update_sigma(k):
        # sigma^2_k = (1/N_k) * sum_{n=1}^{N} g(x_n; mu_k, sigma_k) * (x_n - mu_k)^2
        summand = np.array([g(x_n, k) * (x_n - mu[k])**2 for x_n in x])
        return np.sqrt(summand.sum() / N(k))

    def update_r(k):
        # r_k = N_k / N
        return N(k) / N_x

    def log_L():
        # ln(L) = sum_{n=1}^{N} ln(sum_{k=1}^{K} r_k phi(x_n; mu_k, sigma_k))
        out = 0.0
        for n in range(N_x):
            out += np.log((r * gaussian((x[n] - mu)/sigma)).sum())
        return out

    n_iter = 0
    prev_L = log_L()
    while True:
        if n_iter > maxiter:
            # Raise exception when max iterations reached
            raise RuntimeError("Max iterations exceeded.")

        # Update our guess for each parameter
        for k in range(K):
            mu[k] = update_mu(k)
            sigma[k] = update_sigma(k)
            r[k] = update_r(k)

        # Calculate new value of ln(L)
        new_L = log_L()
        if abs(new_L - prev_L) < epsilon:
            # Iteration has converged
            break

        # Update value of ln(L)
        prev_L = new_L

    sigma = sigma
    return (mu, sigma, r)
