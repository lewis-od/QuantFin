"""Methods for generating different types of random walk."""

import numpy as np

def geometric_brownian(N, S0, mu, sigma):
    """Generates a geometric Brownian random walk.

    Args:
        N (int): How many steps to generate.
        S0 (float): Starting value of the walk.
        mu (float): Drift parameter.
        sigma (float): Volatility parameter.

    Returns:
        A numpy array containing the value of the walk at each step.
    """
    S = np.zeros((N,))
    S[0] = S0
    for n in range(1, N):
        S[n] = S[n-1]*np.exp(np.random.normal(mu, sigma))

    return S

def mean_reversion(N, x0, X, alpha, sigma=1.0):
    """Generates a mean reverting random walk.

    Args:
        N (int): How many steps to generate.
        x0 (float): Starting value of the walk.
        X (float): Mean value to revert to.
        alpha (float): Strength of mean reversion.
        sigma (float, optional): Standard deviation of the distribution the noise is
            sampled from.

    Returns:
        A numpy array containing the value of the walk at each step.
    """
    x = np.zeros((N,))
    x[0] = x0
    for n in range(1, N):
        x[n] = x[n-1] + alpha * (X - x[n-1]) + np.random.normal(0.0, sigma)

    return x
