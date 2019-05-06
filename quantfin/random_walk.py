import numpy as np

def geometric_brownian(N, S0, mu, sigma):
    """
    Generates a geometric Brownian random walk

    :param N: How many steps to take
    :param S0: Starting value of the walk
    :param mu: Drift parameter
    :param sigma: Volatility parameter
    """
    S = np.zeros((N,))
    S[0] = S0
    for n in range(1, N):
        S[n] = S[n-1]*np.exp(np.random.normal(mu, sigma))

    return S

def mean_reversion(N, x0, X, alpha, sigma=1.0):
    """
    Generates a mean reverting random walk

    :param N: How many steps to take
    :param x0: Starting value
    :param X: Mean value to revert to
    :param alpha: Strength of mean reversion
    :param sigma: Standard deviation of noise
    """
    x = np.zeros((N,))
    x[0] = x0
    for n in range(1, N):
        x[n] = x[n-1] + alpha * (X - x[n-1]) + np.random.normal(0.0, sigma)

    return x
