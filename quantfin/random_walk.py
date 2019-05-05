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

def mean_reversion(N, x0, X, alpha, mu=0.0, sigma=1.0):
    x = np.zeros((N,))
    x[0] = x0
    for n in range(1, N):
        x[n] = x[n-1] + alpha * (X - x[n-1]) + np.random.normal(mu, sigma)

    return x
