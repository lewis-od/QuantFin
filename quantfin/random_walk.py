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
