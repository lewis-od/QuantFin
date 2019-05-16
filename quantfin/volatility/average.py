"""
# Average
Functions for calulating volatility using averaging techniques.
"""
import numpy as np

def moving(R, M):
    """Calculates volatility using an M-day moving average.

    Args:
        R (numpy.array): Financial returns data from which to calculate volatility.
        M (int): Number of days to use for the moving average.

    Returns:
        An numpy array of size `(len(R) - M)` containing the calculated
        volatilities.
    """
    N = len(R)
    sigma = np.zeros(N - M)
    for i in range(M, N):
        sigma[i - M] = R[i-M:i].std()

    return sigma

def weighted(R, smoothing=0.98):
    """Calculates volatility using an exponentially weighted moving average.

    Args:
        R (numpy.array): Financial returns data from which to calculate volatility.
        smoothing (float): Smoothing parameter (lambda).

    Returns:
        A numpy array of size `len(R)` containing the calculated volatilities.
    """
    N = len(R)
    sigma_sq = np.zeros(N)
    for n in range(N):
        vol = np.array([(smoothing ** i) * (R[n-i] ** 2) for i in range(n)]).sum()
        vol *= (1 - smoothing)
        sigma_sq[n] = vol

    return np.sqrt(sigma_sq)
