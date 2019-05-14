import numpy as np
from scipy.optimize import minimize

def moving_average(R, M):
    """
    Calculates volatility using an M-day moving average
    :param R: Returns to calculate volatility from
    :param M: Number of days to use for the moving average (assuming returns are daily)
    :returns: An array of size (len(R) - M) containing the calculated volatilities
    """
    N = len(R)
    sigma_sq = np.zeros(N - M)
    for i in range(M, N):
        sigma_sq[i - M] = R[i-M:i].std() ** 2
        
    return sigma_sq


def ewma(R, smoothing=0.98):
    """
    Calculates volatility using an exponentially weighted moving average
    :param R: Returns to calculate volatility from
    :param smoothing: Smoothing parameter (lambda)
    :returns: An array of size len(R) containing the calculated volatilities
    """
    N = len(R)
    sigma_sq = np.zeros(N)
    for n in range(N):
        vol = np.array([(smoothing ** i) * (R[n-i] ** 2) for i in range(n)]).sum()
        vol *= (1 - smoothing)
        sigma_sq[n] = vol
    
    return sigma_sq