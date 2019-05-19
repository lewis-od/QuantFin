"""Various utility functions."""

import numpy as np

def annualise_volatility(sigma, N=250):
    """Converts volatility into annualised volatility.

    Args:
        sigma (numpy.array): Volatilities calculated at daily (or other regular)
            intervals.
        N (int, optional): Number of trading days (or weeks, etc) in a year.
    """
    return sigma * np.sqrt(N)

def returns_from_prices(A):
    """Calculates returns from a time series of prices."""
    R = np.array([(A[n] - A[n-1]) / A[n-1] for n in range(1, 250)])
    return R
