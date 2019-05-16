import numpy as np

def annualise(sigma, N=250):
    """Converts volatility into annualised volatility.

    Args:
        sigma (numpy.array): Volatilities calculated at daily (or other regular)
            intervals.
        N (int, optional): Number of trading days (or weeks, etc) in a year.
    """
    return sigma * np.sqrt(N)
