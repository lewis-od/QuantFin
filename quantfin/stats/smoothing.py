"""Various data smoothing methods"""
import numpy as np

def sma(y, n):
    """Simple moving average, SMA(n).

    Args:
        y (numpy.array): Data to calculate moving average from.
        n (int): Number of points to use when calculating moving avg.

    Returns:
        An array of size `len(y) - n`.

    Raises:
        ValueError: If `n > len(y)`.
    """
    N = len(y) - n
    if n < 0:
        raise ValueError("Input doesn't contain enough data for moving average.")

    out = [y[i:i+n].mean() for i in range(len(y) - n)]
    out = np.array(out)

    return out

def ewma(y, alpha):
    """Exponentialy weighted moving average.

    Args:
        y (numpy.array): Data to calculate the moving average from.
        alpha (float): Smoothing parameter.

    Returns:
        An array of size `len(y)`.
    """
    avg = np.zeros(len(y))
    avg[0] = y[0]
    for i in range(1, len(y)):
        avg[i] = alpha * y[i] + (1 - alpha) * avg[i - 1]

    return avg
