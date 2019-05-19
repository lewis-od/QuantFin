"""Methods for calculating quantities associaled with inetrest-rate swaps."""

import numpy as np

def swap_rate(T, rs, x=0.5):
    """Calculates the fixed rate for an interest-rate swap.

    Args:
        T (float): Time to maturity of the swap.
        rs (numpy.array): Current rates for relevant maturities. I.e. for a
            time to maturity of 2 years with payments twice a year, the 6 month,
            12 month, 18 month, and 24 months rates must be provided.
        x (float, optional): Fraction of the year payments are made. Defualts to
            semi-annually.

    Returns:
        The calculated swap rate as a float.
    """
    taus = np.arange(x, T + x, x) # Payment dates

    if len(taus) != len(rs):
        raise ValueError(
            "Not enough interest rates provided. Expected {} but got {}."
            .format(len(taus), len(rs))
        )

    # F = (1 - exp(-tau_N * r_N)) / (x * sum_{i=1}^{N} exp(-tau_i * r_i))
    F = 1 - np.exp(-taus[-1] * rs[-1]) # Numerator
    denom = x * np.exp(-taus * rs).sum() # Denominator
    F /= denom

    return F
