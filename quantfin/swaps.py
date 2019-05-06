import numpy as np

def swap_rate(T, rs, x=0.5):
    """
    Calculates the fixed rate for an interest-rate swap

    :param T: Maturity
    :param rs: Current rates for relevant maturities. I.e. for a maturity
        of 2 years with payments twice a year, the 6 month, 12 month, 18 month,
        and 24 months rates must be provided.
    :param x: Fraction of the year payments are made. Defualts to semi-annually.
    """
    taus = np.arange(x, T + x, x) # Payment dates

    if len(taus) != len(rs):
        raise ValueError("Not enough interest rates provided")

    F = 1 - np.exp(-taus[-1] * rs[-1])
    denom = 0
    for i in range(len(taus)):
        denom += np.exp(-taus[i] * rs[i])
    denom *= x
    F /= denom

    return F
