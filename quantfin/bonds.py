import numpy as np
from scipy.optimize import root

def yield_to_maturity(M, P, C, T, t, ti):
    """
    Calculates the yield to maturity of a bond

    :param M: Market value
    :param P: Principal
    :param C: Coupon payment
    :param T: Redemtion time
    :param t: Current time
    :param ti: Times at which coupon payments are made
    """
    ti = ti[ti >= t] # Only consider future coupon payments

    def f(y):
        return P*np.exp(-y*(T - t)) + C * np.exp(-y *(ti - t)).sum() - M
    y0 = 3*C / P
    res = root(f, y0)
    if res.success:
        return res.x[0]

    return None
