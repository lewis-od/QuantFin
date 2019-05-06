import numpy as np
from scipy.optimize import root

def yield_to_maturity(M, P, C, T, t, m):
    """
    Calculates the yield to maturity of a bond

    :param M: Market value
    :param P: Principal
    :param C: Coupon payments
    :param T: Redemtion time
    :param t: Current time
    :param m: Fraction of the year coupon payments are made
              (e.g. 0.5 for bi-annually)
    """
    ti = np.arange(t, T, m)

    # M = P*exp[-y(T - t))] + sum_{i}[C*exp(-y(T - t_i))]
    def f(y):
        return P*np.exp(-y*(T - t)) + C*np.exp(-y *(ti - t)).sum() - M

    y0 = 3*C / P # Initial guess - midpoint of 2C/P and 4C/P
    res = root(f, y0)
    if res.success:
        return res.x[0]

    return None
