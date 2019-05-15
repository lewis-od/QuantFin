import numpy as np
from scipy.optimize import root

def yield_to_maturity(M, P, C, T, t, m):
    """Calculates the yield to maturity of a bond.

    Scipy is used to solve the implicit equation relating the bond's value and
    its yield to maturity: `M = P*exp[-y(T - t))] + sum_{i}[C*exp(-y(T - t_i))]`
    where `t_i` are the dates of the coupon payments.

    Args:
        M (float): Market value of bond.
        P (float): Principal of bond.
        C (float): The value of each coupon payment.
        T (float): Redemtion time/date of the bond.
        t (float): Current time (in same units as `T`).
        m (float): Fraction of the year coupon payments are made (e.g. 0.5
            for bi-annually).

    Returns:
        The yield to maturity value calculated.

    Raises:
        RuntimeError: If the root finding fails.
    """
    ti = np.arange(t, T, m)

    # M = P*exp[-y(T - t))] + sum_{i}[C*exp(-y(T - t_i))]
    def f(y):
        return P*np.exp(-y*(T - t)) + C*np.exp(-y *(ti - t)).sum() - M

    y0 = 3*C / P # Initial guess - midpoint of 2C/P and 4C/P
    res = root(f, y0)

    if not res.success:
        raise RuntimeError("Failed to find root: " + res.message)

    return res.x[0]
