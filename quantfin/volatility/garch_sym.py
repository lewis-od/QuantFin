"""
# Symmetric GARCH
Fits the symmetric GARCH(1,1) model to returns data.

## Usage
```python
import quantfin as qf

params = qf.volatility.garch_sym.fit_model(R)
volatilities = qf.volatilty.garch_sym.sigma(R, params)
```
"""
import warnings
import numpy as np
from scipy.optimize import minimize

def sigma(R, params):
    """Calculates volatility using the symmetric GARCH(1,1) model

    The symmetric GARCH(1,1) model is defined by the recursion relation:

        sigma^{2}_{n+1} = omega + alpha * R^{2}_{n} + beta * sigma^{2}_{n}

    where `sigma_n` is the volatilty at time-step `n`, and `R_n` are the returns
    of the asset at time-step `n`. We assume that `sigma_0 = omega`.

    Args:
        R (numpy.array): Financial returns data from which to calculate
            volatility.
        params (numpy.array): The parameters of the GARCH model to use. Should
            be in the form `[alpha, beta, omega]`.

    Returns:
        A numpy array of volatilties.
    """
    alpha, beta, omega = params
    N = len(R)

    # Recursion relation for calculating sigma^2 using GARCH(1,1)
    variance = np.zeros(N)
    variance[0] = omega
    for n in range(1, N):
        variance[n] = omega + alpha * R[n-1]**2 + beta * variance[n - 1]

    volatility = np.sqrt(variance)

    return volatility


def fit_model(R, init=[5, 5, 5], verbose=False):
    """Calculates volatility using the symmetric GARCH(1,1) model.

    For a definition of the model, see `sigma`.
    The parameters for the model are cauluated using the maximum likelihood
    method. The likelihood function is maximised using scipy.

    Args:
        R (numpy.array): Financial returns data from which to calculate
            volatility.
        init (list, optional): Initial values of the parameters to use in
            optimisation
        verbose (bool, optional): Whether or not to print warnings and
            convergence information.

    Returns:
        An array containing the parameters of the model in the form
        `[alpha, beta, omega]`.

    Raises:
        RuntimeError: If the optimisation algorithm fails to converge.
    """
    N = len(R)

    # Log of likelihood function
    def L(params):
        # Calculate variance using model
        sigma_sq = sigma(R, params) ** 2

        # ln(L) = -0.5 * sum_{n} {ln(sigma^2_n) + (r^2_n / sigma^2_n)}
        value = np.log(sigma_sq) + ((R ** 2) / sigma_sq)
        value = -0.5 * value.sum()

        # Return -L instead of L (minimise -L => maximise L)
        return -value

    # Constraints for optimisation
    # Use def instead of lambdas for speed
    def w_con(x): # omega > 0
        return np.array([x[2]])
    def a_con(x): # alpha > 0
        return np.array([x[0]])
    def b_con(x): # beta > 0
        return np.array([x[1]])
    def ab_con(x): # alpha + beta < 1
        return np.array([ 1 - x[0] - x[1] ])
    omega_con = { 'type': 'ineq', 'fun': w_con }
    alpha_con = { 'type': 'ineq', 'fun': a_con }
    beta_con = { 'type': 'ineq', 'fun': b_con }
    alpha_beta_con = { 'type': 'ineq', 'fun': ab_con }
    cons = (alpha_con, beta_con, omega_con)

    # Don't show warnings unless verbose flag specified
    if not verbose:
        warnings.filterwarnings('ignore')

    # Minimise -L to find values of alpha, beta, and omega
    res = minimize(L, init, constraints=cons,
                   options={ 'maxiter': 500, 'disp': verbose })

    # Turn warnings back on if they were turned off above
    if not verbose:
        warnings.resetwarnings()

    if res.success == False:
        raise RuntimeError("Unable to fit GARCH(1,1) model: " + res.message)

    return res.x

def expected(params):
    """Calculate the expected volatility predicted by the model.

    Args:
        params (numpy.array): The parameters from the fitted model of the form
            `[alpha, beta, omega]`.

    Raises:
        ValueError: If the expected volatility is not well defined.
    """
    alpha, beta, omega = params
    if omega < 0:
        raise ValueError("Expected volatility not defined for omega < 0")

    if alpha + beta > 1:
        raise ValueError("Expected volatility not defined for alpha + beta < 1")

    return np.sqrt(omega / (1 - alpha - beta))
