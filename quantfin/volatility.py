import warnings
import numpy as np
import scipy.special
from scipy.optimize import minimize

def moving_average(R, M):
    """Calculates volatility using an M-day moving average.

    Args:
        R (numpy.array): Financial returns data from which to calculate volatility.
        M (int): Number of days to use for the moving average.

    Returns:
        An numpy array of size `(len(R) - M)` containing the calculated
        volatilities.
    """
    N = len(R)
    sigma_sq = np.zeros(N - M)
    for i in range(M, N):
        sigma_sq[i - M] = R[i-M:i].std() ** 2

    return np.sqrt(sigma_sq)


def ewma(R, smoothing=0.98):
    """Calculates volatility using an exponentially weighted moving average.

    Args:
        R (numpy.array): Financial returns data from which to calculate volatility.
        smoothing (float): Smoothing parameter (lambda).

    Returns:
        A numpy array of size `len(R)` containing the calculated volatilities.
    """
    N = len(R)
    sigma_sq = np.zeros(N)
    for n in range(N):
        vol = np.array([(smoothing ** i) * (R[n-i] ** 2) for i in range(n)]).sum()
        vol *= (1 - smoothing)
        sigma_sq[n] = vol

    return np.sqrt(sigma_sq)


def garch(R, verbose=False):
    """Calculates volatility using the symmetric GARCH(1,1) model.

    The parameters for the model are cauluated using the maximum likelihood
    method. The likelihood function is maximised using scipy.

    This slows down rapidly as R increases in size due to the recursive
    nature of GARCH.

    Args:
        R (numpy.array): Financial returns data from which to calculate
            volatility.
        verbose (bool, optional): Whether or not to print warnings and
            convergence information.

    Returns:
        A tuple `(volatility, params)`, where `volatility` is a numpy array
        of size `len(R)` containing the calculated volatilities, and
        `params = [alpha, beta, omega]` are the calculated values of the model
        parameters.

    Raises:
        RuntimeError: If the optimisation algorithm fails to converge.
    """
    N = len(R)

    # sigma^2 using GARCH model
    def sigma_sq(n, alpha, beta, omega):
        prev = 0 if (n == 0) else sigma_sq(n-1, alpha, beta, omega)
        return omega + alpha * R[n-1]**2 + beta * prev

    # Log of likelihood function
    def L(params):
        alpha, beta, omega = params
        value = 0
        for n in range(N):
            sigma_n_sq = sigma_sq(n, alpha, beta, omega)
            value += np.log(sigma_n_sq + (R[n]**2)/sigma_n_sq)
        value *= -0.5
        # Return -L instead of L (minimise -L => maximise L)
        return -value

    # Constrain alpha and beta to be positive
    # Use def instead of lambdas for speed
    def a_con(x):
        return np.array([x[0]])
    def b_con(x):
        return np.array([x[1]])
    alpha_con = { 'type': 'ineq', 'fun': a_con }
    beta_con = { 'type': 'ineq', 'fun': b_con }
    cons = (alpha_con, beta_con)

    # Don't show warnings unless verbose flag specified
    if not verbose:
        warnings.filterwarnings('ignore')

    # Minimise -L to find values of alpha, beta, and omega
    res = minimize(L, [0.5, 0.5, 0.0], constraints=cons,
                   options={ 'maxiter': 500, 'disp': verbose })

    # Turn warnings back on if they were turned off above
    if not verbose:
        warnings.resetwarnings()

    if res.success == False:
        raise RuntimeError("Unable to fit GARCH(1,1) model: " + res.message)

    # Calculate the volatility using the parameters found by minimize
    params = res.x
    volatility = np.array([sigma_sq(n, params[0], params[1], params[2]) for n in range(N)])
    volatility = np.sqrt(volatilty)

    return (volatility, params)

def garch_asym(R, verbose=False):
    """Calculates volatility using the asymmetric GARCH(1,1) model.

    The parameters for the model are cauluated using the maximum likelihood
    method. The likelihood function is maximised using scipy.

    This slows down rapidly as R increases in size due to the recursive
    nature of GARCH. The symmetric GARCH(1,1) model `garch` will be slightly
    faster due to optimising over less parameters.

    Args:
        R (numpy.array): Financial returns data from which to calculate volatility.
        verbose (bool, optional): Whether or not to print warnings and
            convergence information.

    Returns:
        A tuple `(volatility, params)`, where `volatility` is a numpy array
        of size `len(R)` containing the calculated volatilities, and
        `params = [alpha, beta, omega, delta]` are the calculated values of the
        model parameters.

    Raises:
        RuntimeError: If the optimisation algorithm fails to converge.
    """
    N = len(R)

    # sigma^2 using asymmetric GARCH model
    def sigma_sq(n, alpha, beta, omega, delta):
        prev = 0 if (n == 0) else sigma_sq(n-1, alpha, beta, omega, delta)
        return omega + alpha * (R[n-1] - delta)**2 + beta * prev

    # Log of likelihood function
    def L(params):
        alpha, beta, omega, delta = params
        value = 0
        for n in range(N):
            sigma_n_sq = sigma_sq(n, alpha, beta, omega, delta)
            value += np.log(sigma_n_sq + (R[n]**2)/sigma_n_sq)
        value *= -0.5
        # Return -L instead of L (minimise -L => maximise L)
        return -value

    # Constrain alpha, beta, and delta to be positive
    # Use def instead of lambdas for speed
    def a_con(x):
        return np.array([x[0]])
    def b_con(x):
        return np.array([x[1]])
    def d_con(x):
        return np.array([x[3]])

    alpha_con = { 'type': 'ineq', 'fun': a_con }
    beta_con = { 'type': 'ineq', 'fun': b_con }
    delta_con = { 'type': 'ineq', 'fun': d_con }
    cons = (alpha_con, beta_con, delta_con)

    # Don't show warnings unless verbose flag specified
    if not verbose:
        warnings.filterwarnings('ignore')

    # Minimise -L to find values of alpha, beta, and omega
    res = minimize(L, [0.5, 0.5, 0.0, 0.0], constraints=cons,
                   options={ 'maxiter': 500, 'disp': verbose })

    # Turn warnings back on if they were turned off above
    if not verbose:
        warnings.resetwarnings()

    if res.success == False:
        raise RuntimeError("Unable to fit GARCH(1,1) model: " + res.message)

    # Calculate the volatility using the parameters found by minimize
    params = res.x
    volatility = np.array([sigma_sq(n, params[0], params[1], params[2], params[3]) for n in range(N)])
    volatility = np.sqrt(volatility)

    return (volatility, params)

def annualise(sigma, N=250):
    """Converts volatility into annualised volatility.

    Args:
        sigma (numpy.array): Volatilities calculated at daily (or other regular)
            intervals.
        N (int, optional): Number of trading days (or weeks, etc) in a year.
    """
    return sigma * np.sqrt(N)
