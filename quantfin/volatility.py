import warnings
import numpy as np
import scipy.special
from scipy.optimize import minimize

def moving_average(R, M):
    """
    Calculates volatility using an M-day moving average
    :param R: Returns to calculate volatility from
    :param M: Number of days to use for the moving average (assuming returns are daily)
    :returns: An array of size (len(R) - M) containing the calculated volatilities
    """
    N = len(R)
    sigma_sq = np.zeros(N - M)
    for i in range(M, N):
        sigma_sq[i - M] = R[i-M:i].std() ** 2
        
    return sigma_sq


def ewma(R, smoothing=0.98):
    """
    Calculates volatility using an exponentially weighted moving average
    :param R: Returns to calculate volatility from
    :param smoothing: Smoothing parameter (lambda)
    :returns: An array of size len(R) containing the calculated volatilities
    """
    N = len(R)
    sigma_sq = np.zeros(N)
    for n in range(N):
        vol = np.array([(smoothing ** i) * (R[n-i] ** 2) for i in range(n)]).sum()
        vol *= (1 - smoothing)
        sigma_sq[n] = vol
    
    return sigma_sq


def garch(R, verbose=False):
    """
    Calculates volatility using the GARCH(1,1) model
    Warning: This slows down rapidly as R increases in size due to the recursive
    nature of GARCH
    :param R: Returns to calculate volatility from
    :param verbose: Whether or not to print warnings and convergence information
    :returns: A tuple (volatility, params), where params = [alpha, beta, omega] 
    are the calculated values of the model parameters
    :raises RuntimeError: When the optimisation fails to converge 
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
    
    return (volatility, params)
    
    