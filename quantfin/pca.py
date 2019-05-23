"""Principal component analysis."""

import numpy as np

def pca(data, epsilon=1e-5):
    """Principal component analysis.

    Performs PCA using the NIPALS algorithm.

    Args:
        data (numpy.array): A matrix of data on which to perform PCA.
        epsilon (float, optional): Required accuracy for convergence.

    Returns:
        A tuple `(T, P)`, where T and P are matrices. The columns of T are the
        principal components, and P is the matrix of weights.
    """
    # Standardise data
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    X = (data - mu) / sigma

    # Create empty T and P matrices
    N, M = X.shape
    T = np.zeros((N, M))
    P = np.zeros((M, M))

    # NIPALS algorithm
    for j in range(M):
        t_old = X[:, j] # Initial guess
        while True:
            p = X.T @ t_old
            p /= np.linalg.norm(p)
            t_new = X @ p
            delta = np.linalg.norm(t_old - t_new) ** 2
            if delta < epsilon: # Required accuracy reached
                break
            t_old = t_new # Update guess for t

        # Vectors t and p are columns of T and P
        T[:, j] = t_new
        P[:, j] = p

        # Need to convert arrays of shape (N,) -> (N,1) for matmul
        X = X - (np.expand_dims(t_new, 1) @ np.expand_dims(p, 1).T)

    return T, P
