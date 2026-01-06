import numpy as np

####### Norms computation:
def norm1(x):
    """Compute L1-norm of a vector

    Args:
        x (numpy.array): vector

    Returns:
        float : L1-norm of the vector
    """
    return np.linalg.norm(x, ord=1)

def norm2(x):
    """Compute L2-norm of a vector

    Args:
        x (numpy.array): vector

    Returns:
        float : L2-norm of the vector
    """
    return np.linalg.norm(x, ord=2)

####### Soft threshold:
def soft_threshold(Z, nu):
    """Compute the result of a soft threshold on an array

    Args:
        Z (np.array): array
        u (float): threshold
    """
    return np.vectorize(lambda u: np.sign(u)*max(abs(u)-nu, 0))(Z)

def center_matrix(M):
    """Center matrix computing means on columns

    Args:
        M (np.array): 2D-matrix

    Returns:
        np.array: centered 2D-matrix
    """
    means = np.mean(M, axis=0)
    return M - means
