# Public libraries importations
import numpy as np

# Local importations
import utils


def lasso(X, y, ncp, ppnu):
    """_summary_

    Args:
        X (np.array): 2D-array containing the data
        y (np.array): labels vector
        ncp (float): desired number of components 
        ppnu (float): shrinking ratio
    """
    ###################################
    # Dimensions
    ###################################
    N, p = X.shape[0], X.shape[1] # nbr of observations, nbr of variables

    ###################################
    # Centering Data
    ###################################
    Xc = utils.center_matrix(X)
    yc = y - np.mean(y)

    ###################################
    # Dual-SPLS
    ###################################
    Xdef=Xc.copy() # initializing X for Deflation Step
    for k in range(ncp):
        zm = np.transpose(Xdef) @ yc # covariance vector = correlation variables vs. label vector
        print(zm)

lasso(np.array([[1, 2, 2.5], [2, 2, 2], [3, 2, 1.5]]), np.array([1, 2, 3]), 1, 1)