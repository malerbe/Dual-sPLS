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
    WW = np.zeros((p,ncp)) # initialising WW, the matrix of loadings

    Xdef=Xc.copy() # initializing X for Deflation Step
    for k in range(ncp):
        zm = np.transpose(Xdef) @ yc # covariance vector = correlation variables vs. label vector
        print(zm)

        # Optimize nu adaptively according to the shrinking ratio
        # See Figure 1 in the paper
        abs_zm = np.sort(abs(zm)) # firstly, take the abs value for all the correlations within zm and sort them
        print(abs_zm)
        nu = np.quantile(abs_zm, ppnu, method='lower') 
        print(nu)

        # Compute z_nu applying the threshold
        z_nu = utils.soft_threshold(zm, nu)
        print(z_nu)

        # Compute mu and _lambda according to the result:
        mu = utils.norm2(z_nu) # needed to compute the weight vector w
        _lambda = nu/mu # regularization paramter corresponding to what we usually use instead of the dual method
        print(_lambda, mu)

        # Compute w:
        w = (mu/(nu * utils.norm1(z_nu) + mu**2))*z_nu
        print("w:", w)
        
        # Build W k-th column for this iteration:
        WW[:, k] = w

        print(WW)


        


lasso(np.array([[1, 2, 2.5], [2, 2, 2], [3, 2, 1.5]]), np.array([1, 2, 3]), 1, 0.8)