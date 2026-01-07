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

    mean_X = np.mean(X, axis=0) 
    mean_y = np.mean(y)

    yc = y - mean_y

    ###################################
    # Dual-SPLS
    ###################################
    WW = np.zeros((p,ncp)) # initialising WW, the matrix of loadings
    TT = np.zeros((N,ncp)) # initialising TT, the matrix of scores
    Bhat = np.zeros((p, ncp)) # Matrix to store Beta for each ncp step
    Intercept = np.zeros(ncp) # To store intercepts

    # Additionnal stuff to check the results (see R code) source @ Gemini
    YY = np.zeros((N, ncp))        # Valeurs prédites (Fitted values)
    RES = np.zeros((N, ncp))       # Résidus (y - y_pred)
    zerovar = np.zeros(ncp)        # Nombre de variables à zéro
    listelambda = np.zeros(ncp)    # Valeur de lambda à chaque itération
    ind_diff0 = []

    Xdef=Xc.copy() # initializing X for Deflation Step
    for k in range(ncp):
        zm = np.transpose(Xdef) @ yc # covariance vector = correlation variables vs. label vector

        # Optimize nu adaptively according to the shrinking ratio
        # See Figure 1 in the paper
        abs_zm = np.sort(abs(zm)) # firstly, take the abs value for all the correlations within zm and sort them
        nu = np.quantile(abs_zm, ppnu, method='lower') 

        # Compute z_nu applying the threshold
        z_nu = utils.soft_threshold(zm, nu)

        # Compute mu and _lambda according to the result:
        mu = utils.norm2(z_nu) # needed to compute the weight vector w
        _lambda = nu/mu # regularization paramter corresponding to what we usually use instead of the dual method

        # Compute w:
        w = (mu/(nu * utils.norm1(z_nu) + mu**2))*z_nu
        
        # Build W k-th column for this iteration:
        WW[:, k] = w

        # Build T, the matrix of scores
        t = Xdef @ w
        t=t/utils.norm2(t)
        TT[:, k] = t

        # Deflation
        t = t.reshape(-1, 1) # prepare t to be in the right dimension (N, ) -> (N, 1)
        Xdef = Xdef - (t @ np.transpose(t) @ Xdef)

        ## Compute intermediate Beta = W * (T^T@T)^-1 * T'y
        # Current matrixes
        W_curr = WW[:, :k+1] 
        T_curr = TT[:, :k+1]

        # T^T@T
        Gram = T_curr.T @ T_curr

        # Invert --> (T^T@T)^-1
        inv_Gram = np.linalg.pinv(Gram) 

        # Computation and storage:
        beta_k = W_curr @ inv_Gram @ T_curr.T @ yc
        Bhat[:, k] = beta_k

        ## Compute intercept:
        Intercept[k] = mean_y - (mean_X @ beta_k)

        ## Predictions (YY) et Residus (RES) source @ Gemini
        y_pred = X @ beta_k + Intercept[k]
        YY[:, k] = y_pred
        RES[:, k] = y - y_pred
        nb_zeros = np.sum(np.abs(beta_k) < 1e-10) # Compte les quasi-zéros 
        zerovar[k] = nb_zeros
        
        # Indices des variables NON nulles (ce que le chercheur veut savoir)
        indices_non_null = np.where(np.abs(beta_k) > 1e-10)[0]
        ind_diff0.append(indices_non_null)

        print(f"Dual PLS component={k+1} lambda={_lambda:.4f} mu={mu:.4f} nu={nu:.4f} nbzeros={int(nb_zeros)}")

    return {
        "Xmean": mean_X,
        "scores": TT,
        "loadings": WW,
        "Bhat": Bhat,
        "intercept": Intercept,
        "fitted_values": YY,
        "residuals": RES,
        "lambda": listelambda,
        "zerovar": zerovar,
        "ind_diff0": ind_diff0,
        "type": "lasso"
    }

print(lasso(np.array([[1, 2, 2.5], [2, 2, 2], [3, 2, 1.5]]), np.array([1, 2, 3]), 2, 0.8))