import numpy as np
import utils

def d_spls_ridge(X, y, n_components, ppnu, nu2, verbose=True):
    """
    Dual-SPLS Ridge regression algorithm.
    
    Combines L2 regularization (Ridge) on the covariance structure with 
    L1-bases sparsity (Soft-Thresholding) on the weights.

    Args:
        X (np.ndarray): Predictor matrix (n_samples, n_features).
        y (np.ndarray): Response vector (n_samples,).
        n_components (int): Number of components.
        ppnu (float): Sparsity parameter (percentile).
        nu2 (float): Ridge regularization parameter (weight of L2 norm).
        verbose (bool): Whether to print progress.

    Returns:
        dict: Standard Dual-SPLS results dictionary.
    """
    
    #### Center data
    E, F = X.copy(), y.copy()
    E, E_mean = utils.center_matrix(E)
    F, F_mean = utils.center_matrix(F)

    if F.ndim == 1:
        F = F.reshape(-1, 1)

    #### Initializations
    N, p = X.shape[0], X.shape[1] 
    
    WW = np.zeros((p, n_components)) 
    TT = np.zeros((N, n_components)) 
    Bhat = np.zeros((p, n_components))
    YY_pred = np.zeros((N, n_components)) 
    RES = np.zeros((N, n_components)) 
    intercept = np.zeros(n_components)
    zerovar = np.zeros(n_components, dtype=int)
    listelambda1 = np.zeros(n_components)
    ind_diff0 = {} 

    # Copy X for deflation steps
    Xi = E.copy() 

    for k in range(n_components):
        
        # z = Xi.T * y_centered
        zi = Xi.T @ F
        zi = zi.flatten()

        ######################################################################
        # Ridge Inversion Step
        ######################################################################
        # Computing (nu2 * Xi^T Xi + I)^(-1)
        # Note: In high dimensions (p > n), this is computationally expensive (p*p matrix).
        XtX = Xi.T @ Xi
        
        # Ridge matrix construction: nu2 * XtX + I
        # np.eye(p) creates the identity matrix of size p
        temp_matrix = (nu2 * XtX) + np.eye(p)
        
        # Inversion. 
        # R uses SVD to invert, which is robust. 
        # Since temp_matrix is Symmetric Positive Definite (due to +I), 
        # standard inv is usually fine, but let's use robust inversion if needed.
        try:
            inv_matrix = np.linalg.inv(temp_matrix)
        except np.linalg.LinAlgError:
            inv_matrix = np.linalg.pinv(temp_matrix)
        
        # delta calculation (sign of z) - unused in R logic flow but computed
        # delta = np.sign(zi) 
        
        # ZN = inv * z
        ZN = inv_matrix @ zi

        ######################################################################
        # Optimizing nu and finding weights
        ######################################################################
        
        # Sort absolute values to find threshold
        ZN_abs = np.abs(ZN)
        nu1 = utils.quantile_d(ZN_abs, ppnu)
        
        # z12 = Soft Thresholding of ZN by nu1
        z12 = np.sign(ZN) * np.maximum(ZN_abs - nu1, 0)
        
        # Compute mu = Norm2(z12)
        mu = np.linalg.norm(z12)
        
        if mu > 0:
            lambda1 = nu1 / mu
        else:
            lambda1 = 0.0
            
        # Calculating w scalar factor
        # w = (mu / (nu2 * ||Xi z12||^2 + nu1 * ||z12||_1 + mu^2)) * z12
        
        if mu > 0:
            # Norm2 of X * z12
            norm_Xz12 = np.linalg.norm(Xi @ z12)
            
            # Norm1 of z12
            norm1_z12 = np.sum(np.abs(z12))
            
            denominator = (nu2 * (norm_Xz12**2)) + (nu1 * norm1_z12) + (mu**2)
            
            w = (mu / denominator) * z12
        else:
            w = z12 # is zero vector

        ######################################################################
        # Deflation and Storage
        ######################################################################
        
        WW[:, k] = w
        listelambda1[k] = lambda1
        
        # Compute t
        t = Xi @ w
        norm_t = np.linalg.norm(t)
        if norm_t > 1e-10:
            t = t / norm_t
        else:
            t = np.zeros_like(t)
        TT[:, k] = t.reshape(-1)

        # Deflation: Xi = Xi - t t^T Xi
        Xi = Xi - t.reshape(-1, 1) @ (t.reshape(-1, 1).T @ Xi)

        # Coefficients calculation (Backsolve strategy)
        W_k = WW[:, :k+1]
        T_k = TT[:, :k+1]
        
        # R matrix
        R_mat = T_k.T @ E @ W_k
        R_mat = np.triu(R_mat) 

        try:
            L_inv = np.linalg.inv(R_mat)
        except:
            L_inv = np.linalg.pinv(R_mat)

        # Bhat
        bk = W_k @ L_inv @ T_k.T @ F
        bk_flat = bk.flatten() 
        Bhat[:, k] = bk_flat

        intercept[k] = (F_mean - E_mean @ bk).item()
        
        # Count zeros and find indices
        is_zero = np.isclose(bk_flat, 0)
        zerovar[k] = np.sum(is_zero)
        ind_diff0[f"in.diff0_{k+1}"] = np.where(~is_zero)[0].tolist()

        # Predictions
        pred_k = (X @ bk_flat) + intercept[k]
        YY_pred[:, k] = pred_k
        RES[:, k] = y.flatten() - pred_k

        if verbose:
            print(f"Dual PLS Ridge ic={k+1} lambda1={lambda1:.4f} mu={mu:.4f} nu2={nu2} nbzeros={zerovar[k]}")

    return {
        "Xmean": E_mean,
        "scores": TT,
        "loadings": WW,
        "Bhat": Bhat,
        "intercept": intercept,
        "fitted_values": YY_pred,
        "residuals": RES,
        "lambda1": listelambda1,
        "zerovar": zerovar,
        "ind_diff0": ind_diff0,
        "type": "ridge"
    }
