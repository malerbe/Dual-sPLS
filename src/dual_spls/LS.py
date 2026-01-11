import numpy as np
import utils

def d_spls_ls(X, y, n_components, ppnu, verbose=True):
    """
    Dual-SPLS Least Squares (LS) regression algorithm.
    
    This variant approximates the weight vector using a generalized inverse 
    of the covariance matrix (XtX)^-1 applied to the correlation vector, 
    followed by soft-thresholding.

    Args:
        X (np.ndarray): Predictor matrix (n_samples, n_features).
        y (np.ndarray): Response vector (n_samples,).
        n_components (int): Number of components.
        ppnu (float): Sparsity parameter (percentile for thresholding).
        verbose (bool): Whether to print progress.

    Returns:
        dict: Methods results (scores, loadings, coefficients, etc.).
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
    listelambda = np.zeros(n_components)
    ind_diff0 = {} 

    # Copy X for deflation steps
    Xi = E.copy() 

    for k in range(n_components):
        
        # Correlation vector z = Xi.T * y_centered
        zi = Xi.T @ F
        zi = zi.flatten()

        ######################################################################
        # Computed (Xi^T Xi)^-1 via SVD to handle singularity
        ######################################################################
        # SVD of Xi: Xi = U @ S @ Vt
        # Then (Xi^T Xi)^-1 = V @ S^-2 @ V^T
        U, s, Vt = np.linalg.svd(Xi, full_matrices=False)
        V = Vt.T
        
        # Inversion of singular values with stability check
        # R code logic: checks if smallest singular values are negligible relative to max
        # and zeros them out (especially for deflated matrices).
        max_s = np.max(s)
        invD = np.zeros_like(s)
        
        # Threshold similar to R's "1e-16" ratio check
        tolerance = 1e-16
        
        # We compute 1/s, setting very small values to 0 effectively
        # (simulating the pseudo-inverse logic of the R code)
        valid_indices = s > (max_s * tolerance)
        invD[valid_indices] = 1.0 / s[valid_indices]
        
        # If we are strictly following R's deflation logic:
        # The rank decreases by 1 at each step. We essentially force 
        # the reciprocal of the smallest singular values to 0.
        if k > 0:
            # In R: invD[((len-ic+2):len)] = 0
            # This masks the smallest singular values corresponding to deflation directions.
            # In Python (sorted desc): mask the last k entries if they are small.
            # We add a check to be safe, but SVD usually puts small values at the end.
            invD[-(k):] = 0.0 # Force zeros on the theoretical null space of deflation

        # Compute XtX_inv = V @ diag(s^-2) @ V^T
        XtX_inv = V @ np.diag(invD**2) @ Vt

        ######################################################################
        # Optimizing nu and finding weights
        ######################################################################
        
        # w_LS = (XtX)^-1 * z
        w_ls = XtX_inv @ zi
        
        # Determine threshold nu based on 'ppnu' percentile
        w_ls_abs = np.abs(w_ls)
        nu =  utils.quantile_d(w_ls_abs, ppnu)
        
        # Soft thresholding to get znu
        # znu = sign(w_ls) * max(|w_ls| - nu, 0)
        znu = np.sign(w_ls) * np.maximum(w_ls_abs - nu, 0)
        
        # Calculate mu and lambda (for consistency with framework output)
        XZnu = Xi @ znu
        mu = np.linalg.norm(XZnu) # Euclidean norm
        
        if mu > 0:
            val_lambda = nu / mu
        else:
            val_lambda = 0.0
            
        # Weights w are directly the thresholded vector
        w = znu.copy()

        # Compute t
        t = Xi @ w
        norm_t = np.linalg.norm(t)
        if norm_t > 1e-10:
            t = t / norm_t
        else:
            t = np.zeros_like(t)

        ######################################################################
        # Deflation and Storage (Standard PLS)
        ######################################################################

        # Store results
        WW[:, k], TT[:, k] = w, t.reshape(-1)
        listelambda[k] = val_lambda

        # Deflation: Xi = Xi - t t^T Xi
        Xi = Xi - t.reshape(-1, 1) @ (t.reshape(-1, 1).T @ Xi)

        # Coefficients calculation (Backsolve strategy)
        W_k = WW[:, :k+1]
        T_k = TT[:, :k+1]
        
        # R matrix from code: t(TT) @ Xc @ WW
        R_mat = T_k.T @ E @ W_k
        # Enforce upper triangular structure for stability (as in R)
        R_mat = np.triu(R_mat) 

        try:
            L_inv = np.linalg.inv(R_mat)
        except:
            L_inv = np.linalg.pinv(R_mat)

        # Bhat = WW * (R^-1) * TT^T * yc
        bk = W_k @ L_inv @ T_k.T @ F
        bk_flat = bk.flatten() 
        Bhat[:, k] = bk_flat

        intercept[k] = (F_mean - E_mean @ bk).item()
        
        # Count zeros
        is_zero = np.isclose(bk_flat, 0)
        zerovar[k] = np.sum(is_zero)
        
        # Store non-zero indices
        ind_diff0[f"in.diff0_{k+1}"] = np.where(~is_zero)[0].tolist()

        # Predictions
        pred_k = (X @ bk_flat) + intercept[k]
        YY_pred[:, k] = pred_k
        RES[:, k] = y.flatten() - pred_k

        if verbose:
            print(f"Dual PLS LS, ic={k+1} nu={nu:.4f} nbzeros={zerovar[k]}")

    return {
        "Xmean": E_mean,
        "scores": TT,
        "loadings": WW,
        "Bhat": Bhat,
        "intercept": intercept,
        "fitted_values": YY_pred,
        "residuals": RES,
        "lambda": listelambda,
        "zerovar": zerovar,
        "ind_diff0": ind_diff0,
        "type": "LS"
    }
