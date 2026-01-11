import numpy as np
import utils

def dual_spls_glc(X, y, n_components, ppnu, indG, gamma, verbose=True):
    """Dual-SPLS Group Lasso C regression algorithm.

    Args:
        X (np.ndarray): 2D-array containing the input data (n_samples, n_features).
        y (np.ndarray): 1D-array or vector containing the response/labels.
        n_components (int): Number of PLS components to extract.
        ppnu (list or np.ndarray): Sparsity parameters (one per group).
        indG (np.ndarray): Vector of group indices (integers) matching X columns.
        gamma (list or np.ndarray): Vector of weights for each group (must sum to 1).

    Returns:
        dict: A dictionary containing the model results matching the R list structure.
    """
    
    #### Specific Validation for GLC
    nG = np.max(indG)
    if len(gamma) != nG:
        raise ValueError(f"Incorrect length of gamma. Expected {nG}, got {len(gamma)}")
    
    if not np.isclose(np.sum(gamma), 1.0):
        raise ValueError(f"Sum of gamma must be 1. Got {np.sum(gamma)}")

    #### Center data
    E, F = X.copy(), y.copy()
    E, E_mean = utils.center_matrix(E)
    F, F_mean = utils.center_matrix(F)

    if F.ndim == 1:
        F = F.reshape(-1, 1)

    #### Initializations
    N, p = X.shape[0], X.shape[1] 
    
    # Intializations
    PP = np.array([np.sum(indG == u) for u in range(1, nG + 1)])
    
    WW = np.zeros((p, n_components)) 
    TT = np.zeros((N, n_components)) 
    listeLambda = np.zeros((nG, n_components)) 
    listeAlpha = np.zeros((nG, n_components)) # Algo uses Alpha explicitly here
    Bhat = np.zeros((p, n_components))
    intercept = np.zeros(n_components)
    RES = np.zeros((N, n_components)) 
    zerovar = np.zeros((nG, n_components), dtype=int)
    YY_pred = np.zeros((N, n_components)) 
    ind_diff0 = {} 

    # Temporary variables
    nu = np.zeros(nG)
    Znu = np.zeros(p)
    w = np.zeros(p)
    norm1Znu = np.zeros(nG)
    norm2Znu = np.zeros(nG)
    
    Ec = E.copy() 

    for k in range(n_components):
        # Step 1: dual-spls GLC logic:
        ##############################################################################
        F_col = F.reshape(-1, 1)
        Z = np.transpose(E) @ F_col
        Z = Z.reshape(-1)

        # 1. Optimizing nu(g) per group AND building Znu
        for ig in range(1, nG + 1):
            idx_group = np.where(indG == ig)[0]
            
            # --- Adaptative nu ---
            Zs = np.sort(np.abs(Z[idx_group]))
            d = len(Zs)
            Zsp = np.arange(1, d + 1) / d
            iz = np.argmin(np.abs(Zsp - ppnu[ig-1])) 
            
            nu[ig-1] = Zs[iz]

            # --- Soft Thresholding ---
            val_group = Z[idx_group]
            Znu[idx_group] = np.sign(val_group) * np.maximum(np.abs(val_group) - nu[ig-1], 0)

            # Norms per group
            norm1Znu[ig-1] = np.linalg.norm(Znu[idx_group], 1)
            norm2Znu[ig-1] = np.linalg.norm(Znu[idx_group], 2)

        # --- Modification 1: Global Parameters (Norm C specific) ---
        # mu is sum of local L2 norms
        mu = np.sum(norm2Znu)

        # --- Modification 2: Calculating w with gamma weights ---
        # Iterate again to compute w based on the computed mu
        for ig in range(1, nG + 1):
            idx_group = np.where(indG == ig)[0]
            
            if mu > 1e-15:
                alpha = norm2Znu[ig-1] / mu
                _lambda = nu[ig-1] / mu
            else:
                alpha = 0
                _lambda = 0
            
            # Store parameters
            listeAlpha[ig-1, k] = alpha
            listeLambda[ig-1, k] = _lambda

            # Formula: w_g = (gamma_g * Znu_g) / (alpha_g * norm2 + lambda_g * norm1)
            numerator = gamma[ig-1] * Znu[idx_group]
            denominator = (alpha * norm2Znu[ig-1]) + (_lambda * norm1Znu[ig-1])
            
            if denominator > 1e-15:
                w[idx_group] = numerator / denominator
            else:
                w[idx_group] = 0.0

        #### Compute t
        t = E @ w
        norm_t = np.linalg.norm(t)
        if norm_t > 1e-10:
            t = t / norm_t
        else:
            t = np.zeros_like(t)

        ##############################################################################

        # Store results
        WW[:, k], TT[:, k] = w.reshape(-1), t.reshape(-1)
        
        # Deflate E: 
        E = E - t.reshape(-1, 1) @ (t.reshape(-1, 1).T @ E)

        W_k = WW[:, :k+1]
        T_k = TT[:, :k+1]
        L = np.transpose(T_k) @ Ec @ W_k # "backsolve"
        L = np.triu(L) 

        try:
            L_inv = np.linalg.inv(L)
        except:
            L_inv = np.linalg.pinv(L)

        bk = W_k @ L_inv @ T_k.T @ F
        bk_flat = bk.flatten() 
        Bhat[:, k] = bk_flat

        intercept[k] = (F_mean - E_mean @ bk).item()

        # Zero variables : count almost zero coefficients per group
        for ig in range(1, nG + 1):
            idx_group = np.where(indG == ig)[0]
            is_zero_g = np.isclose(Bhat[idx_group, k], 0)
            zerovar[ig-1, k] = np.sum(is_zero_g)

        # non-zero indices 
        is_zero_global = np.isclose(bk_flat, 0)
        indices_non_zero = np.where(~is_zero_global)[0]
        ind_diff0[f"in.diff0_{k+1}"] = indices_non_zero.tolist()

        # Predictions
        pred_k = (X @ bk_flat) + intercept[k]
        YY_pred[:, k] = pred_k

        # Residuals
        RES[:, k] = y.flatten() - pred_k
        
        if verbose:
             print(f'Dual PLS GL-C ic={k+1} nbzeros={zerovar[:, k]}')

    return {
        "Xmean": E_mean,
        "scores": TT,
        "loadings": WW,
        "Bhat": Bhat,
        "intercept": intercept,
        "fitted_values": YY_pred,
        "residuals": RES,
        "lambda": listeLambda,
        "alpha": listeAlpha,
        "zerovar": zerovar,
        "PP": PP,
        "ind_diff0": ind_diff0,
        "type": "GLC"
    }
