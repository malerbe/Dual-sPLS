import numpy as np
import utils

def dual_spls_glb(X, y, n_components, ppnu, indG):
    """Dual-SPLS Group Lasso B regression algorithm.

    Args:
        X (np.ndarray): 2D-array containing the input data (n_samples, n_features).
        y (np.ndarray): 1D-array or vector containing the response/labels.
        n_components (int): Number of PLS components to extract.
        ppnu (list or np.ndarray): Sparsity parameters (one per group).
        indG (np.ndarray): Vector of group indices (integers) matching X columns.

    Returns:
        dict: A dictionary containing the model results matching the R list structure.
    """
    #### Center data
    E, F = X.copy(), y.copy()
    E, E_mean = utils.center_matrix(E)
    F, F_mean = utils.center_matrix(F)

    if F.ndim == 1:
        F = F.reshape(-1, 1)

    #### Initializations
    N, p = X.shape[0], X.shape[1] 
    
    # Specific GLB Initializations
    nG = np.max(indG) 
    PP = np.array([np.sum(indG == u) for u in range(1, nG + 1)])
    
    WW = np.zeros((p, n_components)) 
    TT = np.zeros((N, n_components)) 
    listeLambda = np.zeros((nG, n_components)) 
    Bhat = np.zeros((p, n_components))
    intercept = np.zeros(n_components)
    RES = np.zeros((N, n_components)) 
    zerovar = np.zeros((nG, n_components), dtype=int)
    YY_pred = np.zeros((N, n_components)) 
    ind_diff0 = {} 

    # Specific GLB temporary variables
    nu = np.zeros(nG)
    Znu = np.zeros(p)
    norm1Znu = np.zeros(nG)
    # Note: GLB uses a single global mu, not a sum of local norms like GLA roughly implies

    Ec = E.copy() 

    for k in range(n_components):
        # Step 1: dual-spls GLB logic:
        ##############################################################################
        F_col = F.reshape(-1, 1)
        Z = np.transpose(E) @ F_col
        Z = Z.reshape(-1)

        # 1. Optimizing nu(g) per group AND building Znu
        for ig in range(1, nG + 1):
            idx_group = np.where(indG == ig)[0]
            
            # --- Modification 1: Adaptative nu (Same as GLA) ---
            Zs = np.sort(np.abs(Z[idx_group]))
            d = len(Zs)
            Zsp = np.arange(1, d + 1) / d
            iz = np.argmin(np.abs(Zsp - ppnu[ig-1])) 
            
            nu[ig-1] = Zs[iz]

            # --- Modification 2: Soft Thresholding ---
            val_group = Z[idx_group]
            Znu[idx_group] = np.sign(val_group) * np.maximum(np.abs(val_group) - nu[ig-1], 0)

            # Norms per group
            norm1Znu[ig-1] = np.linalg.norm(Znu[idx_group], 1)
            # Note: GLB doesn't store norm2Znu per group for calculation, mainly norm1

        # --- Modification 3: Global Parameters (Norm B specific) ---
        
        # mu is the global L2 norm of the thresholded vector Znu
        mu = np.linalg.norm(Znu, 2)
        mu2 = mu**2

        # lambda calculation per group 
        # (Handle division by zero if mu is 0, though unlikely with data)
        if mu > 1e-15:
            _lambda_vec = nu / mu
        else:
            _lambda_vec = np.zeros(nG)

        # --- Modification 4: Calculating w at the optimum ---
        # Formula: w = (mu / (mu^2 + nu^T @ norm1Znu)) * Znu
        
        dot_nu_norm1 = np.dot(nu, norm1Znu)
        denominator = mu2 + dot_nu_norm1

        if denominator > 1e-15:
            scaling_factor = mu / denominator
            w = scaling_factor * Znu
        else:
            w = np.zeros(p)
            
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
        listeLambda[:, k] = _lambda_vec

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

    return {
        "Xmean": E_mean,
        "scores": TT,
        "loadings": WW,
        "Bhat": Bhat,
        "intercept": intercept,
        "fitted_values": YY_pred,
        "residuals": RES,
        "lambda": listeLambda,
        "zerovar": zerovar,
        "PP": PP,
        "ind_diff0": ind_diff0,
        "type": "GLB"
    }
