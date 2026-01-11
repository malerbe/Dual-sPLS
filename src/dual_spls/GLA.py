import numpy as np
import utils
import itertools

def dual_spls_gla(X, y, n_components, ppnu, indG, verbose=True):
    """Dual-SPLS Group Lasso A regression algorithm.

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
    N, p = X.shape[0], X.shape[1] # nbr of observations, nbr of variables
    
    # Specific GLA Initializations
    nG = np.max(indG) # number of groups (assuming indG is 1-based integers like in R)
    PP = np.array([np.sum(indG == u) for u in range(1, nG + 1)])
    
    WW = np.zeros((p, n_components)) # W: X weights
    TT = np.zeros((N, n_components)) # T: X scores
    listeLambda = np.zeros((nG, n_components)) # Matrix of lambda values for each group/comp
    listeAlpha = np.zeros((nG, n_components))  # Matrix of alpha values for each group/comp
    Bhat = np.zeros((p, n_components)) # Matrix to store Beta for each n_components step
    intercept = np.zeros(n_components)
    RES = np.zeros((N, n_components)) 
    zerovar = np.zeros((nG, n_components), dtype=int) # Zeros per group
    YY_pred = np.zeros((N, n_components)) # Fitted values
    ind_diff0 = {} 

    # Specific GLA temporary variables
    nu = np.zeros(nG)
    Znu = np.zeros(p)
    w_vec = np.zeros(p)
    norm2Znu = np.zeros(nG)
    norm1Znu = np.zeros(nG)

    Ec = E.copy() # Keeps original centered X for Deflation

    for k in range(n_components):
        # Step 1: dual-spls GLA logic:
        ##############################################################################
        F_col = F.reshape(-1, 1) # Ensure column vector for Z computation
        Z = np.transpose(E) @ F_col
        Z = Z.reshape(-1) # Flatten for easy indexing

        # 1. Optimizing nu(g) per group
        for ig in range(1, nG + 1):
            idx_group = np.where(indG == ig)[0] # Extract indices for this group
            
            # Sorting absolute Z values for the group to find quantile
            Zs = np.sort(np.abs(Z[idx_group]))
            d = len(Zs)
            Zsp = np.arange(1, d + 1) / d
            iz = np.argmin(np.abs(Zsp - ppnu[ig-1])) # Python 0-indexed for ppnu list
            
            nu[ig-1] = Zs[iz]

            # Finding Znu (soft thresholding logic applied per element)
            # Implemented manually here as it depends on group specific nu
            val_group = Z[idx_group]
            Znu[idx_group] = np.sign(val_group) * np.maximum(np.abs(val_group) - nu[ig-1], 0)

            # Norms per group
            norm1Znu[ig-1] = np.linalg.norm(Znu[idx_group], 1)
            norm2Znu[ig-1] = np.linalg.norm(Znu[idx_group], 2)

        #### finding mu
        mu = np.sum(norm2Znu)

        #### finding alpha and lambda
        alpha = norm2Znu / mu
        # Avoid division by zero if alpha is 0 (though unlikely if Znu has content)
        with np.errstate(divide='ignore', invalid='ignore'):
            _lambda_vec = nu / (mu * alpha)
            # max_norm2w computation
            max_norm2w = 1.0 / alpha / (1.0 + (nu * norm1Znu / (mu * alpha)**2))
        
        # Determine sampling for grid search
        # Only iterate up to nG-1 groups, last one is deduced
        ranges = []
        for ig in range(nG - 1):
             # 10 points between 0 and max_norm2w[ig]
             ranges.append(np.linspace(0, max_norm2w[ig], 10))
        
        # Create all combinations (grid)
        comb = np.array(list(itertools.product(*ranges)))
        
        # Calculate last column (nG) based on linear constraint
        # Denominator for the last group
        denom = alpha[nG-1] * (1 + (nu[nG-1] * norm1Znu[nG-1] / (mu * alpha[nG-1])**2))
        
        # Numerator calculation for all combinations
        num_terms = np.zeros((comb.shape[0], nG - 1))
        for ig_idx in range(nG - 1):
             num_terms[:, ig_idx] = alpha[ig_idx] * comb[:, ig_idx] * \
                                    (1 + (nu[ig_idx] * norm1Znu[ig_idx] / (mu * alpha[ig_idx])**2))
        
        num = 1 - np.sum(num_terms, axis=1)
        comb_last = (num / denom).reshape(-1, 1)
        
        # Add last column and filter invalid rows (< 0)
        full_comb = np.hstack((comb, comb_last))
        valid_indices = np.where(full_comb[:, -1] >= 0)[0]
        full_comb = full_comb[valid_indices, :]
        
        ncomb = full_comb.shape[0]
        RMSE = np.zeros(ncomb)
        tempw = np.zeros((p, ncomb))

        # Grid search loop for minimizing RMSE
        # Need to temporarily compute regression param to check RMSE
        for icomb in range(ncomb):
            
            # Construct w for this combination
            w_curr = np.zeros(p)
            for ig in range(1, nG + 1):
                idx_group = np.where(indG == ig)[0]
                w_curr[idx_group] = (full_comb[icomb, ig-1] / (mu * alpha[ig-1])) * Znu[idx_group]
            
            # Finding T
            t_curr = E @ w_curr
            norm_t = np.linalg.norm(t_curr)
            if norm_t > 1e-10:
                t_curr = t_curr / norm_t
            else:
                t_curr = np.zeros_like(t_curr)

            # Store temp candidates
            WW_temp = WW.copy()
            WW_temp[:, k] = w_curr
            TT_temp = TT.copy()
            TT_temp[:, k] = t_curr

            # Temporary Coefficients (Same logic as Loop Deflation end)
            W_k_temp = WW_temp[:, :k+1]
            T_k_temp = TT_temp[:, :k+1]
            
            # Backsolve logic (R style translation to numpy linear algebra)
            L_temp = np.transpose(T_k_temp) @ Ec @ W_k_temp
            L_temp = np.triu(L_temp)
            try:
                L_inv_temp = np.linalg.inv(L_temp)
            except:
                L_inv_temp = np.linalg.pinv(L_temp)
            
            bk_temp = W_k_temp @ L_inv_temp @ T_k_temp.T @ F
            bk_flat_temp = bk_temp.flatten()
            intercept_k_temp = (F_mean - E_mean @ bk_temp).item()
            
            # Quick RMSE check
            yy_pred_temp = (X @ bk_flat_temp) + intercept_k_temp
            res_temp = y.flatten() - yy_pred_temp
            RMSE[icomb] = np.sum(res_temp**2) / N
            tempw[:, icomb] = w_curr

        # Choosing the optimal w
        indwmax = np.argmin(RMSE)
        w = tempw[:, indwmax]
        
        #### Compute t (Final for this step)
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
        listeAlpha[:, k] = alpha

        # Deflate E: 
        E = E - t.reshape(-1, 1) @ (t.reshape(-1, 1).T @ E)

        W_k = WW[:, :k+1]
        T_k = TT[:, :k+1]
        L = np.transpose(T_k) @ Ec @ W_k # "backsolve"
        L = np.triu(L) # "R[row>col]=0"

        try:
            L_inv = np.linalg.inv(L)
        except:
            L_inv = np.linalg.pinv(L)

        bk = W_k @ L_inv @ T_k.T @ F
        bk_flat = bk.flatten() 
        Bhat[:, k] = bk_flat

        intercept[k] = (F_mean - E_mean @ bk).item()

        # Zero variables : count almost zero coefficients per group
        # Corresponds to: sum(Bhat[indu,ic]==0)
        for ig in range(1, nG + 1):
            idx_group = np.where(indG == ig)[0]
            is_zero_g = np.isclose(Bhat[idx_group, k], 0)
            zerovar[ig-1, k] = np.sum(is_zero_g)

        # non-zero indices 
        is_zero_global = np.isclose(bk_flat, 0)
        indices_non_zero = np.where(~is_zero_global)[0]
        ind_diff0[f"in.diff0_{k+1}"] = indices_non_zero.tolist()

        # Predictions (Fitted Values) 
        # Y_hat = X * beta + intercept
        pred_k = (X @ bk_flat) + intercept[k]
        YY_pred[:, k] = pred_k

        # Residuals
        RES[:, k] = y.flatten() - pred_k
        
        if verbose:
             print(f'Dual PLS ic={k+1} nbzeros={zerovar[:, k]}')

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
        "type": "GLA"
    }
