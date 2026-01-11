import numpy as np
import utils

def dual_spls_lasso(X, y, n_components=3, ppnu=0.8):
    """Dual-SPLS Lasso regression algorithm.

    Args:
        X (np.ndarray): 2D-array containing the input data (n_samples, n_features).
        y (np.ndarray): 1D-array or vector containing the response/labels.
        n_components (int, optional): Number of PLS components to extract. Defaults to 3.
        ppnu (float, optional): Sparsity parameter (0 < ppnu < 1) defining the quantile threshold. Higher values imply more sparsity. Defaults to 0.8.

    Returns:
        dict: A dictionary containing the model results: 
            coefficients ('Bhat', 'intercept'),
            'scores', 
            'loadings', 
            'fitted_values', 
            and residuals.
    """
    #### Center data
    E, F = X.copy(), y.copy()
    E, E_mean = utils.center_matrix(E)
    F, F_mean = utils.center_matrix(F)

    if F.ndim == 1:
        F = F.reshape(-1, 1)

    #### Initializations
    N, p = X.shape[0], X.shape[1] # nbr of observations, nbr of variables
    WW = np.zeros((p, n_components)) # W: X weights
    TT = np.zeros((N, n_components)) # T: X scores
    listeLambda = np.zeros((n_components))
    Bhat = np.zeros((p, n_components)) # Matrix to store Beta for each n_components step
    intercept = np.zeros(n_components)
    RES = np.zeros((N, n_components)) 
    zerovar = np.zeros(n_components, dtype=int)
    YY_pred = np.zeros((N, n_components)) # Fitted values
    ind_diff0 = {} 

    Ec = E.copy()

    for k in range(n_components):
        # Step 1: base dual-spls:
        ##############################################################################
        F = F.reshape(-1, 1)

        z = np.transpose(E) @ F  

        #### Modification 1: compute the adaptative nu
        nu = np.quantile(np.abs(z), ppnu)

        #### Modification 2: different soft-thresholding
        z_nu = utils.soft_thresholding(z, nu)

        #### Modification 3: Find paramters 
        z_nu_1 = np.linalg.norm(z_nu, 1)
        z_nu_2 = np.linalg.norm(z_nu, 2)

        mu=z_nu_2 
        _lambda = nu/mu

        #### Compute w
        scaling_factor = mu / (nu * z_nu_1 + mu**2) # Scaling factor, see theory
        w = scaling_factor*z_nu

        #### Compute t, same as sPLS
        t = E @ w

        # + Modification 4: Normalize t instead of w and c 
        norm_t = np.linalg.norm(t)
        if norm_t > 1e-10:
            t = t / norm_t
        else:
            t = np.zeros_like(t)

        # c is not used by the R package


        ##############################################################################

        # Store results
        WW[:, k], TT[:, k] = w.reshape(-1), t.reshape(-1)
        listeLambda[k] = _lambda

        # Deflate E: 
        E = E - t @ (t.T @ E)

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

        # Zero variables : count almost zero coefficients
        is_zero = np.isclose(bk_flat, 0)
        zerovar[k] = np.sum(is_zero)

        # non-zero indices 
        indices_non_zero = np.where(~is_zero)[0]
        ind_diff0[f"in.diff0_{k+1}"] = indices_non_zero.tolist()

        # Predictions (Fitted Values) 
        # Y_hat = X * beta + intercept
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
        "ind_diff0": ind_diff0,
        "type": "lasso"
    }
