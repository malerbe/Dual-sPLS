import numpy as np
from dual_spls import utils

def dual_spls_glc(X, y, n_components, ppnu, indG, gamma, noise_level=1e-6, verbose=True):
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
        raise ValueError(f"gamma length must match nG ({nG})")
    if not np.isclose(np.sum(gamma), 1.0):
        raise ValueError(f"gamma must sum to 1")

    E = X.copy()
    F = y.copy().reshape(-1,1)
    E_mean = E.mean(axis=0)
    F_mean = F.mean(axis=0)
    E -= E_mean
    F -= F_mean

    N, p = X.shape
    PP = np.array([np.sum(indG==u) for u in range(1,nG+1)])

    WW = np.zeros((p,n_components))
    TT = np.zeros((N,n_components))
    Bhat = np.zeros((p,n_components))
    YY_pred = np.zeros((N,n_components))
    RES = np.zeros((N,n_components))
    intercept = np.zeros(n_components)
    zerovar = np.zeros((nG,n_components),dtype=int)
    listeLambda = np.zeros((nG,n_components))
    listeAlpha = np.zeros((nG,n_components))
    ind_diff0 = {}

    Ec = E.copy()

    for k in range(n_components):
        Z = (E.T @ F).flatten()
        nu = np.zeros(nG)
        Znu = np.zeros(p)
        norm1Znu = np.zeros(nG)
        norm2Znu = np.zeros(nG)
        w = np.zeros(p)

        for ig in range(1,nG+1):
            idx_group = np.where(indG==ig)[0]
            if len(idx_group)==0: continue
            Zs = np.sort(np.abs(Z[idx_group]))
            d = len(Zs)
            Zsp = np.arange(1,d+1)/d
            iz = np.argmin(np.abs(Zsp - ppnu[ig-1]))
            nu[ig-1] = Zs[iz]
            val_group = Z[idx_group]
            Znu[idx_group] = np.sign(val_group) * np.maximum(np.abs(val_group)-nu[ig-1],0)
            norm1Znu[ig-1] = np.linalg.norm(Znu[idx_group],1)
            norm2Znu[ig-1] = np.linalg.norm(Znu[idx_group],2)

        mu = max(np.sum(norm2Znu),1e-6)
        alpha = np.maximum(norm2Znu/mu, 1e-6)
        listeAlpha[:,k] = alpha
        listeLambda[:,k] = nu / mu

        for ig in range(1,nG+1):
            idx_group = np.where(indG==ig)[0]
            if len(idx_group)==0: continue
            numerator = gamma[ig-1] * Znu[idx_group]
            denominator = alpha[ig-1]*norm2Znu[ig-1] + listeLambda[ig-1,k]*norm1Znu[ig-1]
            w[idx_group] = numerator / max(denominator,1e-12)

        # Add small noise & normalize
        w += np.random.normal(0, noise_level, size=p)
        w /= max(np.linalg.norm(w), 1e-6)

        t = E @ w
        t /= max(np.linalg.norm(t), 1e-6)
        WW[:,k] = w
        TT[:,k] = t

        # Deflation
        E -= t.reshape(-1,1) @ (t.reshape(-1,1).T @ E)

        # Backsolve
        W_k = WW[:, :k+1]
        T_k = TT[:, :k+1]
        L = np.triu(T_k.T @ Ec @ W_k)
        try: L_inv = np.linalg.inv(L)
        except: L_inv = np.linalg.pinv(L)
        bk = W_k @ L_inv @ T_k.T @ F
        bk_flat = bk.flatten()
        Bhat[:,k] = bk_flat
        intercept[k] = (F_mean - E_mean @ bk).item()

        for ig in range(1,nG+1):
            idx_group = np.where(indG==ig)[0]
            zerovar[ig-1,k] = np.sum(np.isclose(Bhat[idx_group,k],0))

        ind_diff0[f"in.diff0_{k+1}"] = np.where(~np.isclose(bk_flat,0))[0].tolist()
        YY_pred[:,k] = (X @ bk_flat) + intercept[k]
        RES[:,k] = y.flatten() - YY_pred[:,k]

        if verbose:
            print(f"Dual PLS GLC ic={k+1}, nbzeros={np.sum(np.isclose(w,0))}, max|w|={np.max(np.abs(w))}")

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
