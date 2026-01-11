import GLA
import GLB
import GLC

def d_spls_GL(X, y, n_components, ppnu, indG, gamma=None, norm="A", verbose=False):
    """Wrapper dispatcher for Dual-SPLS Group Lasso variants."""
    
    if norm == "A":
        return GLA.dual_spls_gla(X, y, n_components, ppnu, indG)

    if norm == "B":
        return GLB.dual_spls_glb(X, y, n_components, ppnu, indG)

    if norm == "C":
        if gamma is None:
            raise ValueError("gamma is needed for GLC")
            
        return GLC.dual_spls_glc(X, y, n_components, ppnu, indG, gamma)

    return None
