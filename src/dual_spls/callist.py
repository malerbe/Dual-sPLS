import numpy as np
from dual_spls.get_bin_indices import get_bin_indices

def get_calList(bin_indices, pcal, n_bins):
    """Determine the number of observations to select from each cell splitted by type.

    This will potentially use the whole available data in a cell and not keep any for the
    validation. 

    Args:
        datatype (list): y-sized array. For i in len(y), datatype[i] gives the split to which the
                    i-th element of y is attributed. Typically computed by type (see type.py)
        pcal (int): a positive integer between 0 and 100. pcal is the percentage
                    of calibration samples to be selected.
    """
    assert (pcal >= 0) and (pcal <= 100), "pcal must be within [0, 100]"

    n = len(bin_indices)

    # count the number of occurence for each index
    y_counts = np.bincount(bin_indices.astype(int), minlength=n_bins)

    l = np.zeros(n_bins)

    rep = np.floor(n*pcal/100) # number of observations to select

    i = 0
    while (i < rep): # select observations until the right amount is selected
        for j in range(n_bins): # for each cell 
            
            if y_counts[j] > 0 and i < rep:
                # if there is at least one observation available 
                y_counts[j] -= 1 # "take it" from original datatype
                l[j]+=1 # "put it" in the callist
                i+=1
        
    callist = l
    return callist.astype(int)

if __name__ == "__main__":
    from dual_spls.simulate import simulate
    ## Configuration:
    # Simulation configuration:
    coefs = [0, 1, 0, 1, 1, 0, 0, 1, 1] 
    n = 300
    p = [1000]
    nondes = [30]
    sigmaondes = [0.40]
    sigma_y = 0.05

    # Split configuration:
    split_mode = "random"

    # Dual-sPLS configuration
    norm = "pseudo-lasso"

    simulation_results = simulate(n=n, p=p, nondes=nondes , sigmaondes=sigmaondes, sigma_y=sigma_y, coefs=coefs)
    X, y = simulation_results["X"], simulation_results["y"]

    bin_indices = get_bin_indices(y, n_bins=10)
    print(bin_indices)
    calList = get_calList(bin_indices, pcal=90)
    print(calList, calList.shape)