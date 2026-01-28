import numpy as np
from dual_spls.datatype import datatype

def calList(datatype, pcal):
    """Determine the number of observations to select from each cell splitted by type.

    Args:
        datatype (list): y-sized array. For i in len(y), datatype[i] gives the split to which the
                    i-th element of y is attributed. Typically computed by type (see type.py)
        pcal (int): a positive integer between 0 and 100. pcal is the percentage
                    of calibration samples to be selected.
    """
    assert (pcal >= 0) and (pcal <= 100), "pcal must be within [0, 100]"

    n_cells = len(np.unique(datatype))

    # TO BE REPLACED BY THE MORE EFFECTIVE NUMPY FUNCTION
    #####################################################
    y_counts = np.zeros(n_cells)
    for k in range(len(datatype)):
        y_counts[datatype[k]] += 1
    ######################################################

    l = np.zeros(n_cells)

    rep = np.floor(n*pcal/100) # number of observations to select

    # TO BE REPLACED BY A MORE EFFECTIVE ALGORITHM. ANALYTIC SOLUTION ? 
    # AT LEAST IN THE BEST CASE WHERE REP/n_cells < y_counts[j] for all j ? 
    #####################################################
    i = 0
    while (i < rep): # select observations until the right amount is selected
        for j in range(n_cells): # for each cell 
            if y_counts[j] > 0 and i < rep: # is the second condition really needed ? 
                # if there is at least one observation available 
                y_counts[j] -= 1 # "take it" from original datatype
                l[j]+=1 # "put it" in the callist
                i+=1
        
    callist = l
    return callist

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

    _datatype = datatype(y, n_cells=10)
    print(callist(_datatype, pcal=90))