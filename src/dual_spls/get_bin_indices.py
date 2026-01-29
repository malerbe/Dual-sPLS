import numpy as np

"""
Ported from R to Python. Original name: d.spls.type.R
"""

def get_bin_indices(y, n_bins):
    """Divides the response vector y in n_cells cells of
    equal range and attributes a index type to the observations
    according to the corresponding cell.

    Args:
        y (np.array): response vector
        n_bins (int): Number of cells to split the response vector in


    Returns:
        (np.array): y-sized array. For i in len(y), datatype[i] gives the split to which the
                    i-th element of y is attributed.     
    """
    # compute range
    ymin, ymax = np.min(y), np.max(y)
    _range = ymax - ymin
    bin_size = _range/n_bins

    # sort y 
    indices_sorted = np.argsort(y)
    
    # run though y indices and associate them with the right bin
    current_sorted_index = 0
    bin_indices = np.zeros(len(y))
    for k in range(0, n_bins):
        # upper bound of the current bin 
        upper_bound = ymin + (k + 1) * bin_size

        # make sure to take everything in the last bin 
        if k == n_bins - 1:
            upper_bound = ymax + 1.0 

        # add indices as long as the corresponding value in y is below the current bin's upper_bound
        while current_sorted_index < len(y) and y[indices_sorted[current_sorted_index]] <= upper_bound:
            original_idx = indices_sorted[current_sorted_index]
            bin_indices[original_idx] = k

            current_sorted_index += 1

    return bin_indices

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

    for k in range(len(bin_indices)):
        print(y[k], bin_indices[k])

    import matplotlib.pyplot as plt

    plt.plot(bin_indices, y, "+")
    plt.show()