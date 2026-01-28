import numpy as np


def datatype(y, n_cells):
    """Divides the response vector y in n_cells cells of
    equal range and attributes a index type to the observations
    according to the corresponding cell.

    Args:
        y (np.array): response vector
        n_cells (int): Number of cells to split the response vector in


    Returns:
        (np.array): y-sized array. For i in len(y), datatype[i] gives the split to which the
                    i-th element of y is attributed.     
    """
    # sort y 
    indices_sorted = np.argsort(y)
    
    # split indices in equal sized arrays
    splits = np.array_split(indices_sorted, n_cells)

    datatype = np.zeros(y.shape[0], dtype=int)
    for i, split in enumerate(splits):
        datatype[split] = i
    
    return datatype

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

    print(datatype(y, n_cells=10))