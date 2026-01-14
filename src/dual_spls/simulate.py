import numpy as np

def simulate(n=200,p=[100],nondes=[50],sigmaondes=[0.05], sigma_y=0.5, coefs=[1, 2, 3, 4, 5]):
    """Simulates synthetic spectral data (X) and a response vector (y).

    Args:
        n (int, optional): Number of samples. Defaults to 200.
        p (int | list, optional): Length of the signal(s). Defaults to [100].
        nondes (int | list, optional): number of gaussian used to make the signal. Defaults to [50].
        sigmaondes (int | list, optional): Standard deviation (width) of the gaussians. Defaults to 0.05.
        sigmay (float, optional): noise_level. Defaults to 0.5.
        coefs (list, optional): Regression coefficients applied to partial sums of X to generate y. Defaults to [1, 2, 3, 4, 5].
    """

    if isinstance(p, int):
        p = [p]
    if isinstance(nondes, int):
        nondes = [nondes]
    if isinstance(sigmaondes, (int, float)):
        sigmaondes = [sigmaondes]
    coefs = np.array(coefs)

    assert len(p) == len(nondes), "p and nondes must be of same length"
    assert len(p) == len(sigmaondes), "p and sigmaondes must be of same length"

    # X Initialisation 
    X = np.zeros((n, np.sum(p)))
    
    # generate n signals
    for i in range(n):
        # generate len(p) signals to be concatenated:
        begin = 0
        for k in range(len(p)):
            # generation of random amplitudes and mode for the gaussians
            x_local = np.linspace(0, 1, p[k])
            amplitudes = np.random.uniform(0, 1, nondes[k])
            modes = np.random.uniform(0, 1, nondes[k])

            # build signal
            for j in range(nondes[k]):
                X[i, begin:begin+p[k]] += amplitudes[j] * np.exp(-(x_local - modes[j])**2 / (2 * sigmaondes[k]**2))    

            begin += p[k] # shift of p[k] to be able to concatenate the signals in the next iteration

    # generate labels y
    y0 = np.zeros((n)) # initializing the response vector without noise y0
    
    # setting the interval limits for y0
    limits_pct = np.linspace(10, 100, len(coefs))
    pif = np.round(limits_pct * np.sum(p) / 100).astype(int)
    pif = np.insert(pif, 0, 0) # pif = [0, linspace(10, len(100))*np.sum(p)] = [0, 10*np.sum(p)/100, ..., np.sum(p)]

    # computing y0 as a sum of intervals of X
    sumX = np.zeros((n, len(coefs)))
    for i in range(len(coefs)):
        sumX[:, i] = np.sum(X[:, pif[i]:pif[i+1]], axis=1)

    y0 = sumX @ coefs

    #adding noise to y0
    y = y0 + sigma_y * np.random.normal(0, 1, n)

    return {"X": X, 
            "y": y,
            "y0": y0,
            "sigma_y": sigma_y,
            "sigmaondes": sigmaondes,
            "G": len(p),
            "sumX": sumX}

simulate(n=200, p=[100, 50], nondes=[50, 50], sigmaondes=[0.05, 0.10])