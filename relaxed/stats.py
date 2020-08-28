import numpy as np
from scipy.stats import spearmanr


def get_corrs(param_values):
    """This function returns the correlation matrix of a list of params."""
    n_params = len(param_values)
    corrs = np.zeros(n_params, n_params)

    for i, value1 in enumerate(param_values):
        for j, value2 in enumerate(param_values):
            corr, p = spearmanr(value1, value2)
            corrs[i, j] = corr

    return corrs
