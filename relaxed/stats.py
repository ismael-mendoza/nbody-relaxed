import numpy as np
from scipy.stats import spearmanr


def get_corrs(plot_params, param_values):
    """This function returns the correlation matrix of a list of params."""
    n_params = len(param_values)
    corrs = np.zeros(n_params, n_params)

    for i, param1 in enumerate(plot_params):
        for j, param2 in enumerate(plot_params):
            value1 = param_values[param1]
            value2 = param_values[param2]
            corr, p = spearmanr(value1, value2)
            corrs[i, j] = corr

    return corrs
