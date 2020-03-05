import numpy as np
from scipy.stats import spearmanr


def get_corrs(params, cat):
    """
    This function returns the correlation matrix of a list of
    :param params:
    :param cat:
    :return:
    """
    corrs = np.zeros((len(params), len(params)))

    for i, param1 in enumerate(params):
        for j, param2 in enumerate(params):
            corr, p = spearmanr(param1.get_values(cat), param2.get_values(cat))
            corrs[i, j] = corr

    return corrs

