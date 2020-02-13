import numpy as np

def get_corrs(params, cat):
    """
    This function returns the correlation matrix of a list of
    :param params:
    :return:
    """
    corrs = np.zeros((len(params), len(params)))
    for i, param1 in enumerate(params):
        for j, param2 in enumerate(params):
            corr, p = spearmanr(cat[param1], cat[param2])
            corrs[i, j] = corr