"""Functions used to calculate correlations and errors on correlations."""
import numpy as np
from scipy import stats


def spearmanr(*args, **kwargs):
    return stats.spearmanr(*args, **kwargs).correlation


def get_ma_corrs(cat, param, ma):
    corrs = []
    n_scales = ma.shape[1]
    for k in range(n_scales):
        keep = (~np.isnan(ma[:, k])) & (ma[:, k] > 0)
        ma_k = ma[:, k][keep]

        # get correlation.
        assert np.all(ma_k > 0) and np.all(~np.isnan(ma_k))
        corr = spearmanr(ma_k, cat[param][keep])
        corrs.append(corr)

    return np.array(corrs)


def get_am_corrs(cat, param, am, box_keep=None):
    if box_keep is None:
        box_keep = np.ones(am.shape[0]).astype(bool)

    corrs = []
    n_mass_bins = am.shape[1]
    for k in range(n_mass_bins):
        corrs.append(spearmanr(cat[param][box_keep], am[:, k][box_keep], nan_policy="omit"))
    return np.array(corrs)


def add_box_indices(cat, boxes=8, box_size=250):
    # box_size is in Mpc
    # create a new row add it to the catalogue for which box it is in.
    assert int(boxes ** (1.0 / 3)) == boxes ** (1.0 / 3)
    box_per_dim = int(boxes ** (1.0 / 3))
    divides = np.linspace(0, box_size, box_per_dim + 1)[1:-1]  # only use the middle.
    cat.add_column(np.zeros(len(cat)), name="ibox")
    for k, dim in enumerate(["x", "y", "z"]):
        for d in divides:
            cat["ibox"] += 2**k * (d < cat[dim])


def vol_jacknife_err(y_true, y_est, ibox, fn):
    n_boxes = int(np.max(ibox) + 1)
    values = []
    for b in range(n_boxes):
        box_keep = ibox != b
        y1 = y_est[box_keep]
        y2 = y_true[box_keep]
        value = fn(y1, y2)
        values.append(value)
    values = np.array(values)
    return np.sqrt(values.var(axis=0) * (n_boxes - 1))
