"""Functions used to calculate correlations and errors on correlations."""
import numpy as np
from scipy import stats


def get_ma_corrs(cat, param, indices):
    corrs = []
    for k in indices:
        colname = f"mvir_a{k}"
        keep = (~np.isnan(cat[colname])) & (cat[colname] > 0)

        # get mass fraction at this scale
        mvir = cat["mvir"][keep]
        ms = cat[colname][keep]
        ms = ms / mvir
        pvalue = cat[param][keep]

        # get correlation.
        assert np.all(ms > 0) and np.all(~np.isnan(ms))
        assert np.all(mvir > 0)
        corr = stats.spearmanr(ms, pvalue)[0]
        corrs.append(corr)

    return np.array(corrs)


def get_am_corrs(cat, param, am, box_keep=None):
    if box_keep is None:
        box_keep = np.ones(am.shape[0]).astype(bool)

    corrs = []
    n_mass_bins = am.shape[1]
    for k in range(n_mass_bins):
        corrs.append(
            stats.spearmanr(cat[param][box_keep], am[:, k][box_keep], nan_policy="omit")[0]
        )
    return np.array(corrs)


def _add_box_indices(cat, boxes=8, box_size=250):
    # box_size is in Mpc
    # create a new row add it to the catalogue for which box it is in.
    assert int(boxes ** (1.0 / 3)) == boxes ** (1.0 / 3)
    box_per_dim = int(boxes ** (1.0 / 3))
    divides = np.linspace(0, box_size, box_per_dim + 1)[1:-1]  # only use the middle.
    cat.add_column(np.zeros(len(cat)), name="ibox")
    for k, dim in enumerate(["x", "y", "z"]):
        for d in divides:
            cat["ibox"] += 2**k * (d < cat[dim])


def vol_jacknife_err(cat, fn, *args, mode="dict"):
    # assumes cat has had its box indices added with the function above.
    if "ibox" not in cat.colnames:
        _add_box_indices(cat)

    n_boxes = int(np.max(cat["ibox"]) + 1)
    values = []
    for b in range(n_boxes):
        box_keep = cat["ibox"] != b
        value = fn(*args, box_keep=box_keep)
        values.append(value)

    if mode == "dict":
        d = {}
        for val in values:
            for k, v in val.items():
                d[k] = d[k] + [v] if d.get(k, None) is not None else [v]

        d = {k: np.array(v) for k, v in d.items()}
        return {k: np.sqrt(v.var(axis=0) * (n_boxes - 1)) for k, v in d.items()}

    if mode == "array":
        values = np.array(values)
        return np.sqrt(values.var(axis=0) * (n_boxes - 1))

    raise NotImplementedError()
