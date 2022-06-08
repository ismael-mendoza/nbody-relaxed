"""Functions related to loading and filtering catalogs."""
import warnings
from pathlib import Path

import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack
from pminh import minh

from relaxed import parameters


def intersect(ids1, ids2):
    """Intersect two np.array IDs.

    Args:
        Both inputs should be np.arrays.

    Returns:
        An boolean array `indx_ok` corresponding to `ids1` s.t. `indx_ok[i]` is true iff
        `ids1[i]` is contained in `ids2`.

    Notes:
        - Full intersection by repeating operation but switching order.
    """
    assert type(ids1) == type(ids2) == np.ndarray
    assert np.all(np.sort(ids1) == ids1)
    assert np.all(np.sort(ids2) == ids2)
    indx = np.searchsorted(ids2, ids1)
    indx_ok = indx < len(ids2)
    indx_ok[indx_ok] &= ids2[indx[indx_ok]] == ids1[indx_ok]

    return indx_ok


def get_id_filter(ids):
    assert isinstance(ids, list) or isinstance(ids, np.ndarray)
    ids = np.array(ids)
    return {"id": lambda x: intersect(np.array(x), ids)}


def filter_cat(cat, filters: dict):
    # Always do filtering in real space NOT log space.
    for param, filt in filters.items():
        cat = cat[filt(cat[param])]
    return cat


def load_cat_csv(cat_file: Path):
    assert isinstance(cat_file, Path)
    assert cat_file.name.endswith(".csv")
    return ascii.read(cat_file, format="csv", fast_reader=True)


def save_cat_csv(cat, cat_file: Path):
    assert isinstance(cat_file, Path)
    assert cat_file.suffix == ".csv", "format supported will be csv for now"
    ascii.write(cat, cat_file, format="csv")


def load_cat_minh(minh_file: str, params: list, filters: dict, verbose=False):
    """Return astropy table of Halo present-day parameters from .minh catalog.

    Parameters are filtered on the fly to avoid memory errors.
    """
    assert Path(minh_file).name.endswith(".minh")
    assert set(filters.keys()).issubset(set(params))
    if verbose:
        warnings.warn("Divide by zero errors are ignored, and filtered out.")

    mcat = minh.open(minh_file)
    cats = []
    for b in range(mcat.blocks):
        cat = Table()

        # obtain all params from minh and their values.
        with np.errstate(divide="ignore", invalid="ignore"):
            for param in params:
                if param in mcat.names:
                    [value] = mcat.block(b, [param])
                else:
                    value = parameters.derive(param, mcat, b)
                cat.add_column(value, name=param)

        # make sure it's sorted by ID in case using id_filter
        cat.sort("id")

        # filter to reduce size of each block.
        cat = filter_cat(cat, filters)
        cats.append(cat)

    fcat = vstack(cats)
    fcat.sort("id")

    return fcat
