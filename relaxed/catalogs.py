import warnings
from pathlib import Path

import numpy as np
from astropy.io import ascii
from astropy.table import Table
from astropy.table import vstack
from pminh import minh

from relaxed import halo_parameters

default_params = {
    "id",
    "pid",
    "mvir",
    "rvir",
    "rs",
    "xoff",
    "voff",
    "x",
    "y",
    "z",
    "x0",
    "v0",
    "cvir",
    "spin",
    "q",
    "vvir",
    "t/|u|",
    "eta",
    "phi_l",
    "gamma_tdyn",
    "tdyn",
    "scale_of_last_mm",
    "cvir_klypin",
    "gamma_tdyn",
    "tdyn",
    "scale_of_last_mm",
    "cvir_klypin",
    "b_to_a",
    "c_to_a",
    "spin_bullock",
}


def filter_cat(cat, filters):
    # Always do filtering in real space NOT log space.
    for param, filt in filters:
        hparam = halo_parameters.get_hparam(param, log=False)
        cat = cat[filt(hparam.get_values(cat))]
    return cat


def load_cat_csv(cat_file: Path):
    assert isinstance(cat_file, Path)
    assert cat_file.name.endswith(".csv")
    return ascii.read(cat_file, format="csv", fast_reader=True)


def save_cat_csv(cat, cat_file: Path):
    assert isinstance(cat_file, Path)
    assert cat_file.suffix == ".csv", "format supported will be csv for now"
    ascii.write(cat, cat_file, format="csv")


def load_cat_minh(self, minh_file: str, params: list, filters: dict, verbose=False):
    """Return astropy table of Halo present-day parameters from .minh catalog.

    Parameters are filtered on the fly to avoid memory errors.
    """
    assert self.cat_file.name.endswith(".minh")
    assert set(filters.keys()).issubset(set(params))
    if verbose:
        warnings.warn("Divide by zero errors are ignored, and filtered out.")

    with minh.open(minh_file) as mcat:
        cats = []
        for b in range(mcat.blocks):
            cat = Table()

            # obtain all params from minh and their values.
            with np.errstate(divide="ignore", invalid="ignore"):
                for param in params:
                    hparam = halo_parameters.get_hparam(param, log=False)
                    values = hparam.get_values_minh_block(mcat, b)
                    cat.add_column(values, name=param)

            # make sure it's sorted by ID in case using id_filter
            cat.sort("id")

            # filter to reduce size of each block.
            cat = filter_cat(cat, filters)
            cats.append(cat)

    fcat = vstack(cats)
    fcat.sort("id")

    return fcat
