import warnings
from pathlib import Path
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii

from pminh import minh

from . import halo_filters
from . import halo_parameters

# particle mass (Msun/h), total particles, box size (Mpc/h).
_props = {
    "Bolshoi": (1.35e8, 2048 ** 3, 250),
    "BolshoiP": (1.55e8, 2048 ** 3, 250),
    "MDPL2": (1.51e9, 3840 ** 3, 1000),
}

props = {
    key: {"particle_mass": value[0], "total_particles": value[1], "box_size": value[2]}
    for key, value in _props.items()
}


def intersection(cat, sub_cat):
    """Intersect two catalogs by their id attribute.
    * Returns all rows of cat whose ids are in sub_cat.
    * Full intersection by repeating operation but switching order.
    * Both catalogs should be astropy tables and have 'id' as one of their columns.
    """
    cat.sort("id")
    sub_cat.sort("id")

    ids = cat["id"]
    sub_ids = sub_cat["id"]

    indx = np.searchsorted(sub_ids, ids)
    indx_ok = indx < len(sub_ids)
    indx_ok[indx_ok] &= sub_ids[indx[indx_ok]] == ids[indx_ok]

    new_cat = cat[indx_ok]

    return new_cat


class HaloCatalog(object):
    def __init__(
        self,
        name="Bolshoi",
        cat_file="bolshoi.minh",
        minh_params=None,
        hfilter=None,
        subhalos=False,
        verbose=False,
        label="all haloes",
    ):
        """
        * cat_name: Should be one of `Bolshoi / BolshoiP / MDPL2`
        * add_progenitor: filename of summary progenitor table.
        * add_subhalo: add catalog halo properties that depend on their subhalos.
        * labels: useful when plotting (titles, etc.)
        * minh_params: list of keys (params) to add and be read from minh catalog.
        """
        cat_file = Path(cat_file)
        assert name in props, "Catalog name is not recognized."
        assert subhalos is False, "Not implemented subhalo functionality."
        assert cat_file.name.endswith(".minh") or cat_file.name.endswith(".csv")

        self.name = name
        self.cat_file = cat_file
        self.cat_props = props[self.name]
        self.verbose = verbose
        self.subhalos = subhalos
        self.label = label

        self.minh_params = minh_params if minh_params else self.get_default_params()
        self.hfilter = hfilter if hfilter else self.get_default_hfilter()
        assert set(self.hfilter.filters.keys()).issubset(set(self.minh_params))

        self.cat = None  # will be loaded later.

    def __len__(self):
        return len(self.cat)

    @staticmethod
    def get_default_params():
        params = ["id", "upid", "mvir", "rvir", "rs", "xoff", "voff"]
        params += ["x0", "v0", "cvir", "spin", "q", "vrms", "t/|u|", "eta", "phi_l"]
        return params

    def get_default_hfilter(self):
        default_filters = halo_filters.get_default_filters(
            self.cat_props["particle_mass"], self.subhalos
        )
        hfilter = halo_filters.HaloFilter(default_filters)
        return hfilter

    def load_cat_csv(self):
        assert self.cat_file.name.endswith(".csv")
        self.cat = ascii.read(self.cat_file, format="csv", fast_reader=True)

    def load_cat_minh(self):
        assert self.cat_file.name.endswith(".minh")
        if self.verbose:
            warnings.warn("Divide by zero errors are ignored, but filtered out.")

        # do filter on the fly, to avoid memory errors.

        with minh.open(self.cat_file) as mcat:

            cats = []

            for b in range(mcat.blocks):
                cat = Table()

                # obtain all params from minh and their values.
                with np.errstate(divide="ignore", invalid="ignore"):
                    for param in self.minh_params:
                        hparam = halo_parameters.get_hparam(param, log=False)
                        values = hparam.get_values_minh_block(mcat, b)
                        cat.add_column(values, name=param)

                # filter to reduce size of each block.
                cat = self.hfilter.filter_cat(cat)
                cats.append(cat)

            self.cat = vstack(cats)

    def save_cat(self, cat_path):
        assert self.cat is not None, "cat must be loaded"
        assert cat_path.suffix == ".csv", "format supported will be csv for now"
        ascii.write(self.cat, cat_path, format="csv")
