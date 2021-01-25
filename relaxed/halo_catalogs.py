import warnings
from pathlib import Path, PosixPath
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii

from pminh import minh

from . import halo_filters
from . import halo_parameters

# particle mass (Msun/h), total particles, box size (Mpc/h).
# bolshoi log_m_low: 11.13
_props = {
    "Bolshoi": (1.35e8, 2048 ** 3, 250),
    "BolshoiP": (1.55e8, 2048 ** 3, 250),
    "MDPL2": (1.51e9, 3840 ** 3, 1000),
}

all_props = {
    key: {"particle_mass": value[0], "total_particles": value[1], "box_size": value[2]}
    for key, value in _props.items()
}


class HaloCatalog(object):
    def __init__(
        self,
        name="Bolshoi",
        cat_file="bolshoi.minh",
        label="all haloes",
        subhalos=False,
        verbose=False,
    ):
        """
        * cat_name: Should be one of `Bolshoi / BolshoiP / MDPL2`
        * add_progenitor: filename of summary progenitor table.
        * add_subhalo: add catalog halo properties that depend on their subhalos.
        * labels: useful when plotting (titles, etc.)
        * minh_params: list of keys (params) to be loaded when loading from minh catalog.
        """
        cat_file = Path(cat_file)
        assert name in all_props, "Catalog name is not recognized."
        assert subhalos is False, "Not implemented subhalo functionality."
        assert cat_file.name.endswith(".minh") or cat_file.name.endswith(".csv")

        self.name = name
        self.cat_file = cat_file
        self.cat_props = all_props[self.name]
        self.verbose = verbose
        self.subhalos = subhalos
        self.label = label

        self.cat = None  # loaded later

    def __len__(self):
        return len(self.cat)

    @staticmethod
    def get_default_params():
        params = ["id", "upid", "mvir", "rvir", "rs", "xoff", "voff", "x", "y", "z"]
        params += ["x0", "v0", "cvir", "spin", "q", "vvir", "t/|u|", "eta", "phi_l"]
        params += ["gamma_tdyn"]
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

    def load_cat_minh(
        self,
        params=None,
        hfilter=None,
    ):
        assert self.cat_file.name.endswith(".minh")
        if self.verbose:
            warnings.warn("Divide by zero errors are ignored, but filtered out.")

        # do filter on the fly, to avoid memory errors.
        params = params if params else self.get_default_params()
        hfilter = hfilter if hfilter else self.get_default_hfilter()
        assert set(hfilter.filters.keys()).issubset(set(params))

        with minh.open(self.cat_file) as mcat:
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
                cat = hfilter.filter_cat(cat)
                cats.append(cat)

            self.cat = vstack(cats)
            self.cat.sort("id")

    def save_cat(self, cat_path):
        assert type(cat_path) is PosixPath
        assert self.cat is not None, "cat must be loaded"
        assert cat_path.suffix == ".csv", "format supported will be csv for now"
        ascii.write(self.cat, cat_path, format="csv")
