import warnings
from contextlib import contextmanager

import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii

import halo_filter, halo_param
from subhalos import subhalo
from pminh import minh

# particle mass (Msun/h), total particles, box size (Mpc/h).
catalog_props = {
    "Bolshoi": (1.35e8, 2048 ** 3, 250),
    "BolshoiP": (1.55e8, 2048 ** 3, 250),
    "MDPL2": (1.51e9, 3840 ** 3, 1000),
}

catalog_props = {
    key: {"particle_mass": value[0], "total_particles": value[1], "box_size": value[2]}
    for key, value in catalog_props.items()
}


def intersection(cat, sub_cat):
    """Intersect two catalogs by their id attribute.
    * Returns all rows of cat whose ids are in sub_cat.
    * Full intersection by repeating operation but switching order.
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
        cat_path,
        cat_name,
        params=None,
        hfilter=None,
        subhalos=False,
        verbose=False,
        label="all halos",
    ):
        """
        * cat_name: Should be one of `Bolshoi / BolshoiP / MDPL2`
        * add_progenitor: filename of summary progenitor table.
        * add_subhalo: add catalog halo properties that depend on their subhalos.
        * labels: useful when plotting (titles, etc.)
        """
        assert cat_name in catalog_props, "Catalog name is not recognized."
        assert subhalos is False, "Not implemented subhalo functionality."
        assert cat_path.name.endswith(".minh") or cat_path.name.endswith(".csv")

        self.cat_path = cat_path
        self.cat_name = cat_name
        self.cat_props = catalog_props[self.cat_name]
        self.verbose = verbose
        self.subhalos = subhalos
        self.label = label

        self.params = params if params else self.get_default_params()
        self.hfilter = hfilter if hfilter else self.get_default_hfilter()
        assert set(self.filters.keys()).issubset(set(self.params))

        self.cat = self.load_cat()

    def __len__(self):
        return len(self._cat)

    @staticmethod
    def get_default_params():
        return ["id", "mvir", "rvir", "rs", "xoff", "voff", "x0", "v0", "cvir"]

    def get_default_hfilter(self):
        default_filters = halo_filter.get_default_filters(
            self.cat_props["particle_mass"], self.subhalos
        )
        hfilter = halo_filter.HaloFilters(default_filters)
        return hfilter

    def save_cat(self, cat_path):
        assert self.cat is not None, "cat must be loaded"
        assert cat_path.suffix == ".csv", "format supported will be csv for now"
        ascii.write(self.cat, cat_path, format="csv", fast_writer=True)

    def load_cat_csv(self):
        assert self.cat_path.name.endswith(".csv")
        self.cat = ascii.read(self.cat_path, format="csv", fast_reader=True)

    def load_cat_minh(self):
        assert self.cat_path.name.endswith(".minh")
        if self.verbose:
            warnings.warn("Divide by zero errors are ignored, but filtered out.")

        # do filter on the fly, to avoid memory errors.
        mcat = minh.open(self.cat_path)
        cats = []
        hfilter = halo_filter.HaloFilters(self.filters)

        for b in range(minh_cat.blocks):
            cat = Table()

            # obtain all params from minh and their values.
            with np.errstate(divide="ignore", invalid="ignore"):
                for param in self.params:
                    hparam = halo_param.get_hparam(param, log=False)
                    values = hparam.get_values_minh_block(mcat, b)
                    cat.add_column(values, name=param)

            # filter to reduce size.
            cat = self.hfilter.filter_cat(cat)
            cats.append(cat)

        return vstack(cats)

    # TODO: Might need to change if require lower mass host halos.
    # TODO: make process more similar to adding other parameters.
    @staticmethod
    def _extract_subhalo(host_cat, minh_cat):
        # now we also want to add subhalo fraction and we follow Phil's lead

        host_ids = host_cat["id"]
        host_mvir = host_cat["mvir"]
        M_sub_sum = np.zeros(len(host_mvir))

        for b in range(minh_cat.blocks):
            upid, mvir = minh_cat.block(b, ["upid", "mvir"])

            # need to contain only ids of host_ids for it to work.
            sub_pids = upid[upid != -1]
            sub_mvir = mvir[upid != -1]
            M_sub_sum += subhalo.m_sub(host_ids, sub_pids, sub_mvir)

        f_sub = M_sub_sum / host_mvir  # subhalo mass fraction.
        subhalo_cat = Table(data=[host_ids, f_sub], names=["id", "f_sub"])

        return subhalo_cat
