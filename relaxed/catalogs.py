import warnings
from contextlib import contextmanager

import numpy as np
import astropy
from astropy.table import Table
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


#  ToDo: Make everything work together property if not default filters or params
#   to include.
#   ToDo: Delay obtaining filters so that we can use parameters of catalog in
#    the user-defined filters. (Necessary?)


class HaloCatalog(object):
    def __init__(
        self,
        cat_path,
        cat_name,
        params=None,
        filters=None,
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
        assert cat_path.name.endswith(".minh")

        self.cat_path = cat_path
        self.cat_name = cat_name
        self.cat_props = catalog_props[self.cat_name]
        self.verbose = verbose
        self.subhalos = subhalos
        self.label = label

        self.params = params if params else self.get_default_params()
        self.filters = filters if filters else self.get_default_filters()
        assert set(self.filters.keys()).issubset(set(self.params))

        # will be potentially defined later.
        self.use_minh = None
        self._bcat = None  # base cat.
        self._cat = None  # cat to actually return.

    @staticmethod
    def get_default_params():
        return ["id", "mvir", "rvir", "rs", "xoff", "voff", "x0", "v0", "cvir"]

    @staticmethod
    def get_default_filters():
        return {}

    @contextmanager
    def using_filters(self, myfilters, label="filtered catalog"):
        old_label = self.label

        try:
            self.label = label
            self._cat = self._filter_cat(myfilters, self._cat)
            yield self
        finally:
            self._cat = self._bcat
            self.label = old_label

    @contextmanager
    def using_relaxed_filters(self, relaxed_name=None):
        self.using_filters(
            halo_filter.get_relaxed_filters(relaxed_name),
            label=f"{relaxed_name} relaxed",
        )

    def with_filters(self, myfilters, label="filtered catalog"):
        self.label = label
        self._cat = self._filter_cat(myfilters, self._cat,)

    def with_relaxed_filters(self, relaxed_name=None):
        self.with_filters(
            halo_filter.get_relaxed_filters(relaxed_name),
            label=f"{relaxed_name} relaxed",
        )

    def save_base_cat(self, filepath):
        assert self._cat is not None, "cat must be loaded"
        assert filepath.suffix == ".csv", "format supported will be csv for now"
        ascii.write(self._bcat, filepath, format="csv", fast_writer=True)

    def load_base_cat(self, use_minh=False, bcat=None):
        """
        This function is used to set the cat attribute in the hcat to the catalog so it
        can be used in the future.
        :param use_minh:
        :param bcat:
        :return:
        """
        assert use_minh is False, (
            "Not implemented this functionality yet, for now just return "
            "full catalog. "
        )
        assert (
            self._bcat is None
        ), "Overriding catalog that is already created. (probably wasteful)"

        self.use_minh = use_minh
        if not bcat:
            self._bcat = self._load_cat()

        else:
            self._bcat = bcat

        self._cat = self._bcat  # filtering will create a copy later if necessary.

    def _get_minh_cat(self):
        return minh.open(self.cat_path)

    def _load_cat(self):
        """
        Return either the catalog as a table or as a generator. Should only be called by set_cat.
        NOTE: We filter using the cfilters (cat filters).
        """
        minh_cat = self._get_minh_cat()

        if self.use_minh:
            # will do operations in each of the blocks through another interface.
            return minh_cat

        else:
            if self.verbose:
                warnings.warn(
                    "Ignoring dividing by zero and invalid errors that should "
                    "be filtered out anyways."
                )

            # actually extract the data from gcats and read it into memory.
            # do filtering on the fly so don't actually ever read unfiltered catalog.
            cats = []

            for b in range(minh_cat.blocks):
                new_cat = Table()

                # * First obtain all the parameters that we want to have.
                # each block in minh is complete so all parameters can
                # be obtained in any order.
                # * Ignore warning of possible parameters that are divided by zero,
                # this will be filtered out later.
                with np.errstate(divide="ignore", invalid="ignore"):
                    for param in self.param_names:
                        values = self._get_not_log_value_minh(param, minh_cat, b)
                        new_cat.add_column(values, name=param)

                # once all needed params are in new_cat, we filter it out to reduce size.
                new_cat = self._filter_cat(self._filters, new_cat)

                cats.append(new_cat)

                if self.verbose:
                    if b % 10 == 0:
                        print(b)

            warnings.warn(
                "We only include parameters in `params.default_params_to_include`"
            )
            fcat = fcat[self.params_to_include]

            return fcat

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

    def _filter_cat(self, myfilters, cat, copy=False):
        """
        * Do all the appropriate filtering required when not reading the generator
        expression, in particular cat is assumed to contain all the parameter in
        param_names before value filters are applied.

        * Not all parameters are actually required once filtering is complete so they
        can be (optionally) removed according to self.params_to_include.

        NOTE: All filters assumed no logging has been done on the raw catalog columns.
        """
        new_cat = cat.copy() if copy else cat

        for param_name, myfilter in myfilters.items():
            new_cat = new_cat[myfilter(self._get_not_log_value(param_name, new_cat))]

        return new_cat

    @staticmethod
    def _get_not_log_value_minh(key, mcat, b=None):
        return halo_param.HaloParam(key, log=False).get_values_minh(mcat, b)

    @staticmethod
    def _get_not_log_value(key, cat):
        """
        Only purpose is for the filters.
        :param key:
        :return:
        """
        return halo_param.HaloParam(key, log=False).get_values(cat)

    def __len__(self):
        return len(self._cat)

    @classmethod
    def create_filtered_from_base(cls, old_hcat, myfilters, label="filtered cat"):
        # This will copy the `_cat` attribute of the old_hcat.
        assert (
            old_hcat.get_cat() is not None
        ), "Catalog of old_hcat should already be set."
        assert set(myfilters.keys()).issubset(set(old_hcat.get_cat().colnames)), (
            "This will fail because the "
            "cat of old_hcat does "
            "not contain filtered parameters."
        )

        new_hcat = cls(
            old_hcat.cat_path,
            old_hcat.cat_name,
            subhalos=old_hcat.subhalos,
            base_filters=old_hcat.get_filters(),
            label=label,
            params_to_include=old_hcat.params_to_include,
        )

        # it is ok to have a view for the base cat, since filtering will create a copy.
        new_hcat.load_base_cat(use_minh=False, bcat=old_hcat.get_cat())
        new_hcat.with_filters(myfilters, label=label)
        return new_hcat

    @classmethod
    def create_relaxed_from_base(cls, old_hcat, relaxed_name):
        return cls.create_filtered_from_base(
            old_hcat,
            halo_filter.get_relaxed_filters(relaxed_name),
            label=f"{relaxed_name} relaxed",
        )

    @classmethod
    def create_from_saved_cat(cls, cat_file, *args, **kwargs):
        """
        Create catalog from saved (smaller/filtered) cat to reduce waiting time of having
        to filter every time, etc.

        :param cat_file: The file location specified as Path object and save using
        :return:
        """
        hcat = cls(cat_file, *args, **kwargs)
        cat = ascii.read(cat_file, format="csv")
        hcat.load_base_cat(use_minh=False, bcat=cat)
        return hcat
