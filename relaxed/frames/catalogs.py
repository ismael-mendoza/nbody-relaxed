from typing import List
import warnings
from contextlib import contextmanager

import numpy as np
import astropy
from astropy.table import Table
from astropy.io import ascii

from copy import deepcopy

from . import filters
from . import params
from ..subhalos import subhalo
from pminh import minh

# particle mass (Msun/h), total particles, box size (Mpc/h).
catalog_properties = {
    "Bolshoi": (1.35e8, 2048 ** 3, 250),
    "BolshoiP": (1.55e8, 2048 ** 3, 250),
    "MDPL2": (1.51e9, 3840 ** 3, 1000),
}


def intersection(cat, scat):
    # intersect two catalogs by their id attribute.
    # scat ids are ideally a subset of cat ids
    # can also just repeat with switched order and will work in the end.
    cat.sort("id")
    scat.sort("id")

    ids = cat["id"]
    sids = scat["id"]

    indx = np.searchsorted(sids, ids)
    indx_ok = indx < len(sids)
    indx_ok[indx_ok] &= sids[indx[indx_ok]] == ids[indx_ok]

    new_cat = cat[indx_ok]

    return new_cat


class HaloCatalog(object):
    def __init__(
        self,
        filepath,
        catalog_name,
        subhalos=False,
        add_subhalo=False,
        add_progenitor=None,
        base_filters=None,
        params_to_include: List[str] = None,
        verbose=False,
        catalog_label="all halos",
    ):
        """

        :param filepath:
        :param catalog_name: Should be one of `Bolshoi / BolshoiP / MDPL2`
        :param add_progenitor: filename of summary progenitor table.
        :param base_filters:
        """
        assert catalog_name in catalog_properties, "Catalog name is not recognized."
        assert subhalos is False, "Not implemented subhalo functionality."

        self.filepath = filepath
        assert self.filepath.name.endswith(
            ".minh"
        ), "Using Phil's format exclusively now."

        self.catalog_name = catalog_name
        self.verbose = verbose
        self.add_progenitor = add_progenitor

        self.subhalos = subhalos
        self.add_subhalo = add_subhalo
        assert not add_subhalo or not subhalos, "Cannot have both."

        self.particle_mass, self.total_particles, self.box_size = catalog_properties[
            self.catalog_name
        ]
        self.catalog_label = catalog_label  # for use in things like legends.

        # name of all params that will be needed for filtering and
        # params we actually want to have at the end in the output catalog.
        self.param_names = deepcopy(params.param_names)
        self.params_to_include = (
            params_to_include
            if params_to_include
            else deepcopy(params.params_to_include)
        )
        self.params_add_later = deepcopy(params.params_add_later)

        self._filters = (
            base_filters
            if base_filters is not None
            else filters.get_default_base_filters(self.particle_mass, self.subhalos)
        )

        if not set(self._filters.keys()).issubset(set(self.param_names)):
            raise ValueError(
                "filtering will fail since not all params are in self.param_names,"
                "need to update params.py"
            )

        # will be potentially defined later.
        self.use_minh = None
        self._bcat = None  # base cat.
        self._cat = None  # cat to actually return.

        # ToDo: For the generator case we want to returned a modified generator with the filtered
        #  values.
        #  ToDo: Make everything work together property if not default filters or params
        #   to include.
        #   ToDo: Delay obtaining filters so that we can use parameters of catalog in
        #    the user-defined filters. (Necessary?)
        #    ToDo: Change everything in place or create
        #     copy for every catalog (relaxed, mass bins, etc.)? for now everything is copy.

    def get_cat(self):
        return self._cat

    def get_filters(self):
        return self._filters

    @contextmanager
    def using_filters(self, myfilters, catalog_label="filtered catalog"):
        old_label = self.catalog_label

        try:
            self.catalog_label = catalog_label
            self._cat = self._filter_cat(myfilters, self._cat)
            yield self
        finally:
            self._cat = self._bcat
            self.catalog_label = old_label

    @contextmanager
    def using_relaxed_filters(self, relaxed_name=None):
        self.using_filters(
            filters.get_relaxed_filters(relaxed_name),
            catalog_label=f"{relaxed_name} relaxed",
        )

    def with_filters(self, myfilters, catalog_label="filtered catalog"):
        self.catalog_label = catalog_label
        self._cat = self._filter_cat(myfilters, self._cat,)

    def with_relaxed_filters(self, relaxed_name=None):
        self.with_filters(
            filters.get_relaxed_filters(relaxed_name),
            catalog_label=f"{relaxed_name} relaxed",
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
        return minh.open(self.filepath)

    def _load_cat(self):
        """
        Return either the catalog as a table or as a generator. Should only be called by set_cat.

        NOTE: We filter using the cfilters (cat filters).
        :return:
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
                        if param not in self.params_add_later:
                            values = self._get_not_log_value_minh(param, minh_cat, b)
                            new_cat.add_column(values, name=param)

                # once all needed params are in new_cat, we filter it out to reduce size.
                new_cat = self._filter_cat(self._filters, new_cat)

                cats.append(new_cat)

                if self.verbose:
                    if b % 10 == 0:
                        print(b)

            # TODO: add f1 = fraction of most massive subhalo to host halo mass.
            fcat = astropy.table.vstack(cats)
            if self.add_subhalo:
                if self.verbose:
                    print("extracting subhalo properties")
                assert np.all(fcat["upid"] == -1), "Needs to be a host catalog"
                subhalo_cat = self._extract_subhalo(fcat, minh_cat)
                fcat = astropy.table.join(fcat, subhalo_cat, keys="id")
                self.params_to_include.append("f_sub")

            if self.add_progenitor:
                if self.verbose:
                    print("Adding progenitor properties")

                # catalog with progenitor summary.
                pcat = Table.read(self.add_progenitor)

                pcat = intersection(pcat, fcat)
                fcat = intersection(fcat, pcat)

                fcat = astropy.table.join(fcat, pcat, keys="id")
                pcat_params = pcat.colnames
                pcat_params.remove("id")
                self.params_to_include += pcat_params

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

        :param cat:
        :param myfilters:
        :return:
        """
        new_cat = cat.copy() if copy else cat

        for param_name, myfilter in myfilters.items():
            new_cat = new_cat[myfilter(self._get_not_log_value(param_name, new_cat))]

        return new_cat

    @staticmethod
    def _get_not_log_value_minh(key, mcat, b=None):
        return params.Param(key, log=False).get_values_minh(mcat, b)

    @staticmethod
    def _get_not_log_value(key, cat):
        """
        Only purpose is for the filters.
        :param key:
        :return:
        """
        return params.Param(key, log=False).get_values(cat)

    def __len__(self):
        return len(self._cat)

    @classmethod
    def create_filtered_from_base(
        cls, old_hcat, myfilters, catalog_label="filtered cat"
    ):
        """
        This will copy the `_cat` attribute of the old_hcat.
        :param old_hcat:
        :param myfilters:
        :param catalog_label:
        :return:
        """
        assert (
            old_hcat.get_cat() is not None
        ), "Catalog of old_hcat should already be set."
        assert set(myfilters.keys()).issubset(set(old_hcat.get_cat().colnames)), (
            "This will fail because the "
            "cat of old_hcat does "
            "not contain filtered parameters."
        )

        new_hcat = cls(
            old_hcat.filepath,
            old_hcat.catalog_name,
            subhalos=old_hcat.subhalos,
            base_filters=old_hcat.get_filters(),
            catalog_label=catalog_label,
            params_to_include=old_hcat.params_to_include,
        )

        # it is ok to have a view for the base cat, since filtering will create a copy.
        new_hcat.load_base_cat(use_minh=False, bcat=old_hcat.get_cat())
        new_hcat.with_filters(myfilters, catalog_label=catalog_label)
        return new_hcat

    @classmethod
    def create_relaxed_from_base(cls, old_hcat, relaxed_name):
        return cls.create_filtered_from_base(
            old_hcat,
            filters.get_relaxed_filters(relaxed_name),
            catalog_label=f"{relaxed_name} relaxed",
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
