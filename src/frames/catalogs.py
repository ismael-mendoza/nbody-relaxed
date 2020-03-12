import astropy
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from typing import List

from src.frames import params

# particle mass (Msun/h), total particles, box size (Mpc/h).
catalog_properties = {
    'Bolshoi': (1.35e8, 2048**3, 250),
    'BolshoiP': (1.55e8, 2048**3, 250),
    'MDPL2': (1.51e9, 3840**3, 1000)
}


class HaloCatalog(object):

    def __init__(self, filename, catalog_name, subhalos=False, relaxed=None, filters=None,
                 params_to_include: List[str] = None, verbose=False, catalog_label='all halos'):
        """

        :param filename:
        :param catalog_name: Bolshoi / BolshoiP / MDPL2
        :param filters:
        """
        assert catalog_name in catalog_properties, "Catalog name is not recognized."
        assert subhalos is False, "Not implemented subhalo functionality."

        self.filename = filename
        self.catalog_name = catalog_name
        self.verbose = verbose
        self.subhalos = subhalos
        self.relaxed = relaxed
        self.particle_mass, self.total_particles, self.box_size = catalog_properties[self.catalog_name]
        self.catalog_label = catalog_label  # for use in things like legends.

        # name of all params that will be needed for filtering and
        # params we actually want to have at the end in the output catalog.
        self.param_names = params.param_names
        self.params_to_include = params_to_include if params_to_include else params.default_params_to_include

        self.filters = filters if filters else self.get_default_filters()
        assert set(self.filters.keys()).issubset(set(self.param_names)), "filtering will fail since " \
                                                                         "not all params are in self.param_names" \
                                                                         "need to update params.py"

        # update filters from requested relaxed criteria
        if self.relaxed:
            self.filters.update(self._get_relaxed_filters(self.relaxed))
            self.catalog_label = f"{self.relaxed} relaxed"

        # will be defined later.
        self.use_generator = None
        self.cat = None

        # ToDo: For the generator case we want to returned a modified generator with the filtered values.
        # ToDo: Make everything work together property if not default filters or params to include.
        # ToDo: Create a filtered_from_cat method. (general filtering not only relaxed)
        # ToDo: Delay obtaining filters so that we can use parameters of catalog in the user-defined filters.
        #  (Necessary?)

    def set_cat(self, use_generator=False):
        """
        This function is used to set the cat attribute in the hcat to the catalog so it can be used in the future.
        :param use_generator:
        :return:
        """
        assert use_generator is False, "Not implemented this functionality yet, for now just return full catalog."
        assert self.cat is None, "Overriding catalog that is already created. (probably wasteful)"

        self.use_generator = use_generator
        self.cat = self._load_cat()  # could be a generator or an astropy.Table object.

    def _get_cat_generator(self):
        """
        This will eventually contain all the complexities of Phil's code and return a generator to access
        the catalog in chunks. For now it offers the chunkified version of my catalog how I've been doing it.
        :return:
        """

        # 100 Mb chunks of maximum memory in each iteration.
        # this returns a generator.
        return ascii.read(self.filename, format='csv', guess=False,
                          fast_reader={'chunk_size': 100 * 1000000, 'chunk_generator': True})

    def _load_cat(self):
        """
        Return either the catalog as a table or as a generator. Should only be called by set_cat.
        :return:
        """
        gcats = self._get_cat_generator()

        if self.use_generator:
            return gcats

        else:
            if self.verbose:
                print("Ignoring dividing by zero and invalid errors that should be filtered out anyways.")

            # actually extract the data from gcats and read it into memory.
            # do filtering on the fly so don't actually ever read unfiltered catalog.
            cats = []

            for i, cat in enumerate(gcats):
                new_cat = Table()

                # ignore warning of possible parameters that are divided by zero, this will be filtered out later.
                with np.errstate(divide='ignore', invalid='ignore'):
                    for param in self.param_names:
                        new_cat.add_column(self.get_values(cat, param), name=param)

                new_cat = self.filter_cat(new_cat, self.filters)

                cats.append(new_cat)

                if self.verbose:
                    if i % 10 == 0:
                        print(i)

            return astropy.table.vstack(cats)

    def filter_cat(self, cat, filters):
        """
        * Do all the appropriate filtering required when not reading the generator expression, in particular cat is
        assumed to contain all the parameter in param_names before value filters are applied.

        * Not all parameters are actually required once filtering is complete so they are removed according
        to self.params_to_include.

        * All filters assumed no logging is done.
        :param cat:
        :param filters:
        :return:
        """
        for param_name, my_filter in filters.items():
            cat = cat[my_filter(self._get_not_log_value(cat, param_name))]

        cat = cat[self.params_to_include]
        return cat

    @staticmethod
    def _get_relaxed_filters(relaxed_name):
        """
        For now only relaxed criteria is (cat['xoff'] < 0.04), according to Power 2011
        :return:
        """

        if relaxed_name == 'power2011':
            return {
                'xoff': lambda x: x < 0.04,
            }

        else:
            raise ValueError("The required relaxed_name has not be specified.")

    def get_default_filters(self):
        """
        NOTE: Always assume that the values of the catalog are returned without log10ing first.

        * x in the lambda functions represents the values of the keys.

        * upid >=0 indicates a subhalo, upid=-1 indicates a distinct halo. Phil's comment: "This is -1 for distinct
        halos and a halo ID for subhalos."
        >> cat_distinct = cat[cat['upid'] == -1]
        >> cat_sub = cat[cat['upid'] >= 0]
        :return:
        """
        return {
            'mvir': HaloCatalog.mass_default_filter(self.particle_mass, self.catalog_name),
            'upid': lambda x: (x == -1 if not self.subhalos else x >= 0),
            # the ones after seem to have no effect after.
            'Spin': lambda x: x != 0,
            'q': lambda x: x != 0,
            'vrms': lambda x: x != 0,
            'mag2_A': lambda x: x != 0,
            'mag2_J': lambda x: x != 0,
        }

    @staticmethod
    def mass_default_filter(particle_mass, catalog_name):
        """
        * The cuts on mvir are based on Phil's comment that Bolshoi/BolshoiP only give reasonable results up to
        log10(Mvir) ~ 13.5 - 13.75.
        :return:
        """

        def mass_filter(mvirs):
            # first, we need a minimum number of particles, this log mass is around 11.13 for Bolshoi.
            conds = np.log10(mvirs) > np.log10(particle_mass * 1e3)

            if catalog_name == 'Bolshoi' or catalog_name == 'BolshoiP':
                conds = conds & (np.log10(mvirs) < 13.75)

            else:
                raise NotImplementedError("Implemented other catalogs yet.")

            return conds

        return mass_filter

    @staticmethod
    def _get_not_log_value(cat, key):
        """
        Only purpose is for the filters.
        :param cat:
        :param key:
        :return:
        """
        return params.Param(key, log=False).get_values(cat)

    @classmethod
    def create_relaxed_from_complete(cls, old_hcat, relaxed=None):
        # Todo: Make it more flexible and transparent. (No need to have all parameters in all catalog for example)

        assert old_hcat.relaxed is None, "This is only to be used from an old hcat which is not relaxed initially."
        assert old_hcat.cat is not None, "Catalog of old_hcat should already be set."
        assert relaxed is not None, "Need relaxed to be specified."

        new_hcat = cls(old_hcat.filename, old_hcat.catalog_name, subhalos=old_hcat.subhalos, relaxed=relaxed,
                       filters=old_hcat.filters, params_to_include=old_hcat.params_to_include)

        relaxed_filters = new_hcat.get_relaxed_filters()

        assert set(relaxed_filters.keys()).issubset(set(old_hcat.cat.colnames)), "This will fail because the " \
                                                                                 "final catalog of old_hcat does " \
                                                                                 "not contain relaxed parameters."

        new_hcat.cat = new_hcat.filter_cat(old_hcat.cat, relaxed_filters)


