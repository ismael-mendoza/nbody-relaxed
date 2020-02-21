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

    def __init__(self, filename, catalog_name, filters=None, params_to_include: List[str] = None,
                 use_generator=False, verbose=False, subhalos=False):
        """

        :param filename:
        :param catalog_name: Bolshoi / BolshoiP / MDPL2
        :param filters:
        :param use_generator:
        """
        assert use_generator is False, "Not implemented this functionality yet, for now just return full catalog."
        assert catalog_name in catalog_properties, "Catalog name is not recognized."

        self.filename = filename
        self.catalog_name = catalog_name
        self.verbose = verbose
        self.subhalos = subhalos
        self.particle_mass, self.total_particles, self.box_size = catalog_properties[self.catalog_name]

        # name of all params.
        self.param_names = params.param_names
        self.params_to_include = params_to_include if params_to_include else params.default_params_to_include

        self.use_generator = use_generator
        self.filters = filters if filters else self.get_default_filters()
        assert set(self.filters.keys()).issubset(set(self.param_names)), "filtering will fail."

        self.cat = self.get_cat()  # could be a generator or an astropy.Table object.

        # ToDo: For the generator case we want to returned a modified generator with the filtered values.
        # ToDo: Add assert that all fundamental params are in catalog.
        # ToDo: Make everything work together property if not default filters or params to include.

    def get_cat_generator(self):
        """
        This will eventually contain all the complexities of Phil's code and return a generator to access
        the catalog in chunks. For now it offers the chunkified version of my catalog how I've been doing it.
        :return:
        """

        # 100 Mb chunks of maximum memory in each iteration.
        # this returns a generator.
        return ascii.read(self.filename, format='csv', guess=False,
                          fast_reader={'chunk_size': 100 * 1000000, 'chunk_generator': True})

    def get_cat(self):
        gcats = self.get_cat_generator()

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

                new_cat = self.filter_cat(new_cat)

                cats.append(new_cat)

                if self.verbose:
                    if i % 10 == 0:
                        print(i)

            return astropy.table.vstack(cats)

    def get_default_filters(self):
        """
        NOTE: Always assume that the values of the catalog are returned without log10ing first.
        :return:
        """
        return {
            'mvir': lambda cat: np.log10(self.get_values(cat, 'mvir')) > np.log10(self.particle_mass * 1e3),
            'Spin': lambda cat: self.get_values(cat, 'Spin') != 0,
            'q': lambda cat: self.get_values(cat, 'q') != 0,
            'vrms': lambda cat: self.get_values(cat, 'vrms') != 0,
            'mag2_A': lambda cat: self.get_values(cat, 'mag2_A') != 0,
            'mag2_J': lambda cat: self.get_values(cat, 'mag2_J') != 0,
            'upid': lambda cat: (self.get_values(cat, 'upid') == -1 if self.subhalos else
                                 self.get_values(cat, 'upid') >= 0)
        }

    def filter_cat(self, cat):
        """
        Do all the appropriate filtering required when not reading the generator expression , in particular cat is
        assumed to contain all the parameter in param_names before value filters are applied. Not all parameters are
        actually required once filtering is complete so they are removed according to self.params_to_include.
        :param cat:
        :return:
        """
        for my_filter in self.filters.values():
            cat = cat[my_filter(cat)]

        cat = cat[self.params_to_include]
        return cat

    # ToDo: Implement this in some way?
    def relaxed(self):
        """
        For now relaxed = (cat['Xoff'] < 0.04)
        :return:
        """
        pass

    @staticmethod
    def get_values(cat, key):
        """
        NOTE: Always return without using logs.
        :param cat:
        :param key:
        :return:
        """
        # ToDo: maybe this can be a module level function?
        return params.Param(key, log=False).get_values(cat)


