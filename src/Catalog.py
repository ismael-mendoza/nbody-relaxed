import astropy
from astropy.table import Table
from astropy.io import ascii
import numpy as np

# particle mass (Msun/h), total particles, box size (Mpc/h).
catalog_properties = {
    'Bolshoi': (1.35e8, 2048**3, 250),
    'BolshoiP': (1.55e8, 2048**3, 250),
    'MDPL2': (1.51e9, 3840**3, 1000)
}


class HaloCatalog(object):

    def __init__(self, filename, catalog_name, param_names=None, filters=None, use_generator=False):
        """

        :param filename:
        :param catalog_name: Bolshoi / BolshoiP / MDPL2
        :param param_names:
        :param filters:
        :param use_generator:
        """
        assert catalog_name in catalog_properties, "Catalog type is not recognized."

        # 100 Mb chunks of maximum memory in each iteration.
        # this returns a generator.
        self.filename = filename
        self.catalog_name = catalog_name
        self.particle_mass, self.total_particles, self.box_size = catalog_properties[self.catalog_name]
        self.required_params = []

        self.use_generator = use_generator
        self.param_names = param_names if param_names else self.get_default_param_names()
        self.filters = filters if filters else self.get_default_filters()

        # ToDo: Deal with the fact that filters might contain non native parameters so need to automatically update
        # with the non-native parameters to account for this.
        # ToDo: Decide which parameters should be require for catalog? (Do it here or in the dict of parameters)
        self.required_params.extend(self.filters.keys())

        self.cat = self.get_cat()  # could be a generator or an astropy.Table object.

        # ToDo: Check if this is possible with a generator.
        assert self.required_params in self.cat.colnames

        if not self.use_generator:
            self.cat = self.filter_cat(self.cat)

        # ToDo: For the generator case we want to returned a modified generator with the filtered values.

    def get_cat_generator(self):
        """
        This will eventually contain all the complexities of Phil's code and return a generator to access
        the catalog in chunks. For now it offers the chunkified version of my catalog how I've been doing it.
        :return:
        """
        return ascii.read(self.filename, format='csv', guess=False,
                          fast_reader={'chunk_size': 100 * 1000000, 'chunk_generator': True})

    def get_cat(self):
        gcats = self.get_cat_generator()

        if self.use_generator:
            return gcats

        else:
            # actually extract the data from gcats and read it into memory.
            cats = []
            for i, cat in enumerate(gcats):
                cats.append(cat[self.param_names])

            return astropy.table.vstack(cats)

    def add_param(self, param):
        pass

    def get_default_filters(self):
        return {
            'mvir': lambda cat: np.log10(cat['mvir']) > np.log10(self.particle_mass * 1e3),
            'Spin': lambda cat: (cat['Spin'] != 0),
            'q': lambda cat: cat['q'] != 0
        }

    def filter_cat(self, cat):
        for my_filter in self.filters.values():
            cat = cat[my_filter(cat)]
        return cat

    @staticmethod
    def get_default_param_names():
        return ['mvir', 'rvir', ]


