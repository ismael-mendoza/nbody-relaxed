import astropy
from astropy.table import Table
from astropy.io import ascii
import numpy as np

# particle mass (Msun), total particles, box size (Mpc/h).
catalog_properties = {
    'Bolshoi': (1.35e8, 2048**3, 250),
    'BolshoiP': (),
    'MDPL2': ()
}


class HaloCatalog(object):

    # ToDo: Eventually handle chunkification by using a generator.
    def __init__(self, filename, catalog_type, params=None, filters=None, use_generator=False):
        """

        :param filename:
        :param catalog_type: Bolshoi / BolshoiP / MDPL2
        :param params:
        :param filters:
        :param use_generator:
        """
        assert catalog_type in catalog_properties

        # 100 Mb chunks of maximum memory in each iteration.
        # this returns a generator.
        self.filename = filename
        self.catalog_type = catalog_type
        self.particle_mass, self.total_particles, self.box_size = catalog_properties[catalog_type]

        self.use_generator = use_generator
        self.params = params if params else self.get_default_params()
        self.filter = filters if filters else self.get_default_filters()
        self.gcats, self.cat = self.get_cat()

        if self.cat:
            self.cat = self.cat[self.filter(self.cat)]

    def get_cat(self):
        gcats = ascii.read(self.filename, format='csv', guess=False,
                           fast_reader={'chunk_size': 100 * 1000000, 'chunk_generator': True})
        if self.use_generator:
            return gcats, None

        else:
            # actually extract the data from gcats.
            cats = []
            for i, cat in enumerate(gcats):
                cats.append(cat[self.params])

            return None, astropy.table.vstack(cats)

    def add_param(self, param):
        pass

    def get_default_filters(self):
        return lambda cat: ((np.log10(cat['mvir']) > np.log10(self.particle_mass * 1e3)) &
                            (cat['Spin'] != 0) & (cat['q'] != 0))

    @staticmethod
    def get_default_params():
        pass


