"""
Manages the interface for returning filters that can be used in catalogs.
A filter is always a dictionary of keys which are catalog parameters and values of lambda functions,
check `get_default_base_filters` below for an example.
"""
import numpy as np


def get_default_base_filters(particle_mass, subhalos):
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
        **particle_mass_filter(particle_mass),
        'upid': lambda x: (x == -1 if not subhalos else x >= 0),
        # the ones after seem to have no effect after for not subhalos.
        'Spin': lambda x: x != 0,
        'q': lambda x: x != 0,
        'vrms': lambda x: x != 0,
        'mag2_A': lambda x: x != 0,
        'mag2_J': lambda x: x != 0,
    }


def particle_mass_filter(particle_mass):
    """
    We introduce two default cuts on mass:
        * The first part is to account for having too few particles (<1000).
         * The second is too account for bins that are undersampled in Bolshoi.

    :param particle_mass: The mass of each particle in the halo catalog.
    :return:
    """

    return {'mvir': lambda mvirs: (np.log10(mvirs) > np.log10(particle_mass * 1e3)) & (
                np.log10(mvirs) < 14.18)}


def catalog_mass_filter(catalog_name):
    """
    * The cuts on mvir are based on Phil's comment that Bolshoi/BolshoiP only give reasonable results up to
    log10(Mvir) ~ 13.5 - 13.75.
    :return:
    """

    if catalog_name == 'Bolshoi' or catalog_name == 'BolshoiP':
        def myfilter(mvirs):
            return np.log10(mvirs) < 13.75

    else:
        raise NotImplementedError("Implemented other catalogs yet.")

    return {'mvir': myfilter}


def get_relaxed_filters(relaxed_name):
    """
    For now only relaxed criteria is (cat['xoff'] < 0.04), according to Power 2011
    :return:
    """

    if relaxed_name == 'power2011':
        return {
            'xoff': lambda x: x < 0.04,
        }

    if relaxed_name == 'neto2007':
        return {
            # 'fsub': lambda x: x < 0.1,
            'xoff': lambda x: x < 0.07,
            'eta': lambda x: x < 1.35
        }

    else:
        raise NotImplementedError("The required relaxed name has not been implemented.")
