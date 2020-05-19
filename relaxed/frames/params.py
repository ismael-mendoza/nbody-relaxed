import numpy as np
from astropy.table import Column, Table


# functions to get derived quantities.
def get_phi_l(cat):
    """
    * JX/JY/JZ: Halo angular momenta ((Msun/h) * (Mpc/h) * km/s (physical))
    * A[x],A[y],A[z]: Largest shape ellipsoid axis (kpc/h comoving)
    :param cat:
    :return: Value of phi_l for each row of the catalog cat.
    :rtype: astropy.Column
    """
    return np.arccos(
        ((cat['A[x]'] * cat['Jx'] + cat['A[y]'] * cat['Jy'] + cat['A[z]'] * cat['Jz'])
         /
         (np.sqrt(cat['A[x]'] ** 2 + cat['A[y]'] ** 2 + cat['A[z]'] ** 2) * np.sqrt(
             cat['Jx'] ** 2 + cat['Jy'] ** 2 + cat['Jz'] ** 2))
         )
    )


def get_fsub(cat):
    """
    Return the substructure fraction of each entry of the catalog has value of -1 for
    halos without subhalos.
    This is defined in Neto2007: "We compute the mass fraction in resolved substructures
    whose centres lie inside r_vir".
    :param cat:
    :return: astropy.Column
    """
    fsubs = []
    for row in cat:
        halo_id = row['id']
        mvir = row['mvir']

        # find all rows that have this id as the upid
        upids = (cat['upid'] == halo_id)

        if not np.sum(upids):
            fsubs.append(-1)
            continue

        substructure_mass = np.sum(cat[upids]['mvir'])
        fsubs.append(substructure_mass / mvir)

    return Column(data=np.array(fsubs), name='fsub')


class Param(object):

    def __init__(self, key, log=False, modifiers=None, text=None):
        """
        Class implementing a Param object which manages how data is accessed from catalog
        and attributes like its name and its text representation for plots.


        :param key: is the actual string used to access the corresponding parameter from
                    the catalogue.
        :param log: Whether to log the values when returning them and change the label to
                    indicate that there is a log.
        :param modifiers: Extra modifiers to the values passed in as a list of lambda
                         functions. This will be applied  after logging.
        """
        self.key = key
        self.latex_param = params_dict[key]['latex_param']

        # units.
        self.units, self.latex_units = '', ''
        unit_pair = params_dict[key]['units']
        if unit_pair:
            self.units = unit_pair[0]
            self.latex_units = unit_pair[1]

        # deriving parameter.
        self.derive_func, self.required_derive_params = None, None
        derivation_pair = params_dict[key]['derive']
        if derivation_pair:
            self.derive_func = derivation_pair[0]
            self.required_derive_params = derivation_pair[1]

        self.log = log
        self.modifiers = modifiers

        self.text = self.get_text() if not text else text
        self.values = None

    def get_values_minh(self, mcat, b=None):

        if not self.derive_func:
            if b is None:
                return mcat.read([self.key])

            else:
                return mcat.block(b, [self.key])

        else:
            if b is None:
                t = Table([Column(data=mcat.read([self.required_derive_params]))])
            else:
                t = Table([Column(data=mcat.block(b, [self.required_derive_params]))])

            self.get_values(t)

    def get_values(self, cat):
        if self.key not in cat.colnames and self.derive_func is None:
            raise ValueError(f"Cannot obtained the parameter {self.key} for "
                             f"the given catalog.")

        if self.values is not None:
            return self.values
        elif self.key in cat.colnames:
            values = cat[self.key]
        else:
            values = self.derive_func(cat)

        if self.log:
            values = np.log10(values)

        if self.modifiers:
            for modifier in self.modifiers:
                values = modifier(values)

        return values

    def get_text(self, only_param=False):
        """
        Obtain the text that will be used in the produce_plots.
        :return:
        """
        template = '${}{}{}$'
        log_tex = ''
        units_tex = ''

        if only_param:
            return template.format(log_tex, self.latex_param, units_tex)

        if self.log:
            log_tex = '\\log_{10}'
        elif self.latex_units is not None:
            units_tex = '\\; [{}]'.format(self.latex_units)

        return template.format(log_tex, self.latex_param, units_tex)


# ToDo: Change T/U to eta globally.
# non-derived quantities are by default included.
info_params = {
    # fundamental quantities in the catalog.
    # The key is the actual way name to access from the catalog.
    'id': (None, None, ''),
    'mvir': (None, ('Msun/h', 'h^{-1} \\, M_{\\odot}'), 'M_{\\rm vir}'),
    'rvir': (None, ('kpc/h', 'h^{-1} \\, kpc'), 'R_{\\rm vir}'),
    'rs': (None, ('kpc/h', 'h^{-1} \\, kpc'), 'R_{\\rm vir}'),
    'Xoff': (None, ('kpc/h', 'h^{-1} \\, kpc'), 'X_{\\rm off}'),
    'Voff': (None, ('km/s', 'km \\, s^{-1}'), 'V_{\\rm off}'),
    'vrms': (None, ('km/s', 'km \\, s^{-1}'), 'V_{\\rm rms}'),
    'Acc_Rate_1*Tdyn': (None, ('Msun/h/yr', 'h^{-1}\\, yr^{-1} \\, M_{\\odot}'),
                        '\\alpha_{\\tau_{\\rm dyn}}'),
    'Acc_Rate_Inst': (
        None, ('Msun/h/yr', 'h^{-1}\\, yr^{-1} \\, M_{\\odot}'), '\\alpha_{\\rm inst}'),

    'T/|U|': (None, None, 'T/|U|'),
    'Spin': (None, None, '\\lambda'),
    'scale_of_last_MM': (None, None, '\\delta_{\\rm MM}'),

    # derived quantities.
    'cvir': ((lambda cat: cat['rvir'] / cat['rs'], ('rvir', 'rs')), None, 'c_{\\rm vir}'),
    'eta': ((lambda cat: 2 * cat['T/|U|'], 'T/U'), None, '\\eta'),
    'q': (
        (lambda cat: (1 / 2) * (cat['b_to_a'] + cat['c_to_a']), ('b_to_a', 'c_to_a')),
        None,
        'q'),
    'phi_l': ((get_phi_l, ('A[x]', 'A[y]', 'A[z]', 'Jx', 'Jy', 'Jz')), None, '\\Phi_{l}'),
    'xoff': (
        (lambda cat: cat['Xoff'] / cat['rvir'], ('Xoff', 'rvir')), None, 'x_{\\rm off}'),
    'voff': (
        (lambda cat: cat['Voff'] / cat['vrms'], ('Voff', 'vrms')), None, 'v_{\\rm off}'),
    # 'fsub': (get_fsub, '', '', 'f_{\\rm sub}'),
    # 'tdyn': (lambda cat: np.sqrt(2) * cat['rvir'] / cat['vrms'], 'kpc/h / km/s',
    # '', '\\tau_{\\rm dyn}'), (notesheet)

    # usually excluded quantities necessary for filtering
    'upid': (None, '', '', ''),
    'mag2_A': ((lambda cat: cat['A[x]'] ** 2 + cat['A[y]'] ** 2 + cat['A[z]'] ** 2,
                ('A[x]', 'A[y]', 'A[z]')),
               None, None),
    'mag2_J': (
        (
            lambda cat: cat['Jx'] ** 2 + cat['Jy'] ** 2 + cat['Jz'] ** 2,
            ('Jx', 'Jy', 'Jz')),
        None, None),
}

# nicer format.
params_dict = {
    key: {'derive': value[0], 'units': value[1], 'latex_param': value[2]}
    for (key, value) in info_params.items()
}

param_names = params_dict.keys()
default_params_to_exclude = {'mag2_A', 'mag2_J'}
default_params_to_include = [param for param in param_names if
                             param not in default_params_to_exclude]
