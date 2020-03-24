import numpy as np
from astropy.table import Column


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
    Return the substructure fraction of each entry of the catalog has value of -1 for halos without subhalos.
    This is defined as in Neto2007: "We compute the mass fraction in resolved substructures
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
        fsubs.append(substructure_mass/mvir)

    return Column(data=np.array(fsubs), name='fsub')


class Param(object):

    def __init__(self, key, log=False, modifiers=None, text=None):
        """
        Class implementing a Param object which manages how data is accessed from catalog and attributes like its name
        and its text representation for plots.


        :param key: is the actual string used to access the corresponding parameter from the catalogue.
        :param log: Whether to log the values when returning them and change the label to indicate that there is a log.
        :param modifiers: Extra modifiers to the values passed in as a list of lambda functions. This will be applied
                         after logging.
        """
        self.key = key
        self.latex_param = params_dict[key]['latex_param']
        self.latex_units = params_dict[key]['latex_units']
        self.units = params_dict[key]['units']
        self.derive = params_dict[key]['derive']
        self.log = log
        self.modifiers = None

        self.text = self.get_text() if not text else text
        self.values = None

    def get_values(self, cat):
        assert self.key in cat.colnames or self.derive is not None, f"Cannot obtained the parameter {self.key} for " \
                                                                    f"the given catalog."

        if self.values is not None:
            return self.values
        elif self.key in cat.colnames:
            values = cat[self.key]
        else:
            values = self.derive(cat)

        if self.log:
            values = np.log10(values)

        if self.modifiers:
            for modifier in self.modifiers:
                values = modifier(values)

        return values

    def set_values(self, cat):
        self.values = self.get_values(cat)

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
# Can adjust by making it a 'false' derived.
info_params = {
    # fundamental quantities in the catalog.
    # The key is the actual way name to access from the catalog.
    'id': (None, '', '', ''),
    'mvir': (None, 'Msun/h',  'h^{-1} \\, M_{\\odot}', 'M_{\\rm vir}'),
    'rvir': (None, 'kpc/h', 'h^{-1} \\, kpc', 'R_{\\rm vir}'),
    'rs': (None, 'kpc/h', 'h^{-1} \\, kpc', 'R_{\\rm vir}'),
    'Xoff': (None, 'kpc/h', 'h^{-1} \\, kpc', 'X_{\\rm off}'),
    'Voff': (None, 'km/s', 'km \\, s^{-1}', 'V_{\\rm off}'),
    'vrms': (None, 'km/s', 'km \\, s^{-1}', 'V_{\\rm rms}'),
    'Acc_Rate_1*Tdyn': (None, 'Msun/h/yr', 'h^{-1}\\, yr^{-1} \\, M_{\\odot}', '\\alpha_{\\tau_{\\rm dyn}}'),
    'Acc_Rate_Inst': (None, 'Msun/h/yr', 'h^{-1}\\, yr^{-1} \\, M_{\\odot}', '\\alpha_{\\rm inst}'),

    'T/|U|': (None, '', '', 'T/|U|'),
    'Spin': (None, '', '', '\\lambda'),
    'scale_of_last_MM': (None, '', '', '\\delta_{\\rm MM}'),

    # derived quantities.
    'cvir': (lambda cat: cat['rvir'] / cat['rs'], '', '', 'c_{\\rm vir}'),
    'eta': (lambda cat: 2*cat['T/|U|'], '', '', '\\eta'),
    'q': (lambda cat: (1/2)*(cat['b_to_a'] + cat['c_to_a']), '', '', 'q'),
    'phi_l': (get_phi_l, '', '', '\\Phi_{l}'),
    'xoff': (lambda cat: cat['Xoff']/cat['rvir'], '', '', 'x_{\\rm off}'),
    'voff': (lambda cat: cat['Voff']/cat['vrms'], '', '', 'v_{\\rm off}'),
    # 'fsub': (get_fsub, '', '', 'f_{\\rm sub}'),
    # 'tdyn': (lambda cat: np.sqrt(2) * cat['rvir'] / cat['vrms'], 'kpc/h / km/s', '', '\\tau_{\\rm dyn}'), (notesheet)

    # usually excluded quantities necessary for filtering
    'upid': (None, '', '', ''),
    'mag2_A': (lambda cat: cat['A[x]'] ** 2 + cat['A[y]'] ** 2 + cat['A[z]'] ** 2, '', '', ''),
    'mag2_J': (lambda cat: cat['Jx'] ** 2 + cat['Jy'] ** 2 + cat['Jz'] ** 2, '', '', ''),
}


# nicer format.
params_dict = {
    key: {'derive': value[0], 'units': value[1], 'latex_units': value[2], 'latex_param': value[3]}
    for (key, value) in info_params.items()
}

param_names = params_dict.keys()
default_params_to_exclude = {'mag2_A', 'mag2_J'}
default_params_to_include = [param for param in param_names if param not in default_params_to_exclude]


