import numpy as np
from astropy.table import Column, Table

from .. import utils


# functions to get derived quantities.
def get_phi_l(cat):
    """
    * JX/JY/JZ: Halo angular momenta ((Msun/h) * (Mpc/h) * km/s (physical))
    * A[x],A[y],A[z]: Largest shape ellipsoid axis (kpc/h comoving)
    :param cat:
    :return: Value of phi_l for each row of the catalog cat.
    :rtype: astropy.Column
    """
    numerator = cat["ax"] * cat["jx"] + cat["ay"] * cat["jy"] + cat["az"] * cat["jz"]
    denominator = np.sqrt(cat["ax"] ** 2 + cat["ay"] ** 2 + cat["az"] ** 2)
    denominator *= np.sqrt(cat["jx"] ** 2 + cat["jy"] ** 2 + cat["jz"] ** 2)
    return np.arccos(numerator / denominator)


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
        self.latex_param = params_dict[key]["latex_param"]

        # units.
        self.units, self.latex_units = None, None
        unit_pair = params_dict[key]["units"]
        if unit_pair:
            self.units = unit_pair[0]
            self.latex_units = unit_pair[1]

        # deriving parameter.
        self.derive_func, self.required_derive_params = None, None
        derivation_pair = params_dict[key]["derive"]
        if derivation_pair:
            self.derive_func = derivation_pair[0]
            self.required_derive_params = derivation_pair[1]
            assert utils.is_iterable(self.required_derive_params)

        self.log = log
        self.modifiers = modifiers

        self.text = self.get_text() if not text else text
        self.values = None

    def get_values_minh(self, mcat, b=None):

        if not self.derive_func:
            if b is None:
                data = mcat.read([self.key]).pop()

            else:
                data = mcat.block(b, [self.key]).pop()

            return Column(data=data, name=self.key)

        else:
            if b is None:
                t = Table(
                    mcat.read(self.required_derive_params),
                    names=self.required_derive_params,
                )
            else:
                t = Table(
                    mcat.block(b, self.required_derive_params),
                    names=self.required_derive_params,
                )

            return self.get_values(t)

    def get_values(self, cat):
        if self.key not in cat.colnames and self.derive_func is None:
            raise ValueError(
                f"Cannot obtained the parameter {self.key} for " f"the given catalog."
            )

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
        template = "${}{}{}$"
        log_tex = ""
        units_tex = ""

        if only_param:
            return template.format(log_tex, self.latex_param, units_tex)

        if self.log:
            log_tex = "\\log_{10}"
        if self.latex_units is not None:
            units_tex = "\\; [{}]".format(self.latex_units)

        return template.format(log_tex, self.latex_param, units_tex)


# ToDo: Change T/U to eta globally.
# TODO: Remove 'mvir' from f_sub.
# non-derived quantities are by default included.
info_params = {
    # fundamental quantities in the catalog.
    # The key is the actual way name to access from the catalog.
    "id": (None, None, ""),
    "mvir": (None, ("Msun/h", "h^{-1} \\, M_{\\odot}"), "M_{\\rm vir}"),
    "rvir": (None, ("kpc/h", "h^{-1} \\, \\rm kpc"), "R_{\\rm vir}"),
    "rs": (None, ("kpc/h", "h^{-1} \\, \\rm kpc"), "R_{\\rm vir}"),
    "xoff": (None, ("kpc/h", "h^{-1} \\, \\rm kpc"), "X_{\\rm off}"),
    "voff": (None, ("km/s", "\\rm km \\, s^{-1}"), "V_{\\rm off}"),
    "vrms": (None, ("km/s", "\\rm km \\, s^{-1}"), "V_{\\rm rms}"),
    "gamma_tdyn": (
        None,
        ("Msun/h/yr", "h^{-1}\\, yr^{-1} \\, M_{\\odot}"),
        "\\alpha_{\\tau_{\\rm dyn}}",
    ),
    "gamma_inst": (
        None,
        ("Msun/h/yr", "h^{-1}\\, yr^{-1} \\, M_{\\odot}"),
        "\\alpha_{\\rm inst}",
    ),
    "t/|u|": (None, None, "T/|U|"),
    "spin": (None, None, "\\lambda"),
    "scale_of_last_mm": (None, None, "\\delta_{\\rm MM}"),
    # derived quantities.
    "cvir": (
        (lambda cat: cat["rvir"] / cat["rs"], ("rvir", "rs")),
        None,
        "c_{\\rm vir}",
    ),
    "eta": ((lambda cat: 2 * cat["t/|u|"], ("t/|u|",)), None, "\\eta"),
    "q": (
        (lambda cat: (1 / 2) * (cat["b_to_a"] + cat["c_to_a"]), ("b_to_a", "c_to_a")),
        None,
        "q",
    ),
    "phi_l": ((get_phi_l, ("ax", "ay", "az", "jx", "jy", "jz")), None, "\\Phi_{l}"),
    "x0": (
        (lambda cat: cat["xoff"] / cat["rvir"], ("xoff", "rvir")),
        None,
        "x_{\\rm off}",
    ),
    "v0": (
        (lambda cat: cat["voff"] / cat["vrms"], ("voff", "vrms")),
        None,
        "v_{\\rm off}",
    ),
    # 'tdyn': (lambda cat: np.sqrt(2) * cat['rvir'] / cat['vrms'], 'kpc/h / km/s',
    # '', '\\tau_{\\rm dyn}'), (notesheet)
    # will be added later via a more complicated procedure.
    "f_sub": (None, None, "f_{\\rm sub}"),
    "a2": (None, None, "a_{1/2}"),
    "alpha": (None, None, "\\alpha"),
    # usually excluded quantities necessary for filtering
    "upid": (None, None, None),
    "mag2_a": (
        (
            lambda cat: cat["ax"] ** 2 + cat["ay"] ** 2 + cat["ax"] ** 2,
            ("ax", "ay", "az"),
        ),
        None,
        None,
    ),
    "mag2_j": (
        (
            lambda cat: cat["jx"] ** 2 + cat["jy"] ** 2 + cat["jz"] ** 2,
            ("jx", "jy", "jz"),
        ),
        None,
        None,
    ),
}

# nicer format.
params_dict = {
    key: {"derive": value[0], "units": value[1], "latex_param": value[2]}
    for (key, value) in info_params.items()
}

# but can still add units, etc.
param_names = list(params_dict.keys())
params_to_exclude = {"mag2_a", "mag2_j"}
params_add_later = {"a2", "f_sub", "alpha"}
params_to_include = [
    param
    for param in param_names
    if param not in params_to_exclude and param not in params_add_later
]
