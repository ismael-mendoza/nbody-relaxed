import numpy as np
from astropy.table import Column, Table
from abc import ABC, abstractmethod

from relaxed import utils


class HaloParam(ABC):
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

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def latex(self):
        return {}

    @property
    def derive(self):
        return {}


class ID(HaloParam):
    units = ""

    @property
    def name(self):
        return "id"


class Mvir(HaloParam):
    units = "Msun/h"

    @property
    def name(self):
        return "mvir"

    @property
    def latex(self):
        return {"units": "h^{-1} \\, M_{\\odot}", "form": "M_{\\rm vir}"}


class Rvir(HaloParam):
    units = "kpc/h"

    @property
    def name(self):
        return "kpc/h"

    @property
    def latex(self):
        return {"units": "h^{-1} \\, \\rm kpc", "form": "R_{\\rm vir}"}


class Rs(HaloParam):
    units = "kpc/h"

    @property
    def name(self):
        return "rs"

    @property
    def latex(self):
        return {
            "units": "h^{-1} \\, \\rm kpc",
            "form": "R_{s}",
        }


class Xoff(HaloParam):
    units = "kpc/h"

    @property
    def name(self):
        return "xoff"

    @property
    def latex(self):
        return {
            "units": "h^{-1} \\, \\rm kpc",
            "form": "X_{\\rm off}",
        }


class Voff(HaloParam):
    units = "km/s"

    @property
    def name(self):
        return "voff"

    @property
    def latex(self):
        return {
            "units": "h^{-1} \\, \\rm kpc",
            "form": "V_{\\rm off}",
        }


class Vrms(HaloParam):
    units = "km/s"

    @property
    def name(self):
        return "vrms"

    @property
    def latex(self):
        return {
            "units": "\\rm km \\, s^{-1}",
            "form": "V_{\\rm rms}",
        }


class GammaTDyn(HaloParam):
    units = "Msun/h/yr"

    @property
    def name(self):
        return "gamma_tdyn"

    @property
    def latex(self):
        return {
            "units": "h^{-1}\\, yr^{-1} \\, M_{\\odot}",
            "form": "\\alpha_{\\tau_{\\rm dyn}}",
        }


class GammaInst(HaloParam):
    units = "Msun/h/yr"

    @property
    def name(self):
        return "gamma_inst"

    @property
    def latex(self):
        return {
            "units": "h^{-1}\\, yr^{-1} \\, M_{\\odot}",
            "form": "\\alpha_{\\rm inst}",
        }


class ToverU(HaloParam):
    units = ""

    @property
    def name(self):
        return "t/|u|"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "T/|U|",
        }


class Spin(HaloParam):
    units = ""

    @property
    def name(self):
        return "spin"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "\\lambda",
        }


class ScaleOfLastMM(HaloParam):
    units = ""

    @property
    def name(self):
        return "scale_of_last_mm"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "\\delta_{\\rm MM}",
        }


class Cvir(HaloParam):
    units = ""

    @property
    def name(self):
        return "cvir"

    @property
    def latex(self):
        return {
            "units": "c_{\\rm vir}",
            "form": "\\delta_{\\rm MM}",
        }

    @property
    def derive(self):
        return {"func": lambda cat: cat["rvir"] / cat["rs"], "depends": ("rvir", "rs")}


class Eta(HaloParam):
    units = ""

    @property
    def name(self):
        return "cvir"

    @property
    def latex(self):
        return {
            "units": "c_{\\rm vir}",
            "form": "\\delta_{\\rm MM}",
        }

    @property
    def derive(self):
        return {"func": lambda cat: 2 * cat["t/|u|"], "depends": ("t/|u|",)}


class Q(HaloParam):
    units = ""

    @property
    def name(self):
        return "cvir"

    @property
    def latex(self):
        return {
            "units": "c_{\\rm vir}",
            "form": "\\delta_{\\rm MM}",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: (1 / 2) * (cat["b_to_a"] + cat["c_to_a"]),
            "depends": ("b_to_a", "c_to_a"),
        }


class Phi_L(HaloParam):
    units = ""

    @staticmethod
    def get_phi_l(cat):
        """
        * JX/JY/JZ: Halo angular momenta ((Msun/h) * (Mpc/h) * km/s (physical))
        * A[x],A[y],A[z]: Largest shape ellipsoid axis (kpc/h comoving)
        :return: Value of phi_l for each row of the catalog cat.
        :rtype: astropy.Column
        """
        numerator = (
            cat["ax"] * cat["jx"] + cat["ay"] * cat["jy"] + cat["az"] * cat["jz"]
        )
        denominator = np.sqrt(cat["ax"] ** 2 + cat["ay"] ** 2 + cat["az"] ** 2)
        denominator *= np.sqrt(cat["jx"] ** 2 + cat["jy"] ** 2 + cat["jz"] ** 2)
        return np.arccos(numerator / denominator)

    @property
    def name(self):
        return "phi_l"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "\\Phi_{l}",
        }

    @property
    def derive(self):
        return {
            "func": self.get_phi_l,
            "depends": ("ax", "ay", "az", "jx", "jy", "jz"),
        }


class X0(HaloParam):
    units = ""

    @property
    def name(self):
        return "x0"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "x_{\\rm off}",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: cat["xoff"] / cat["rvir"],
            "depends": ("xoff", "rvir"),
        }


class V0(HaloParam):
    units = ""

    @property
    def name(self):
        return "v0"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "v_{\\rm off}",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: cat["voff"] / cat["vrms"],
            "depends": ("voff", "vrms"),
        }


class Fsub(HaloParam):
    units = ""

    @property
    def name(self):
        return "f_sub"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "f_{\\rm sub}",
        }


class A2(HaloParam):
    units = ""

    @property
    def name(self):
        return "a2"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "a_{1/2}",
        }


class UPID(HaloParam):
    units = ""

    @property
    def name(self):
        return "upid"


# nicer format.
params_dict = {
    key: {"derive": value[0], "units": value[1], "latex_param": value[2]}
    for (key, value) in info_params.items()
}
param_names = list(params_dict.keys())


# functions to get derived quantities.


# params_to_exclude = {"mag2_a", "mag2_j"}
# params_add_later = {"a2", "f_sub", "alpha"}
# params_to_include = [
#     param
#     for param in param_names
#     if param not in params_to_exclude and param not in params_add_later
# ]
