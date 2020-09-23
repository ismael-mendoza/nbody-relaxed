import numpy as np
from astropy.table import Table
from abc import ABC, abstractmethod


class HaloParam(ABC):
    def __init__(self, log=False, modifiers=(lambda x: x,)):
        """
        Class implementing a Param object which manages how data is accessed from catalog
        and attributes like its name and its text representation for plots.

        :param log: Whether to log the values when returning them and change the label to
                    indicate that there is a log.
        :param modifiers: Extra modifiers to the values passed in as a list of lambda
                         functions. This will be applied after np.log() if log=True. Should be
                         linear function in number of data points.
        """
        self.log = log
        self.modifiers = modifiers
        self.text = self.get_text()

    def get_values_minh_block(self, mcat, b):
        t = Table(mcat.block(b, self.derive["requires"]), names=self.derive["requires"])
        return self.get_values(t)

    def get_values_minh(self, mcat):
        raise NotImplementedError()

    def get_values(self, cat):
        if self.name in cat.colnames:
            values = cat[self.name]

        else:
            values = self.derive["func"](cat)

        if self.log:
            values = np.log10(values)

        # apply modifiers
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
            return template.format(log_tex, self.latex["form"], units_tex)

        if self.log:
            log_tex = "\\log_{10}"
        if self.latex["units"]:
            units_tex = "\\; [{}]".format(self.latex["units"])

        return template.format(log_tex, self.latex["form"], units_tex)

    @property
    @abstractmethod
    def name(self):
        return ""

    @property
    def latex(self):
        return {"form": "", "units": ""}

    @property
    def derive(self):
        return {
            "func": lambda cat: cat[self.name],
            "requires": (self.name,),
        }


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
        return "rvir"

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


class BToA(HaloParam):
    units = ""

    @property
    def name(self):
        return "b_to_a"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "b/a",
        }


class CToA(HaloParam):
    units = ""

    @property
    def name(self):
        return "c_to_a"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "c/a",
        }


class Cvir(HaloParam):
    units = ""

    @property
    def name(self):
        return "cvir"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "c_{\\rm vir}",
        }

    @property
    def derive(self):
        return {"func": lambda cat: cat["rvir"] / cat["rs"], "requires": ("rvir", "rs")}


class Eta(HaloParam):
    units = ""

    @property
    def name(self):
        return "eta"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "\\eta",
        }

    @property
    def derive(self):
        return {"func": lambda cat: 2 * cat["t/|u|"], "requires": ("t/|u|",)}


class Q(HaloParam):
    units = ""

    @property
    def name(self):
        return "q"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "q",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: (1 / 2) * (cat["b_to_a"] + cat["c_to_a"]),
            "requires": ("b_to_a", "c_to_a"),
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
            "requires": ("ax", "ay", "az", "jx", "jy", "jz"),
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
            "requires": ("xoff", "rvir"),
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
            "requires": ("voff", "vrms"),
        }


class UPID(HaloParam):
    units = ""

    @property
    def name(self):
        return "upid"


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

    def get_values_minh(self, mcat, b=None):
        raise NotImplementedError("Cannot obtain f_sub from minh")


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

    def get_values_minh(self, mcat, b=None):
        raise NotImplementedError("Cannot obtain a2 from minh")


# map from parameter name -> class
hparam_dict = {c().name: c for c in HaloParam.__subclasses__()}


def get_hparam(param, **kwargs):
    return hparam_dict[param](**kwargs)
